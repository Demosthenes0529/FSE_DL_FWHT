"""Bounded-memory GPU routing for the four-owner projected k=32 histogram.

``compact_owner_indices`` treats each signed-int32 input as a uint32 bit
pattern, selects its owner from bits 31..30, and emits its low 30 bits.  The
output contains four consecutive, stable owner segments.  The only routing
metadata the caller needs to copy to the host is four int32 segment lengths.
There are no CPU sample scans, host transfers, or synchronizations here.

Routing uses three GPU passes: per-block owner counts, an exclusive prefix
over those counts, and stable scatter.  No sample reserves an output slot
through a contended global owner counter.  An independently allocated
workspace is required for each concurrently active batch/stream slot.

``histogram_add_`` increments an owner histogram directly from compacted
int32 indices, without a sample-sized int64 cast or tensor of ones.  Histogram
storage may be signed int32 or int64; callers remain responsible for the
overall integer-count bound (the N=2**34 experiment requires int64).
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


NUM_OWNERS = 4
SHARD_BITS = 30
SHARD_SIZE = 1 << SHARD_BITS
SHARD_MASK = SHARD_SIZE - 1

# Prefixing the tiny [4, blocks] table in four CTAs is sufficient for the
# intended 2**20-sample batches.  Bound the scan's compile-time vector size
# explicitly instead of silently creating an enormous single-CTA prefix.
MAX_PREFIX_BLOCKS = 16384


@triton.jit
def _count_owners_kernel(
    Z, BlockCounts,
    N: tl.constexpr, STRIDE: tl.constexpr, BLOCK: tl.constexpr,
):
    block = tl.program_id(0)
    positions = block.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = positions < N
    values = tl.load(Z + positions, mask=valid, other=0).to(tl.uint32)
    owners = values >> 30
    for owner in tl.static_range(4):
        count = tl.sum((valid & (owners == owner)).to(tl.int32), axis=0)
        tl.store(BlockCounts + owner * STRIDE + block, count)


@triton.jit
def _prefix_owners_kernel(
    BlockCounts, BlockOffsets, Counts,
    BLOCKS: tl.constexpr, STRIDE: tl.constexpr, PREFIX: tl.constexpr,
):
    owner = tl.program_id(0)
    blocks = tl.arange(0, PREFIX)
    active = blocks < BLOCKS
    counts = tl.load(BlockCounts + owner * STRIDE + blocks,
                     mask=active, other=0)
    inclusive = tl.cumsum(counts, axis=0)
    tl.store(BlockOffsets + owner * STRIDE + blocks,
             inclusive - counts, mask=active)
    tl.store(Counts + owner, tl.sum(counts, axis=0))


@triton.jit
def _scatter_owners_kernel(
    Z, Packed, BlockOffsets, Counts,
    N: tl.constexpr, STRIDE: tl.constexpr, BLOCK: tl.constexpr,
):
    block = tl.program_id(0)
    positions = block.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = positions < N
    values = tl.load(Z + positions, mask=valid, other=0).to(tl.uint32)
    owners = values >> 30
    local_indices = (values & 0x3fffffff).to(tl.int32)
    owner_base = tl.full((), 0, tl.int64)
    for owner in tl.static_range(4):
        selected = valid & (owners == owner)
        local_rank = tl.cumsum(selected.to(tl.int32), axis=0) - 1
        block_offset = tl.load(BlockOffsets + owner * STRIDE + block)
        output_position = owner_base + block_offset.to(tl.int64) + local_rank
        tl.store(Packed + output_position, local_indices, mask=selected)
        owner_base += tl.load(Counts + owner).to(tl.int64)


@triton.jit(do_not_specialize=["N"])
def _histogram_add_kernel(
    Histogram, Indices,
    N, BLOCK: tl.constexpr,
):
    # Owner segment lengths change on every batch.  Keep N runtime-valued
    # so those lengths do not trigger one JIT compilation per segment.
    positions = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = positions < N
    # Promote *before* pointer arithmetic: an int64 histogram spans 8 GiB,
    # even though every valid local bin fits in a nonnegative signed int32.
    local_indices = tl.load(Indices + positions, mask=valid, other=0).to(tl.int64)
    one = tl.full((), 1, Histogram.dtype.element_ty)
    tl.atomic_add(Histogram + local_indices, one, mask=valid, sem="relaxed")


@dataclass(frozen=True)
class RouteWorkspace:
    """Reusable buffers owned by one routing slot on one CUDA device.

    ``packed`` has ``capacity`` int32 entries; ``block_counts`` and
    ``block_offsets`` each have shape ``[4, ceil(capacity / block_size)]``.
    The returned segments/counts alias these buffers and are overwritten by
    the next use.  Retain the workspace until all dependent asynchronous
    operations, including any D2H copies, have finished.
    """

    capacity: int
    block_size: int
    packed: torch.Tensor
    block_counts: torch.Tensor
    block_offsets: torch.Tensor
    counts: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.packed.device

    @property
    def scratch_bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in (
            self.packed, self.block_counts, self.block_offsets, self.counts,
        ))


def _check_block_size(block_size: int) -> int:
    block_size = int(block_size)
    if block_size < 128 or block_size > 2048 or block_size & (block_size - 1):
        raise ValueError("block_size must be a power of two in [128, 2048]")
    return block_size


def _check_cuda_vector(tensor: torch.Tensor, name: str, dtypes) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype not in dtypes:
        raise TypeError(f"{name} has unsupported dtype {tensor.dtype}")
    if tensor.ndim != 1 or not tensor.is_contiguous():
        raise ValueError(f"{name} must be a contiguous one-dimensional tensor")
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must reside on a CUDA device")


def allocate_route_workspace(
    capacity: int,
    device,
    *,
    block_size: int = 1024,
) -> RouteWorkspace:
    """Allocate routing scratch without touching samples or synchronizing.

    Integer ``device`` values are CUDA ordinals; strings and torch.device
    values must name CUDA.  Capacity can be zero and may not exceed
    ``block_size * MAX_PREFIX_BLOCKS``.  With the defaults, a 2**20-sample
    workspace occupies 4 MiB + 32 KiB + 16 bytes.  No histogram is allocated.
    """
    capacity = int(capacity)
    block_size = _check_block_size(block_size)
    if capacity < 0 or capacity > block_size * MAX_PREFIX_BLOCKS:
        raise ValueError(
            f"capacity must be in [0, {block_size * MAX_PREFIX_BLOCKS}] "
            "for the selected block_size"
        )
    device = torch.device(f"cuda:{device}" if isinstance(device, int) else device)
    if device.type != "cuda":
        raise ValueError("routing workspaces must reside on a CUDA device")
    blocks = triton.cdiv(capacity, block_size)
    with torch.cuda.device(device):
        packed = torch.empty(capacity, dtype=torch.int32, device=device)
        block_counts = torch.empty((NUM_OWNERS, blocks), dtype=torch.int32, device=device)
        block_offsets = torch.empty_like(block_counts)
        counts = torch.empty(NUM_OWNERS, dtype=torch.int32, device=device)
    return RouteWorkspace(capacity, block_size, packed, block_counts, block_offsets, counts)


def compact_owner_indices(
    z: torch.Tensor,
    workspace: Optional[RouteWorkspace] = None,
    *,
    block_size: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(packed_local_indices, owner_counts)`` on the input GPU.

    Input ``z`` must be contiguous 1-D torch.int32.  Negative values encode
    uint32 patterns with bit 31 set and are routed to owner 2 or 3.  Counts
    are four int32 values, whose sum equals ``z.numel()``; owner ``r`` occupies
    ``packed[sum(counts[:r]):sum(counts[:r+1])]``.  Within each segment, sample
    order is unchanged.  Empty input produces four zero counts.

    Pass a reusable workspace to exclude allocation from timing.  A smaller
    final batch reuses the same buffers; stale block counts are not included.
    When a workspace is supplied its block size takes precedence.  Calls use
    the current CUDA stream of z's device and do not inspect CUDA values on
    the CPU.  Cross-stream consumers need explicit event dependencies.
    """
    _check_cuda_vector(z, "z", (torch.int32,))
    count = int(z.numel())
    if workspace is None:
        workspace = allocate_route_workspace(count, z.device, block_size=block_size)
    if not isinstance(workspace, RouteWorkspace):
        raise TypeError("workspace must be a RouteWorkspace")
    if workspace.device != z.device:
        raise ValueError("z and workspace must reside on the same CUDA device")
    if count > workspace.capacity:
        raise ValueError("z exceeds the routing workspace capacity")
    # Scatter is not in-place: reject any overlap, including sliced views.
    if count:
        input_start = z.data_ptr()
        output_start = workspace.packed.data_ptr()
        size_bytes = count * z.element_size()
        if input_start < output_start + size_bytes and output_start < input_start + size_bytes:
            raise ValueError("z must not overlap the workspace packed output")
    block_size = workspace.block_size
    blocks = triton.cdiv(count, block_size)
    stride = workspace.block_counts.shape[1]
    with torch.cuda.device(z.device):
        if not count:
            workspace.counts.zero_()
        else:
            _count_owners_kernel[(blocks,)](
                z, workspace.block_counts, count, stride, block_size, num_warps=4,
            )
            _prefix_owners_kernel[(NUM_OWNERS,)](
                workspace.block_counts, workspace.block_offsets, workspace.counts,
                blocks, stride, triton.next_power_of_2(blocks), num_warps=4,
            )
            _scatter_owners_kernel[(blocks,)](
                z, workspace.packed, workspace.block_offsets, workspace.counts,
                count, stride, block_size, num_warps=4,
            )
    return workspace.packed[:count], workspace.counts


def histogram_add_(
    histogram: torch.Tensor,
    local_indices: torch.Tensor,
    *,
    block_size: int = 256,
) -> torch.Tensor:
    """Add one for each local int32 index, asynchronously and in place.

    Histogram must be contiguous 1-D int32/int64 on the same GPU as indices.
    A production shard has 2**30 bins.  Smaller histograms are permitted for
    tests, provided every supplied index is in ``[0, histogram.numel())``.
    This precondition is intentionally not verified by reading samples on
    the CPU; compacted indices are always valid for a full-size owner shard.
    The histogram must not alias the index buffer and must be initialized by
    the caller.  The caller also owns stream ordering and overflow policy.
    """
    _check_cuda_vector(histogram, "histogram", (torch.int32, torch.int64))
    _check_cuda_vector(local_indices, "local_indices", (torch.int32,))
    if histogram.device != local_indices.device:
        raise ValueError("histogram and local_indices must reside on the same CUDA device")
    if not 1 <= histogram.numel() <= SHARD_SIZE:
        raise ValueError("histogram must contain between 1 and 2**30 bins")
    block_size = _check_block_size(block_size)
    count = int(local_indices.numel())
    if count:
        histogram_start = histogram.data_ptr()
        histogram_end = histogram_start + histogram.numel() * histogram.element_size()
        indices_start = local_indices.data_ptr()
        indices_end = indices_start + count * local_indices.element_size()
        if histogram_start < indices_end and indices_start < histogram_end:
            raise ValueError("histogram must not overlap local_indices")
        with torch.cuda.device(histogram.device):
            _histogram_add_kernel[(triton.cdiv(count, block_size),)](
                histogram, local_indices, count, block_size, num_warps=4,
            )
    return histogram
