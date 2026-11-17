"""Four-GPU projected 32-bit histogram and FWHT for empirical DL scans.

The 2^32-vector is partitioned by its two most significant index bits.  Each
GPU owns one contiguous shard with 2^30 entries.  For N <= 2^30 signed int32
is exact.  Larger supported runs automatically use signed int64 because every
histogram/FWHT intermediate has absolute value at most N.
"""

from __future__ import annotations

import math
import threading
import time


from Framework.common.gpu import _random_words


SHARD_BITS = 30
SHARD_SIZE = 1 << SHARD_BITS
SHARD_MASK = SHARD_SIZE - 1
NUM_SHARDS = 4
INT32_MAX_SAFE_SAMPLES = 1 << 30
MAX_SUPPORTED_SAMPLES = 1 << 34


def _torch():
    import torch
    return torch


def dtype_for_samples(samples, force_int64=False):
    torch = _torch()
    samples = int(samples)
    if samples <= 0:
        raise ValueError("samples must be positive")
    if samples > MAX_SUPPORTED_SAMPLES:
        raise ValueError(f"k=32 backend currently supports N <= 2^34, got {samples}")
    return torch.int64 if force_int64 or samples > INT32_MAX_SAFE_SAMPLES else torch.int32


def allocate_shards(gpus, samples=INT32_MAX_SAFE_SAMPLES, force_int64=False):
    torch = _torch()
    if len(gpus) != NUM_SHARDS:
        raise ValueError("k=32 backend requires exactly four GPUs")
    dtype = dtype_for_samples(samples, force_int64=force_int64)
    element_size = torch.empty((), dtype=dtype).element_size()
    shards = []
    reports = []
    for gpu in gpus:
        torch.cuda.set_device(gpu)
        torch.cuda.reset_peak_memory_stats(gpu)
        shard = torch.zeros(SHARD_SIZE, dtype=dtype, device=f"cuda:{gpu}")
        torch.cuda.synchronize(gpu)
        shards.append(shard)
        reports.append({
            "gpu": int(gpu), "bins": SHARD_SIZE,
            "dtype": str(dtype).replace("torch.", ""),
            "bytes_per_element": element_size,
            "bytes": SHARD_SIZE * element_size,
            "allocated_bytes": int(torch.cuda.memory_allocated(gpu)),
            "reserved_bytes": int(torch.cuda.memory_reserved(gpu)),
        })
    return shards, reports


def _paper_bit_plane(packed, paper_bit, block_size, word_size=None):
    """Return one canonical paper-coordinate bit plane.

    Up to 64 bits, primitives may return one signed int64 tensor.  Wider
    primitives return a tuple of tensors in low-paper-word first order; this
    avoids silently truncating a 96/128-bit state to CUDA int64.
    """
    paper_bit = int(paper_bit)
    if paper_bit < 0 or paper_bit >= int(block_size):
        raise ValueError(f"paper bit {paper_bit} outside [0,{int(block_size)})")
    if isinstance(packed, (tuple, list)):
        if word_size is None:
            if int(block_size) % len(packed):
                raise ValueError("word_size is required for uneven packed state words")
            word_size = int(block_size) // len(packed)
        word, offset = divmod(paper_bit, int(word_size))
        if word >= len(packed):
            raise ValueError("packed state has too few paper-order words")
        return (packed[word] >> offset) & 1
    return (packed >> paper_bit) & 1


def primitive_paper_bit_plane(primitive, packed, paper_bit):
    """Read a named paper coordinate without imposing an engine convention.

    Legacy ARX primitives use the canonical low-bit-first packing handled by
    ``_paper_bit_plane``.  A primitive with a different published convention
    (notably PRESENT's MSB-first state) must provide ``paper_bit_plane``.
    """
    custom = getattr(primitive, "paper_bit_plane", None)
    if custom is not None:
        return custom(packed, int(paper_bit))
    return _paper_bit_plane(
        packed, paper_bit, primitive.config.block_size,
        primitive.config.word_size,
    )


def primitive_paper_bit_matrix(primitive, packed):
    """Unpack all paper coordinates as one vectorized ``[B, block]`` matrix.

    The default contract is the same low-paper-bit-first representation used
    by :func:`primitive_paper_bit_plane`.  Primitives with a different public
    convention (for example PRESENT and KNOT, which number bits MSB first)
    provide ``paper_bit_matrix``.  No Python loop over coordinates or samples
    is used; the only default loop is over the small number of state words.
    """
    torch = _torch()
    custom = getattr(primitive, "paper_bit_matrix", None)
    if custom is not None:
        matrix = custom(packed)
    else:
        block_size = int(primitive.config.block_size)
        word_size = int(primitive.config.word_size)
        tuple_packed = isinstance(packed, (tuple, list))
        words = list(packed) if tuple_packed else [packed]
        columns = []
        remaining = block_size
        for word in words:
            # A scalar tensor is an already-packed complete state even when
            # the cipher's arithmetic word is narrower (e.g. Speck64).
            width = remaining if not tuple_packed else min(word_size, remaining)
            offsets = torch.arange(width, dtype=torch.int64, device=word.device)
            columns.append(((word[:, None] >> offsets[None, :]) & 1).to(torch.uint8))
            remaining -= width
        if remaining != 0:
            raise ValueError(
                f"{primitive.config.name} packed output exposes "
                f"{block_size - remaining}/{block_size} paper bits"
            )
        matrix = torch.cat(columns, dim=1)
    expected = (packed[0].shape[0] if isinstance(packed, (tuple, list))
                else packed.shape[0], int(primitive.config.block_size))
    if tuple(matrix.shape) != expected:
        raise ValueError(
            f"{primitive.config.name} paper bit matrix has shape "
            f"{tuple(matrix.shape)}, expected {expected}"
        )
    return matrix


def project_packed_state(packed, panel_bits, block_size, word_size=None, primitive=None):
    """Project canonical paper-bit coordinates and pack them in panel order."""
    panel = [int(bit) for bit in panel_bits]
    if not 1 <= len(panel) <= 32 or len(set(panel)) != len(panel):
        raise ValueError("GPU projection requires 1..32 distinct panel bits")
    if min(panel) < 0 or max(panel) >= int(block_size):
        raise ValueError(f"panel bits must lie in [0,{int(block_size) - 1}]")
    base = packed[0] if isinstance(packed, (tuple, list)) else packed
    z = base.new_zeros(base.shape, dtype=_torch().int64)
    for position, paper_bit in enumerate(panel):
        plane = (primitive_paper_bit_plane(primitive, packed, paper_bit)
                 if primitive is not None else
                 _paper_bit_plane(packed, paper_bit, block_size, word_size))
        z |= plane.to(_torch().int64) << position
    return z


def _key_widths(config):
    if getattr(config, "key_word_bits", None) is not None:
        return tuple(int(width) for width in config.key_word_bits)
    return (int(config.word_size),) * (int(config.key_size) // int(config.word_size))


def _random_state_and_key(torch, primitive, current, device, generator):
    config = primitive.config
    state = tuple(_random_words(
        torch, current, int(config.state_words), int(config.word_size), device, generator
    ))
    key = tuple(
        _random_words(torch, current, 1, width, device, generator)[0]
        for width in _key_widths(config)
    )
    return state, key


def _generate_z(torch, gpu, current, rounds, difference, generator, primitive,
                panel_bits=None, round_start=0):
    device = torch.device(f"cuda:{gpu}")
    state, keys = _random_state_and_key(torch, primitive, current, device, generator)
    pair_encrypt = getattr(primitive, "encrypt_pair_difference_batch", None)
    if pair_encrypt is not None:
        delta_state = pair_encrypt(
            state, keys, difference, rounds, round_start=round_start
        )
        packed = primitive.output_to_uint(delta_state)
        panel = list(range(int(primitive.config.block_size))) if panel_bits is None else panel_bits
        return project_packed_state(
            packed, panel, primitive.config.block_size, primitive.config.word_size,
            primitive=primitive,
        )
    if getattr(primitive, "supports_round_start", False):
        cx = primitive.encrypt_batch(state, keys, rounds, round_start=round_start)
    elif int(round_start):
        raise ValueError(f"{primitive.config.name} does not support nonzero round_start")
    else:
        cx = primitive.encrypt_batch(state, keys, rounds)
    paired = primitive.apply_difference(state, difference)
    if getattr(primitive, "supports_round_start", False):
        cx2 = primitive.encrypt_batch(paired, keys, rounds, round_start=round_start)
    else:
        cx2 = primitive.encrypt_batch(paired, keys, rounds)
    delta_state = tuple(a ^ b for a, b in zip(cx, cx2))
    packed = primitive.output_to_uint(delta_state)
    panel = list(range(int(primitive.config.block_size))) if panel_bits is None else panel_bits
    return project_packed_state(
        packed, panel, primitive.config.block_size, primitive.config.word_size,
        primitive=primitive,
    )


def distributed_histogram(
    shards, gpus, samples, rounds, difference, seed, batch_size,
    *, primitive, progress=None, panel_bits=None, round_start=0,
):
    """Build four histogram shards with GPU compaction and bounded host relay.

    Only remote segments use pinned-host transfers, including on systems
    without CUDA peer access. No CPU sample filtering is performed.
    """
    torch = _torch()
    from .kernels import allocate_route_workspace, compact_owner_indices, histogram_add_

    if len(gpus) != 4 or len(shards) != 4 or len(set(gpus)) != 4:
        raise ValueError("GPU k32 routing requires four distinct GPUs and four shards")
    if not 1 <= int(batch_size) <= (1 << 24):
        raise ValueError("GPU compact routing requires batch_size in [1, 2^24]")
    samples, batch_size = int(samples), int(batch_size)
    expected_dtype = dtype_for_samples(samples)
    dtype = shards[0].dtype
    if dtype not in (torch.int32, torch.int64):
        raise TypeError("histogram shards must be signed int32 or int64")
    if expected_dtype == torch.int64 and dtype != torch.int64:
        raise ValueError("N > 2^30 requires signed int64 histogram/FWHT shards")
    for gpu, shard in zip(gpus, shards):
        if (shard.dtype != dtype or shard.device != torch.device(f"cuda:{gpu}")
                or shard.ndim != 1 or shard.numel() != SHARD_SIZE
                or not shard.is_contiguous()):
            raise ValueError("histogram shard shape, dtype or device mismatch")
    assignments = [(gpu, samples * rank // 4,
                    samples * (rank + 1) // 4 - samples * rank // 4)
                   for rank, gpu in enumerate(gpus)]
    wall0 = time.perf_counter()
    generators, generation, relay, accumulation_streams = [], [], [], []
    slots, receive = [], []
    for rank, (gpu, begin, count) in enumerate(assignments):
        with torch.cuda.device(gpu):
            # Respect caller-side zeroing before consuming shards on private streams.
            torch.cuda.current_stream(gpu).synchronize()
            gen = torch.Generator(device=f"cuda:{gpu}").manual_seed(
                       (int(seed) + 0x1E3779B97F4A7C15 * (gpu + 1) + begin)
                       & ((1 << 63) - 1))
            generators.append(gen)
            generation.append(torch.cuda.Stream(device=gpu))
            relay.append(torch.cuda.Stream(device=gpu))
            accumulation_streams.append(torch.cuda.Stream(device=gpu))
            capacity = min(batch_size, max(count, 1))
            source_slots = []
            for _ in range(2):
                source_slots.append({
                    "workspace": allocate_route_workspace(capacity, f"cuda:{gpu}"),
                    "host": torch.empty(capacity, dtype=torch.int32, pin_memory=True),
                    "host_counts": torch.empty(4, dtype=torch.int32, pin_memory=True),
                })
            slots.append(source_slots)
            # All producers may route their entire batch to this owner. Reuse
            # a single receive buffer sequentially on the owner stream.
            receive.append(torch.empty(min(batch_size, max(x[2] for x in assignments)),
                                       dtype=torch.int32, device=f"cuda:{gpu}"))

    processed, scheduled = [0] * 4, [0] * 4
    generation_sec = route_cuda_sec = transfer_cuda_sec = accumulation_sec = 0.0
    host_dispatch_sec = 0.0
    remote_samples = local_samples = 0
    batches_done = 0

    def schedule(slot_index):
        pending = [None] * 4
        for rank, (gpu, _, count) in enumerate(assignments):
            remaining = count - scheduled[rank]
            if remaining <= 0:
                continue
            current = min(batch_size, remaining)
            slot = slots[rank][slot_index]
            with torch.cuda.device(gpu), torch.cuda.stream(generation[rank]):
                events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                done = torch.cuda.Event()
                events[0].record()
                z = _generate_z(torch, gpu, current, rounds, difference, generators[rank],
                                primitive, panel_bits, round_start).to(torch.int32)
                events[1].record()
                packed, counts = compact_owner_indices(z, slot["workspace"])
                events[2].record()
                slot["host_counts"].copy_(counts, non_blocking=True)
                done.record()
            pending[rank] = {"slot": slot, "packed": packed, "current": current,
                             "events": events, "done": done, "z": z}
            scheduled[rank] += current
        return pending if any(x is not None for x in pending) else None

    pending = schedule(0)
    slot_index = 1
    while pending is not None:
        # No workspace or pinned buffer is overwritten until all consumers of
        # its previous use have finished at the end of the preceding loop.
        following = schedule(slot_index)
        slot_index ^= 1
        gen_times, route_times = [], []
        payload_events = []
        dispatch_start = time.perf_counter()
        for rank, item in enumerate(pending):
            if item is None:
                continue
            item["done"].synchronize()
            events = item["events"]
            gen_times.append(events[0].elapsed_time(events[1]) / 1000)
            route_times.append(events[1].elapsed_time(events[2]) / 1000)
            counts = item["slot"]["host_counts"].tolist()
            if min(counts) < 0 or sum(counts) != item["current"]:
                raise AssertionError(f"GPU routing count mismatch: {counts}")
            offsets = [0]
            for count in counts:
                offsets.append(offsets[-1] + count)
            item["offsets"] = offsets
            local_samples += counts[rank]
            remote_samples += item["current"] - counts[rank]
            gpu = gpus[rank]
            # Remote segments are the two ranges around this producer's
            # local-owner range. Copy no local samples to host.
            with torch.cuda.device(gpu), torch.cuda.stream(relay[rank]):
                started, ended = (torch.cuda.Event(enable_timing=True),
                                  torch.cuda.Event(enable_timing=True))
                started.record()
                for lo, hi in ((0, offsets[rank]), (offsets[rank + 1], item["current"])):
                    if hi > lo:
                        item["slot"]["host"][lo:hi].copy_(item["packed"][lo:hi],
                                                          non_blocking=True)
                ended.record()
                payload_events.append((started, ended))
            with torch.cuda.device(gpu), torch.cuda.stream(accumulation_streams[rank]):
                histogram_add_(shards[rank], item["packed"][offsets[rank]:offsets[rank + 1]])
        host_dispatch_sec += time.perf_counter() - dispatch_start
        generation_sec += max(gen_times, default=0.0)
        route_cuda_sec += max(route_times, default=0.0)
        for _, ended in payload_events:
            ended.synchronize()
        transfer_cuda_sec += max((a.elapsed_time(b) / 1000 for a, b in payload_events), default=0.0)
        accumulation_start = time.perf_counter()
        for owner, gpu in enumerate(gpus):
            with torch.cuda.device(gpu), torch.cuda.stream(accumulation_streams[owner]):
                for source_rank, item in enumerate(pending):
                    if item is None or source_rank == owner:
                        continue
                    lo, hi = item["offsets"][owner:owner + 2]
                    if hi == lo:
                        continue
                    target = receive[owner][:hi - lo]
                    target.copy_(item["slot"]["host"][lo:hi], non_blocking=True)
                    histogram_add_(shards[owner], target)
        for stream in accumulation_streams:
            stream.synchronize()
        accumulation_sec += time.perf_counter() - accumulation_start
        for rank, item in enumerate(pending):
            if item is not None:
                processed[rank] += item["current"]
        batches_done += 1
        if progress and (batches_done == 1 or batches_done % 16 == 0):
            progress(sum(processed), samples, time.perf_counter() - wall0)
        pending = following
    if progress:
        progress(sum(processed), samples, time.perf_counter() - wall0)

    totals = []
    for shard in shards:
        total = 0
        for start in range(0, SHARD_SIZE, 1 << 24):
            total += int(shard[start:start + (1 << 24)].sum(dtype=torch.int64).item())
        totals.append(total)
    if sum(totals) != samples or local_samples + remote_samples != samples:
        raise AssertionError(f"histogram count mismatch: {totals} != {samples}")
    return {
        "assignments": [{"gpu": int(gpu), "start": int(begin), "samples": int(count),
                         "seed": (int(seed) + 0x1E3779B97F4A7C15 * (gpu + 1) + begin)
                                 & ((1 << 63) - 1)} for gpu, begin, count in assignments],
        "owner_counts": totals,
        "dtype": str(dtype).replace("torch.", ""),
        "integer_safety": f"all |partial/final Walsh| <= N={samples}",
        "histogram_pipeline_mode": "double_buffer_gpu_compact_host_relay_v1",
        "sample_generation_cuda_sec": generation_sec,
        "histogram_routing_cuda_sec": route_cuda_sec,
        "histogram_host_dispatch_and_wait_sec": host_dispatch_sec,
        "histogram_remote_d2h_cuda_sec": transfer_cuda_sec,
        "histogram_accumulation_sec": accumulation_sec,
        "histogram_total_sec": time.perf_counter() - wall0,
        "timing_note": "stage times overlap; histogram_total_sec is authoritative wall time",
        "local_samples": local_samples, "remote_samples": remote_samples,
        "remote_d2h_bytes": 4 * remote_samples, "remote_h2d_bytes": 4 * remote_samples,
    }


def _local_fwht_worker(shard, gpu, chunk_elements, report):
    torch = _torch()
    torch.cuda.set_device(gpu)
    start = time.perf_counter()
    width = 1
    stage = 0
    while width < SHARD_SIZE:
        pair_span = width * 2
        if pair_span <= chunk_elements:
            # Thousands/millions of small butterflies must be issued as one
            # vectorized reshape operation, never as Python loops per block.
            for start_index in range(0, SHARD_SIZE, chunk_elements):
                count = min(chunk_elements, SHARD_SIZE - start_index)
                count -= count % pair_span
                blocks = shard[start_index:start_index + count].view(-1, pair_span)
                left = blocks[:, :width].clone()
                right = blocks[:, width:].clone()
                blocks[:, :width] = left + right
                blocks[:, width:] = left - right
        else:
            # For the last large stages, split corresponding halves into
            # bounded slices so temporary storage remains far below 12 GiB.
            for base in range(0, SHARD_SIZE, pair_span):
                for offset in range(0, width, chunk_elements):
                    count = min(chunk_elements, width - offset)
                    lo = base + offset
                    hi = base + width + offset
                    left = shard[lo:lo + count].clone()
                    right = shard[hi:hi + count].clone()
                    shard[lo:lo + count] = left + right
                    shard[hi:hi + count] = left - right
        width <<= 1
        stage += 1
        if report.get("progress"):
            report["progress"](stage, gpu)
    torch.cuda.synchronize(gpu)
    report[gpu] = time.perf_counter() - start


def _parallel_workers(worker, arguments):
    """Propagate worker failures before a partial transform can be consumed."""
    errors = []
    def run(args):
        try:
            worker(*args)
        except BaseException as exc:
            errors.append(exc)
    threads = [threading.Thread(target=run, args=(args,)) for args in arguments]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise RuntimeError(f"Distributed FWHT worker failed: {errors[0]}") from errors[0]


def local_fwht(shards, gpus, chunk_elements=1 << 24, progress=None):
    report = {"progress": progress}
    wall0 = time.perf_counter()
    _parallel_workers(_local_fwht_worker,
        [(shard, gpu, chunk_elements, report) for shard, gpu in zip(shards, gpus)])
    return {"wall_sec": time.perf_counter() - wall0,
            "per_gpu_sec": {str(gpu): report[gpu] for gpu in gpus}}


def _cross_pair(a, b, gpu_a, gpu_b, chunk_elements):
    torch = _torch()
    # Bounded pinned-memory transport also supports hosts without peer access.
    for start in range(0, SHARD_SIZE, chunk_elements):
        count = min(chunk_elements, SHARD_SIZE - start)
        if a.dtype != b.dtype:
            raise TypeError("cross-FWHT shard dtypes must match")
        host_a = torch.empty(count, dtype=a.dtype, pin_memory=True)
        host_b = torch.empty(count, dtype=b.dtype, pin_memory=True)
        host_a.copy_(a[start:start + count])
        host_b.copy_(b[start:start + count])
        original_b_on_a = host_b.to(f"cuda:{gpu_a}", non_blocking=False)
        original_a_on_b = host_a.to(f"cuda:{gpu_b}", non_blocking=False)
        a[start:start + count].add_(original_b_on_a)
        torch.sub(original_a_on_b, b[start:start + count], out=b[start:start + count])
    torch.cuda.synchronize(gpu_a)
    torch.cuda.synchronize(gpu_b)


def cross_fwht_stage(shards, gpus, pairs, chunk_elements=1 << 24):
    wall0 = time.perf_counter()
    _parallel_workers(_cross_pair,
        [(shards[left], shards[right], gpus[left], gpus[right], chunk_elements)
         for left, right in pairs])
    return time.perf_counter() - wall0


def _popcount32(indices):
    """Vectorized Hamming weight of nonnegative 32-bit mask indices."""
    x = indices - ((indices >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    return ((x * 0x01010101) >> 24) & 0xFF


_PER_HW_SCAN_CHUNK = 1 << 22


def _paper_mask_lex_keys(masks, panel_bits):
    """Larger keys mean earlier canonical paper-bit tuples at a fixed HW."""
    keys = masks.new_zeros(masks.shape)
    positions = sorted(range(len(panel_bits)), key=panel_bits.__getitem__)
    for rank, position in enumerate(positions):
        keys |= ((masks >> position) & 1) << (len(positions) - 1 - rank)
    return keys


def _stable_topk_abs_masks(masks, coefficients, count, panel_bits=None):
    """Select by descending |Walsh| with a deterministic boundary tie rule.

    Only the retained candidates are sorted.  The spectrum-sized input uses
    top-k to find the cutoff; tied masks have distinct integer values, so a
    second top-k on their tie keys makes the cutoff deterministic as well.
    A supplied panel maps fixed-HW ties to canonical paper-bit tuple order.
    """
    torch = _torch()
    count = min(int(count), int(masks.numel()))
    if count <= 0:
        return masks[:0], coefficients[:0]
    absolute = coefficients.abs()
    if count < masks.numel():
        cutoff = torch.topk(absolute, k=count, sorted=False).values.min()
        stronger = torch.nonzero(absolute > cutoff, as_tuple=True)[0]
        ties = torch.nonzero(absolute == cutoff, as_tuple=True)[0]
        remaining = count - int(stronger.numel())
        tie_keys = (masks[ties] if panel_bits is None else
                    _paper_mask_lex_keys(masks[ties], panel_bits))
        tie_order = torch.topk(tie_keys, k=remaining, largest=panel_bits is not None,
                               sorted=False).indices
        positions = torch.cat((stronger, ties[tie_order]))
        masks, coefficients = masks[positions], coefficients[positions]
    # Stable sorting of at most k retained entries gives the same ordering
    # across chunk boundaries and CPU/CUDA implementations.
    order = (torch.argsort(masks) if panel_bits is None else
             torch.argsort(_paper_mask_lex_keys(masks, panel_bits), descending=True))
    order = order[torch.argsort(coefficients[order].abs(), descending=True,
                               stable=True)]
    return masks[order], coefficients[order]


def extract_top_per_hw(shards, gpus, samples, per_hw_k=1000,
                       hw_min=1, hw_max=8, rho_threshold=None, panel_bits=None):
    """Exact per-HW top-k for a small single vector or 2^30-entry shards.

    There is no global cap.  Zero masks and zero coefficients are excluded;
    ties use canonical paper-bit tuples when panel_bits is supplied, otherwise
    integer masks for compatibility.  Both are independent of chunk boundaries.
    """
    wall0 = time.perf_counter()
    if not (1 <= int(hw_min) <= int(hw_max) <= 32):
        raise ValueError("per-HW extraction requires 1 <= hw_min <= hw_max <= 32")
    if int(per_hw_k) < 0:
        raise ValueError("per_hw_k must be nonnegative")
    if int(samples) <= 0:
        raise ValueError("samples must be positive")
    if len(shards) != len(gpus):
        raise ValueError("one device label is required per shard")
    if panel_bits is not None:
        panel_bits = [int(bit) for bit in panel_bits]
        if (not 1 <= len(panel_bits) <= 32 or len(set(panel_bits)) != len(panel_bits)
                or min(panel_bits) < 0):
            raise ValueError("panel_bits must contain 1..32 distinct nonnegative coordinates")
        if any(((owner << SHARD_BITS) + int(shard.numel())) > 1 << len(panel_bits)
               for owner, shard in enumerate(shards)):
            raise ValueError("spectrum masks exceed the supplied panel dimension")
    tie_metadata = {
        "tie_break": ("paper_bits_lexicographic" if panel_bits is not None
                      else "mask_integer_ascending"),
        "panel_bits": panel_bits,
    }
    if rho_threshold is not None and (not math.isfinite(float(rho_threshold))
                                      or not 0 <= float(rho_threshold) <= 1):
        raise ValueError("rho_threshold must be finite and in [0,1]")
    # The paper and the shared final selector require |rho| > tau.  Taking
    # ceil(tau*N) with >= would include the equality boundary when tau*N is
    # an integer, and report threshold hits that the final selector rejects.
    threshold = (None if rho_threshold is None else
                 int(math.floor(float(rho_threshold) * int(samples))) + 1)
    if int(per_hw_k) == 0:
        return [], time.perf_counter() - wall0, {
            "selection_policy": "per_hw_topk_no_global_cap_v2",
            "threshold_comparison": "strictly_greater",
            "zero_mask_excluded": True,
            "rho_threshold": rho_threshold, "walsh_threshold": threshold,
            "num_above_threshold": None, "per_hw_limit": 0,
            "hw_min": int(hw_min), "hw_max": int(hw_max),
            "num_selected_per_hw": {str(hw): 0 for hw in range(int(hw_min), int(hw_max) + 1)},
            **tie_metadata,
        }
    torch = _torch()
    # Keep only k candidates per weight and GPU on-device.  The complete
    # spectrum is inspected, but never copied to host memory.
    winners = {hw: [] for hw in range(int(hw_min), int(hw_max) + 1)}
    num_above_threshold = 0
    scan_chunk = _PER_HW_SCAN_CHUNK
    print("[stage] Em per-HW mask extraction started", flush=True)
    for owner, (gpu, shard) in enumerate(zip(gpus, shards)):
        if shard.device.type == "cuda":
            torch.cuda.set_device(gpu)
        best = {hw: None for hw in winners}
        for start in range(0, shard.numel(), scan_chunk):
            if start % (64 * scan_chunk) == 0:
                print(
                    f"\r[progress] Em per-HW mask extraction: "
                    f"{100.0 * (owner + start / max(1, shard.numel())) / len(shards):5.1f}% "
                    f"(shard {owner + 1}/{len(shards)})",
                    end="", flush=True,
                )
            part = shard[start:start + scan_chunk]
            absolute = part.abs()
            if int(absolute.max().item()) > int(samples):
                raise ArithmeticError("FWHT coefficient magnitude exceeds the sample count")
            if owner == 0 and start == 0:
                absolute[0] = -1  # Exclude the zero mask.
            if threshold is not None:
                num_above_threshold += int(
                    ((absolute > 0) & (absolute >= threshold)).sum(dtype=torch.int64).item())
            masks = torch.arange(start, start + part.numel(),
                                 dtype=torch.int64, device=part.device) + (owner << SHARD_BITS)
            hw_values = _popcount32(masks)
            for hw in winners:
                eligible = torch.nonzero((hw_values == hw) & (masks != 0)
                                         & (absolute > 0), as_tuple=True)[0]
                if threshold is not None and eligible.numel():
                    eligible = eligible[absolute[eligible] >= threshold]
                if not eligible.numel():
                    continue
                new_masks, new_coefficients = _stable_topk_abs_masks(
                    masks[eligible], part[eligible], per_hw_k, panel_bits)
                previous = best[hw]
                if previous is not None:
                    new_masks = torch.cat((previous[0], new_masks))
                    new_coefficients = torch.cat((previous[1], new_coefficients))
                best[hw] = _stable_topk_abs_masks(
                    new_masks, new_coefficients, per_hw_k, panel_bits)
        for hw, pair in best.items():
            if pair is not None:
                winners[hw].extend(zip(pair[0].cpu().tolist(), pair[1].cpu().tolist()))
        print(
            f"\r[progress] Em per-HW mask extraction: {100.0 * (owner + 1) / len(shards):5.1f}% "
            f"(shard {owner + 1}/{len(shards)})",
            end="", flush=True,
        )
    print("\r[stage] Em per-HW mask extraction completed: 100.0%", flush=True)
    selected = []
    counts = {}

    def mask_tie_key(mask):
        if panel_bits is None:
            return mask
        return tuple(sorted(bit for position, bit in enumerate(panel_bits)
                            if (mask >> position) & 1))

    for hw, pairs in winners.items():
        pairs.sort(key=lambda pair: (-abs(pair[1]), mask_tie_key(pair[0])))
        kept = pairs[:int(per_hw_k)]
        counts[str(hw)] = len(kept)
        selected.extend((mask, coefficient, hw) for mask, coefficient in kept)
    selected.sort(key=lambda row: (-abs(row[1]), mask_tie_key(row[0])))
    rows = []
    sqrt_n = math.sqrt(samples)
    for rank, (mask, coefficient, hw) in enumerate(selected, 1):
        rho = coefficient / float(samples)
        rows.append({
            "rank": rank, "Gamma_m": mask,
            "Gamma_m_hex": f"0x{mask:08x}",
            "Gamma_m_paper_bits": (list(mask_tie_key(mask)) if panel_bits is not None
                                   else [bit for bit in range(32) if (mask >> bit) & 1]),
            "HW": hw, "Walsh": coefficient, "rho": rho,
            "abs_rho_sqrt_N": abs(rho) * sqrt_n,
            "w_Em": -math.log2(abs(rho)) if coefficient else math.inf,
        })
    return rows, time.perf_counter() - wall0, {
        "selection_policy": "per_hw_topk_no_global_cap_v2",
        "threshold_comparison": "strictly_greater",
        "zero_mask_excluded": True,
        "rho_threshold": rho_threshold,
        "walsh_threshold": threshold,
        "num_above_threshold": num_above_threshold if threshold is not None else None,
        "per_hw_limit": int(per_hw_k),
        "hw_min": int(hw_min), "hw_max": int(hw_max),
        "num_selected_per_hw": counts,
        **tie_metadata,
    }


