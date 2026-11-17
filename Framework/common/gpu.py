"""Shared tensor cipher configuration, device selection, random words and FWHT."""
import os

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class CipherGPUConfig:
    name: str
    block_size: int
    word_size: int
    key_size: int
    state_words: int
    # Most ARX ciphers use equally-sized key words.  SPNs such as
    # PRESENT-80 do not: its canonical tensor key is (hi16, lo64).  Keeping
    # the true widths here prevents a nominal 80-bit key from being silently
    # represented as (and sampled like) a 128-bit key.
    key_word_bits: Optional[Tuple[int, ...]] = None


def validate_primitive(primitive) -> None:
    config = primitive.config
    if config.block_size < 32:
        raise ValueError(
            "The distributed k=32 engine requires a state of at least 32 bits"
        )
    if not 1 <= config.word_size <= 64 or config.state_words <= 0:
        raise ValueError("invalid cipher primitive configuration")
    if config.word_size * config.state_words != config.block_size:
        raise ValueError("word_size * state_words must equal block_size")
    if config.key_size < 0:
        raise ValueError("key_size must be nonnegative")
    widths = config.key_word_bits
    if config.key_size == 0:
        if widths not in (None, ()):
            raise ValueError("unkeyed primitives must use key_word_bits=() or None")
    elif widths is None:
        if config.key_size % config.word_size:
            raise ValueError("key_size must be a multiple of word_size unless key_word_bits is set")
    else:
        if not widths or any(int(width) <= 0 or int(width) > 64 for width in widths):
            raise ValueError("key_word_bits must contain widths in 1..64")
        if sum(int(width) for width in widths) != int(config.key_size):
            raise ValueError("sum(key_word_bits) must equal key_size")

def disabled_gpu_ids():
    raw = os.environ.get("DL_FWHT_DISABLED_GPUS", "")
    devices = {int(value.strip()) for value in raw.split(",") if value.strip()}
    if any(device < 0 for device in devices):
        raise ValueError("disabled GPU IDs must be nonnegative CUDA-visible indices")
    return devices


def resolve_gpu_ids(spec, required_count=None):
    """Resolve logical CUDA ordinals after CUDA_VISIBLE_DEVICES remapping."""
    requested = []
    for value in str(spec).split(","):
        if value.strip():
            device = int(value)
            if device < 0:
                raise ValueError("GPU IDs must be nonnegative CUDA-visible indices")
            if device not in requested:
                requested.append(device)
    disabled = disabled_gpu_ids()
    selected = [device for device in requested if device not in disabled]

    target = int(required_count) if required_count is not None else None
    if target is not None and target < 1:
        raise ValueError("required GPU count must be positive")
    import torch
    visible = torch.cuda.device_count()
    if not torch.cuda.is_available() or visible < 1:
        raise RuntimeError("CUDA is unavailable; GPU indices refer to devices visible through CUDA_VISIBLE_DEVICES")
    invalid = [device for device in selected if device >= visible]
    if invalid:
        raise ValueError(f"GPU IDs {invalid} are outside CUDA-visible indices 0..{visible - 1}")
    if target is not None and len(selected) < target:
        for device in range(visible):
            if device not in disabled and device not in selected:
                selected.append(device)
                if len(selected) == target:
                    break
    if target is not None and len(selected) != target:
        raise ValueError(
            f"CUDA backend requires {target} usable GPUs; requested={requested}, "
            f"disabled={sorted(disabled)}, resolved={selected}"
        )
    if not selected:
        raise ValueError(
            f"no usable CUDA GPU remains; requested={requested}, "
            f"disabled={sorted(disabled)}"
        )
    excluded = [device for device in requested if device in disabled]
    if excluded:
        print(
            f"[GPU] excluded unstable/disabled devices {excluded}; using {selected}",
            flush=True,
        )
    return selected


def fwht_inplace_int64(values, progress=None):
    """Transform a contiguous power-of-two int64 vector in place.

    The small-panel CUDA scan and CPU tensor checks call this same butterfly.
    Histogram/FWHT counts have magnitude at most the sample count; callers
    must keep that count within the signed int64 range. Large k=32 shards use
    their separate bounded-memory transform in fwht.distributed.
    """
    import torch
    if values.dtype != torch.int64:
        raise TypeError("FWHT requires int64 counts")
    size = int(values.numel())
    if values.ndim != 1 or not values.is_contiguous() or size < 1 or size & (size - 1):
        raise ValueError("FWHT requires a contiguous power-of-two vector")
    stages = size.bit_length() - 1
    width = 1
    for stage in range(1, stages + 1):
        blocks = values.view(-1, 2, width)
        left = blocks[:, 0, :].clone()
        right = blocks[:, 1, :]
        blocks[:, 0, :].add_(right)
        blocks[:, 1, :].copy_(left - right)
        width <<= 1
        if progress is not None:
            progress(stage, stages)
    if values.is_cuda:
        torch.cuda.synchronize(values.device)
    return values


def _random_words(torch, current, count, word_size, device, generator):
    if not 1 <= int(word_size) <= 64:
        raise ValueError("random word width must be in 1..64")
    if int(word_size) < 63:
        return [torch.randint(0, 1 << int(word_size), (current,), dtype=torch.int64, device=device, generator=generator) for _ in range(count)]
    result = []
    for _ in range(count):
        low = torch.randint(0, 1 << 32, (current,), dtype=torch.int64, device=device, generator=generator)
        high = torch.randint(0, 1 << (int(word_size) - 32), (current,), dtype=torch.int64, device=device, generator=generator)
        result.append((high << 32) | low)
    return result


