"""Fused Triton kernels for PRESENT-64 encryption and pair differences."""

import torch
import triton
import triton.language as tl


@triton.jit
def _sbox(x):
    m = 0x1111111111111111
    x0, x1 = x & m, (x >> 1) & m
    x2, x3 = (x >> 2) & m, (x >> 3) & m
    x01, x02, x12 = x0 & x1, x0 & x2, x1 & x2
    x03, x13, x23 = x0 & x3, x1 & x3, x2 & x3
    x012, x013, x023 = x01 & x2, x01 & x3, x02 & x3
    y0 = x0 ^ x2 ^ x12 ^ x3
    y1 = x1 ^ x012 ^ x3 ^ x13 ^ x013 ^ x23 ^ x023
    y2 = m ^ x01 ^ x2 ^ x3 ^ x03 ^ x13 ^ x013 ^ x023
    y3 = m ^ x0 ^ x1 ^ x12 ^ x012 ^ x3 ^ x013 ^ x023
    return y0 | (y1 << 1) | (y2 << 2) | (y3 << 3)


@triton.jit
def _compact4(x):
    x = x & 0x1111111111111111
    x = (x | (x >> 3)) & 0x0303030303030303
    x = (x | (x >> 6)) & 0x000f000f000f000f
    x = (x | (x >> 12)) & 0x000000ff000000ff
    return (x | (x >> 24)) & 0xffff


@triton.jit
def _player(x):
    return (_compact4(x) | (_compact4(x >> 1) << 16)
            | (_compact4(x >> 2) << 32) | (_compact4(x >> 3) << 48))


@triton.jit
def _nibble_sbox(x):
    return _sbox(x) & 0xf


@triton.jit
def _present_kernel(state, key0, key1, output, n: tl.constexpr,
                    START: tl.constexpr, ROUNDS: tl.constexpr,
                    VERSION: tl.constexpr, PAIR: tl.constexpr,
                    DIFF: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < n
    x0 = tl.load(state + offsets, mask=active).to(tl.uint64)
    if PAIR:
        x1 = x0 ^ DIFF
    hi = tl.load(key0 + offsets, mask=active).to(tl.uint64)
    lo = tl.load(key1 + offsets, mask=active).to(tl.uint64)

    # Index denotes the round-key index.  PRESENT needs key START..START+ROUNDS.
    for index in tl.static_range(0, START + ROUNDS + 1):
        if VERSION == 80:
            rk = (hi << 48) | (lo >> 16)
        else:
            rk = hi
        if index >= START:
            if index < START + ROUNDS:
                x0 = _player(_sbox(x0 ^ rk))
                if PAIR:
                    x1 = _player(_sbox(x1 ^ rk))
            else:
                x0 ^= rk
                if PAIR:
                    x1 ^= rk
        if index < START + ROUNDS:
            counter = index + 1
            # Materialize the round counter as uint64 before shifts by 15/62;
            # leaving it as a constexpr int lets Triton infer int32 first.
            counter64 = tl.full((BLOCK,), counter, tl.uint64)
            if VERSION == 80:
                new_hi = (lo >> 3) & 0xffff
                new_lo = ((lo & 7) << 61) | (hi << 45) | (lo >> 19)
                hi = (_nibble_sbox(new_hi >> 12) << 12) | (new_hi & 0xfff)
                lo = new_lo ^ (counter64 << 15)
            else:
                new_hi = ((hi & 7) << 61) | (lo >> 3)
                new_lo = ((lo & 7) << 61) | (hi >> 3)
                hi = ((_nibble_sbox(new_hi >> 60) << 60)
                      | (_nibble_sbox((new_hi >> 56) & 0xf) << 56)
                      | (new_hi & 0x00ffffffffffffff))
                lo = new_lo ^ ((counter64 & 3) << 62)
                hi ^= (counter64 >> 2) & 7
    if PAIR:
        x0 ^= x1
    tl.store(output + offsets, x0, mask=active)


def run(state, key, rounds, round_start, version, difference=None):
    """Run one fused kernel; when difference is set, return the pair XOR."""
    if not state.is_cuda:
        raise ValueError("the fused PRESENT kernel requires a CUDA tensor")
    result = torch.empty_like(state)
    block = 256
    grid = (triton.cdiv(state.numel(), block),)
    raw_diff = 0 if difference is None else int(difference) & ((1 << 64) - 1)
    # constexpr integers above INT64_MAX are rejected by Triton's launcher.
    signed_diff = raw_diff if raw_diff < (1 << 63) else raw_diff - (1 << 64)
    _present_kernel[grid](
        state, key[0], key[1], result, state.numel(),
        START=int(round_start), ROUNDS=int(rounds), VERSION=int(version),
        PAIR=difference is not None, DIFF=signed_diff, BLOCK=block,
        num_warps=4,
    )
    return result
