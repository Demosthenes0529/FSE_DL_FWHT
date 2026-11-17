"""Vectorized SIMECK reference implementation for DL path experiments.

Passing ``None`` as the key keeps the historical keyless internal permutation.
Passing a real master key uses the standard SIMECK key schedule.
"""

from Cipher.Simon.reference import (  # noqa: F401
    MASK_VAL,
    WORD_SIZE,
    _block_from_version,
)


import numpy as np


def rol(x, k, version=64):
    w = WORD_SIZE(version)
    mask = MASK_VAL(version)
    mask_u = np.uint64(mask)
    x = x & mask_u
    return (np.left_shift(x, np.uint64(k)) & mask_u) | np.right_shift(x, np.uint64(w - k))


def enc_one_round(p, k=0, version=64):
    x, y = p
    x = np.asarray(x, dtype=np.uint64)
    y = np.asarray(y, dtype=np.uint64)
    k = np.asarray(k, dtype=np.uint64)
    mask = MASK_VAL(version)
    mask_u = np.uint64(mask)
    z = (y ^ (x & rol(x, 5, version)) ^ rol(x, 1, version) ^ k) & mask_u
    return z, x & mask_u


SIMECK_ROUNDS = {
    (32, 64): 32,
    (48, 96): 36,
    (64, 128): 44,
}

# Low-bit-first constants used by the reference implementation. Z0 has period
# 31 and is used by SIMECK32/64 and SIMECK48/96; the Z1 prefix below covers the
# full 44 rounds of SIMECK64/128.
SIMECK_Z0 = 0x9A42BB1F
SIMECK_Z1_PREFIX = 0x938BCA3083F

SIMECK_TEST_VECTORS = [
    ((32, 64), (0x6565, 0x6877), (0x1918, 0x1110, 0x0908, 0x0100), (0x770D, 0x2C76)),
    ((64, 128), (0x656B696C, 0x20646E75), (0x1B1A1918, 0x13121110, 0x0B0A0908, 0x03020100), (0x45CE6902, 0x5F7AB7ED)),
]


def _resolve_variant(version=64, block_size=None, key_size=None):
    block = _block_from_version(block_size if block_size is not None else version)
    if key_size is None:
        key_size = 2 * block
    key_size = int(key_size)
    if (block, key_size) not in SIMECK_ROUNDS:
        raise ValueError(f"unsupported SIMECK variant: block_size={block}, key_size={key_size}")
    return block, key_size


def rounds_for_variant(block_size=64, key_size=128):
    return SIMECK_ROUNDS[_resolve_variant(block_size, key_size=key_size)]


def _z_bit(block_size, index):
    if int(block_size) in (32, 48):
        return (SIMECK_Z0 >> (int(index) % 31)) & 1
    if int(index) >= 44:
        raise ValueError("SIMECK64/128 Z1 prefix is only defined for the standard 44 rounds")
    return (SIMECK_Z1_PREFIX >> int(index)) & 1


def _as_key_words(k, mask):
    words = [np.asarray(word, dtype=np.uint64) & np.uint64(mask) for word in k]
    if len(words) != 4:
        raise ValueError(f"SIMECK requires exactly four key words, got {len(words)}")
    return words


def expand_key(k, t, version=64, block_size=None, key_size=None):
    block = _block_from_version(block_size if block_size is not None else version)
    t = int(t)
    if t <= 0:
        return []
    if k is None:
        return [np.uint64(0)] * t
    block, key_size = _resolve_variant(block, key_size=key_size)
    mask = MASK_VAL(block)
    key_words = _as_key_words(k, mask)
    if key_size != 4 * WORD_SIZE(block):
        raise ValueError(f"SIMECK{block} requires key_size={4 * WORD_SIZE(block)}, got {key_size}")

    # Public vectors write K as (t2, t1, t0, k0).
    t_state = [
        key_words[3] & np.uint64(mask),
        key_words[2] & np.uint64(mask),
        key_words[1] & np.uint64(mask),
        key_words[0] & np.uint64(mask),
    ]
    round_keys = []
    base_constant = np.uint64(mask ^ 3)
    for i in range(t):
        round_keys.append(t_state[0])
        constant = base_constant | np.uint64(_z_bit(block, i))
        t_state[1], t_state[0] = enc_one_round((t_state[1], t_state[0]), constant, version=block)
        moved = t_state[1]
        t_state[1] = t_state[2]
        t_state[2] = t_state[3]
        t_state[3] = moved
    return round_keys


def encrypt(p, ks, version=64):
    x, y = p
    for k in ks:
        x, y = enc_one_round((x, y), k, version=version)
    mask_u = np.uint64(MASK_VAL(version))
    return x & mask_u, y & mask_u


def encrypt_rounds(
    p, rounds_or_keys, version=64, round_start=0, start_round=None,
    num_rounds=None, block_size=None,
):
    block = _block_from_version(block_size if block_size is not None else version)
    start = int(round_start if start_round is None else start_round)
    if isinstance(rounds_or_keys, (int, np.integer)):
        rounds = int(rounds_or_keys)
        ks = expand_key(None, start + rounds, version=block)[start:start + rounds]
    else:
        if num_rounds is None:
            raise ValueError("num_rounds is required when round keys are supplied")
        ks = list(rounds_or_keys)[start:start + int(num_rounds)]
    return encrypt(p, ks, version=block)


def encrypt_block(plaintext, key, block_size=64, key_size=128, rounds=None):
    block, key_size = _resolve_variant(block_size, key_size=key_size)
    total_rounds = rounds_for_variant(block, key_size) if rounds is None else int(rounds)
    return encrypt(plaintext, expand_key(key, total_rounds, version=block, key_size=key_size), version=block)
