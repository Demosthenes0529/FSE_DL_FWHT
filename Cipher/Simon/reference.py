"""Vectorized SIMON reference implementation for DL path experiments.

Passing ``None`` as the key keeps the historical keyless internal permutation.
Passing a real master key uses the standard SIMON key schedule.
"""


import numpy as np


def _block_from_version(version):
    version = int(version)
    if version in (32, 48, 64, 96, 128):
        return version
    raise ValueError(f"SIMON permutation version must be block size 32/48/64/96/128, got {version}")


def WORD_SIZE(version=64):
    return _block_from_version(version) // 2


def MASK_VAL(version=64):
    return (1 << WORD_SIZE(version)) - 1


def rol(x, k, version=64):
    w = WORD_SIZE(version)
    mask = MASK_VAL(version)
    mask_u = np.uint64(mask)
    x = x & mask_u
    return (np.left_shift(x, np.uint64(k)) & mask_u) | np.right_shift(x, np.uint64(w - k))


def ror(x, k, version=64):
    w = WORD_SIZE(version)
    mask = MASK_VAL(version)
    mask_u = np.uint64(mask)
    x = x & mask_u
    return np.right_shift(x, np.uint64(k)) | (np.left_shift(x, np.uint64(w - k)) & mask_u)


def enc_one_round(p, k=0, version=64):
    x, y = p
    x = np.asarray(x, dtype=np.uint64)
    y = np.asarray(y, dtype=np.uint64)
    k = np.asarray(k, dtype=np.uint64)
    mask = MASK_VAL(version)
    mask_u = np.uint64(mask)
    z = (y ^ (rol(x, 1, version) & rol(x, 8, version)) ^ rol(x, 2, version) ^ k) & mask_u
    return z, x & mask_u


SIMON_ROUNDS = {
    (32, 64): 32,
    (48, 72): 36,
    (48, 96): 36,
    (64, 96): 42,
    (64, 128): 44,
    (96, 96): 52,
    (96, 144): 54,
    (128, 128): 68,
    (128, 192): 69,
    (128, 256): 72,
}

SIMON_Z = {
    0: 0b01100111000011010100100010111110110011100001101010010001011111,
    1: 0b01011010000110010011111011100010101101000011001001111101110001,
    2: 0b11001101101001111110001000010100011001001011000000111011110101,
    3: 0b11110000101100111001010001001000000111101001100011010111011011,
    4: 0b11110111001001010011000011101000000100011011010110011110001011,
}

SIMON_Z_BY_VARIANT = {
    (32, 64): 0,
    (48, 72): 0,
    (48, 96): 1,
    (64, 96): 2,
    (64, 128): 3,
    (96, 96): 2,
    (96, 144): 3,
    (128, 128): 2,
    (128, 192): 3,
    (128, 256): 4,
}

SIMON_DEFAULT_KEY_SIZE = {
    32: 64,
    48: 96,
    64: 128,
    96: 144,
    128: 256,
}

SIMON_TEST_VECTORS = [
    ((32, 64), (0x6565, 0x6877), (0x1918, 0x1110, 0x0908, 0x0100), (0xC69B, 0xE9BB)),
    ((48, 72), (0x612067, 0x6E696C), (0x121110, 0x0A0908, 0x020100), (0xDAE5AC, 0x292CAC)),
    ((48, 96), (0x726963, 0x20646E), (0x1A1918, 0x121110, 0x0A0908, 0x020100), (0x6E06A5, 0xACF156)),
    ((64, 96), (0x6F722067, 0x6E696C63), (0x13121110, 0x0B0A0908, 0x03020100), (0x5CA2E27F, 0x111A8FC8)),
    ((64, 128), (0x656B696C, 0x20646E75), (0x1B1A1918, 0x13121110, 0x0B0A0908, 0x03020100), (0x44C8FC20, 0xB9DFA07A)),
    ((96, 96), (0x2072616C6C69, 0x702065687420), (0x0D0C0B0A0908, 0x050403020100), (0x602807A462B4, 0x69063D8FF082)),
    ((96, 144), (0x746168742074, 0x73756420666F), (0x151413121110, 0x0D0C0B0A0908, 0x050403020100), (0xECAD1C6C451E, 0x3F59C5DB1AE9)),
    ((128, 128), (0x6373656420737265, 0x6C6C657661727420), (0x0F0E0D0C0B0A0908, 0x0706050403020100), (0x49681B1E1E54FE3F, 0x65AA832AF84E0BBC)),
    ((128, 192), (0x206572656874206E, 0x6568772065626972), (0x1716151413121110, 0x0F0E0D0C0B0A0908, 0x0706050403020100), (0xC4AC61EFFCDC0D4F, 0x6C9C8D6E2597B85B)),
    ((128, 256), (0x74206E69206D6F6F, 0x6D69732061207369), (0x1F1E1D1C1B1A1918, 0x1716151413121110, 0x0F0E0D0C0B0A0908, 0x0706050403020100), (0x8D2B5579AFC8A3A0, 0x3BF72A87EFE7B868)),
]


def _resolve_variant(version=64, block_size=None, key_size=None, key_words=None):
    block = _block_from_version(block_size if block_size is not None else version)
    if key_size is None:
        if key_words is not None:
            key_size = int(key_words) * WORD_SIZE(block)
        else:
            key_size = SIMON_DEFAULT_KEY_SIZE[block]
    key_size = int(key_size)
    if (block, key_size) not in SIMON_ROUNDS:
        raise ValueError(f"unsupported SIMON variant: block_size={block}, key_size={key_size}")
    return block, key_size


def rounds_for_variant(block_size=64, key_size=128):
    return SIMON_ROUNDS[_resolve_variant(block_size, key_size=key_size)]


def _as_key_words(k, mask):
    words = [np.asarray(word, dtype=np.uint64) & np.uint64(mask) for word in k]
    if not words:
        raise ValueError("SIMON key must contain at least one word")
    return words


def expand_key(k, t, version=64, block_size=None, key_size=None):
    block = _block_from_version(block_size if block_size is not None else version)
    t = int(t)
    if t <= 0:
        return []
    if k is None:
        return [np.uint64(0)] * t
    key_words = _as_key_words(k, MASK_VAL(block))
    block, key_size = _resolve_variant(block, key_size=key_size, key_words=len(key_words))
    word_size = WORD_SIZE(block)
    mask = MASK_VAL(block)
    m = len(key_words)
    if m != key_size // word_size:
        raise ValueError(f"SIMON{block}/{key_size} requires {key_size // word_size} key words, got {m}")

    round_keys = [(key_words[m - 1 - i] & np.uint64(mask)) for i in range(min(m, t))]
    c = np.uint64(mask ^ 3)
    z = SIMON_Z[SIMON_Z_BY_VARIANT[(block, key_size)]]
    for i in range(m, t):
        tmp = ror(round_keys[i - 1], 3, version=block)
        if m == 4:
            tmp = tmp ^ round_keys[i - 3]
        tmp = tmp ^ ror(tmp, 1, version=block)
        zbit = np.uint64((z >> ((i - m) % 62)) & 1)
        round_keys.append((c ^ zbit ^ round_keys[i - m] ^ tmp) & np.uint64(mask))
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
