"""Reference CHAM-64/128 and CHAM-128/128 implementations.

State words are stored in public paper order ``v0, v1, v2, v3`` with the
least-significant bit of each word addressed first.  Byte serialization uses
little-endian words, matching the public CHAM reference implementations.

Only 128-bit master-key variants are implemented here.  Trail models may ignore
round-key XOR and constants, but this reference cipher always expands and uses
the real 128-bit master key.
"""

from __future__ import annotations


import numpy as np

class UnsupportedVariantError(ValueError):
    pass


def require_variant(cipher, block_size, key_size, supported):
    pair = (int(block_size), int(key_size))
    if pair not in set(supported):
        raise UnsupportedVariantError(f"unsupported {cipher} variant {pair}")
    return pair


SUPPORTED_VARIANTS = {(64, 128), (128, 128)}
BRANCHES = 4
KEY_SIZE = 128
DEFAULT_ROUNDS = {64: 88, 128: 112}
REFERENCE_ROUNDS = {64: 80, 128: 80}


def WORD_SIZE(block_size=128):
    block_size = int(block_size)
    if block_size == 64:
        return 16
    if block_size == 128:
        return 32
    raise UnsupportedVariantError(f"unsupported CHAM block size: {block_size}")


def _mask(word_size):
    return (1 << int(word_size)) - 1


def _as_word(value, word_size):
    return np.asarray(value, dtype=np.uint64) & np.uint64(_mask(word_size))


def rol(x, amount, word_size):
    amount %= int(word_size)
    x = _as_word(x, word_size)
    if amount == 0:
        return x
    mask = np.uint64(_mask(word_size))
    return ((x << np.uint64(amount)) & mask) | (x >> np.uint64(int(word_size) - amount))


def ror(x, amount, word_size):
    amount %= int(word_size)
    x = _as_word(x, word_size)
    if amount == 0:
        return x
    mask = np.uint64(_mask(word_size))
    return (x >> np.uint64(amount)) | ((x << np.uint64(int(word_size) - amount)) & mask)


def expand_key(master_key, rounds=None, version=128, block_size=None, key_size=128):
    block_size = int(block_size or (64 if version == 64 else 128))
    require_variant("CHAM", block_size, key_size, SUPPORTED_VARIANTS)
    word_size = WORD_SIZE(block_size)
    key_words = int(key_size) // word_size
    if len(master_key) != key_words:
        raise ValueError(f"CHAM-{block_size}/{key_size} key must contain {key_words} words")
    k = [_as_word(word, word_size) for word in master_key]
    rk = [None] * (2 * key_words)
    for i, word in enumerate(k):
        rk[i] = word ^ rol(word, 1, word_size) ^ rol(word, 8, word_size)
        rk[(i + key_words) ^ 1] = word ^ rol(word, 1, word_size) ^ rol(word, 11, word_size)
    return tuple(_as_word(word, word_size) for word in rk)


def enc_one_round(state, round_key, round_index, word_size):
    x0, x1, x2, x3 = (_as_word(word, word_size) for word in state)
    if int(round_index) & 1:
        alpha, beta = 8, 1
    else:
        alpha, beta = 1, 8
    tmp = (x0 ^ np.uint64(int(round_index))) + (rol(x1, alpha, word_size) ^ round_key)
    new_word = rol(tmp, beta, word_size)
    return (x1, x2, x3, _as_word(new_word, word_size))


def dec_one_round(state, round_key, round_index, word_size):
    y0, y1, y2, y3 = (_as_word(word, word_size) for word in state)
    if int(round_index) & 1:
        alpha, beta = 8, 1
    else:
        alpha, beta = 1, 8
    x1, x2, x3 = y0, y1, y2
    pre = ror(y3, beta, word_size)
    modulus = np.uint64(1 << int(word_size))
    x0 = _as_word(pre + modulus - _as_word(rol(x1, alpha, word_size) ^ round_key, word_size), word_size)
    x0 = x0 ^ np.uint64(int(round_index))
    return (_as_word(x0, word_size), x1, x2, x3)


def encrypt_rounds(state, round_keys, start_round=0, num_rounds=None, block_size=128):
    word_size = WORD_SIZE(block_size)
    num_rounds = len(round_keys) if num_rounds is None else int(num_rounds)
    current = tuple(_as_word(word, word_size) for word in state)
    for offset in range(num_rounds):
        round_index = int(start_round) + offset
        current = enc_one_round(current, round_keys[round_index % len(round_keys)], round_index, word_size)
    return current


def decrypt_rounds(state, round_keys, start_round=0, num_rounds=None, block_size=128):
    word_size = WORD_SIZE(block_size)
    num_rounds = len(round_keys) if num_rounds is None else int(num_rounds)
    current = tuple(_as_word(word, word_size) for word in state)
    for offset in reversed(range(num_rounds)):
        round_index = int(start_round) + offset
        current = dec_one_round(current, round_keys[round_index % len(round_keys)], round_index, word_size)
    return current


def encrypt(state, round_keys, version=128, block_size=None, start_round=0):
    block_size = int(block_size or (64 if version == 64 else 128))
    return encrypt_rounds(state, round_keys, start_round=start_round, num_rounds=len(round_keys), block_size=block_size)


def decrypt(state, round_keys, version=128, block_size=None, start_round=0):
    block_size = int(block_size or (64 if version == 64 else 128))
    return decrypt_rounds(state, round_keys, start_round=start_round, num_rounds=len(round_keys), block_size=block_size)


def _words_from_bytes(data, word_size, count):
    step = word_size // 8
    if len(data) != step * count:
        raise ValueError(f"expected {step * count} bytes")
    return tuple(int.from_bytes(data[i * step:(i + 1) * step], "little") for i in range(count))


def _words_to_bytes(words, word_size):
    step = word_size // 8
    return b"".join(int(word).to_bytes(step, "little") for word in words)


def encrypt_block(plaintext, key, rounds=None, block_size=128, key_size=128):
    require_variant("CHAM", block_size, key_size, SUPPORTED_VARIANTS)
    word_size = WORD_SIZE(block_size)
    rounds = DEFAULT_ROUNDS[int(block_size)] if rounds is None else int(rounds)
    if isinstance(plaintext, (bytes, bytearray)):
        plaintext = _words_from_bytes(bytes(plaintext), word_size, BRANCHES)
    if isinstance(key, (bytes, bytearray)):
        key = _words_from_bytes(bytes(key), word_size, key_size // word_size)
    round_keys = expand_key(key, rounds=rounds, block_size=block_size, key_size=key_size)
    out = encrypt_rounds(plaintext, round_keys, 0, rounds, block_size)
    return tuple(int(np.asarray(word, dtype=np.uint64)) for word in out)


def decrypt_block(ciphertext, key, rounds=None, block_size=128, key_size=128):
    require_variant("CHAM", block_size, key_size, SUPPORTED_VARIANTS)
    word_size = WORD_SIZE(block_size)
    rounds = DEFAULT_ROUNDS[int(block_size)] if rounds is None else int(rounds)
    if isinstance(ciphertext, (bytes, bytearray)):
        ciphertext = _words_from_bytes(bytes(ciphertext), word_size, BRANCHES)
    if isinstance(key, (bytes, bytearray)):
        key = _words_from_bytes(bytes(key), word_size, key_size // word_size)
    round_keys = expand_key(key, rounds=rounds, block_size=block_size, key_size=key_size)
    out = decrypt_rounds(ciphertext, round_keys, 0, rounds, block_size)
    return tuple(int(np.asarray(word, dtype=np.uint64)) for word in out)


def encrypt_block_bytes(plaintext, key, rounds=None, block_size=128, key_size=128):
    words = encrypt_block(plaintext, key, rounds=rounds, block_size=block_size, key_size=key_size)
    return _words_to_bytes(words, WORD_SIZE(block_size))


def decrypt_block_bytes(ciphertext, key, rounds=None, block_size=128, key_size=128):
    words = decrypt_block(ciphertext, key, rounds=rounds, block_size=block_size, key_size=key_size)
    return _words_to_bytes(words, WORD_SIZE(block_size))
