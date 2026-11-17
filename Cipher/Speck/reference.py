"""One NumPy reference implementation for all standard SPECK versions.

State is (left, right); master-key words are in most-significant-word-first
order. Paper bit 0 is the least-significant bit of the right state word.
"""
import numpy as np

SPECK_ROUNDS = {(32, 64): 22, (48, 72): 22, (48, 96): 23,
                (64, 96): 26, (64, 128): 27, (96, 96): 28,
                (96, 144): 29, (128, 128): 32, (128, 192): 33,
                (128, 256): 34}
DEFAULT_KEY_SIZE = {32: 64, 48: 96, 64: 128, 96: 96, 128: 128}


def _variant(block_size, key_size=None):
    block_size = int(block_size)
    key_size = DEFAULT_KEY_SIZE.get(block_size) if key_size is None else int(key_size)
    if (block_size, key_size) not in SPECK_ROUNDS:
        raise ValueError(f"unsupported SPECK{block_size}/{key_size}")
    return block_size, key_size


def WORD_SIZE(version=128):
    return _variant(version)[0] // 2


def rol(value, shift, version=128):
    width = WORD_SIZE(version)
    shift %= width
    mask = np.uint64((1 << width) - 1)
    value = np.asarray(value, dtype=np.uint64) & mask
    if not shift:
        return value
    return ((value << np.uint64(shift)) | (value >> np.uint64(width - shift))) & mask


def ror(value, shift, version=128):
    return rol(value, -int(shift), version)


def enc_one_round(state, key, version=128):
    width = WORD_SIZE(version)
    alpha, beta = (7, 2) if width == 16 else (8, 3)
    mask = np.uint64((1 << width) - 1)
    left, right = (np.asarray(v, dtype=np.uint64) for v in state)
    with np.errstate(over="ignore"):
        left = ((ror(left, alpha, version) + right) & mask) ^ np.asarray(key, dtype=np.uint64)
    right = rol(right, beta, version) ^ left
    return left & mask, right & mask


def dec_one_round(state, key, version=128):
    width = WORD_SIZE(version)
    alpha, beta = (7, 2) if width == 16 else (8, 3)
    mask = np.uint64((1 << width) - 1)
    left, right = (np.asarray(v, dtype=np.uint64) for v in state)
    right = ror(right ^ left, beta, version)
    with np.errstate(over="ignore"):
        left = rol(((left ^ np.asarray(key, dtype=np.uint64)) - right) & mask, alpha, version)
    return left, right


def expand_key(key, rounds, version=128, block_size=None, key_size=None):
    block = int(version if block_size is None else block_size)
    key_size = len(key) * (block // 2) if key_size is None else key_size
    block, key_size = _variant(block, key_size)
    if len(key) != key_size // (block // 2):
        raise ValueError("incorrect SPECK master-key word count")
    rounds = int(rounds)
    if rounds < 0:
        raise ValueError("rounds must be nonnegative")
    if rounds == 0:
        return []
    mask = np.uint64((1 << (block // 2)) - 1)
    current = np.asarray(key[-1], dtype=np.uint64) & mask
    left = [np.asarray(v, dtype=np.uint64) & mask for v in reversed(key[:-1])]
    keys = [current]
    for index in range(rounds - 1):
        slot = index % len(left)
        left[slot], current = enc_one_round((left[slot], current), index, block)
        keys.append(current)
    return keys


def encrypt(state, round_keys, version=128):
    state = tuple(np.asarray(v, dtype=np.uint64) for v in state)
    for key in round_keys:
        state = enc_one_round(state, key, version)
    return state


def decrypt(state, round_keys, version=128):
    state = tuple(np.asarray(v, dtype=np.uint64) for v in state)
    for key in reversed(round_keys):
        state = dec_one_round(state, key, version)
    return state


def encrypt_block(plaintext, key, block_size=128, key_size=None, rounds=None):
    block_size, key_size = _variant(block_size, key_size)
    rounds = SPECK_ROUNDS[block_size, key_size] if rounds is None else int(rounds)
    return encrypt(plaintext, expand_key(key, rounds, block_size=block_size, key_size=key_size), block_size)


def encrypt_segment(state, key, rounds, round_start=0, block_size=128, key_size=None):
    if int(round_start) < 0 or int(rounds) < 0:
        raise ValueError("round interval must be nonnegative")
    keys = expand_key(key, int(round_start) + int(rounds), block_size=block_size, key_size=key_size)
    return encrypt(state, keys[int(round_start):], block_size)


def cpu_reference_encrypt(x, y, key_words, rounds=5, round_start=0, word_size=16):
    """Scalar compatibility entry used by the shared FWHT diagnostic."""
    return tuple(int(v) for v in encrypt_segment((x, y), key_words, rounds,
                                                round_start, 2 * int(word_size)))
