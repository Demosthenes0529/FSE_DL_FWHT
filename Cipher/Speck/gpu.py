"""Parameterized SPECK tensor implementation for 32/48/64/96/128-bit blocks."""
from Framework.common.gpu import CipherGPUConfig
from Cipher.tensor_ops import WordPrimitive, rol, ror, word_mask
from .reference import DEFAULT_KEY_SIZE, SPECK_ROUNDS


def speck_encrypt_batch(x, y, key_words, rounds=5, round_start=0, word_size=16):
    mask = word_mask(word_size)
    alpha, beta = (7, 2) if int(word_size) == 16 else (8, 3)
    if len(key_words) < 2 or int(rounds) < 0 or int(round_start) < 0:
        raise ValueError("invalid SPECK key or round interval")
    current = key_words[-1] & mask
    left = [value & mask for value in reversed(key_words[:-1])]
    keys = [current]
    for index in range(int(round_start) + int(rounds) - 1):
        slot = index % len(left)
        left[slot] = ((ror(left[slot], alpha, word_size) + current) & mask) ^ index
        current = rol(current, beta, word_size) ^ left[slot]
        keys.append(current)
    for index in range(int(round_start), int(round_start) + int(rounds)):
        x = ((ror(x, alpha, word_size) + y) & mask) ^ keys[index]
        y = rol(y, beta, word_size) ^ x
    return x & mask, y & mask


def speck32_encrypt_batch(x, y, key_words, rounds=5, round_start=0):
    return speck_encrypt_batch(x, y, key_words, rounds, round_start, 16)


class SpeckGPUPrimitive(WordPrimitive):
    def __init__(self, block_size=32, key_size=None):
        block_size = int(block_size)
        key_size = DEFAULT_KEY_SIZE.get(block_size) if key_size is None else int(key_size)
        if (block_size, key_size) not in SPECK_ROUNDS:
            raise ValueError(f"unsupported SPECK{block_size}/{key_size}")
        self.config = CipherGPUConfig(f"Speck{block_size}/{key_size}", block_size,
                                      block_size // 2, key_size, 2)

    def encrypt_batch(self, state, key, rounds, round_start=0):
        if len(key) != self.config.key_size // self.config.word_size:
            raise ValueError("incorrect SPECK master-key word count")
        return speck_encrypt_batch(*state, key, rounds, round_start, self.config.word_size)


class Speck32GPUPrimitive(SpeckGPUPrimitive):
    def __init__(self): super().__init__(32)


class Speck48GPUPrimitive(SpeckGPUPrimitive):
    def __init__(self): super().__init__(48)


class Speck64GPUPrimitive(SpeckGPUPrimitive):
    def __init__(self): super().__init__(64)


class Speck96GPUPrimitive(SpeckGPUPrimitive):
    def __init__(self): super().__init__(96)


class Speck128GPUPrimitive(SpeckGPUPrimitive):
    def __init__(self): super().__init__(128)
