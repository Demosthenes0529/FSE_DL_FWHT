"""One tensor implementation for all standard SIMON block/key variants."""
from Framework.common.gpu import CipherGPUConfig
from Cipher.tensor_ops import WordPrimitive, rol, ror, word_mask
from .reference import SIMON_DEFAULT_KEY_SIZE, SIMON_ROUNDS, SIMON_Z, SIMON_Z_BY_VARIANT


class SimonGPUPrimitive(WordPrimitive):
    def __init__(self, block_size=32, key_size=None):
        block_size = int(block_size)
        key_size = SIMON_DEFAULT_KEY_SIZE.get(block_size) if key_size is None else int(key_size)
        if (block_size, key_size) not in SIMON_ROUNDS:
            raise ValueError(f"unsupported SIMON{block_size}/{key_size}")
        self.config = CipherGPUConfig(f"Simon{block_size}/{key_size}", block_size,
                                      block_size // 2, key_size, 2)
        self.z = SIMON_Z[SIMON_Z_BY_VARIANT[block_size, key_size]]

    def expand_key(self, key, rounds):
        width = self.config.word_size
        mask = word_mask(width)
        count = self.config.key_size // width
        if len(key) != count or int(rounds) < 0:
            raise ValueError("invalid SIMON key word count or rounds")
        keys = [value & mask for value in reversed(key)][:int(rounds)]
        for index in range(count, int(rounds)):
            value = ror(keys[index - 1], 3, width)
            if count == 4:
                value ^= keys[index - 3]
            value ^= ror(value, 1, width)
            keys.append(((mask ^ 3) ^ ((self.z >> ((index - count) % 62)) & 1)
                         ^ keys[index - count] ^ value) & mask)
        return keys

    def encrypt_batch(self, state, key, rounds, round_start=0):
        if int(round_start) < 0 or int(rounds) < 0:
            raise ValueError("round interval must be nonnegative")
        width = self.config.word_size
        mask = word_mask(width)
        x, y = state
        for round_key in self.expand_key(key, int(round_start) + int(rounds))[int(round_start):]:
            x, y = (y ^ (rol(x, 1, width) & rol(x, 8, width)) ^ rol(x, 2, width) ^ round_key) & mask, x
        return x & mask, y & mask


class Simon32GPUPrimitive(SimonGPUPrimitive):
    def __init__(self): super().__init__(32)


class Simon48GPUPrimitive(SimonGPUPrimitive):
    def __init__(self): super().__init__(48)


class Simon64GPUPrimitive(SimonGPUPrimitive):
    def __init__(self): super().__init__(64)


class Simon96GPUPrimitive(SimonGPUPrimitive):
    def __init__(self): super().__init__(96)


class Simon128GPUPrimitive(SimonGPUPrimitive):
    def __init__(self): super().__init__(128)
