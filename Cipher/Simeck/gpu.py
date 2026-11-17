"""One tensor implementation for SIMECK32/64, SIMECK48/96 and SIMECK64/128."""
from Framework.common.gpu import CipherGPUConfig
from Cipher.tensor_ops import WordPrimitive, rol, word_mask
from .reference import SIMECK_ROUNDS, SIMECK_Z0, SIMECK_Z1_PREFIX


class SimeckGPUPrimitive(WordPrimitive):
    def __init__(self, block_size=32, key_size=None):
        block_size = int(block_size)
        key_size = 2 * block_size if key_size is None else int(key_size)
        if (block_size, key_size) not in SIMECK_ROUNDS:
            raise ValueError(f"unsupported SIMECK{block_size}/{key_size}")
        self.config = CipherGPUConfig(f"Simeck{block_size}/{key_size}", block_size,
                                      block_size // 2, key_size, 2)

    def _round(self, left, right, key):
        width = self.config.word_size
        return (right ^ (left & rol(left, 5, width)) ^ rol(left, 1, width) ^ key) & word_mask(width), left & word_mask(width)

    def expand_key(self, key, rounds):
        rounds = int(rounds)
        maximum = SIMECK_ROUNDS[self.config.block_size, self.config.key_size]
        if len(key) != 4 or not 0 <= rounds <= maximum:
            raise ValueError(f"SIMECK requires four key words and rounds in 0..{maximum}")
        mask = word_mask(self.config.word_size)
        state = [value & mask for value in reversed(key)]
        keys = []
        for index in range(rounds):
            keys.append(state[0])
            zbit = ((SIMECK_Z1_PREFIX >> index) & 1 if self.config.block_size == 64
                    else (SIMECK_Z0 >> (index % 31)) & 1)
            moved, first = self._round(state[1], state[0], (mask ^ 3) | zbit)
            state = [first, state[2], state[3], moved]
        return keys

    def encrypt_batch(self, state, key, rounds, round_start=0):
        if int(round_start) < 0 or int(rounds) < 0:
            raise ValueError("round interval must be nonnegative")
        left, right = state
        for round_key in self.expand_key(key, int(round_start) + int(rounds))[int(round_start):]:
            left, right = self._round(left, right, round_key)
        return left, right


class Simeck32GPUPrimitive(SimeckGPUPrimitive):
    def __init__(self): super().__init__(32)


class Simeck48GPUPrimitive(SimeckGPUPrimitive):
    def __init__(self): super().__init__(48)


class Simeck64GPUPrimitive(SimeckGPUPrimitive):
    def __init__(self): super().__init__(64)
