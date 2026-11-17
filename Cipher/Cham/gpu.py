"""Shared CHAM tensor rounds and key schedule for 64/128 and 128/128."""
from Framework.common.gpu import CipherGPUConfig
from Cipher.tensor_ops import WordPrimitive, rol, word_mask


class ChamGPUPrimitive(WordPrimitive):
    def __init__(self, block_size=64, key_size=128):
        block_size, key_size = int(block_size), int(key_size)
        if block_size not in (64, 128) or key_size != 128:
            raise ValueError("supported CHAM variants are 64/128 and 128/128")
        self.config = CipherGPUConfig(f"CHAM-{block_size}/128", block_size,
                                      block_size // 4, 128, 4)
        self.max_rounds = 88 if block_size == 64 else 112

    def expand_key(self, key):
        width = self.config.word_size
        count = 128 // width
        mask = word_mask(width)
        if len(key) != count:
            raise ValueError(f"CHAM key requires {count} words")
        keys = [None] * (2 * count)
        for index, value in enumerate(key):
            value = value & mask
            keys[index] = (value ^ rol(value, 1, width) ^ rol(value, 8, width)) & mask
            keys[(index + count) ^ 1] = (value ^ rol(value, 1, width) ^ rol(value, 11, width)) & mask
        return keys

    def encrypt_batch(self, state, key, rounds, round_start=0):
        rounds, round_start = int(rounds), int(round_start)
        if min(rounds, round_start) < 0 or rounds + round_start > self.max_rounds:
            raise ValueError("CHAM round interval outside supported bounds")
        width = self.config.word_size
        mask = word_mask(width)
        x0, x1, x2, x3 = state
        keys = self.expand_key(key)
        for index in range(round_start, round_start + rounds):
            alpha, beta = (8, 1) if index & 1 else (1, 8)
            tmp = ((x0 ^ index) + (rol(x1, alpha, width) ^ keys[index % len(keys)])) & mask
            x0, x1, x2, x3 = x1, x2, x3, rol(tmp, beta, width)
        return tuple(v & mask for v in (x0, x1, x2, x3))

    def output_to_uint(self, state):
        return tuple(state)

    def paper_bits_to_difference(self, paper_bits):
        words = [0] * 4
        for raw in paper_bits:
            bit = int(raw)
            if not 0 <= bit < self.config.block_size:
                raise ValueError(f"paper bit out of range: {bit}")
            word, offset = divmod(bit, self.config.word_size)
            words[word] |= 1 << offset
        return tuple(words)


class Cham64GPUPrimitive(ChamGPUPrimitive):
    def __init__(self): super().__init__(64)


class Cham128GPUPrimitive(ChamGPUPrimitive):
    def __init__(self): super().__init__(128)
