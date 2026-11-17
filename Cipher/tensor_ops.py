"""Word operations shared by tensor cipher implementations (no GPU import)."""

def word_mask(width):
    return -1 if int(width) == 64 else (1 << int(width)) - 1


def logical_rshift(value, shift, width):
    shift %= int(width)
    return value & word_mask(width) if shift == 0 else (value >> shift) & ((1 << (int(width) - shift)) - 1)


def rol(value, shift, width):
    shift %= int(width)
    if shift == 0:
        return value & word_mask(width)
    return ((value << shift) | logical_rshift(value, int(width) - shift, width)) & word_mask(width)


def ror(value, shift, width):
    return rol(value, -int(shift), width)


class WordPrimitive:
    """Two-word, right-word-first paper coordinates for SPECK/SIMON/SIMECK."""

    supports_round_start = True

    def apply_difference(self, state, difference):
        if len(state) != self.config.state_words or len(difference) != len(state):
            raise ValueError("state and difference must match the configured word count")
        return tuple(value ^ int(delta) for value, delta in zip(state, difference))

    def output_to_uint(self, state):
        if self.config.block_size > 64:
            return state[1], state[0]
        return (state[0] << self.config.word_size) | state[1]

    def paper_bits_to_difference(self, paper_bits):
        words = [0, 0]
        width = self.config.word_size
        for raw in paper_bits:
            bit = int(raw)
            if not 0 <= bit < self.config.block_size:
                raise ValueError(f"paper bit out of range: {bit}")
            words[1 - bit // width] |= 1 << (bit % width)
        if width == 64:
            words = [v if v < (1 << 63) else v - (1 << 64) for v in words]
        return tuple(words)
