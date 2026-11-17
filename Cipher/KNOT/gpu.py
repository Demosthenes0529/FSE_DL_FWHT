"""Vectorized KNOT-256 permutation for the cipher-neutral CUDA Em backend.

The state is four 64-bit rows in public ``row0||row1||row2||row3`` order.
Paper coordinates are MSB-first inside each row.  KNOT is unkeyed, so the
common engine passes an empty key tuple and validation repetitions represent
independent sample batches rather than independent keys.
"""

from Framework.common.gpu import CipherGPUConfig
from .reference import ROUND_CONSTANTS


def _i64(value):
    value = int(value) & ((1 << 64) - 1)
    return value if value < (1 << 63) else value - (1 << 64)


def _logical_rshift(value, amount):
    amount = int(amount)
    return value if amount == 0 else (value >> amount) & ((1 << (64 - amount)) - 1)


def _rol64(value, amount):
    amount = int(amount) & 63
    if amount == 0:
        return value
    return (value << amount) | _logical_rshift(value, 64 - amount)


def _sbox(a, b, c, d):
    t1 = ~a
    t2 = b & t1
    t3 = c ^ t2
    h = d ^ t3
    t5 = b | c
    t6 = d ^ t1
    g = t5 ^ t6
    t8 = b ^ d
    t9 = t3 & t6
    e = t8 ^ t9
    t11 = g & t8
    f = t3 ^ t11
    return e, f, g, h


def encrypt_batch(state, key, rounds, round_start=0):
    if len(state) != 4 or key:
        raise ValueError("KNOT-256 requires four state rows and an empty key tuple")
    start, rounds = int(round_start), int(rounds)
    if start < 0 or rounds < 0 or start + rounds > len(ROUND_CONSTANTS):
        raise ValueError("KNOT-256 round interval outside 0..63")
    x0, x1, x2, x3 = state
    for absolute_round in range(start, start + rounds):
        x0 = x0 ^ int(ROUND_CONSTANTS[absolute_round])
        x0, x1, x2, x3 = _sbox(x0, x1, x2, x3)
        x1, x2, x3 = _rol64(x1, 1), _rol64(x2, 8), _rol64(x3, 25)
    return x0, x1, x2, x3


def apply_difference(state, difference):
    if len(state) != 4 or len(difference) != 4:
        raise ValueError("KNOT-256 state/difference must contain four rows")
    return tuple(word ^ _i64(delta) for word, delta in zip(state, difference))


def output_to_uint(state):
    if len(state) != 4:
        raise ValueError("KNOT-256 output must contain four rows")
    return tuple(state)


def paper_bit_plane(packed, paper_bit):
    bit = int(paper_bit)
    if bit < 0 or bit >= 256:
        raise ValueError(f"KNOT-256 paper bit out of range: {bit}")
    row, offset = divmod(bit, 64)
    return (packed[row] >> (63 - offset)) & 1


def paper_bit_matrix(packed):
    """Vectorized row-major, MSB-first unpack for all 256 paper bits."""
    import torch
    if len(packed) != 4:
        raise ValueError("KNOT-256 packed output must contain four rows")
    offsets = 63 - torch.arange(64, dtype=torch.int64, device=packed[0].device)
    return torch.cat([
        ((row[:, None] >> offsets[None, :]) & 1).to(torch.uint8)
        for row in packed
    ], dim=1)


def paper_bits_to_difference(paper_bits):
    rows = [0, 0, 0, 0]
    for raw in paper_bits:
        bit = int(raw)
        if bit < 0 or bit >= 256:
            raise ValueError(f"KNOT-256 paper bit out of range: {bit}")
        row, offset = divmod(bit, 64)
        rows[row] |= 1 << (63 - offset)
    return tuple(rows)


def paper_bits_to_hex_words(paper_bits):
    """Return the public four-row representation used by KNOT reports."""
    return [f"0x{word:016x}" for word in paper_bits_to_difference(paper_bits)]


CONFIG = CipherGPUConfig(
    name="KNOT-256", block_size=256, word_size=64, key_size=0,
    state_words=4, key_word_bits=(),
)


class KNOT256GPUPrimitive:
    config = CONFIG
    supports_round_start = True
    encrypt_batch = staticmethod(encrypt_batch)
    apply_difference = staticmethod(apply_difference)
    output_to_uint = staticmethod(output_to_uint)
    paper_bit_plane = staticmethod(paper_bit_plane)
    paper_bit_matrix = staticmethod(paper_bit_matrix)
    paper_bits_to_difference = staticmethod(paper_bits_to_difference)
    paper_bits_to_hex_words = staticmethod(paper_bits_to_hex_words)
