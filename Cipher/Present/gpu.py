"""Vectorized PRESENT-64 CUDA primitives for the shared k=32 Em backend.

The SPN framework numbers state coordinates MSB first: paper bit 0 is raw
integer bit 63.  The distributed engine therefore accesses bits only through
``paper_bit_plane``; the raw uint64 state is never mistaken for an LSB-first
ARX paper representation.
"""

from Framework.common.gpu import CipherGPUConfig


SBOX_VALUES = (0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD,
               0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2)
NIBBLE_LSB = 0x1111111111111111
MAX_ROUNDS = 31


def _i64(value):
    value = int(value) & ((1 << 64) - 1)
    return value if value < (1 << 63) else value - (1 << 64)


def _logical_rshift(value, shift):
    shift = int(shift)
    if shift == 0:
        return value
    return (value >> shift) & ((1 << (64 - shift)) - 1)


def _sbox_layer(x):
    """PRESENT S-box ANF evaluated on all sixteen nibbles in parallel."""
    x0 = x & NIBBLE_LSB
    x1 = (x >> 1) & NIBBLE_LSB
    x2 = (x >> 2) & NIBBLE_LSB
    x3 = (x >> 3) & NIBBLE_LSB
    x01, x02, x12 = x0 & x1, x0 & x2, x1 & x2
    x03, x13, x23 = x0 & x3, x1 & x3, x2 & x3
    x012, x013, x023 = x01 & x2, x01 & x3, x02 & x3
    y0 = x0 ^ x2 ^ x12 ^ x3
    y1 = x1 ^ x012 ^ x3 ^ x13 ^ x013 ^ x23 ^ x023
    y2 = NIBBLE_LSB ^ x01 ^ x2 ^ x3 ^ x03 ^ x13 ^ x013 ^ x023
    y3 = NIBBLE_LSB ^ x0 ^ x1 ^ x12 ^ x012 ^ x3 ^ x013 ^ x023
    return y0 | (y1 << 1) | (y2 << 2) | (y3 << 3)


def _p_layer(x):
    """Transpose the 16x4 nibble-bit matrix into four 16-bit planes.

    PRESENT maps raw bit ``4*j+b`` to ``j+16*b``.  The staged masks compact
    every fourth source bit without a per-bit/sample loop and need only four
    independent CUDA vector expressions.
    """
    def compact_every_four(value):
        value = value & 0x1111111111111111
        value = (value | (value >> 3)) & 0x0303030303030303
        value = (value | (value >> 6)) & 0x000f000f000f000f
        value = (value | (value >> 12)) & 0x000000ff000000ff
        return (value | (value >> 24)) & 0xffff
    return (compact_every_four(x)
            | (compact_every_four(x >> 1) << 16)
            | (compact_every_four(x >> 2) << 32)
            | (compact_every_four(x >> 3) << 48))


def _lookup_sbox(nibble):
    import torch
    table = torch.tensor(SBOX_VALUES, dtype=torch.int64, device=nibble.device)
    return table[nibble.to(torch.int64)]


def _expand_key80(key, last_round):
    if len(key) != 2:
        raise ValueError("PRESENT-80 key must be (hi16, lo64)")
    hi, lo = key[0] & 0xffff, key[1]
    round_keys = []
    for counter in range(1, int(last_round) + 2):
        round_keys.append((hi << 48) | _logical_rshift(lo, 16))
        if counter > int(last_round):
            break
        new_hi = _logical_rshift(lo, 3) & 0xffff
        new_lo = ((lo & 0x7) << 61) | (hi << 45) | _logical_rshift(lo, 19)
        hi = (_lookup_sbox(_logical_rshift(new_hi, 12) & 0xf) << 12) | (new_hi & 0x0fff)
        lo = new_lo ^ (counter << 15)
    return round_keys


def _expand_key128(key, last_round):
    if len(key) != 2:
        raise ValueError("PRESENT-128 key must be (hi64, lo64)")
    hi, lo = key
    round_keys = []
    for counter in range(1, int(last_round) + 2):
        round_keys.append(hi)
        if counter > int(last_round):
            break
        new_hi = ((hi & 0x7) << 61) | _logical_rshift(lo, 3)
        new_lo = ((lo & 0x7) << 61) | _logical_rshift(hi, 3)
        top0 = _lookup_sbox(_logical_rshift(new_hi, 60) & 0xf)
        top1 = _lookup_sbox(_logical_rshift(new_hi, 56) & 0xf)
        hi = (top0 << 60) | (top1 << 56) | (new_hi & 0x00ffffffffffffff)
        lo = new_lo ^ _i64((counter & 0x3) << 62)
        hi = hi ^ ((counter >> 2) & 0x7)
    return round_keys


def _encrypt_batch_torch(state, key, rounds, round_start, version):
    rounds, round_start = int(rounds), int(round_start)
    if len(state) != 1:
        raise ValueError("PRESENT state must contain one 64-bit tensor")
    if rounds < 0 or round_start < 0 or round_start + rounds > MAX_ROUNDS:
        raise ValueError(f"PRESENT invalid round range: start={round_start}, rounds={rounds}")
    expand = _expand_key80 if int(version) == 80 else _expand_key128
    round_keys = expand(key, round_start + rounds)
    x = state[0]
    for index in range(round_start, round_start + rounds):
        x = _p_layer(_sbox_layer(x ^ round_keys[index]))
    return (x ^ round_keys[round_start + rounds],)


def _encrypt_batch(state, key, rounds, round_start, version):
    if state[0].is_cuda:
        from .triton import run
        return (run(state[0], key, rounds, round_start, version),)
    return _encrypt_batch_torch(state, key, rounds, round_start, version)


def _encrypt_pair_difference_batch(state, key, difference, rounds, round_start, version):
    """Encrypt a pair in one kernel, sharing the complete key schedule."""
    if state[0].is_cuda:
        from .triton import run
        return (run(state[0], key, rounds, round_start, version,
                    difference=int(difference[0])),)
    left = _encrypt_batch_torch(state, key, rounds, round_start, version)
    right = _encrypt_batch_torch(apply_difference(state, difference), key,
                                 rounds, round_start, version)
    return tuple(a ^ b for a, b in zip(left, right))


def apply_difference(state, difference):
    if len(difference) != 1:
        raise ValueError("PRESENT difference must contain one 64-bit word")
    return (state[0] ^ _i64(difference[0]),)


def output_to_uint(state):
    return state[0]


def paper_bit_plane(packed, paper_bit):
    bit = int(paper_bit)
    if bit < 0 or bit >= 64:
        raise ValueError(f"PRESENT paper bit out of range: {bit}")
    return (packed >> (63 - bit)) & 1


def paper_bit_matrix(packed):
    """Vectorized MSB-first unpack used by the HW1 coordinate selector."""
    import torch
    offsets = 63 - torch.arange(64, dtype=torch.int64, device=packed.device)
    return ((packed[:, None] >> offsets[None, :]) & 1).to(torch.uint8)


def paper_bits_to_difference(paper_bits):
    value = 0
    for raw_bit in paper_bits:
        bit = int(raw_bit)
        if bit < 0 or bit >= 64:
            raise ValueError(f"PRESENT paper bit out of range: {bit}")
        value |= 1 << (63 - bit)
    return (value,)


def paper_bits_to_hex_words(paper_bits):
    """Serialize masks using PRESENT's MSB-first public bit numbering."""
    return [f"0x{value:016x}" for value in paper_bits_to_difference(paper_bits)]


def _warmup_cuda(device, rounds, round_start, version):
    """Initialize Triton's CUDA driver and PRESENT kernel on the main thread.

    The generic panel engine subsequently launches one Python worker per GPU.
    Lazy first-use initialization from several workers can race inside older
    Triton releases, so perform one tiny sequential launch before threading.
    """
    import torch
    with torch.cuda.device(int(device)):
        state = (torch.zeros(1, dtype=torch.int64, device=f"cuda:{int(device)}"),)
        key = (
            torch.zeros(1, dtype=torch.int64, device=state[0].device),
            torch.zeros(1, dtype=torch.int64, device=state[0].device),
        )
        _encrypt_pair_difference_batch(
            state, key, (1,), int(rounds), int(round_start), int(version)
        )
        torch.cuda.synchronize(int(device))


CONFIG80 = CipherGPUConfig(
    name="PRESENT-64/80", block_size=64, word_size=64, key_size=80,
    state_words=1, key_word_bits=(16, 64),
)
CONFIG128 = CipherGPUConfig(
    name="PRESENT-64/128", block_size=64, word_size=64, key_size=128,
    state_words=1, key_word_bits=(64, 64),
)


class Present80GPUPrimitive:
    config = CONFIG80
    paper_bits_to_hex_words = staticmethod(paper_bits_to_hex_words)
    supports_round_start = True
    encrypt_batch = staticmethod(lambda state, key, rounds, round_start=0:
                                 _encrypt_batch(state, key, rounds, round_start, 80))
    encrypt_pair_difference_batch = staticmethod(
        lambda state, key, difference, rounds, round_start=0:
        _encrypt_pair_difference_batch(state, key, difference, rounds, round_start, 80))
    apply_difference = staticmethod(apply_difference)
    output_to_uint = staticmethod(output_to_uint)
    paper_bit_plane = staticmethod(paper_bit_plane)
    paper_bit_matrix = staticmethod(paper_bit_matrix)
    paper_bits_to_difference = staticmethod(paper_bits_to_difference)
    warmup_cuda = staticmethod(lambda device, rounds, round_start=0:
                               _warmup_cuda(device, rounds, round_start, 80))


class Present128GPUPrimitive:
    config = CONFIG128
    paper_bits_to_hex_words = staticmethod(paper_bits_to_hex_words)
    supports_round_start = True
    encrypt_batch = staticmethod(lambda state, key, rounds, round_start=0:
                                 _encrypt_batch(state, key, rounds, round_start, 128))
    encrypt_pair_difference_batch = staticmethod(
        lambda state, key, difference, rounds, round_start=0:
        _encrypt_pair_difference_batch(state, key, difference, rounds, round_start, 128))
    apply_difference = staticmethod(apply_difference)
    output_to_uint = staticmethod(output_to_uint)
    paper_bit_plane = staticmethod(paper_bit_plane)
    paper_bit_matrix = staticmethod(paper_bit_matrix)
    paper_bits_to_difference = staticmethod(paper_bits_to_difference)
    warmup_cuda = staticmethod(lambda device, rounds, round_start=0:
                               _warmup_cuda(device, rounds, round_start, 128))
