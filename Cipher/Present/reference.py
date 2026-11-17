"""PRESENT-80/128 reference rounds, state mapping and known-answer vectors."""


import numpy as np


BLOCK_BITS = 64
MASK64_INT = (1 << 64) - 1
MASK64 = np.uint64(MASK64_INT)
SBOX = np.asarray([0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2], dtype=np.uint64)
SBOX_BYTE = np.asarray(
    [int(SBOX[value & 0xF]) | (int(SBOX[value >> 4]) << 4) for value in range(256)],
    dtype=np.uint64,
)
PBOX = tuple((16 * bit) % 63 if bit < 63 else 63 for bit in range(64))
PBOX_BYTE = np.zeros((8, 256), dtype=np.uint64)
for _byte_pos in range(8):
    for _value in range(256):
        _acc = 0
        for _bit in range(8):
            if (_value >> _bit) & 1:
                _acc |= 1 << PBOX[8 * _byte_pos + _bit]
        PBOX_BYTE[_byte_pos, _value] = np.uint64(_acc)


def int_to_hex(value):
    return f"{int(value) & MASK64_INT:016x}"


def get_cell(value, cell):
    """Cell 0 is the leftmost hexadecimal nibble."""
    if not 0 <= int(cell) < 16:
        raise ValueError("PRESENT cell index must be in 0..15")
    return (int(value) >> (4 * (15 - int(cell)))) & 15


def set_cell(value, cell, nibble):
    if not 0 <= int(cell) < 16 or not 0 <= int(nibble) < 16:
        raise ValueError("invalid PRESENT cell")
    shift = 4 * (15 - int(cell))
    return (int(value) & ~(15 << shift)) | (int(nibble) << shift)


def linear_layer(value):
    return sum(((int(value) >> bit) & 1) << target for bit, target in enumerate(PBOX))


def inverse_linear_layer(value):
    return sum(((int(value) >> target) & 1) << bit for bit, target in enumerate(PBOX))


def WORD_SIZE():
    return BLOCK_BITS


def _validate_version(version):
    version = int(version)
    if version not in (80, 128):
        raise ValueError(f"PRESENT supports key sizes 80 and 128, got version={version}")
    return version


def _as_u64_array(value):
    return np.asarray(value, dtype=np.uint64)


def _maybe_scalar(array, scalar):
    return int(np.asarray(array, dtype=np.uint64).item()) if scalar else array


def _sbox_layer(state):
    state = _as_u64_array(state)
    out = np.zeros_like(state, dtype=np.uint64)
    for byte_pos in range(8):
        idx = ((state >> np.uint64(8 * byte_pos)) & np.uint64(0xFF)).astype(np.uint16, copy=False)
        out |= SBOX_BYTE[idx] << np.uint64(8 * byte_pos)
    return out


def _p_layer(state):
    state = _as_u64_array(state)
    out = np.zeros_like(state, dtype=np.uint64)
    for byte_pos in range(8):
        idx = ((state >> np.uint64(8 * byte_pos)) & np.uint64(0xFF)).astype(np.uint16, copy=False)
        out |= PBOX_BYTE[byte_pos, idx]
    return out


def _split_key80(key):
    if isinstance(key, (tuple, list)) and len(key) == 2:
        hi, lo = key
        return _as_u64_array(hi) & np.uint64(0xFFFF), _as_u64_array(lo)
    key = int(key)
    return np.asarray((key >> 64) & 0xFFFF, dtype=np.uint64), np.asarray(key & MASK64_INT, dtype=np.uint64)


def _split_key128(key):
    if isinstance(key, (tuple, list)) and len(key) == 2:
        hi, lo = key
        return _as_u64_array(hi), _as_u64_array(lo)
    key = int(key)
    return np.asarray((key >> 64) & MASK64_INT, dtype=np.uint64), np.asarray(key & MASK64_INT, dtype=np.uint64)


def _expand_key80(key, rounds):
    hi, lo = _split_key80(key)
    round_keys = []
    for round_counter in range(1, int(rounds) + 2):
        round_keys.append(((hi << np.uint64(48)) | (lo >> np.uint64(16))).astype(np.uint64, copy=False))
        if round_counter > int(rounds):
            break
        new_hi = (lo >> np.uint64(3)) & np.uint64(0xFFFF)
        new_lo = (
            ((lo & np.uint64(0x7)) << np.uint64(61))
            | ((hi & np.uint64(0xFFFF)) << np.uint64(45))
            | (lo >> np.uint64(19))
        )
        hi = (SBOX[(new_hi >> np.uint64(12)).astype(np.uint16)] << np.uint64(12)) | (new_hi & np.uint64(0x0FFF))
        lo = new_lo ^ (np.uint64(round_counter & 0x1F) << np.uint64(15))
    return round_keys


def _expand_key128(key, rounds):
    hi, lo = _split_key128(key)
    round_keys = []
    for round_counter in range(1, int(rounds) + 2):
        round_keys.append(hi.astype(np.uint64, copy=False))
        if round_counter > int(rounds):
            break
        new_hi = ((hi & np.uint64(0x7)) << np.uint64(61)) | (lo >> np.uint64(3))
        new_lo = ((lo & np.uint64(0x7)) << np.uint64(61)) | (hi >> np.uint64(3))
        top0 = SBOX[(new_hi >> np.uint64(60)).astype(np.uint16)]
        top1 = SBOX[((new_hi >> np.uint64(56)) & np.uint64(0xF)).astype(np.uint16)]
        hi = (
            (top0 << np.uint64(60))
            | (top1 << np.uint64(56))
            | (new_hi & np.uint64(0x00FFFFFFFFFFFFFF))
        )
        lo = new_lo ^ (np.uint64(round_counter & 0x3) << np.uint64(62))
        hi = hi ^ np.uint64((round_counter >> 2) & 0x7)
    return round_keys


def expand_key(key, rounds, version=80):
    version = _validate_version(version)
    if int(rounds) < 0:
        raise ValueError("rounds must be non-negative")
    if version == 80:
        return _expand_key80(key, rounds)
    return _expand_key128(key, rounds)


def encrypt(plaintext, round_keys):
    scalar = np.asarray(plaintext).ndim == 0
    state = _as_u64_array(plaintext)
    if not round_keys:
        return _maybe_scalar(state, scalar)
    for round_key in round_keys[:-1]:
        state = state ^ _as_u64_array(round_key)
        state = _p_layer(_sbox_layer(state))
    state = state ^ _as_u64_array(round_keys[-1])
    return _maybe_scalar(state, scalar)


def encrypt_block(plaintext, key, rounds=31, version=80):
    return encrypt(np.uint64(int(plaintext) & MASK64_INT), expand_key(key, rounds, version=version))


def check_testvector():
    return (
        encrypt_block(0x0000000000000000, 0x00000000000000000000, rounds=31, version=80)
        == 0x5579C1387B228445
        and encrypt_block(0x0000000000000000, 0x00000000000000000000000000000000, rounds=31, version=128)
        == 0x96DB702A2E6900AF
    )


if __name__ == "__main__":
    print("PRESENT test vectors:", "ok" if check_testvector() else "failed")
