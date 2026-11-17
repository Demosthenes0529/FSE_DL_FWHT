"""PRESENT metadata and representation for the shared OCP S-box adapter."""

from .reference import SBOX, encrypt, expand_key, get_cell, int_to_hex, inverse_linear_layer, linear_layer, set_cell
from Framework.search.adapter import require_version
from Framework.search.models import CipherMetadata
from Framework.search.adapter import SBoxOCPAdapter


class PRESENTAdapter(SBoxOCPAdapter):
    sbox = tuple(int(value) for value in SBOX)
    word_size = 64
    format_state = staticmethod(int_to_hex)
    get_cell = staticmethod(get_cell)
    set_cell = staticmethod(set_cell)
    linear = staticmethod(linear_layer)
    inverse_linear = staticmethod(inverse_linear_layer)
    transpose_linear = staticmethod(inverse_linear_layer)

    def __init__(self, version=80):
        version = int(version)
        if version not in (80, 128):
            raise ValueError("PRESENT version must be 80 or 128")
        self.version = version
        self.gpu_name = f"present{version}"
        self.metadata = CipherMetadata(
            name=self.gpu_name, variant=f"PRESENT-64/{version}", block_bits=64, key_bits=version,
            max_rounds=31, cell_bits=4,
            cells=tuple(tuple(range(4 * cell, 4 * cell + 4)) for cell in range(16)))

    def encrypt_segment(self, plaintext, key, round_start, rounds, version=None):
        version = self.version if version is None else version
        require_version(self, version)
        keys = expand_key(key, int(round_start) + int(rounds), version=int(version))
        return int(encrypt(int(plaintext), keys[int(round_start):int(round_start) + int(rounds) + 1]))
