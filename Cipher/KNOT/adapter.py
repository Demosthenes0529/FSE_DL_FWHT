"""KNOT-256 representation and AEAD endpoint rules for the shared OCP adapter."""

from .reference import SBOX, encrypt_segment, get_cell, int_to_hex, inverse_linear_layer, linear_layer, set_cell
from Framework.search.adapter import require_version
from Framework.search.models import CipherMetadata
from Framework.search.adapter import SBoxOCPAdapter


class KNOT256Adapter(SBoxOCPAdapter):
    # This is an unkeyed permutation; the registry also accepts 256 as a
    # state-size selector.  AEAD endpoint restrictions do not add round keys.
    metadata = CipherMetadata(
        name="knot256", variant="KNOT-256 (unkeyed permutation)", block_bits=256,
        key_bits=0, max_rounds=63, cell_bits=4,
        cells=tuple((column, 64 + column, 128 + column, 192 + column) for column in range(64)))
    sbox = SBOX
    gpu_name = "knot256"
    word_size = 64
    format_state = staticmethod(int_to_hex)
    get_cell = staticmethod(get_cell)
    set_cell = staticmethod(set_cell)
    linear = staticmethod(linear_layer)
    inverse_linear = staticmethod(inverse_linear_layer)
    transpose_linear = staticmethod(inverse_linear_layer)

    def encrypt_segment(self, plaintext, key, round_start, rounds, version=0):
        require_version(self, version)
        return encrypt_segment(plaintext, key, round_start, rounds)

    def endpoint_constraints(self, stage, config):
        if config.knot_mode == "permutation":
            return {}
        if config.knot_mode != "aead":
            raise ValueError("KNOT mode must be aead or permutation")
        # Initialization has S = K || N with an unchanged key; output masks
        # select only the first 64-bit rate word observable through AEAD.
        return {"input_zero_bits": list(range(128))} if stage == "E1" else {"output_zero_bits": list(range(64, 256))}
