"""Canonical cipher metadata plus interchangeable CPU and tensor interfaces.

``version`` denotes the master-key size; the block size is part of ``name``.
Names such as ``speck128/256`` and ``PRESENT-80`` are also accepted.
Importing this module never imports PyTorch, Triton, or a GPU implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import re


@dataclass(frozen=True)
class CipherSpec:
    name: str
    family: str
    block_size: int
    word_size: int
    key_size: int
    state_words: int
    key_word_bits: tuple | None = None

    @property
    def key_words(self):
        return len(self.key_word_bits) if self.key_word_bits is not None else self.key_size // self.word_size

    @property
    def max_rounds(self):
        if self.family == "present":
            return 31
        if self.family == "knot":
            return 63
        module = import_module(f"Cipher.{_DIRECTORIES[self.family]}.reference")
        if self.family == "cham":
            return module.DEFAULT_ROUNDS[self.block_size]
        return getattr(module, f"{self.family.upper()}_ROUNDS")[self.block_size, self.key_size]


_DIRECTORIES = {"speck": "Speck", "simon": "Simon", "simeck": "Simeck",
                "cham": "Cham", "present": "Present", "knot": "KNOT"}
_DEFAULT_KEYS = {
    "speck": {32: 64, 48: 96, 64: 128, 96: 96, 128: 128},
    "simon": {32: 64, 48: 96, 64: 128, 96: 144, 128: 256},
    "simeck": {32: 64, 48: 96, 64: 128}, "cham": {64: 128, 128: 128},
}
_KEY_SIZES = {
    "speck": {32: (64,), 48: (72, 96), 64: (96, 128), 96: (96, 144), 128: (128, 192, 256)},
    "simon": {32: (64,), 48: (72, 96), 64: (96, 128), 96: (96, 144), 128: (128, 192, 256)},
    "simeck": {32: (64,), 48: (96,), 64: (128,)}, "cham": {64: (128,), 128: (128,)},
}


def available_ciphers():
    """Return the 18 canonical names using the release's default key sizes."""
    return tuple(f"{family}{block}" for family, blocks in _DEFAULT_KEYS.items()
                 for block in blocks) + ("present80", "present128", "knot256")


def available_variants():
    """Return metadata for all 28 supported block/key combinations."""
    specs = [get_cipher_spec(f"{family}{block}", key)
             for family, blocks in _KEY_SIZES.items()
             for block, keys in blocks.items() for key in keys]
    return tuple(specs) + tuple(get_cipher_spec(name) for name in ("present80", "present128", "knot256"))


def get_cipher_spec(name, version=None):
    text = re.sub(r"[-_ ]", "", str(name).lower())
    match = re.fullmatch(r"(speck|simon|simeck|cham|present|knot)(\d+)?(?:/(\d+))?", text)
    if match is None:
        raise ValueError(f"unknown cipher: {name}; available: {', '.join(available_ciphers())}")
    family, size, explicit_key = match.groups()
    if explicit_key is not None:
        if version is not None and int(version) != int(explicit_key):
            raise ValueError("conflicting key sizes in name and version")
        version = int(explicit_key)
    if family == "present":
        key_size = int(version if version is not None else (size or 80))
        if key_size not in (80, 128) or (size and int(size) not in (64, 80, 128)):
            raise ValueError("PRESENT supports 80- or 128-bit keys")
        return CipherSpec(f"present{key_size}", family, 64, 64, key_size, 1,
                          (16, 64) if key_size == 80 else (64, 64))
    if family == "knot":
        if (size is not None and int(size) != 256) or version not in (None, 0, 256, "0", "256"):
            raise ValueError("only the unkeyed KNOT-256 permutation is included")
        return CipherSpec("knot256", family, 256, 64, 0, 4, ())
    block = int(size or (128 if family == "cham" else 32))
    key_size = _DEFAULT_KEYS[family].get(block) if version is None else int(version)
    if key_size not in _KEY_SIZES[family].get(block, ()):
        raise ValueError(f"unsupported {family}{block}/{key_size}")
    words = 4 if family == "cham" else 2
    return CipherSpec(f"{family}{block}", family, block, block // words, key_size, words)


def get_gpu_primitive(name, version=None):
    spec = get_cipher_spec(name, version)
    module = import_module(f"Cipher.{_DIRECTORIES[spec.family]}.gpu")
    if spec.family == "present":
        return getattr(module, f"Present{spec.key_size}GPUPrimitive")()
    if spec.family == "knot":
        return module.KNOT256GPUPrimitive()
    cls = getattr(module, f"{_DIRECTORIES[spec.family]}GPUPrimitive")
    return cls(spec.block_size, spec.key_size)


class ReferenceCipher:
    """Bound reference with the same word/key layout as its tensor primitive.

    ``encrypt_batch`` accepts NumPy word arrays or scalar words. The integer
    ``encrypt_segment`` interface packs ARX left/right in natural order, CHAM
    word 0 in the low word, and KNOT row 0 in the high word. PRESENT includes
    the final key XOR, matching its existing reduced-round primitive.
    """
    def __init__(self, spec):
        self.config = spec
        self.module = import_module(f"Cipher.{_DIRECTORIES[spec.family]}.reference")

    def encrypt_batch(self, state, key, rounds, round_start=0):
        import numpy as np
        spec, ref = self.config, self.module
        start, rounds = int(round_start), int(rounds)
        if min(start, rounds) < 0:
            raise ValueError("round interval must be nonnegative")
        if len(state) != spec.state_words or len(key) != spec.key_words:
            raise ValueError("state/key word count disagrees with the cipher configuration")
        state = tuple(np.asarray(v).astype(np.uint64) for v in state)
        key = tuple(np.asarray(v).astype(np.uint64) for v in key)
        if spec.family == "speck":
            return ref.encrypt_segment(state, key, rounds, start, spec.block_size, spec.key_size)
        if spec.family in ("simon", "simeck"):
            keys = ref.expand_key(key, start + rounds, block_size=spec.block_size, key_size=spec.key_size)
            return ref.encrypt(state, keys[start:], version=spec.block_size)
        if spec.family == "cham":
            keys = ref.expand_key(key, block_size=spec.block_size, key_size=spec.key_size)
            return ref.encrypt_rounds(state, keys, start, rounds, spec.block_size)
        if spec.family == "present":
            keys = ref.expand_key(key, start + rounds, version=spec.key_size)
            return (np.asarray(ref.encrypt(state[0], keys[start:]), dtype=np.uint64),)
        return ref.encrypt_batch(state, key, rounds, round_start=start)

    def _state_words(self, value):
        spec = self.config
        mask = (1 << spec.word_size) - 1
        words = tuple((int(value) >> (i * spec.word_size)) & mask for i in range(spec.state_words))
        return words if spec.family == "cham" else tuple(reversed(words))

    def _key_words(self, value):
        spec = self.config
        if spec.key_size == 0:
            if int(value or 0):
                raise ValueError("KNOT permutation takes no key")
            return ()
        if spec.family == "present":
            return (int(value) >> 64) & ((1 << (spec.key_size - 64)) - 1), int(value) & ((1 << 64) - 1)
        mask = (1 << spec.word_size) - 1
        words = tuple((int(value) >> (i * spec.word_size)) & mask for i in range(spec.key_words))
        return words if spec.family == "cham" else tuple(reversed(words))

    def encrypt_segment(self, plaintext, key, round_start, rounds, version=None):
        if version is not None and int(version) != self.config.key_size:
            return get_reference(self.config.name, version).encrypt_segment(plaintext, key, round_start, rounds)
        state = self.encrypt_batch(self._state_words(plaintext), self._key_words(key), rounds, round_start)
        if self.config.family == "cham":
            state = tuple(reversed(state))
        value = 0
        for word in state:
            value = (value << self.config.word_size) | (int(word) & ((1 << self.config.word_size) - 1))
        return value


def get_reference(name, version=None):
    return ReferenceCipher(get_cipher_spec(name, version))
