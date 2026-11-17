#!/usr/bin/env python3
"""Shared state conversion, atomic storage and experiment cache paths."""

import hashlib
import json
import platform
import math
import os
import tempfile
from pathlib import Path


N_BITS = 128
WORD_SIZE = 64


def normalize_bits(bits, n_bits=N_BITS):
    values = sorted(set(int(bit) for bit in bits))
    if any(bit < 0 or bit >= n_bits for bit in values):
        raise ValueError(f"paper bits outside [0,{n_bits}): {values}")
    return values


def paper_bits_to_hex_pair(bits, word_size=WORD_SIZE, n_bits=None):
    """Convert paper-indexed bits to public hex words.

    The historical two-word convention is preserved: paper bits in the first
    word are the right/v1 branch and output is ``[left, right]``.  Multi-word
    ciphers use direct paper order ``[v0, v1, ...]``.
    """
    n_bits = 2 * word_size if n_bits is None else int(n_bits)
    if n_bits % int(word_size):
        raise ValueError("n_bits must be a multiple of word_size")
    word_count = n_bits // int(word_size)
    words = [0] * word_count
    for bit in normalize_bits(bits, n_bits):
        word, offset = divmod(bit, int(word_size))
        words[word] |= 1 << offset
    if word_count == 2:
        words = [words[1], words[0]]
    width = (int(word_size) + 3) // 4
    return [f"0x{value:0{width}x}" for value in words]


def probability_to_weight(value):
    value = abs(float(value))
    return math.inf if value <= 0.0 else -math.log2(value)


def weight_to_abs_correlation(weight):
    return 0.0 if not math.isfinite(float(weight)) else 2.0 ** (-float(weight))


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def read_jsonl(path):
    rows = []
    try:
        with open(path, "r") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
    except FileNotFoundError:
        pass
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as handle:
            for row in rows:
                handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise



def collect_fingerprint():
    """Hash executable code and bundled OCP constraints for cache identity."""
    repo = Path(__file__).resolve().parents[2]
    suffixes = {".py", ".cpp", ".c", ".hpp", ".h"}
    files = []
    for folder in ("Cipher", "Framework", "Experiments", "OCP"):
        files.extend(path for path in (repo / folder).rglob("*")
                     if path.is_file() and "build" not in path.parts and (
                         path.suffix in suffixes or
                         (folder == "OCP" and path.parent.name == "sbox_modeling" and path.suffix == ".txt") or
                         path == repo / "Framework" / "requirements.txt"))
    rows = [{"path": str(path.relative_to(repo)),
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(files)]
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    return {"schema": "paper-framework-v2", "combined_sha256": digest,
            "files": rows, "python": platform.python_version()}
