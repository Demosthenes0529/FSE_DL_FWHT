"""Independent small checks for cipher interfaces, search and aggregation.

Default: python -m Experiments.checks --suite interfaces framework
Full CPU/SAT/native checks: add --suite all --cms-bin /path/to/cryptominisat5
Aggregation requires pycryptosat; tensor interface checks require PyTorch.
Dependencies are imported only by the selected checks. All generated files
and native binaries live in temporary directories, not the release package.
These checks establish implementation consistency, not long-round estimates.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import contextlib
from dataclasses import replace
from functools import lru_cache
import io
import itertools
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

FAMILIES = ("speck32", "simon32", "simeck32", "cham64", "present80", "knot256")


def parity(value):
    """Parity also works in the Python 3.9 environment used by CryptoMiniSat."""
    return bin(int(value)).count("1") & 1


# Cipher implementations, coordinates and tensor dispatch.


def _tensor(value, device):
    import numpy as np
    import torch

    return torch.from_numpy(np.asarray(value, dtype=np.uint64).view(np.int64).copy()).to(device)


def _unsigned(value):
    import numpy as np

    return value.detach().cpu().numpy().view(np.uint64)


def check_known_vectors():
    """Retained known-answer values, independent of the new dispatch layer."""
    for family in ("Simon", "Simeck"):
        module = __import__(f"Cipher.{family}.reference", fromlist=["_"])
        for (block, key_size), plain, key, expected in getattr(module, f"{family.upper()}_TEST_VECTORS"):
            actual = module.encrypt_block(plain, key, block_size=block, key_size=key_size)
            assert tuple(int(x) for x in actual) == expected, (family, block, key_size)
    from Cipher.Speck.reference import encrypt_block
    for block, key_size, plain, key, expected in (
        (32, 64, (0x6574, 0x694C), (0x1918, 0x1110, 0x0908, 0x0100), (0xA868, 0x42F2)),
        (128, 128, (0x6C61766975716520, 0x7469206564616D20),
         (0x0F0E0D0C0B0A0908, 0x0706050403020100), (0xA65D985179783265, 0x7860FEDF5C570D18)),
    ):
        actual = encrypt_block(plain, key, block_size=block, key_size=key_size)
        assert tuple(int(x) for x in actual) == expected, ("speck", block, key_size)
    from Cipher.Present.reference import check_testvector
    assert check_testvector(), "PRESENT known-answer vectors"
    from Cipher.KNOT.reference import encrypt_segment
    assert encrypt_segment(0, 0, 0, 9) == int(
        "6c70cb8d8c31177d0f6aca3b2f0eb3e7d6d4b72d609aa8fe8eba21b029edf1c4", 16
    ), "KNOT-256 retained zero-state nine-round vector"


def check_paper_coordinates(spec, primitive, device):
    import torch
    from Framework.fwht.distributed import primitive_paper_bit_matrix, primitive_paper_bit_plane

    block, width = spec.block_size, spec.word_size
    differences = [primitive.paper_bits_to_difference([bit]) for bit in range(block)]
    rows = tuple(_tensor([int(row[word]) & ((1 << width) - 1) for row in differences], device)
                 for word in range(spec.state_words))
    packed = primitive.output_to_uint(rows)
    matrix = primitive_paper_bit_matrix(primitive, packed)
    assert torch.equal(matrix.cpu(), torch.eye(block, dtype=torch.uint8)), (spec.name, "coordinate round trip")
    for bit in (0, width - 1, block - 1):
        assert torch.equal(primitive_paper_bit_plane(primitive, packed, bit), matrix[:, bit]), (spec.name, bit)
    # Physical endpoint anchors prevent mutually inverse but wrong conventions.
    first = tuple(int(v) & ((1 << width) - 1) for v in differences[0])
    if spec.family in ("speck", "simon", "simeck"):
        expected = (0, 1)
    elif spec.family == "cham":
        expected = (1, 0, 0, 0)
    elif spec.family == "present":
        expected = (1 << 63,)
    else:
        expected = (1 << 63, 0, 0, 0)
    assert first == expected, (spec.name, "paper bit zero", first)


def check_variant(spec, device, samples):
    import numpy as np
    import torch
    from Framework.common.gpu import fwht_inplace_int64
    from Cipher.registry import get_gpu_primitive, get_reference
    from Framework.fwht.distributed import project_packed_state

    primitive = get_gpu_primitive(spec.name, spec.key_size)
    reference = get_reference(spec.name, spec.key_size)
    rng = np.random.default_rng(81331)

    def words(widths):
        result = []
        for width in widths:
            value = rng.integers(0, 1 << 32, samples, dtype=np.uint64)
            value |= rng.integers(0, 1 << 32, samples, dtype=np.uint64) << np.uint64(32)
            result.append(value & np.uint64((1 << width) - 1))
        return tuple(result)

    state = words([spec.word_size] * spec.state_words)
    key = words(spec.key_word_bits or [spec.word_size] * spec.key_words)
    tensor_state, tensor_key = tuple(_tensor(x, device) for x in state), tuple(_tensor(x, device) for x in key)
    for start, rounds in ((0, 0), (0, 1), (0, spec.max_rounds), (5, min(7, spec.max_rounds - 5))):
        expected = reference.encrypt_batch(state, key, rounds, round_start=start)
        actual = primitive.encrypt_batch(tensor_state, tensor_key, rounds, round_start=start)
        for index, (cpu, tensor) in enumerate(zip(expected, actual)):
            assert np.array_equal(cpu, _unsigned(tensor)), (spec.name, spec.key_size, start, rounds, index)
    check_paper_coordinates(spec, primitive, device)

    # Projection and FWHT must reproduce direct parity on the same data.
    packed = primitive.output_to_uint(actual)
    panel = sorted(set((0, spec.word_size - 1, spec.block_size // 2, spec.block_size - 1)))
    projected = project_packed_state(packed, panel, spec.block_size, spec.word_size, primitive=primitive)
    histogram = torch.bincount(projected.to(torch.int64), minlength=1 << len(panel)).to(torch.int64)
    walsh = fwht_inplace_int64(histogram).cpu().tolist()
    indices = projected.cpu().tolist()
    direct = [sum(1 if parity(int(value) & mask) == 0 else -1 for value in indices)
              for mask in range(1 << len(panel))]
    assert walsh == direct, (spec.name, "projected FWHT/direct parity")


def check_backend_versions():
    """Stop before any allocation but verify CUDA dispatch uses the given key size."""
    from Cipher.registry import available_variants, get_gpu_primitive

    from Framework.fwht import cuda as backend

    class BeforeAllocation(Exception):
        pass

    for spec in available_variants():
        selected = []

        def primitive(name, version):
            result = get_gpu_primitive(name, version)
            selected.append(result.config.key_size)
            return result

        with patch.object(backend, "get_gpu_primitive", primitive), \
             patch.object(backend, "resolve_gpu_ids", side_effect=BeforeAllocation):
            try:
                backend.run_cuda_em([0], 1, SimpleNamespace(root="."), 16, 42, 0,
                                        spec.name, "0", 16, panel_size=4, version=spec.key_size)
            except BeforeAllocation:
                pass
        assert selected == [spec.key_size], (spec.name, spec.key_size, selected)


# FWHT, complete search, split search and SPN constraints.


def spn_tables(family):
    """Derive probabilities from the executable S-box, independently of OCP."""
    if family == "present":
        from Cipher.Present.reference import SBOX
    else:
        from Cipher.KNOT.reference import SBOX
    box = tuple(map(int, SBOX))
    ddt = [[sum(box[x] ^ box[x ^ a] == b for x in range(16))
            for b in range(16)] for a in range(16)]
    lat = [[sum((-1) ** (parity((a & x) ^ (b & box[x])))
                for x in range(16)) for b in range(16)] for a in range(16)]
    return ddt, lat


def check_spn_templates():
    """Exhaust all local assignments, including the SAT weight variables.

    This checks the shipped constraints themselves, not just OCP's copy of
    the S-box table. No SAT solver, model generator, or cached result is used.
    """
    templates = Path(__file__).resolve().parents[1] / "OCP/files/sbox_modeling"
    for family in ("present", "knot"):
        for model, table in zip(("XORDIFF", "LINEAR"), spn_tables(family)):
            path = templates / f"constraints_sat_{family.upper()}_Sbox_{model}_PR_minimize_logic_0.txt"
            lines = path.read_text().splitlines()
            clauses = ast.literal_eval(next(line.removeprefix("Constraints: ")
                                            for line in lines if line.startswith("Constraints: ")))
            clauses = [clause.split() for clause in clauses]
            weights = sorted(set(re.findall(r"\bp\d+\b", " ".join(lines))))
            assert next(line for line in lines if line.startswith("Weight: ")) == (
                "Weight: " + " + ".join(weights)), (path, "unexpected weight encoding")
            for a, b in itertools.product(range(16), repeat=2):
                accepted = set()
                for weight_bits in itertools.product((0, 1), repeat=len(weights)):
                    values = {f"a{i}": (a >> (3 - i)) & 1 for i in range(4)}
                    values.update({f"b{i}": (b >> (3 - i)) & 1 for i in range(4)})
                    values.update(zip(weights, weight_bits))
                    if all(any(bool(values[literal.lstrip("-")]) != literal.startswith("-")
                               for literal in clause) for clause in clauses):
                        accepted.add(sum(weight_bits))
                expected = {-math.log2(abs(table[a][b]) / 16)} if table[a][b] else set()
                assert accepted == expected, (path.name, a, b, accepted, expected)


def spn_cells(family, state):
    if family == "present":
        return [(state >> (60 - 4 * cell)) & 15 for cell in range(16)]
    return [sum(((state >> (255 - row * 64 - column)) & 1) << row
                for row in range(4)) for column in range(64)]


def spn_state(family, cells):
    if family == "present":
        return sum(value << (60 - 4 * cell) for cell, value in enumerate(cells))
    return sum(((value >> row) & 1) << (255 - row * 64 - column)
               for column, value in enumerate(cells) for row in range(4))


def spn_linear(family, state, inverse=False):
    """Bit permutation written independently of adapter/SAT mapping helpers."""
    if family == "present":
        factor = 4 if inverse else 16
        return sum(((state >> bit) & 1) << (factor * bit % 63 if bit < 63 else 63)
                   for bit in range(64))
    mask, result = (1 << 64) - 1, 0
    for row, shift in enumerate((0, 1, 8, 25)):
        word = (state >> (192 - 64 * row)) & mask
        shift = (-shift if inverse else shift) % 64
        result |= (((word << shift) | (word >> (64 - shift))) & mask) << (192 - 64 * row)
    return result


def spn_short_optimum(family, endpoint, rounds, linear):
    """Enumerate one active endpoint S-box and finish with independent minima.

    Only used on the sparse fixtures below; this is deliberately not a
    general search implementation or a fallback for production searches.
    """
    table = spn_tables(family)[int(linear)]
    def choices(state):
        local = spn_cells(family, state if linear else spn_linear(family, state, True))
        return [[(other, -math.log2(abs(table[value][other] if linear else table[other][value]) / 16))
                 for other in range(16) if (table[value][other] if linear else table[other][value])]
                for value in local]
    def one_round(state):
        return sum(min(weight for _, weight in items) for items in choices(state))
    if rounds == 1:
        return one_round(endpoint)
    candidates = choices(endpoint)
    assert math.prod(map(len, candidates)) <= 16, "fixture must remain small"
    return min(sum(weight for _, weight in combination) + one_round(
        spn_linear(family, spn_state(family, [value for value, _ in combination]))
        if linear else spn_state(family, [value for value, _ in combination]))
        for combination in itertools.product(*candidates))


def check_spn_trail(family, row, endpoint, rounds, linear):
    states = [int(value, 16) for value in row["states"]]
    assert len(states) == rounds + 1
    assert states[0 if linear else -1] == endpoint
    table = spn_tables(family)[int(linear)]
    weights = []
    for before, after in zip(states, states[1:]):
        local_in = spn_cells(family, before)
        local_out = spn_cells(family, spn_linear(family, after, True))
        counts = [abs(table[a][b]) for a, b in zip(local_in, local_out)]
        assert all(counts), (family, "impossible local transition", row)
        weights.append(sum(-math.log2(count / 16) for count in counts))
    assert weights == row["round_weights"] and sum(weights) == row["weight"], row
    assert row["path_verified"] and row["optimum_proven"], row
    if rounds:
        assert "ocp" in row["backend"], row


def check_spn_ocp(directory, cms_bin):
    """Real nonzero SAT queries, exact short-round optima and failure states."""
    from Framework.search.adapter import get_adapter

    for name in ("present80", "present128", "knot256"):
        adapter = get_adapter(name)
        family = "present" if name.startswith("present") else "knot"
        gamma = spn_state(family, [1] + [0] * (len(adapter.metadata.cells) - 1))
        delta = spn_linear(family, gamma)
        for rounds in (0, 1, 2):
            config = configuration(adapter, directory / name, r1=rounds, r2=rounds,
                                   cms_bin=cms_bin, e1_weight_bound=16, e2_weight_bound=16)
            for linear, endpoint in ((False, delta), (True, gamma)):
                result = (adapter.search_e2(endpoint, 16, config) if linear
                          else adapter.search_e1(endpoint, config))
                assert result["status"] == "ok" and len(result["results"]) == 1, (name, rounds, result)
                row = result["results"][0]
                check_spn_trail(family, row, endpoint, rounds, linear)
                optimum = spn_short_optimum(family, endpoint, rounds, linear) if rounds else 0
                assert row["weight"] == optimum, (name, rounds, linear, row, optimum)
        config = configuration(adapter, directory / name, r1=1, r2=1,
                               cms_bin=cms_bin, e1_weight_bound=0, e2_weight_bound=0)
        for result in (adapter.search_e1(delta, config), adapter.search_e2(gamma, 0, config)):
            assert result["status"] == "no_path" and not result["results"], result
        config.cms_bin = str(directory / "solver-does-not-exist")
        for result in (adapter.search_e1(delta, config), adapter.search_e2(gamma, 16, config)):
            assert result["status"] == "solver_unavailable" and not result["results"], result
        config.cms_bin, config.timeout_seconds = cms_bin, 1e-6
        for result in (adapter.search_e1(delta, config), adapter.search_e2(gamma, 16, config)):
            assert result["status"] == "resource_limit" and not result["results"], result
    check_knot_aead(directory, cms_bin)


def check_knot_aead(directory, cms_bin):
    from Framework.search.adapter import get_adapter

    adapter = get_adapter("knot256")
    ddt, lat = spn_tables("knot")
    for rounds in (0, 1, 2):
        config = configuration(adapter, directory / "aead", r1=rounds, r2=rounds,
                               cms_bin=cms_bin, knot_mode="aead", e1_weight_bound=32)
        # Construct a feasible nonce-only input path, then forget its input.
        delta = 1 << 127
        for _ in range(rounds):
            cells = [next(b for b in range(16) if ddt[a][b]) for a in spn_cells("knot", delta)]
            delta = spn_linear("knot", spn_state("knot", cells))
        e1 = adapter.search_e1(delta, config)
        assert e1["status"] == "ok" and e1["results"], e1
        row = e1["results"][0]
        check_spn_trail("knot", row, delta, rounds, False)
        assert int(row["delta_in"], 16) >> 128 == 0, row
        # Construct a feasible path backwards from a rate-only output mask.
        gamma = 1 << 255
        for _ in range(rounds):
            beta = spn_linear("knot", gamma, True)
            cells = [next(a for a in range(16) if lat[a][b]) for b in spn_cells("knot", beta)]
            gamma = spn_state("knot", cells)
        e2 = adapter.search_e2(gamma, 32, config)
        assert e2["status"] == "ok" and e2["results"], e2
        row = e2["results"][0]
        check_spn_trail("knot", row, gamma, rounds, True)
        assert int(row["gamma_out"], 16) & ((1 << 192) - 1) == 0, row
    config.r1 = config.r2 = 0
    assert adapter.search_e1(1 << 255, config)["status"] == "no_path"
    assert adapter.search_e2(1, 32, config)["status"] == "no_path"
    config.r1 = config.r2 = 1
    # Both queries are feasible for the permutation and forbidden by AEAD.
    delta = spn_linear("knot", spn_state("knot", [3] + [0] * 63))
    gamma = 1 << 255
    for mode in ("permutation", "aead"):
        config.knot_mode = mode
        expected = "ok" if mode == "permutation" else "no_path"
        assert adapter.search_e1(delta, config)["status"] == expected
        assert adapter.search_e2(gamma, 32, config)["status"] == expected


def configuration(adapter, directory, **overrides):
    from Framework.search.models import SearchConfig

    values = dict(cipher=adapter.metadata.name, version=adapter.metadata.key_bits,
                  r1=0, rm=1, r2=0, round_start=0, output_dir=str(directory),
                  fixed_deltas=[adapter.from_paper_bits([0])], delta_candidates=1,
                  backend="cpu", panel_size=4, eval_samples_power=6,
                  scout_samples_power=6, parallel_workers=1, validation_threads=1,
                  em_top_k=2, em_top_per_hw=2, leaderboard_size=2,
                  validation_enabled=True, validation_samples_power=6,
                  validation_num_keys=2, validation_top_n=2, resume=False,
                  knot_mode="permutation", timeout_seconds=30)
    values.update(overrides)
    return SearchConfig(**values)


def check_transform():
    import numpy as np
    from Framework.fwht.cpu import fwht

    histogram = np.array([3, 0, 1, 4, 0, 2, 1, 0, 5, 0, 0, 1, 2, 0, 1, 0])
    transformed = fwht(histogram)
    expected = [sum(int(count) * (-1 if parity(value & mask) else 1)
                    for value, count in enumerate(histogram)) for mask in range(16)]
    assert transformed.tolist() == expected, "FWHT differs from explicit parity sums"


def check_zero_round_spectrum(adapter, directory):
    from Framework.fwht.scan import scan

    delta = adapter.from_paper_bits([0, 2, adapter.metadata.block_bits - 1])
    config = configuration(adapter, directory, rm=0, rho_threshold=0)
    result = scan(adapter, delta, config, directory)
    assert result.samples_generated == 64
    assert result.data_generation_passes == 1
    for item in result.domains["direct"]["panels"]:
        rows = item["rows"]
        assert len(rows) == (1 << len(item["panel"].bits)) - 1
        for row in rows:
            mask = int(row["mask_hex"], 16)
            expected = -1 if parity(delta & mask) else 1
            assert row["rho"] == expected and row["Walsh"] == expected * 64, (
                adapter.metadata.name, row, expected)


def check_search(adapter, directory, cms_bin=None):
    from Framework.search.engine import run

    overrides = {}
    if cms_bin:
        overrides.update(r1=1, r2=1, cms_bin=cms_bin,
                         e1_weight_bound=8, e2_weight_bound=8)
    config = configuration(adapter, directory, **overrides)
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        result = run(adapter, config)
    if result["status"] != "completed":
        raise AssertionError(f"{adapter.metadata.name}: {result['unresolved_queries']}\n{capture.getvalue()}")
    assert result["num_complete_paths"] > 0, (adapter.metadata.name, result)
    assert result["num_validated_paths"] > 0, (adapter.metadata.name, result)
    validated = Path(directory) / "paths" / "paths_valid.jsonl"
    rows = [json.loads(line) for line in validated.read_text().splitlines()]
    assert all(len(row["validation"]["per_key"]) == 2 for row in rows)
    candidates = next(Path(directory).glob("delta_runs/*/em_candidates.jsonl"))
    return [json.loads(line) for line in candidates.read_text().splitlines()]



def check_fixed_key_factors(directory):
    """Check signed linear counts and pairing without assuming independent keys."""
    import numpy as np
    from Experiments.backends import stream_seed
    from Experiments.validation import ValidationConfig, paired_summary, validate
    from Framework.search.adapter import get_adapter

    # Zero-round PRESENT must be the identity, with no whitening key added.
    identity = ValidationConfig(rounds=0, round_start=4, samples_power=4, num_keys=2,
                                measurement="linear", sample_seed=8123, batch_size=5)
    keys = [[0x123456789ABCDEF0, 0xFEDCBA9876543210], [0x5555, 0xAAAA]]
    mask = 0x8040000000000001
    result = validate(get_adapter("present128"), None, [mask], identity,
                      directory / "identity", input_mask=mask, master_keys=keys)[0]
    assert all(row["rho"] == row["q_squared"] == row["q_squared_noise_corrected"] ==
               row["q_squared_split_half"] == 1 for row in result["per_key"]), result

    # Recreate only the input samples; use scalar SIMON equations and the
    # directly supplied round-key words, independently of its batch encryptor.
    config = ValidationConfig(rounds=2, round_start=2, samples_power=5, num_keys=2,
                              measurement="linear", sample_seed=0x5566, batch_size=16)
    keys = [[0x1918, 0x1110, 0x0908, 0x0100], [0x1357, 0x2468, 0x369C, 0xABCD]]
    alpha, beta = (0x4001 << 16) | 0x0081, (0x0210 << 16) | 0x4002
    result = validate(get_adapter("simon32"), None, [beta], config, directory / "linear",
                      input_mask=alpha, master_keys=keys)[0]
    rotate = lambda value, amount: ((value << amount) | (value >> (16-amount))) & 65535
    for row, key in zip(result["per_key"], keys):
        assert [int(word, 0) for word in row["master_key_words_hex"]] == key
        seed = stream_seed(config.sample_seed, row["key_index"])
        assert row["stream_seed"] == seed
        rng = np.random.default_rng(seed)
        half_walsh = []
        for _ in range(2):
            left = rng.integers(0, 65535, size=16, dtype=np.uint64, endpoint=True)
            right = rng.integers(0, 65535, size=16, dtype=np.uint64, endpoint=True)
            walsh = 0
            for x, y in zip(map(int, left), map(int, right)):
                input_parity = parity((x & (alpha >> 16)) ^ (y & (alpha & 65535)))
                for round_key in (key[1], key[0]):  # SIMON32 round keys 2 and 3.
                    x, y = y ^ (rotate(x, 1) & rotate(x, 8)) ^ rotate(x, 2) ^ round_key, x
                walsh += 1 - 2 * (input_parity ^ parity((x & (beta >> 16)) ^ (y & (beta & 65535))))
            half_walsh.append(walsh)
        walsh = sum(half_walsh)
        assert row["Walsh"] == walsh and row["odd_parity_count"] == (32-walsh)//2, row
        assert [row["Walsh_first_half"], row["Walsh_second_half"]] == half_walsh, row
        assert row["rho"] == walsh/32 and row["q_squared"] == (walsh/32)**2
        assert row["q_squared_noise_corrected"] == (walsh*walsh/32-1)/31
        assert row["q_squared_split_half"] == half_walsh[0]*half_walsh[1]/256

    # Two correlated factors distinguish a paired mean from two separate means.
    em = {"measurement": "dl", "per_key": [
        {"key_index": i, "rho": r, "master_key_words_hex": list(map(hex, keys[i]))}
        for i, r in enumerate((-0.25, 0.75))]}
    suffix = {"measurement": "linear", "per_key": [
        {"key_index": i, "q_squared": q2, "master_key_words_hex": list(map(hex, keys[i]))}
        for i, q2 in enumerate((1, 0.25))]}
    paired = paired_summary(em, suffix)
    assert paired["mean_paired_abs_r_times_q_squared"] == 0.21875
    assert paired["product_of_separate_means"] == 0.3125
    suffix["per_key"][1]["master_key_words_hex"][0] = hex(keys[1][0] ^ 1)
    try:
        paired_summary(em, suffix)
    except ValueError as error:
        assert "master-key mismatch" in str(error)
    else:
        raise AssertionError("Different master keys were accepted as a pair")

    # KNOT repetitions are unkeyed batches, not fixed-key observations.
    knot = get_adapter("knot256", 0)
    knot_config = ValidationConfig(rounds=0, samples_power=3, num_keys=2)
    unkeyed = validate(knot, 1, [1], knot_config, directory / "knot")[0]
    assert knot.metadata.key_bits == 0 and unkeyed["unkeyed"]
    assert unkeyed["num_keys"] == 0 and unkeyed["num_independent_batches"] == 2
    assert all(row["master_key_words_hex"] == [] for row in unkeyed["per_key"])
    try:
        paired_summary(unkeyed, dict(unkeyed, measurement="linear"))
    except ValueError as error:
        assert "require a keyed cipher" in str(error), error
    else:
        raise AssertionError("KNOT batches were accepted as same-key observations")

def check_candidate_selection(directory):
    """Filtering precedes quotas; duplicate boundary masks share one slot."""
    from Framework.search.models import EmScanResult, Panel
    from Framework.search.adapter import get_adapter
    from Framework.fwht.candidates import scan_candidates

    adapter = get_adapter("present80")
    config = configuration(adapter, directory, rho_threshold=0.25, em_top_k=2,
                           em_top_per_hw=1, mask_hw_min=1, mask_hw_max=2)
    rows = []
    for bits, rho in (((), 1.0), ((0, 1, 2), 1.0), ((0,), 0.9), ((1,), 0.8),
                      ((0, 1), 0.7), ((0, 2), 0.25), ((2,), math.nan)):
        rows.append({"mask_hex": adapter.format_state(adapter.from_paper_bits(bits)),
                     "mask_index": sum(1 << bit for bit in bits), "rho": rho,
                     "source": "test_parity_counts"})
    duplicate = {"mask_hex": adapter.format_state(adapter.from_paper_bits([0, 1])),
                 "mask_index": 3, "rho": -0.75, "source": "test_parity_counts"}
    result = EmScanResult({"direct": {"singletons": [], "panels": [
        {"panel": Panel("direct", (0, 1, 2), "top-bits", "first"), "rows": rows},
        {"panel": Panel("direct", (0, 1, 2), "top-bits", "second"), "rows": [duplicate]},
    ]}}, 64, 1, True, True, "cpu_reference")
    selected = scan_candidates(adapter, result, config)
    assert [(row["gamma_m_paper_bits"], row["rhoEm_search"]) for row in selected] == [
        ([0], 0.9), ([0, 1], -0.75)], selected
    assert all(not {"gammaEm", "rho_search", "w_Em"}.intersection(row) for row in selected)


def check_resume(directory):
    """A failed suffix must retry without reusing a false completion marker."""
    from Framework.search.adapter import get_adapter
    from Framework.search.engine import run

    from Framework.search import engine
    adapter = get_adapter("speck32")
    config = configuration(adapter, directory, validation_enabled=False)
    with contextlib.redirect_stdout(io.StringIO()), \
         patch.object(engine, "ProcessPoolExecutor", ThreadPoolExecutor), \
         patch.object(engine, "_solve_e2_process", return_value={
             "status": "resource_limit", "results": [], "error": "interrupted test query"}):
        interrupted = run(adapter, config)
    assert interrupted["status"] == "incomplete" and interrupted["unresolved_queries"]
    assert not list(directory.glob("delta_runs/*/done.json"))
    cache = next(directory.glob("delta_runs/*/em_scan_done.json"))
    cached_time = cache.stat().st_mtime_ns
    resumed = replace(config, resume=True)
    with contextlib.redirect_stdout(io.StringIO()), \
         patch.object(engine, "ProcessPoolExecutor", ThreadPoolExecutor), \
         patch.object(adapter, "scan_em", side_effect=AssertionError("valid Em cache was not reused")):
        result = run(adapter, resumed)
        repeated = run(adapter, resumed)
    assert result["status"] == repeated["status"] == "completed"
    assert result["num_complete_paths"] == repeated["num_complete_paths"] > 0
    assert cache.stat().st_mtime_ns == cached_time

    def rejected(changed):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                run(adapter, changed)
        except ValueError as error:
            assert "identity mismatch" in str(error), str(error)
        else:
            raise AssertionError("incompatible search cache was accepted")

    rejected(replace(resumed, seed=resumed.seed + 1))
    identity_path = directory / "run_identity.json"
    original = identity_path.read_text()
    identity = json.loads(original)
    identity["implementation_fingerprint"] = {"incompatible_source": "changed"}
    identity_path.write_text(json.dumps(identity))
    try:
        rejected(resumed)
    finally:
        identity_path.write_text(original)


def check_split_endpoints_and_retry(directory):
    from Framework.split_ladder import SplitLadderRunner, build_parser
    args = build_parser().parse_args([
        "--ciphers", "knot256", "--output-root", str(directory), "--backend", "cpu",
        "--panel-size", "4", "--samples", "64", "--panel-samples", "64",
        "--knot-mode", "aead", "--rm-start", "1", "--em-round-start-values", "0",
        "--max-side-rounds-from-total", "1"])
    runner = SplitLadderRunner(args, "knot256")
    config = runner.cfg
    for stage, bit, status in (("E1", 0, "no_path"), ("E1", 128, "ok"),
                                ("E2", 128, "no_path"), ("E2", 0, "ok")):
        row = runner.query_frontier(stage, runner.adapter.from_paper_bits([bit]), config)
        assert row["status"] == status, (stage, bit, row)
        assert runner.certified_no_path(row) if status == "no_path" else row["backend"] == "identity"
    delta = runner.adapter.from_paper_bits([129])
    with patch.object(runner.adapter, "search_e1", return_value={
            "status": "resource_limit", "results": [], "error": "interrupted test query"}):
        interrupted = runner.query_frontier("E1", delta, config)
    assert interrupted["status"] == "resource_limit"
    retried = runner.query_frontier("E1", delta, config)
    assert retried["status"] == "ok" and retried["backend"] == "identity"
    # Run the actual frontier/status path with a finite, legal zero-round pair.
    candidate = {"delta_m": runner.adapter.format_state(delta),
                 "gamma_m": runner.adapter.format_state(1 << 254),
                 "em_start": 0, "rm": 1, "wEm_search": 1.0}
    with patch.object(runner, "run_rm_ladder", return_value=(1, 2, [1])), \
         patch.object(runner, "select_candidates", return_value=[candidate]), \
         patch.object(runner.adapter, "search_e2", return_value={
             "status": "resource_limit", "results": [], "error": "interrupted test query"}), \
         contextlib.redirect_stdout(io.StringIO()):
        try:
            runner.run()
        except RuntimeError:
            pass
        else:
            raise AssertionError("unfinished split was reported as complete")
    status = json.loads((runner.output_dir / "scout_status.json").read_text())
    assert status["status"] == "incomplete" and status["error_type"] == "IncompleteScoutError", status


def check_split(name, directory, main_candidates):
    command = [sys.executable, "-m", "Framework.split_ladder", "--mode", "em-only",
               "--ciphers", name, "--rm-values", "1", "--samples", "64",
               "--panel-samples", "64", "--panel-size", "4", "--backend", "cpu",
               "--em-top-k", "2", "--em-top-per-hw", "2",
               "--mask-hw-max", "4", "--seed", "42", "--knot-mode", "permutation",
               "--resume", "false", "--output-root", str(directory),
               "--em-round-start-values", "0"]
    result = subprocess.run(command, text=True, capture_output=True, timeout=120)
    if result.returncode:
        raise AssertionError(f"{name}: split ladder failed\n{result.stdout}\n{result.stderr}")
    paths = list(Path(directory).rglob("em_candidates.jsonl"))
    assert len(paths) == 4, (name, "expected four representative differences", paths)
    assert all(path.read_text().strip() for path in paths), (name, "no Em candidates")
    initial = next(path for path in paths if path.parent.name == "delta0")
    split_candidates = [json.loads(line) for line in initial.read_text().splitlines()]
    fields = set(main_candidates[0])
    assert [{key: row[key] for key in fields} for row in split_candidates] == main_candidates, (
        name, "main search and split scout selected different middle candidates")


# Projected SAT counts, window DP and recovery accounting.


@lru_cache(None)
def round_table(family, word):
    a, b, c = (1, 8, 2) if family == "simon" else (0, 5, 1)
    mask = (1 << word) - 1
    def rotate(value, amount):
        amount %= word
        return ((value << amount) | (value >> ((word - amount) % word))) & mask
    return tuple((rotate(value, a) & rotate(value, b)) ^ rotate(value, c)
                 for value in range(1 << word))


@lru_cache(None)
def differential_counts(family, word, delta):
    values = round_table(family, word)
    return Counter(values[value] ^ values[value ^ delta] for value in range(1 << word))


def linear_correlation(family, word, alpha, beta):
    return sum(1 - 2 * parity((alpha & value) ^ (beta & output))
               for value, output in enumerate(round_table(family, word))) / (1 << word)


def fixtures():
    for family, shift in (("simon", 2), ("simeck", 1)):
        for rounds in (1, 2):
            start = (1, 0)
            end = (1 << shift, 1) if rounds == 1 else ((1 << (2 * shift)) ^ 1, 1 << shift)
            yield dict(family=family, trail_kind="differential", word_size=16, rounds=rounds,
                       delta_in=start, delta_out=end, weight=2 * rounds, shard_round=1)
    yield dict(family="speck", trail_kind="linear", word_size=16, rounds=1,
               delta_in=(256, 2), delta_out=(2, 0), weight=1, shard_round=1)


def enumerate_model(config):
    from Framework.aggregation.allsat import build_model
    from Framework.aggregation.models import new_solver
    model = build_model(config, {})
    solver = new_solver(model)
    trails = []
    while True:
        satisfiable, solution = solver.solve(time_limit=20)
        assert satisfiable is not None, ("SAT timeout", config)
        if not satisfiable:
            break
        states = tuple(tuple(pair) for pair in model.decode_states(solution))
        assert states not in trails, "projected state sequence counted twice"
        assert model.replay_weight(solution) == config["weight"]
        trails.append(states)
        assert len(trails) <= 1000, "check fixture unexpectedly large"
        solver.add_clause([-var if solution[var] else var for var in model.state_variables()])
    return trails


def check_sat():
    from Framework.aggregation.run import resolve
    from Framework.aggregation.models import OCPLinearModel
    for family in ("simon", "simeck"):
        assert not resolve(family + "32", trail_kind="linear").hull_backend
        try:
            OCPLinearModel(16, 1, (0, 1), (1, 0x4000), 1, family=family)
        except ValueError as error:
            assert "fixed-key" in str(error), error
        else:
            raise AssertionError("Unverified SIMON-like linear AllSAT model was enabled")
    for config in fixtures():
        trails = enumerate_model(config)
        assert len(trails) == 1, (config, trails)
        mass = 1.0
        for before, after in zip(trails[0], trails[0][1:]):
            if config["family"] == "speck":
                # The bit-one addition mask depends only on the two low bits.
                correlation = sum(1 - 2 * parity((2 & x) ^ (2 & y) ^ (2 & ((x + y) & 3)))
                                  for x in range(4) for y in range(4)) / 16
                mass *= correlation * correlation
            elif config["trail_kind"] == "differential":
                assert before[0] == after[1]
                mass *= differential_counts(config["family"], 16, before[0])[after[0] ^ before[1]] / 65536
            else:
                assert before[1] == after[0]
                correlation = linear_correlation(config["family"], 16, before[0] ^ after[1], after[0])
                mass *= correlation * correlation
        exponent = 2 if config["trail_kind"] == "linear" else 1
        assert mass == 2.0 ** (-exponent * config["weight"]), (config, mass)
        assert not enumerate_model(dict(config, weight=config["weight"] - 1))
    # An all-active rotation cycle has one dependent derivative equation.
    # Checking both its rank and output parity catches an overcounted weight
    # as well as impossible transitions accepted by per-bit constraints.
    for family in ("simon", "simeck"):
        config = dict(family=family, trail_kind="differential", word_size=16, rounds=1,
                      delta_in=(65535, 0), delta_out=(0, 65535), weight=15, shard_round=1)
        counts = differential_counts(family, 16, 65535)
        assert counts[0] == 2 and counts[1] == 0
        assert len(enumerate_model(config)) == 1
        assert not enumerate_model(dict(config, weight=16))
        for weight in (15, 16):
            assert not enumerate_model(dict(config, delta_out=(1, 65535), weight=weight))
    # Four rounds provide several genuinely distinct projected trails.
    for family, endpoint, weight, expected in (("simon", (36, 73), 15, 5),
                                               ("simeck", (16, 111), 19, 9)):
        config = dict(family=family, trail_kind="differential", word_size=8, rounds=4,
                      delta_in=(1, 0), delta_out=endpoint, weight=weight, shard_round=2)
        # Enumerate the first two transitions directly. The final endpoint
        # fixes the last two states once the middle state is chosen.
        independent = []
        for a, count1 in differential_counts(family, 8, 1).items():
            for value, count2 in differential_counts(family, 8, a).items():
                middle = (value ^ 1, a)
                penultimate = (endpoint[1], middle[0])
                counts = (count1, count2,
                          differential_counts(family, 8, middle[0])[penultimate[0] ^ middle[1]],
                          differential_counts(family, 8, penultimate[0])[endpoint[0] ^ penultimate[1]])
                if all(counts) and sum(8 - math.log2(count) for count in counts) == weight:
                    independent.append(((1, 0), (a, 1), middle, penultimate, endpoint))
        assert len(independent) == expected
        assert set(enumerate_model(config)) == set(independent), config


def check_partial_replay(directory):
    from Framework.aggregation import allsat
    from Framework.aggregation.models import new_solver
    from Framework.aggregation.window import replay_records, iter_records, strict_union
    config = next(fixtures())
    parent = allsat.make_task({})
    child = allsat.make_task({"0": 0}, parent=parent["id"], depth=1)
    parent["children"] = [child["id"]]
    source = directory / "sat"
    def interrupt_after_one(model):
        actual = new_solver(model)
        class Interrupted:
            calls = 0
            def solve(self, **kwargs):
                self.calls += 1
                return actual.solve(**kwargs) if self.calls == 1 else (None, None)
            def add_clause(self, clause):
                actual.add_clause(clause)
        return Interrupted()
    with patch.object(allsat, "new_solver", side_effect=interrupt_after_one):
        partial = allsat.run_shard((config, child, str(source), 20))
    assert partial["status"] == "timeout" and partial["trail_count"] == 1, partial
    layer = source / "W002"
    identity = dict(config, mode="unrestricted", window_supports=None)
    (layer / "manifest.json").write_text(json.dumps({"config_identity": identity,
                                                    "tasks": {parent["id"]: parent, child["id"]: child}}))
    # A split parent's count overlaps its descendants and must be ignored.
    (layer / f"shard_{parent['id']}.json").write_text(json.dumps({"status": "complete", "trail_count": 1}))
    summary_path = layer / "layer_summary.json"
    summary_path.write_text(json.dumps({"status": "incomplete", "exact_trail_count": None,
                                        "enumerated_trail_lower_bound": 1}))
    records = replay_records(source, directory / "replay", 2, 2, 1)
    assert records["status"] == "partial" and records["exact_through_weight"] is None
    assert records["replayed_count"] == len(list(iter_records(records["files"]))) == 1
    times = {path: Path(path).stat().st_mtime_ns for path in records["files"]}
    again = replay_records(source, directory / "replay", 2, 2, 1)
    assert again["replayed_count"] == 1
    assert times == {path: Path(path).stat().st_mtime_ns for path in records["files"]}
    for kind in ("differential", "linear"):
        # In-window mass must not be added a second time.
        overlap = {"inside_log2_mass": -2, "outside_log2_mass": -3,
                   "all_source_counts_replayed": True}
        union = strict_union(2.0, overlap, kind)
        assert math.isclose(2 ** -union["strict_union_mass_weight"], 0.25 + 0.125)
    complete = allsat.run_shard((config, child, str(source), 20))
    assert complete["status"] == "complete" and complete["trail_count"] == 1
    summary_path.write_text(json.dumps({"status": "complete", "exact_trail_count": 1,
                                        "enumerated_trail_lower_bound": 1}))
    recovered = replay_records(source, directory / "replay", 2, 2, 1)
    assert recovered["status"] == "complete" and recovered["exact_through_weight"] == 2
    assert recovered["replayed_count"] == 1


def independent_window(family, kind, word, rounds, rows, start):
    size = 1 << word
    if kind == "differential":
        table = [[differential_counts(family, word, a)[b] / size for b in range(size)] for a in range(size)]
    else:
        table = [[linear_correlation(family, word, a, b) ** 2 for b in range(size)] for a in range(size)]
    def admitted(pair, rnd):
        left_row, right_row = (rnd + 1, rnd) if kind == "differential" else (rnd, rnd + 1)
        return all(value & ~sum(1 << bit for bit in rows[index]) == 0
                   for value, index in zip(pair, (left_row, right_row)))
    current = {start: (1.0, 1)}
    for rnd in range(rounds):
        following = defaultdict(lambda: [0.0, 0])
        for (left, right), (mass, count) in current.items():
            for value in range(size):
                if kind == "differential":
                    target, probability = (value, left), table[left][value ^ right]
                else:
                    target, probability = (right, value), table[left ^ value][right]
                if probability and admitted(target, rnd + 1):
                    following[target][0] += mass * probability
                    following[target][1] += count
        current = following
    return current


def check_terminal_retry(directory):
    """A larger timeout must retry a leaf even at the partition/depth cap."""
    for limit in ("bits", "depth"):
        output = directory / limit
        command = [sys.executable, "-m", "Framework.aggregation.allsat", "--family", "simon",
                   "--word-size", "8", "--rounds", "4", "--input", "1,0", "--output", "36,73",
                   "--weight-start", "15", "--weight-end", "15", "--workers", "1",
                   "--shard-round", "2", "--shard-bits", "1", "--max-shard-bits",
                   "1" if limit == "bits" else "4", "--adaptive-probe-models", "0",
                   "--probe-models", "2", "--probe-timeout", "2",
                   "--timeout-per-shard", "1e-12", "--output-dir", str(output)]
        if limit == "depth":
            command += ["--max-adaptive-depth", "0"]
        def launch():
            result = subprocess.run(command, capture_output=True, text=True, timeout=90)
            assert result.returncode == 0, result.stdout + result.stderr
            return json.loads((output / "W015/layer_summary.json").read_text())
        partial = launch()
        assert partial["status"] == "incomplete" and partial["exact_trail_count"] is None, partial
        command[command.index("--timeout-per-shard") + 1] = "10"
        complete = launch()
        assert complete["status"] == "complete" and complete["exact_trail_count"] == 5, complete
        shards = {path: path.stat().st_mtime_ns for path in output.glob("W015/shard_*.json")}
        assert launch()["exact_trail_count"] == 5
        assert shards == {path: path.stat().st_mtime_ns for path in shards}


def check_native(directory, binary):
    for family in ("simon", "simeck"):
        for kind in ("differential", "linear"):
            for restricted in (False, True):
                rounds, word = 4, 4
                rows = [list(range(4)) if not restricted else [bit for bit in range(4) if bit != row % 4]
                        for row in range(rounds + 2)]
                start = ((4, 0) if kind == "differential" else (0, 4))
                expected = independent_window(family, kind, word, rounds, rows, start)
                assert expected, (family, kind, rows)
                end = max(expected, key=lambda state: (expected[state][1] > 1, expected[state][0]))
                if not restricted:
                    assert expected[end][1] > 1, "full-window fixture must aggregate distinct paths"
                prefix = directory / f"{family}_{kind}_{int(restricted)}"
                window = prefix.with_suffix(".txt")
                window.write_text(f"{rounds} {len(rows[0])}\n" + "".join(" ".join(map(str, row)) + "\n" for row in rows))
                output = prefix.with_suffix(".json")
                command = [str(binary), "--trail-kind", kind, "--family", family, "--word-size", str(word),
                           "--window", str(window), "--output", str(output), "--threads", "1", "--memory-limit-gb", "1",
                           "--input-left", hex(start[0]), "--input-right", hex(start[1]),
                           "--output-left", hex(end[0]), "--output-right", hex(end[1])]
                subprocess.run(command, check=True, capture_output=True, text=True)
                actual = json.loads(output.read_text())
                assert math.isclose(actual["mass"], expected[end][0], rel_tol=1e-12), (command, actual, expected[end])
                assert math.isclose(actual["mass_weight"], -math.log2(expected[end][0]), abs_tol=1e-9)
                # A nonzero difference/mask cannot propagate to the zero endpoint.
                zero = list(command)
                zero[zero.index("--output-left") + 1] = "0"
                zero[zero.index("--output-right") + 1] = "0"
                subprocess.run(zero, check=True, capture_output=True, text=True)
                actual = json.loads(output.read_text())
                assert actual["mass"] == 0 and actual["mass_weight"] is None, actual
    # Odd rotation cycles have an affine support shift. Exhaust every input
    # mask against an all-active output mask, including zero correlations.
    window = directory / "odd_cycle.txt"
    window.write_text("1 3\n0 1 2\n0 1 2\n0 1 2\n")
    output = directory / "odd_cycle.json"
    for family in ("simon", "simeck"):
        for alpha in range(8):
            command = [str(binary), "--trail-kind", "linear", "--family", family,
                       "--word-size", "3", "--window", str(window), "--output", str(output),
                       "--threads", "1", "--memory-limit-gb", "1",
                       "--input-left", str(alpha), "--input-right", "7",
                       "--output-left", "7", "--output-right", "0"]
            subprocess.run(command, check=True, capture_output=True, text=True)
            actual = json.loads(output.read_text())
            expected = linear_correlation(family, 3, alpha, 7) ** 2
            assert actual["mass"] == expected, (family, alpha, actual, expected)


def check_pipeline(directory, build):
    environment = dict(os.environ, DL_FWHT_BUILD_DIR=str(build))
    for kind, before, after, weight in (("differential", "1,0", "4,1", 2),):
        output = directory / kind
        command = [sys.executable, "-m", "Framework.aggregation.run", "--cipher", "simon32",
                   "--trail-kind", kind, "--rounds", "1", "--input", before, "--output", after,
                   "--weight-range", f"{weight}-{weight + 1}", "--method", "sat-guided-window",
                   "--window-size", "2", "--samples-power", "6", "--top-families", "2",
                   "--max-guided-iterations", "1", "--workers", "1", "--timeout", "20",
                   "--shard-bits", "1", "--max-shard-bits", "2", "--memory-limit-gb", "1",
                   "--output-dir", str(output)]
        result = subprocess.run(command, text=True, capture_output=True, env=environment, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        summary = json.loads((output / "summary.json").read_text())
        assert summary["status"] == "complete" and summary["sat_trails"] == 1, summary
        assert summary["strict_union_mass_weight"] == 2, summary
        shards = {str(path): path.stat().st_mtime_ns for path in output.rglob("shard_*.json")}
        result = subprocess.run(command + ["--resume"], text=True, capture_output=True, env=environment, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        resumed = json.loads((output / "summary.json").read_text())
        assert resumed["sat_trails"] == 1 and resumed["strict_union_mass_weight"] == 2
        assert shards == {str(path): path.stat().st_mtime_ns for path in output.rglob("shard_*.json")}


def run_interfaces(args):
    """Known vectors and all public CPU/tensor cipher variants."""
    import torch
    from Cipher.registry import available_variants

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this Python environment")
    check_known_vectors()
    variants = available_variants()
    for spec in variants:
        check_variant(spec, device, args.samples)
    check_backend_versions()
    print(f"PASS interfaces: known vectors, {len(variants)} cipher variants on {device}, "
          "round offsets, coordinates, FWHT/parity and key-size dispatch", flush=True)


def run_framework(args):
    """Shared search/split behavior and optional real OCP endpoint searches."""
    from Framework.search.adapter import get_adapter

    check_transform()
    check_spn_templates()
    with TemporaryDirectory(prefix="fse-framework-check-") as temporary:
        root = Path(temporary)
        if args.cms_bin:
            check_spn_ocp(root / "spn", args.cms_bin)
            print("PASS PRESENT/KNOT: SAT transitions/weights, exact 0/1/2-round optima, "
                  "AEAD endpoints, bounded UNSAT, unavailable solver and timeout", flush=True)
        if args.spn_only:
            return
        check_fixed_key_factors(root / "fixed_key_factors")
        print("PASS fixed-key factors: linear identity, offset-round counts and same-key paired statistics", flush=True)
        check_candidate_selection(root / "candidates")
        check_resume(root / "resume")
        print("PASS cache: interrupted-query retry, Em reuse and configuration/source identity", flush=True)
        if not args.skip_split:
            check_split_endpoints_and_retry(root / "split_endpoints")
            print("PASS split: zero-round AEAD constraints, incomplete status and query retry", flush=True)
        for name in FAMILIES:
            adapter = get_adapter(name)
            check_zero_round_spectrum(adapter, root / name / "spectrum")
            candidates = check_search(adapter, root / name / "search")
            if not args.skip_split:
                check_split(name, root / name / "split", candidates)
            if args.cms_bin:
                check_search(adapter, root / name / "sat", args.cms_bin)
            print(f"PASS {name}: spectrum, search, fixed-key validation"
                  + (", split ladder" if not args.skip_split else "")
                  + (", SAT 1+1+1" if args.cms_bin else ""), flush=True)


def run_aggregation(args):
    """Independent enumeration and native DP with temporary artifacts only."""
    check_sat()
    print("PASS projected AllSAT: round probabilities, squared Walsh coefficients and distinct paths", flush=True)
    with TemporaryDirectory(prefix="fse-aggregation-check-") as temporary:
        root = Path(temporary)
        check_partial_replay(root / "recovery")
        check_terminal_retry(root / "terminal_retry")
        print("PASS aggregation accounting: terminal leaves, disjoint union, partial status and recovery", flush=True)
        if args.skip_native:
            return
        source = Path(__file__).resolve().parents[1] / "Framework/aggregation"
        build = root / "build"
        build.mkdir()
        for name in ("window_dp", "markov_sampler"):
            subprocess.run([os.environ.get("CC", "cc"), "-O2", "-fopenmp", str(source / f"{name}.c"),
                            "-lm", "-o", str(build / name)], check=True)
        check_native(root, build / "window_dp")
        print("PASS native DP: independent differential/linear enumeration, restricted windows and zero mass", flush=True)
        check_pipeline(root / "pipeline", build)
        print("PASS short differential pipeline and completed-run recovery", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", nargs="+", choices=("interfaces", "framework", "aggregation", "all"),
                        help="default: interfaces framework; all also includes aggregation")
    parser.add_argument("--device", default="cpu", help="tensor interface device: cpu or cuda:N")
    parser.add_argument("--samples", type=int, default=17, help="samples per cipher interface check")
    parser.add_argument("--cms-bin", help="external solver for framework E1/E2 checks")
    parser.add_argument("--skip-split", action="store_true", help="skip framework split-ladder checks")
    parser.add_argument("--spn-only", action="store_true", help="run only the framework PRESENT/KNOT checks")
    parser.add_argument("--skip-native", action="store_true", help="skip aggregation native DP/pipeline checks")
    args = parser.parse_args(argv)
    if args.samples < 1:
        parser.error("--samples must be positive")
    suites = args.suite or (["framework"] if args.spn_only else ["interfaces", "framework"])
    if "all" in suites:
        suites = ["interfaces", "framework", "aggregation"]
    suites = list(dict.fromkeys(suites))
    if args.spn_only and (not args.cms_bin or suites != ["framework"]):
        parser.error("--spn-only requires --cms-bin and only the framework suite")
    runners = {"interfaces": run_interfaces, "framework": run_framework, "aggregation": run_aggregation}
    for suite in suites:
        print(f"[checks] {suite}", flush=True)
        runners[suite](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
