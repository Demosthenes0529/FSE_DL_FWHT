"""Exact Simon/Simeck differential-trail correction through affine-rank replay.

OCP's Simon/Simeck differential model is a local bitwise AND/ANDXOR model.
For word rotations, different AND output bits can share the same underlying
state bit, so the exact whole-word transition probability is not always the
sum of local bit weights.  These helpers replay returned trails exactly for
all supported word sizes and keep the original OCP model weight for auditability.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Tuple

from Framework.common.path_utils import paper_bits_to_hex_pair, weight_to_abs_correlation


Pair = Tuple[int, int]

def bits_to_pair(bits: Iterable[int], word_size: int) -> Pair:
    left = 0
    right = 0
    for bit in bits or []:
        bit = int(bit)
        if bit < int(word_size):
            right |= 1 << bit
        else:
            left |= 1 << (bit - int(word_size))
    return left, right


def exact_diff_round_weight(cipher: str, word_size: int, input_pair: Pair, output_pair: Pair) -> Tuple[bool, float]:
    from Framework.aggregation.models import exact_round_weight
    family = "simeck" if str(cipher).lower().startswith("simeck") else "simon"
    weight = exact_round_weight(word_size, input_pair, output_pair, family)
    return (False, math.inf) if weight is None else (True, float(weight))


def replay_e1_states(cipher: str, word_size: int, states_paper_bits: List[List[int]]) -> Tuple[bool, List[float], float]:
    states = [bits_to_pair(bits, word_size) for bits in states_paper_bits or []]
    if len(states) < 2:
        return False, [], math.inf
    weights = []
    for before, after in zip(states, states[1:]):
        feasible, weight = exact_diff_round_weight(cipher, word_size, before, after)
        if not feasible:
            return False, weights, math.inf
        weights.append(float(weight))
    return True, weights, float(sum(weights))


def _is_simonlike(cipher: str) -> bool:
    cipher = str(cipher or "").lower()
    return cipher.startswith("simon") or cipher.startswith("simeck")


def correct_e1_candidate(candidate, cipher: str, word_size: int):
    if not _is_simonlike(cipher):
        return candidate
    states = candidate.get("states_paper_bits") or (candidate.get("trail") or {}).get("states_paper_bits")
    if not states:
        return candidate
    feasible, exact_round_weights, exact_weight = replay_e1_states(cipher, word_size, states)
    candidate = dict(candidate)
    if not feasible:
        candidate["replay_ok"] = False
        candidate["replay_warning"] = "Simon/Simeck exact replay found an infeasible E1 transition"
        return candidate
    model_weight = float(candidate.get("w_E1", exact_weight))
    model_round_weights = candidate.get("round_weights")
    candidate["replay_ok"] = True
    candidate["exact_replay_word_size"] = int(word_size)
    candidate["model_w_E1"] = model_weight
    candidate["w_E1"] = exact_weight
    candidate["round_weights"] = exact_round_weights
    candidate["model_round_weights"] = model_round_weights
    if abs(model_weight - exact_weight) > 1e-9:
        candidate["replay_correction"] = {
            "kind": "simonlike_exact_e1",
            "model_w_E1": model_weight,
            "exact_w_E1": exact_weight,
            "delta": exact_weight - model_weight,
            "reason": "OCP local AND/ANDXOR model differs from exact whole-word transition probability",
        }
    return candidate


def correct_e1_row(row, cipher: str, word_size: int):
    if not _is_simonlike(cipher) or row.get("status") != "ok":
        return row
    trail = row.get("trail") or {}
    states = trail.get("states_paper_bits")
    if not states:
        return row
    candidate = {
        "w_E1": row.get("model_w_E1", row.get("w_E1")),
        "round_weights": trail.get("model_round_weights", trail.get("round_weights")),
        "states_paper_bits": states,
    }
    corrected = correct_e1_candidate(candidate, cipher, word_size)
    if not corrected.get("replay_ok"):
        updated = dict(row)
        updated["status"] = "replay_failed"
        updated["warning"] = corrected.get("replay_warning")
        return updated
    updated = dict(row)
    updated["model_w_E1"] = corrected.get("model_w_E1")
    updated["w_E1"] = corrected["w_E1"]
    updated["p"] = weight_to_abs_correlation(updated["w_E1"])
    updated["replay_ok"] = True
    updated["exact_replay_word_size"] = int(word_size)
    updated["trail"] = dict(trail)
    updated["trail"]["model_round_weights"] = corrected.get("model_round_weights")
    updated["trail"]["round_weights"] = corrected.get("round_weights")
    if corrected.get("replay_correction"):
        updated["replay_correction"] = corrected["replay_correction"]
    if updated.get("delta_in_paper_bits") is None and states:
        updated["delta_in_paper_bits"] = states[0]
        updated["delta_in_hex_pair"] = paper_bits_to_hex_pair(states[0], word_size)
    if updated.get("delta_m_paper_bits") is None and states:
        updated["delta_m_paper_bits"] = states[-1]
        updated["delta_m_hex_pair"] = paper_bits_to_hex_pair(states[-1], word_size)
    return updated
