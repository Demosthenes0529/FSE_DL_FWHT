#!/usr/bin/env python3
"""Four-coordinate round-split selection using the common search interfaces.

Probe successive middle lengths, retain the last two detected lengths, and
combine their single-trail E1/E2 frontiers. All qualifying splits are retained
within the declared mask domain and resource limits. Timeout is never UNSAT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from Cipher.registry import available_ciphers
from Framework.common.path_utils import (
    atomic_write_json, collect_fingerprint, read_jsonl, write_jsonl,
)
from Framework.fwht.candidates import scan_candidates
from Framework.search.adapter import get_adapter, validate_payload
from Framework.search.models import SearchConfig, compose_weight

FRAMEWORK_ROOT = Path(__file__).resolve().parent
MODEL_VERSION = "dl_split_ladder_v3"
DEFAULT_RM_START = {
    "speck32": 4, "speck48": 6, "speck64": 6, "speck96": 7, "speck128": 8,
    "cham64": 26, "cham128": 32, "simon32": 11, "simeck32": 11,
    "simon48": 13, "simeck48": 16, "simon64": 15, "simeck64": 17,
    "simon96": 16, "simon128": 18,
}


class IncompleteScoutError(RuntimeError):
    """An unresolved solver query requires resuming the saved scout."""


def str2bool(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() in {"1", "true", "yes", "on"}:
        return True
    if str(value).lower() in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean {value!r}")


def parse_int_list(text):
    return list(dict.fromkeys(int(item.strip(), 0) for item in text.replace(";", ",").split(",") if item.strip()))


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def representative_deltas(block_size):
    """Four evenly spaced coordinates in the complete state."""
    if block_size < 4 or block_size % 4:
        raise ValueError("four probes require a block size divisible by four")
    return [index * block_size // 4 for index in range(4)]


class SplitLadderRunner:
    def __init__(self, args, cipher):
        self.args = args
        self.adapter = get_adapter(cipher, args.version)
        self.metadata = self.adapter.metadata
        self.block_size = self.metadata.block_bits
        self.position_dependent = self.metadata.name.startswith(("cham", "knot", "present"))
        self.exact_trail_model = not self.metadata.name.startswith(("simon", "simeck"))
        self.output_dir = Path(args.output_root) / self.adapter.gpu_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = SearchConfig(
            cipher=self.metadata.name, version=self.metadata.key_bits,
            r1=0, rm=0, r2=0, round_start=0, output_dir=str(self.output_dir),
            eval_samples=args.samples,
            scout_samples=args.panel_samples if args.panel_samples is not None else args.samples,
            seed=args.seed, panel_seed=args.panel_seed, backend=args.backend,
            gpus=args.gpus, cuda_batch_size=args.cuda_batch_size,
            panel_size=args.panel_size,
            rho_threshold=args.rho_threshold,
            em_top_k=args.em_top_k, em_top_per_hw=args.em_top_per_hw,
            mask_hw_min=args.mask_hw_min, mask_hw_max=args.mask_hw_max,
            e1_weight_bound=self.block_size, e2_weight_bound=self.block_size,
            cms_bin=args.cms_bin, ocp_root=args.ocp_root,
            solver_threads=args.solver_threads, timeout_seconds=args.timeout_seconds,
            knot_mode=args.knot_mode, resume=args.resume,
        )
        from Framework.sat.search import backend_status
        dependency = backend_status({"cipher": self.adapter.gpu_name,
                                     "cms_bin": self.cfg.cms_bin,
                                     "ocp_root": self.cfg.ocp_root})["ocp"]
        self.cfg = replace(self.cfg, cms_bin=dependency["cms_bin"], ocp_root=dependency["root"])
        if not 1 <= self.cfg.panel_size <= min(32 if args.backend == "cuda" else 24, self.block_size):
            raise ValueError("panel size exceeds the selected backend or state size")
        if min(self.cfg.sample_count, self.cfg.scout_count, self.cfg.cuda_batch_size) <= 0:
            raise ValueError("sample counts and CUDA batch size must be positive")
        if self.cfg.scout_count > self.cfg.sample_count:
            raise ValueError("panel samples must not exceed spectrum samples")
        if args.backend == "cuda" and self.cfg.sample_count > 1 << 34:
            raise ValueError("CUDA sampling currently supports at most 2^34 samples")
        if self.cfg.panel_seed == self.cfg.seed:
            raise ValueError("panel and spectrum samples must use independent seeds")
        if args.solver_threads < 1:
            raise ValueError("thread counts must be positive")
        if not 0 <= args.rho_threshold < 1 or not math.isfinite(args.rho_threshold):
            raise ValueError("rho threshold must be finite and in [0, 1)")
        if not 1 <= args.mask_hw_min <= args.mask_hw_max <= self.block_size:
            raise ValueError("mask weights must satisfy 1 <= min <= max <= block size")
        if not math.isfinite(args.timeout_seconds) or args.timeout_seconds < 0:
            raise ValueError("timeout must be finite and nonnegative; zero disables the timeout")
        for field in ("max_rm_steps", "family_limit", "min_e1_rounds", "min_e2_rounds",
                      "max_e1_rounds", "max_e2_rounds", "max_side_rounds_from_total",
                      "em_top_k", "em_top_per_hw"):
            if getattr(args, field) < 0:
                raise ValueError(f"{field} must be nonnegative")
        if self.rm_start() < 1:
            raise ValueError("middle length must be positive")
        representative_deltas(self.block_size)
        self.stop_events = []
        self._completed_em = set()
        self.source_fingerprint = collect_fingerprint()["combined_sha256"]
        self.run_request = {
            "policy_version": MODEL_VERSION, "source_fingerprint": self.source_fingerprint,
            "parameters": {key: value for key, value in vars(args).items()
                           if key not in {"output_root", "resume", "dry_run", "ciphers"}},
            "config": {key: value for key, value in self.cfg.to_dict().items()
                       if key not in {"output_dir", "resume"}},
        }
        # JSON normalization prevents tuples from differing after reload.
        self.run_request = json.loads(json.dumps(self.run_request, sort_keys=True))
        previous = read_json(self.output_dir / "scout_request.json")
        if args.resume and previous is not None and previous != self.run_request:
            raise ValueError("saved scout parameters or source changed; use a new output directory or --resume false")

    def record_stop(self, stage, reason, **details):
        self.stop_events.append({"stage": stage, "reason": reason, **details})
        atomic_write_json(self.output_dir / "scout_stops.json", self.stop_events)

    def incomplete(self, stage, row):
        self.record_stop(stage, "unresolved_solver", result=row)
        raise IncompleteScoutError(
            f"{stage}: {row.get('status')} does not prove a correlation boundary; resume to retry")

    @staticmethod
    def certified_no_path(row):
        return (row.get("status") in {"no_path", "unsat", "no_path_or_pruned"}
                and not row.get("timed_out")
                and bool(row.get("model_unsat_proven") or row.get("unsat_proven")))

    @staticmethod
    def resolved_optimum(row):
        return (row.get("status") == "ok" and row.get("path_verified") is True
                and row.get("optimum_proven") is True and not row.get("timed_out")
                and math.isfinite(float(row["weight"])) and float(row["weight"]) >= 0)

    def allow_finite_candidate(self, stage, row, fixed_rounds=False):
        # Replayed AND-RX trails can be retained in an explicitly finite scan,
        # but cannot certify pruning under exact whole-word transition weights.
        feasible = (row.get("status") == "ok" and row.get("path_verified") is True
                    and not row.get("timed_out") and row.get("weight") is not None
                    and math.isfinite(float(row["weight"])) and float(row["weight"]) >= 0)
        finite = (fixed_rounds or getattr(self.args, f"max_{stage.lower()}_rounds") > 0
                  or self.args.max_side_rounds_from_total > 0)
        if feasible and finite:
            row["uncertified_frontier"] = True
            self.record_stop(stage, "uncertified_frontier", rounds=row["rounds"],
                             fixed_endpoint=row["fixed_endpoint"])
        return feasible and finite

    def rm_start(self):
        return (int(self.args.rm_start) if self.args.rm_start is not None else
                int(DEFAULT_RM_START.get(self.metadata.name, max(1, self.block_size // 8))))

    def total_round_limit(self):
        return min(self.metadata.max_rounds, self.args.max_side_rounds_from_total or self.metadata.max_rounds)

    def round_start_values_for_em(self, rm):
        available = self.total_round_limit() - rm - self.args.min_e2_rounds
        if available < self.args.min_e1_rounds:
            return []
        if not self.position_dependent:
            return [0]
        requested = parse_int_list(self.args.em_round_start_values)
        if any(value < 0 for value in requested):
            raise ValueError("Em starts must be nonnegative")
        configured = self.args.max_e1_rounds
        maximum = min(available, configured) if configured else available
        if requested:
            return [value for value in requested if self.args.min_e1_rounds <= value <= maximum]
        if not configured and not self.args.max_side_rounds_from_total:
            raise ValueError("round-dependent Em requires explicit --em-round-start-values or a finite round limit")
        return list(range(self.args.min_e1_rounds, maximum + 1))

    def frontier_round_values(self, side, fixed_r1=None, rm=None):
        minimum = getattr(self.args, f"min_{side}_rounds")
        middle = rm if rm is not None else self.minimum_retained_rm
        other = (fixed_r1 if fixed_r1 is not None else self.args.min_e1_rounds) if side == "e2" else self.args.min_e2_rounds
        maximum = self.total_round_limit() - middle - other
        configured = getattr(self.args, f"max_{side}_rounds")
        if configured:
            maximum = min(maximum, configured)
        return range(minimum, maximum + 1)

    def scan_config(self, rm, round_start, directory):
        return replace(self.cfg, rm=rm, round_start=round_start, output_dir=str(directory))

    def em_output_dir(self, rm):
        return self.output_dir / "em" / f"rm{rm}"

    def run_em_for_rm(self, rm):
        output = self.em_output_dir(rm)
        combined = output / "candidates.jsonl"
        request_path = output / "request.json"
        starts = self.round_start_values_for_em(rm)
        request = {"run": self.run_request, "rm": rm, "round_starts": starts}
        if (rm in self._completed_em or self.cfg.resume) and read_json(request_path) == request and combined.is_file():
            self._completed_em.add(rm)
            return read_jsonl(combined)
        candidates = []
        for round_start in starts:
            for delta_id in representative_deltas(self.block_size):
                print(f"[Em] {self.metadata.name} rm={rm} start={round_start} delta={delta_id}", flush=True)
                directory = output / f"start{round_start}" / f"delta{delta_id}"
                config = self.scan_config(rm, round_start, directory)
                delta = self.adapter.from_paper_bits([delta_id])
                result = self.adapter.scan_em(delta, config, directory)
                rows = scan_candidates(self.adapter, result, config)
                rows = [dict(row, cipher=self.metadata.name, version=self.cfg.version,
                             rm=rm, em_start=round_start,
                             delta_m=self.adapter.format_state(delta), delta_m_id=delta_id,
                             delta_m_paper_bits=[delta_id], seed=self.cfg.seed)
                        for row in rows]
                write_jsonl(directory / "em_candidates.jsonl", rows)
                candidates.extend(rows)
        candidates.sort(key=lambda row: (row["wEm_search"], row["hw"], row["delta_m_id"], row["gamma_m"]))
        write_jsonl(combined, candidates)
        # Write completion metadata last, so interrupted partial scans cannot be reused.
        atomic_write_json(request_path, request)
        self._completed_em.add(rm)
        return candidates

    def run_rm_ladder(self):
        ladder, valid = [], []
        rm = self.rm_start()
        first_failed = None
        while self.args.max_rm_steps == 0 or len(ladder) < self.args.max_rm_steps:
            if not self.round_start_values_for_em(rm):
                self.record_stop("Em", "round_start_limit", rm=rm)
                break
            rows = self.run_em_for_rm(rm)
            ladder.append({"rm": rm, "num_valid_candidates": len(rows),
                           "best_wEm_search": min((row["wEm_search"] for row in rows), default=None),
                           "status": "valid" if rows else "empty_four_probe_spectrum"})
            atomic_write_json(self.output_dir / "em_rm_ladder.json", ladder)
            if not rows:
                first_failed = rm
                self.record_stop("Em", "empty_four_probe_spectrum", rm=rm)
                break
            valid.append(rm)
            rm += 1
        else:
            self.record_stop("Em", "step_limit", next_rm=rm)
        if not valid:
            raise RuntimeError(f"{self.metadata.name}: no detectable middle length from {self.rm_start()}")
        maximum = max(valid)
        retained = [maximum]
        if maximum > 1 and self.run_em_for_rm(maximum - 1):
            retained.append(maximum - 1)
        return maximum, first_failed, retained

    def select_candidates(self, rm):
        rows = self.run_em_for_rm(rm)
        unique = {}
        for row in rows:
            key = row["delta_m"], row["gamma_m"], row["em_start"]
            unique.setdefault(key, row)
        selected = sorted(unique.values(), key=lambda row: (row["wEm_search"], row["hw"], row["delta_m_id"], row["gamma_m"]))
        if self.args.family_limit and len(selected) > self.args.family_limit:
            self.record_stop("candidate_pool", "family_limit", rm=rm,
                             before=len(selected), retained=self.args.family_limit)
            selected = selected[:self.args.family_limit]
        return selected

    def query_frontier(self, stage, endpoint, config):
        rounds = config.r1 if stage == "E1" else config.r2
        request = {"stage": stage, "endpoint": self.adapter.format_state(endpoint),
                   "config": config.to_dict(), "source_fingerprint": self.source_fingerprint}
        request = json.loads(json.dumps(request, sort_keys=True))
        digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:24]
        path = self.output_dir / "frontier_cache" / stage.lower() / f"{digest}.json"
        cached = read_json(path) if self.cfg.resume else None
        if isinstance(cached, dict) and cached.get("request") == request:
            if self.resolved_optimum(cached) or self.certified_no_path(cached):
                return cached
        started = time.monotonic()
        # Identity queries deliberately pass through the same adapter, including
        # KNOT nonce/rate endpoint checks and weight-bound handling.
        payload = (self.adapter.search_e1(endpoint, config) if stage == "E1" else
                   self.adapter.search_e2(endpoint, config.e2_weight_bound, config))
        payload = validate_payload(payload, stage, rounds)
        row = {key: value for key, value in payload.items() if key != "results"}
        if payload["results"]:
            row.update(min(payload["results"], key=lambda result: float(result["weight"])))
        row.update(stage=stage, rounds=rounds, fixed_endpoint=self.adapter.format_state(endpoint),
                   r1=config.r1, rm=config.rm, r2=config.r2, round_start=config.round_start,
                   em_start=config.em_start, e2_start=config.e2_start,
                   runtime_seconds=time.monotonic() - started, request=request)
        atomic_write_json(path, row)
        return row

    def search_frontier(self, stage, endpoint, w_em_values, fixed_r1=None, rm=None):
        side = stage.lower()
        values = ([fixed_r1] if stage == "E1" and fixed_r1 is not None else
                  self.frontier_round_values(side, fixed_r1=fixed_r1, rm=rm))
        budget = self.block_size / 2
        rows = []
        endpoint_constrained = self.metadata.name == "knot256" and self.cfg.knot_mode == "aead"
        for rounds in values:
            config = replace(self.cfg, r1=rounds if stage == "E1" else int(fixed_r1 or 0),
                             rm=0 if stage == "E1" else int(rm or 0),
                             r2=rounds if stage == "E2" else 0)
            row = self.query_frontier(stage, endpoint, config)
            rows.append(row)
            if self.certified_no_path(row):
                self.record_stop(stage, "certified_unsat", rounds=rounds,
                                 endpoint=self.adapter.format_state(endpoint))
                # A forbidden identity or short AEAD endpoint may become legal
                # after more rounds; it is not a monotone extension boundary.
                if endpoint_constrained:
                    continue
                if not self.exact_trail_model:
                    self.record_stop(stage, "model_unsat_without_exact_bound", rounds=rounds,
                                     endpoint=self.adapter.format_state(endpoint))
                    if (fixed_r1 is not None and stage == "E1"
                            or getattr(self.args, f"max_{side}_rounds") > 0
                            or self.args.max_side_rounds_from_total > 0):
                        continue
                    self.incomplete(stage, row)
                break
            if not self.resolved_optimum(row):
                if self.allow_finite_candidate(stage, row, fixed_rounds=stage == "E1" and fixed_r1 is not None):
                    continue
                self.incomplete(stage, row)
            factor = 1 if stage == "E1" else 2
            if factor * row["weight"] + min(w_em_values) > budget and not endpoint_constrained:
                self.record_stop(stage, "correlation_bound", rounds=rounds,
                                 endpoint=self.adapter.format_state(endpoint))
                break
        else:
            if not (stage == "E1" and fixed_r1 is not None):
                self.record_stop(stage, "round_limit", endpoint=self.adapter.format_state(endpoint),
                                 total_round_limit=self.total_round_limit())
        return rows

    def frontiers(self, candidates):
        e1_groups, e2_groups = {}, {}
        for row in candidates:
            r1 = row["em_start"] if self.position_dependent else None
            e1_groups.setdefault((row["delta_m"], r1), []).append(row["wEm_search"])
            # Position-invariant models share the same E2 frontier across rm.
            key = (row["gamma_m"], r1, row["rm"] if self.position_dependent else None)
            e2_groups.setdefault(key, []).append(row["wEm_search"])
        e1_rows, e2_rows = [], []
        for (delta, r1), weights in e1_groups.items():
            print(f"[E1] {self.metadata.name} delta={delta} r1={r1}", flush=True)
            e1_rows.extend(self.search_frontier("E1", int(delta, 16), weights, fixed_r1=r1))
        for (gamma, r1, rm), weights in e2_groups.items():
            print(f"[E2] {self.metadata.name} gamma={gamma} start={int(r1 or 0) + int(rm or 0)}", flush=True)
            e2_rows.extend(self.search_frontier("E2", int(gamma, 16), weights, fixed_r1=r1, rm=rm))
        write_jsonl(self.output_dir / "e1_frontiers.jsonl", e1_rows)
        write_jsonl(self.output_dir / "e2_frontiers.jsonl", e2_rows)
        return e1_rows, e2_rows

    def compose(self, candidates, e1_rows, e2_rows):
        prefixes, suffixes = {}, {}
        for row in e1_rows:
            if row["status"] == "ok" and row.get("path_verified"):
                key = (row["fixed_endpoint"], row["r1"] if self.position_dependent else None)
                prefixes.setdefault(key, []).append(row)
        for row in e2_rows:
            if row["status"] == "ok" and row.get("path_verified"):
                key = (row["fixed_endpoint"], row["e2_start"] if self.position_dependent else None)
                suffixes.setdefault(key, []).append(row)
        feasible, boundary = [], []
        for candidate in candidates:
            e1_key = candidate["delta_m"], candidate["em_start"] if self.position_dependent else None
            e2_key = candidate["gamma_m"], candidate["em_start"] + candidate["rm"] if self.position_dependent else None
            for e1 in prefixes.get(e1_key, []):
                for e2 in suffixes.get(e2_key, []):
                    r1, rm, r2 = e1["r1"], candidate["rm"], e2["r2"]
                    if r1 + rm + r2 > self.total_round_limit():
                        continue
                    weight = compose_weight(e1["weight"], candidate["wEm_search"], e2["weight"])
                    row = dict(candidate, r1=r1, r2=r2, round_start=0,
                               em_start=r1, e2_start=r1 + rm, split=[r1, rm, r2],
                               total_rounds=r1 + rm + r2, delta_in=e1["delta_in"],
                               gamma_out=e2["gamma_out"], wE1=e1["weight"], wE2=e2["weight"],
                               W_search=weight, margin_to_half=self.block_size / 2 - weight,
                               formal_feasible=weight < self.block_size / 2,
                               uncertified_frontier=bool(e1.get("uncertified_frontier") or e2.get("uncertified_frontier")))
                    if row["formal_feasible"]:
                        feasible.append(row)
                    elif weight <= self.block_size / 2 + 0.5:
                        boundary.append(row)
        order = lambda row: (-row["total_rounds"], row["W_search"], -abs(row["rhoEm_search"]), row["hw"])
        feasible.sort(key=order)
        boundary.sort(key=order)
        write_jsonl(self.output_dir / "feasible_combinations.jsonl", feasible)
        write_jsonl(self.output_dir / "boundary_combinations.jsonl", boundary)
        return feasible, boundary

    def recommend(self, max_rm, first_failed, retained, candidates, feasible, boundary):
        by_split = {}
        for row in feasible:
            by_split.setdefault(tuple(row["split"]), row)
        qualifying = list(by_split.values())
        best = feasible[0] if feasible else None
        return {
            "cipher": self.metadata.name, "version": self.cfg.version,
            "block_bits": self.block_size, "policy_version": MODEL_VERSION,
            "max_rm": max_rm, "first_failed_rm": first_failed,
            "retained_rm_values": retained,
            "best_total_rounds": best["total_rounds"] if best else None,
            "best_split": best["split"] if best else None,
            "best_W_search": best["W_search"] if best else None,
            "best_candidate": best,
            "qualifying_splits": qualifying, "num_qualifying_splits": len(qualifying),
            "boundary_combinations": boundary,
            "uncertified_frontier": any(event["reason"] in {"uncertified_frontier", "model_unsat_without_exact_bound"}
                                        for event in self.stop_events),
            "frontier_interpretation": "single-trail candidates under the selected model and declared limits; not a global optimality claim",
            "search_scope": {"probe_deltas": representative_deltas(self.block_size),
                             "round_starts_by_rm": {str(rm): self.round_start_values_for_em(rm) for rm in retained},
                             "config": self.cfg.to_dict(),
                             "middle_limit_interpretation": "empirical detection limit under the sampling budget"},
            "stop_events": self.stop_events,
            "next_full_search_action": {"splits": [row["split"] for row in qualifying],
                                        "suggested_delta_m": list(dict.fromkeys(row["delta_m"] for row in candidates)),
                                        "full_search_launched": False},
        }

    def run(self):
        started = time.monotonic()
        self.stop_events = []
        self._completed_em.clear()
        request_path = self.output_dir / "scout_request.json"
        status_path = self.output_dir / "scout_status.json"
        recommendation_path = self.output_dir / "recommended_splits.json"
        if not self.cfg.resume:
            # An interrupted explicit restart must not leave an earlier Em
            # completion marker eligible for reuse on the next resume.
            for marker in self.output_dir.glob("em/rm*/request.json"):
                marker.unlink()
        previous = read_json(recommendation_path)
        if isinstance(previous, dict):
            previous.update(stale=True, search_status="running")
            atomic_write_json(recommendation_path, previous)
        atomic_write_json(request_path, self.run_request)
        atomic_write_json(status_path, {"status": "running", "policy_version": MODEL_VERSION})
        try:
            if self.args.dry_run:
                result = {"cipher": self.metadata.name, "search_status": "planned",
                          "probe_deltas": representative_deltas(self.block_size),
                          "rm_start": self.rm_start(), "max_rm_steps": self.args.max_rm_steps,
                          "round_starts": self.round_start_values_for_em(self.rm_start()),
                          "config": self.cfg.to_dict(), "full_search_launched": False}
                atomic_write_json(self.output_dir / "scout_plan.json", result)
            elif self.args.mode == "em-only":
                rms = parse_int_list(self.args.rm_values) or [self.rm_start()]
                if any(rm < 1 for rm in rms):
                    raise ValueError("middle lengths must be positive")
                for rm in rms:
                    if not self.round_start_values_for_em(rm):
                        raise ValueError(f"no valid round starts for rm={rm}")
                    self.run_em_for_rm(rm)
                result = {"cipher": self.metadata.name, "mode": "em-only", "search_status": "complete", "rm_values": rms}
            else:
                max_rm, first_failed, retained = self.run_rm_ladder()
                self.minimum_retained_rm = min(retained)
                if self.args.max_side_rounds_from_total:
                    self.record_stop("scope", "total_round_limit", total=self.args.max_side_rounds_from_total)
                candidates = [row for rm in retained for row in self.select_candidates(rm)]
                write_jsonl(self.output_dir / "em_candidates_retained.jsonl", candidates)
                e1_rows, e2_rows = self.frontiers(candidates)
                feasible, boundary = self.compose(candidates, e1_rows, e2_rows)
                result = self.recommend(max_rm, first_failed, retained, candidates, feasible, boundary)
                result["search_status"] = ("complete_with_limits" if any(event["reason"].endswith("limit") for event in self.stop_events) else "complete")
                atomic_write_json(recommendation_path, result)
        except Exception as error:
            atomic_write_json(status_path, {"status": "incomplete", "error": str(error),
                                           "error_type": type(error).__name__, "stop_events": self.stop_events})
            if isinstance(previous, dict):
                previous["search_status"] = "incomplete"
                atomic_write_json(recommendation_path, previous)
            raise
        atomic_write_json(status_path, {"status": result["search_status"], "stop_events": self.stop_events,
                                       "elapsed_seconds": time.monotonic() - started})
        return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick-split", "em-only"), default="quick-split")
    parser.add_argument("--ciphers", required=True, help="comma-separated registered cipher names")
    parser.add_argument("--version", type=int, help="key size; inferred from each cipher if omitted")
    parser.add_argument("--rm-start", type=int)
    parser.add_argument("--rm-values", default="", help="middle lengths for em-only mode")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--samples", type=int, default=1 << 30)
    parser.add_argument("--panel-samples", type=int, help="independent coordinate proposal samples; defaults to --samples")
    parser.add_argument("--seed", type=int, default=2026072701)
    parser.add_argument("--panel-seed", type=int)
    parser.add_argument("--rho-threshold", type=float, default=2e-4)
    parser.add_argument("--panel-size", type=int, default=24)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--cuda-batch-size", type=int, default=1 << 20)
    parser.add_argument("--mask-hw-min", type=int, default=1)
    parser.add_argument("--mask-hw-max", type=int, default=6)
    parser.add_argument("--em-top-k", type=int, default=0)
    parser.add_argument("--em-top-per-hw", type=int, default=200)
    parser.add_argument("--family-limit", type=int, default=0, help="optional retained endpoint limit; zero keeps all extracted endpoints")
    parser.add_argument("--max-rm-steps", type=int, default=0, help="zero extends until the first undetectable middle length")
    parser.add_argument("--min-e1-rounds", type=int, default=0)
    parser.add_argument("--min-e2-rounds", type=int, default=0)
    parser.add_argument("--max-e1-rounds", type=int, default=0)
    parser.add_argument("--max-e2-rounds", type=int, default=0)
    parser.add_argument("--max-side-rounds-from-total", type=int, default=0, help="optional total-round resource limit")
    parser.add_argument("--em-round-start-values", default="", help="explicit absolute starts/r1 for CHAM, PRESENT and KNOT")
    parser.add_argument("--ocp-root", default=os.environ.get("OCP_ROOT", str(FRAMEWORK_ROOT.parent / "OCP")))
    parser.add_argument("--cms-bin", default=os.environ.get("CRYPTOMINISAT_BIN") or shutil.which("cryptominisat5") or "cryptominisat5")
    parser.add_argument("--solver-threads", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=300, help="per E1/E2 query timeout; zero disables")
    parser.add_argument("--knot-mode", choices=("aead", "permutation"), default="aead")
    parser.add_argument("--resume", type=str2bool, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if not args.output_root:
        args.output_root = str(FRAMEWORK_ROOT / "runs" / datetime.now().strftime("split_%Y%m%d_%H%M%S"))
    ciphers = list(dict.fromkeys(name.strip().lower() for name in args.ciphers.replace(";", ",").split(",") if name.strip()))
    unknown = set(ciphers) - set(available_ciphers())
    if unknown or not ciphers:
        raise ValueError(f"choose registered cipher names; unknown={sorted(unknown)}")
    results = []
    for cipher in ciphers:
        runner = SplitLadderRunner(args, cipher)
        print(f"[split] {cipher} mode={args.mode}", flush=True)
        results.append(runner.run())
        print(f"[split] saved {runner.output_dir}", flush=True)
    atomic_write_json(Path(args.output_root) / "all_recommended_splits.json", results)


if __name__ == "__main__":
    main()
