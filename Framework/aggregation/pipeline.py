"""One resumable SAT-guided window pipeline for probability and linear potential."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from Framework.aggregation.allsat import atomic_json
from Framework.aggregation.models import parse_pair
from Framework.common.path_utils import collect_fingerprint
from Framework.aggregation.window import (
    analyze_records, evaluate_window, initial_window, native_binary, replay_records,
    replacement_windows, strict_union, window_fingerprint, write_window,
)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_stage(args, name, command, outputs, inputs=(), force=False):
    """A stage is reusable only when its command, inputs and outputs still match."""
    folder = args.output_dir / "stages"
    folder.mkdir(exist_ok=True)
    marker, log = folder / f"{name}.json", folder / f"{name}.log"
    identity = {"command": command, "inputs": {str(path): _digest(path) for path in inputs}}
    if args.resume and not force and marker.exists():
        saved = json.loads(marker.read_text())
        if saved.get("identity") == identity and all(Path(path).is_file() for path in outputs):
            if saved.get("outputs") == {str(path): _digest(path) for path in outputs}:
                print(f"[stage] {name}: reused", flush=True)
                return
    print(f"[stage] {name}: starting", flush=True)
    started = time.monotonic()
    with log.open("a") as handle:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            print(line.rstrip(), flush=True)
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, command)
    if not all(Path(path).is_file() for path in outputs):
        raise RuntimeError(f"Stage {name} did not produce all required outputs")
    atomic_json(marker, {"identity": identity, "status": "complete",
                         "elapsed_sec": time.monotonic() - started,
                         "outputs": {str(path): _digest(path) for path in outputs}})


def _arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trail-kind", choices=("differential", "linear"), default="differential")
    parser.add_argument("--family", choices=("simon", "simeck"), required=True)
    for name in ("word-size", "rounds", "weight-start", "weight-end", "best-trail-weight", "window-size"):
        parser.add_argument(f"--{name}", type=int, required=True)
    parser.add_argument("--input", dest="input_text", required=True)
    parser.add_argument("--output", dest="output_text", required=True)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--timeout-per-shard", type=float, default=900)
    parser.add_argument("--shard-round", type=int)
    parser.add_argument("--shard-bits", type=int, default=6)
    parser.add_argument("--max-shard-bits", type=int, default=16)
    parser.add_argument("--probe-models", type=int, default=1000)
    parser.add_argument("--probe-timeout", type=float, default=600)
    parser.add_argument("--probe-file", type=Path)
    parser.add_argument("--adaptive-probe-models", type=int, default=500)
    parser.add_argument("--adaptive-probe-timeout", type=float, default=300)
    parser.add_argument("--max-adaptive-depth", type=int)
    parser.add_argument("--split-stale-running", action="store_true")
    parser.add_argument("--samples-power", type=int, default=22)
    parser.add_argument("--top-families", type=int, default=8)
    parser.add_argument("--max-guided-iterations", type=int, default=3)
    parser.add_argument("--min-guided-improvement", type=float, default=0.01)
    parser.add_argument("--memory-limit-gb", type=float, default=420)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.trail_kind == "linear":
        parser.error("SIMON/SIMECK linear SAT-guided aggregation requires an exact "
                     "whole-round SAT model; use Experiments.validation fixed-key E2 measurement")
    if not 1 <= args.rounds <= 64 or not 9 <= args.word_size <= 64:
        parser.error("Require rounds in 1..64 and word-size in 9..64")
    if not 0 < args.window_size <= args.word_size or args.window_size > 20:
        parser.error("Require window-size in 1..min(word-size,20)")
    if not 0 <= args.weight_start <= args.weight_end or args.best_trail_weight < 0:
        parser.error("Invalid weight range or best-trail weight")
    if min(args.workers, args.top_families, args.max_guided_iterations) < 1:
        parser.error("Workers, top-families and max-guided-iterations must be positive")
    if args.timeout_per_shard <= 0 or args.memory_limit_gb <= 0 or args.min_guided_improvement < 0:
        parser.error("Invalid timeout, memory cap or improvement threshold")
    if not 1 <= args.samples_power <= 62:
        parser.error("samples-power must lie in 1..62")
    if not 1 <= args.shard_bits <= args.max_shard_bits <= 2 * args.word_size:
        parser.error("Require 1 <= shard-bits <= max-shard-bits <= 2*word-size")
    if args.probe_models < 1 or args.adaptive_probe_models < 0 or min(args.probe_timeout, args.adaptive_probe_timeout) <= 0:
        parser.error("Invalid probe count or timeout")
    if args.max_adaptive_depth is not None and args.max_adaptive_depth < 0:
        parser.error("max-adaptive-depth must be nonnegative")
    args.shard_round = args.rounds // 2 if args.shard_round is None else args.shard_round
    if not 0 <= args.shard_round <= args.rounds:
        parser.error("shard-round is outside the trail")
    try:
        args.delta_in = parse_pair(args.input_text, args.word_size)
        args.delta_out = parse_pair(args.output_text, args.word_size)
    except ValueError as error:
        parser.error(str(error))
    args.output_dir = args.output_dir.resolve()
    return args


def main(argv=None):
    args = _arguments(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    identity = {key: value for key, value in vars(args).items()
                if key not in ("output_dir", "resume", "input_text", "output_text", "probe_file")}
    # Sampling streams depend on the thread count; include workers as well as
    # all numerical/model parameters, rather than reusing a different experiment.
    identity = json.loads(json.dumps(identity))
    identity["source_fingerprint"] = collect_fingerprint()["combined_sha256"]
    if args.probe_file is not None:
        args.probe_file = args.probe_file.resolve()
        identity["probe_file"] = {"path": str(args.probe_file), "sha256": _digest(args.probe_file)}
    binaries = ["window_dp"] + (["markov_sampler"] if args.trail_kind == "differential" else [])
    identity["native_binaries"] = {name: _digest(native_binary(name)) for name in binaries}
    manifest = args.output_dir / "pipeline_manifest.json"
    if manifest.exists():
        if not args.resume:
            raise ValueError("Pipeline output already exists; use --resume")
        previous = json.loads(manifest.read_text())
        if previous != identity:
            changed = sorted(key for key in set(previous) | set(identity) if previous.get(key) != identity.get(key))
            raise ValueError("Resume parameters or implementation changed: " + ", ".join(changed))
    atomic_json(manifest, identity)
    stage = lambda name, command, outputs, inputs=(): run_stage(args, name, command, outputs, inputs)
    sat = args.output_dir / "sat"
    command = [sys.executable, "-m", "Framework.aggregation.allsat", "--trail-kind", args.trail_kind,
        "--family", args.family, "--word-size", str(args.word_size), "--rounds", str(args.rounds),
        "--input", args.input_text, "--output", args.output_text,
        "--weight-start", str(args.weight_start), "--weight-end", str(args.weight_end),
        "--workers", str(args.workers), "--shard-round", str(args.shard_round),
        "--shard-bits", str(args.shard_bits), "--max-shard-bits", str(args.max_shard_bits),
        "--probe-models", str(args.probe_models), "--probe-timeout", str(args.probe_timeout),
        "--adaptive-probe-models", str(args.adaptive_probe_models),
        "--adaptive-probe-timeout", str(args.adaptive_probe_timeout),
        "--timeout-per-shard", str(args.timeout_per_shard), "--continue-incomplete",
        "--best-trail-weight", str(args.best_trail_weight), "--no-window-references",
        "--output-dir", str(sat)]
    if args.probe_file is not None:
        command.extend(("--probe-file", str(args.probe_file)))
    if args.max_adaptive_depth is not None:
        command.extend(("--max-adaptive-depth", str(args.max_adaptive_depth)))
    if args.split_stale_running:
        command.append("--split-stale-running")
    layer_outputs = [sat / f"W{weight:03d}" / "layer_summary.json"
                     for weight in range(args.weight_start, args.weight_end + 1)]
    partial = any(path.exists() and json.loads(path.read_text())["status"] != "complete" for path in layer_outputs)
    run_stage(args, "sat", command, layer_outputs, force=partial)
    records = replay_records(sat, args.output_dir / "trails", args.weight_start, args.weight_end, args.workers)
    rows, scores, required = initial_window(args, records, stage)
    baseline_path = args.output_dir / "windows" / "baseline.txt"
    write_window(baseline_path, args.rounds, args.window_size, rows)
    baseline_weight = evaluate_window(args, baseline_path, "baseline", stage)
    current_weight, current_path = baseline_weight, baseline_path
    seen, history = {window_fingerprint(rows)}, []
    stopping_reason = "maximum_iterations_reached"
    for iteration in range(1, args.max_guided_iterations + 1):
        overlap, escapes = analyze_records(records, rows, args.trail_kind)
        candidates = []
        for index, candidate in enumerate(replacement_windows(args, rows, scores, required, escapes), 1):
            fingerprint = window_fingerprint(candidate["rows"])
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            label = f"iteration_{iteration:02d}_candidate_{index:02d}"
            path = args.output_dir / "windows" / f"{label}.txt"
            write_window(path, args.rounds, args.window_size, candidate["rows"])
            weight = evaluate_window(args, path, label, stage)
            candidates.append({**candidate, "window": str(path), "mass_weight": weight})
        best = min(candidates, key=lambda candidate: candidate["mass_weight"], default=None)
        improvement = current_weight - best["mass_weight"] if best and math.isfinite(best["mass_weight"]) else None
        accepted = improvement is not None and improvement >= args.min_guided_improvement
        record = {"iteration": iteration, "input_window": str(current_path),
                  "strict_union": strict_union(current_weight, overlap, args.trail_kind),
                  "candidates": candidates, "best_improvement_bits": improvement, "accepted": accepted}
        history.append(record)
        atomic_json(args.output_dir / "iterations" / f"{iteration:02d}.json", record)
        print(f"[window] iteration={iteration} candidates={len(candidates)} accepted={accepted}", flush=True)
        if not best:
            stopping_reason = "no_new_candidate_windows"
            break
        if not accepted:
            stopping_reason = "improvement_below_threshold"
            break
        rows, current_path, current_weight = best["rows"], Path(best["window"]), best["mass_weight"]
    # Reclassify the final accepted window even when the last iteration hit the cap.
    overlap, _ = analyze_records(records, rows, args.trail_kind)
    union = strict_union(current_weight, overlap, args.trail_kind)
    exponent = 2 if args.trail_kind == "linear" else 1
    summary = {"status": records["status"], "pipeline_status": "complete", "trail_kind": args.trail_kind,
               "method": "sat-guided-window", "semantics": union["semantics"],
               "stopping_reason": stopping_reason, "sat_weight_range": [args.weight_start, args.weight_end],
               "sat_exact_through_weight": records["exact_through_weight"],
               "sat_exact_prefix_range": ([args.weight_start, records["exact_through_weight"]]
                                          if records["exact_through_weight"] is not None else None),
               "sat_trails": records["replayed_count"], "sat_layers": records["layers"],
               "baseline_window": str(baseline_path), "baseline_mass_weight": baseline_weight,
               "final_window": str(current_path), "final_window_mass_weight": current_weight,
               "strict_union": union, "strict_union_mass_weight": union["strict_union_mass_weight"],
               "aggregation_gain_bits": exponent * args.best_trail_weight - union["strict_union_mass_weight"],
               "iterations": history}
    if args.trail_kind == "linear":
        summary.update(strict_union_potential_weight=union["strict_union_potential_weight"],
                       rms_correlation_weight=union["rms_correlation_weight"])
    atomic_json(args.output_dir / "summary.json", summary)
    print((args.output_dir / "summary.json").read_text(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
