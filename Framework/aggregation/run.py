#!/usr/bin/env python3
"""Unified fixed-endpoint SAT differential/linear-hull accumulation launcher.

The launcher normalizes cipher parameters, records an immutable manifest, and
dispatches to an audited exact-weight projected-AllSAT backend.  Completed
shards are durable, so a resumed run never recounts them.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass

from Cipher.registry import get_cipher_spec
from Framework.aggregation.models import parse_pair
from Framework.common.path_utils import atomic_write_json as atomic_json, collect_fingerprint

@dataclass(frozen=True)
class BackendSpec:
    cipher: str
    block_size: int
    word_size: int
    hull_backend: bool
    backend: str | None
    reason: str = ""
    default_method: str = "sat"
    family: str | None = None

def resolve(cipher, block_size=None, trail_kind="differential"):
    name = str(cipher).strip().lower().replace("-", "").replace("_", "")
    if name in ("simon", "simeck", "speck", "cham", "knot") and block_size:
        name += str(block_size)
    spec = get_cipher_spec(name)
    if block_size is not None and spec.block_size != int(block_size):
        raise ValueError("Cipher name and --block-size disagree")
    if trail_kind not in ("differential", "linear"):
        raise ValueError("trail_kind must be differential or linear")
    supported = ((spec.family in ("simon", "simeck") and trail_kind == "differential") or
                 (spec.family == "speck" and trail_kind == "linear"))
    reason = "No exact-weight projected-AllSAT backend is implemented for this cipher/trail kind."
    if spec.family in ("simon", "simeck") and trail_kind == "linear":
        reason = ("Bitwise OCP linear weights are not exact whole-round correlations; "
                  "use Experiments.validation fixed-key E2 measurement.")
    return BackendSpec(spec.name, spec.block_size, spec.word_size, supported,
                       "exact_projected_allsat" if supported else None,
                       "" if supported else reason,
                       "sat-guided-window" if spec.family in ("simon", "simeck") else "sat", spec.family)

def all_specs(trail_kind="differential"):
    names = [family + str(size) for family, sizes in (
        ("speck", (32, 48, 64, 96, 128)), ("simon", (32, 48, 64, 96, 128)),
        ("simeck", (32, 48, 64)), ("cham", (64, 128)), ("knot", (256,))) for size in sizes]
    names.append("present")
    return [resolve(name, trail_kind=trail_kind) for name in sorted(names)]


def parse_weight_range(text: str) -> tuple[int, int]:
    normalized = str(text).strip().replace("..", "-").replace(":", "-")
    fields = normalized.split("-")
    if len(fields) != 2:
        raise ValueError("weight range must be START-END, e.g. 68-84")
    start, end = map(int, fields)
    if start < 0 or end < start:
        raise ValueError("weight range must satisfy 0 <= START <= END")
    return start, end


def normalized_pair(pair: tuple[int, int], word_size: int) -> str:
    digits = (word_size + 3) // 4
    return f"0x{pair[0]:0{digits}x},0x{pair[1]:0{digits}x}"


def show_backends():
    print("cipher          differential-hull  linear-hull")
    for spec in all_specs():
        linear = resolve(spec.cipher, trail_kind="linear")
        print(f"{spec.cipher:15} {str(spec.hull_backend):18} {str(linear.hull_backend)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-ciphers", action="store_true")
    parser.add_argument("--cipher", help="variant name, e.g. simon128")
    parser.add_argument("--block-size", type=int, help="needed only for unsuffixed names")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--input", dest="input_text", help="fixed input difference or linear mask")
    parser.add_argument("--output", dest="output_text", help="fixed output difference or linear mask")
    parser.add_argument("--weight-range", help="inclusive range, e.g. 68-84")
    parser.add_argument("--weight-start", type=int)
    parser.add_argument("--weight-end", type=int)
    parser.add_argument("--trail-kind", choices=("differential", "linear"),
                        default="differential",
                        help="linear accumulates key-averaged hull potential sum(q_trail^2)")
    parser.add_argument("--method", choices=("auto", "sat", "sat-guided-window"),
                        default="auto",
                        help="auto uses SAT-guided windows for SIMON/SIMECK and SAT otherwise")
    parser.add_argument("--window-size", type=int,
                        help="dynamic-window width (SIMON/SIMECK SAT-guided mode)")
    parser.add_argument("--samples-power", type=int, default=22,
                        help="Reverse-MLW samples per seed as a power of two")
    parser.add_argument("--top-families", type=int, default=8,
                        help="probability-ranked leakage families tested per iteration")
    parser.add_argument("--max-guided-iterations", type=int, default=3)
    parser.add_argument("--min-guided-improvement", type=float, default=0.01,
                        help="stop iterative window replacement below this bit gain")
    parser.add_argument("--memory-limit-gb", type=float, default=420,
                        help="memory cap for one dynamic-window DP")
    parser.add_argument("--workers", "--threads", type=int, default=48)
    parser.add_argument("--timeout", "--timeout-per-shard", dest="timeout", type=float,
                        default=900)
    parser.add_argument("--shard-round", default="auto",
                        help="middle state round, or auto (floor(rounds/2))")
    parser.add_argument("--shard-bits", type=int, default=6)
    parser.add_argument("--max-shard-bits", type=int, default=16)
    parser.add_argument("--max-adaptive-depth", type=int)
    parser.add_argument("--probe-models", type=int, default=1000)
    parser.add_argument("--probe-timeout", type=float, default=600)
    parser.add_argument("--probe-file", type=Path)
    parser.add_argument("--adaptive-probe-models", type=int, default=500)
    parser.add_argument("--adaptive-probe-timeout", type=float, default=300)
    parser.add_argument("--mode", choices=("unrestricted", "window-restricted"),
                        default="unrestricted")
    parser.add_argument("--window", type=Path)
    parser.add_argument("--prior-layers", type=Path,
                        help="CSV containing already completed lower weight layers")
    parser.add_argument("--best-trail-weight", type=int,
                        help="single-trail weight (default: weight-start); differential guided mode protects all trails at this feasible layer")
    parser.add_argument("--reference-weight", type=float,
                        help="SAT only: known window mass weight; for linear this is -log2(ELP)")
    parser.add_argument("--continue-incomplete", action="store_true",
                        help="SAT only: continue after a partial layer; guided mode always visits all requested layers")
    parser.add_argument("--split-stale-running", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--detach", action="store_true",
                        help="start in a new session, write runner.log, then return")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def require(parser: argparse.ArgumentParser, args, names: tuple[str, ...]) -> None:
    missing = [name for name in names if getattr(args, name) is None]
    if missing:
        parser.error("missing required arguments: " + ", ".join("--" + x.replace("_", "-")
                                                                 for x in missing))


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_ciphers:
        show_backends()
        return 0
    require(parser, args, ("cipher", "rounds", "input_text", "output_text"))
    try:
        spec = resolve(args.cipher, args.block_size, args.trail_kind)
    except ValueError as error:
        parser.error(str(error))
    if not spec.hull_backend:
        parser.error(f"{spec.cipher}: {spec.reason} Add a backend adapter before using it; "
                     "the OCP best-trail model alone is insufficient for projected AllSAT.")
    method = spec.default_method if args.method == "auto" else args.method
    if method == "sat-guided-window" and spec.family not in ("simon", "simeck"):
        parser.error("SAT-guided windows are currently defined only for SIMON/SIMECK")
    if method == "sat-guided-window" and args.window_size is None:
        parser.error("SIMON/SIMECK default to SAT-guided-window; provide --window-size "
                     "or explicitly select --method sat")
    if method == "sat-guided-window" and (args.mode != "unrestricted" or args.window or
                                         args.prior_layers or args.reference_weight is not None):
        parser.error("guided windows use unrestricted AllSAT and construct their own windows; "
                     "--mode window-restricted/--window/--prior-layers/--reference-weight require --method sat")
    if method == "sat" and args.window_size is not None:
        parser.error("--window-size requires --method sat-guided-window")
    if args.best_trail_weight is not None and args.best_trail_weight < 0:
        parser.error("--best-trail-weight must be nonnegative")
    if args.rounds <= 0:
        parser.error("--rounds must be positive")
    if args.weight_range:
        if args.weight_start is not None or args.weight_end is not None:
            parser.error("use either --weight-range or --weight-start/--weight-end")
        try:
            weight_start, weight_end = parse_weight_range(args.weight_range)
        except ValueError as error:
            parser.error(str(error))
    else:
        require(parser, args, ("weight_start", "weight_end"))
        weight_start, weight_end = args.weight_start, args.weight_end
        if weight_start < 0 or weight_end < weight_start:
            parser.error("require 0 <= weight-start <= weight-end")
    try:
        delta_in = parse_pair(args.input_text, spec.word_size)
        delta_out = parse_pair(args.output_text, spec.word_size)
    except ValueError as error:
        parser.error(str(error))
    try:
        shard_round = args.rounds // 2 if args.shard_round == "auto" else int(args.shard_round)
    except ValueError:
        parser.error("--shard-round must be an integer or auto")
    if not 0 <= shard_round <= args.rounds:
        parser.error("--shard-round is outside the trail")
    if not 0 < args.workers <= 4096:
        parser.error("--workers must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if not 0 < args.shard_bits <= args.max_shard_bits <= 2 * spec.word_size:
        parser.error("require 0 < shard-bits <= max-shard-bits <= state bits")
    if args.max_adaptive_depth is not None and args.max_adaptive_depth < 0:
        parser.error("--max-adaptive-depth must be nonnegative")
    if args.probe_models < 1 or args.adaptive_probe_models < 0 or min(args.probe_timeout, args.adaptive_probe_timeout) <= 0:
        parser.error("invalid probe budget")
    if args.samples_power < 1:
        parser.error("--samples-power must be positive")
    if args.top_families < 1:
        parser.error("--top-families must be positive")
    if args.max_guided_iterations < 1:
        parser.error("--max-guided-iterations must be positive")
    if args.min_guided_improvement < 0:
        parser.error("--min-guided-improvement must be nonnegative")
    if args.memory_limit_gb <= 0:
        parser.error("--memory-limit-gb must be positive")
    if args.mode == "window-restricted" and args.window is None:
        parser.error("window-restricted mode requires --window")
    if args.window and args.mode != "window-restricted":
        parser.error("--window requires --mode window-restricted")
    if args.trail_kind == "linear" and args.mode != "unrestricted":
        parser.error("linear SAT window restrictions are handled by --method sat-guided-window")
    for path in (args.window, args.prior_layers, args.probe_file):
        if path is not None and not path.is_file():
            parser.error(f"input file does not exist: {path}")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_dir or
                  Path(__file__).resolve().parent / "results" /
                  f"{spec.cipher}_r{args.rounds}_W{weight_start}_{weight_end}_{stamp}")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        parser.error(f"output directory is not empty; add --resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_input = normalized_pair(delta_in, spec.word_size)
    normalized_output = normalized_pair(delta_out, spec.word_size)
    if method == "sat-guided-window":
        engine = "Framework.aggregation.pipeline"
        command = [
            sys.executable, "-m", engine, "--trail-kind", args.trail_kind,
            "--family", str(spec.family),
            "--word-size", str(spec.word_size), "--rounds", str(args.rounds),
            "--input", normalized_input, "--output", normalized_output,
            "--weight-start", str(weight_start), "--weight-end", str(weight_end),
            "--best-trail-weight", str(int(args.best_trail_weight
                                             if args.best_trail_weight is not None
                                             else weight_start)),
            "--window-size", str(args.window_size), "--workers", str(args.workers),
            "--timeout-per-shard", str(args.timeout), "--shard-round", str(shard_round),
            "--shard-bits", str(args.shard_bits),
            "--max-shard-bits", str(args.max_shard_bits),
            "--top-families", str(args.top_families),
            "--max-guided-iterations", str(args.max_guided_iterations),
            "--output-dir", str(output_dir),
        ]
        command.extend((
            "--samples-power", str(args.samples_power),
            "--min-guided-improvement", str(args.min_guided_improvement),
            "--memory-limit-gb", str(args.memory_limit_gb),
            "--probe-models", str(args.probe_models), "--probe-timeout", str(args.probe_timeout),
            "--adaptive-probe-models", str(args.adaptive_probe_models),
            "--adaptive-probe-timeout", str(args.adaptive_probe_timeout),
        ))
        if args.probe_file:
            command.extend(("--probe-file", str(args.probe_file.resolve())))
        if args.max_adaptive_depth is not None:
            command.extend(("--max-adaptive-depth", str(args.max_adaptive_depth)))
        if args.split_stale_running:
            command.append("--split-stale-running")
        if args.resume:
            command.append("--resume")
    else:
        command = [
        sys.executable, "-m", "Framework.aggregation.allsat", "--trail-kind", args.trail_kind,
        "--family", str(spec.family or "simon"),
        "--word-size", str(spec.word_size), "--rounds", str(args.rounds),
        "--input", normalized_input, "--output", normalized_output,
        "--mode", args.mode, "--weight-start", str(weight_start),
        "--weight-end", str(weight_end), "--workers", str(args.workers),
        "--shard-round", str(shard_round), "--shard-bits", str(args.shard_bits),
        "--max-shard-bits", str(args.max_shard_bits),
        "--timeout-per-shard", str(args.timeout),
        "--probe-models", str(args.probe_models), "--probe-timeout", str(args.probe_timeout),
        "--adaptive-probe-models", str(args.adaptive_probe_models),
        "--adaptive-probe-timeout", str(args.adaptive_probe_timeout),
        "--output-dir", str(output_dir),
        ]
        if args.window:
            command.extend(("--window", str(args.window.resolve())))
        if args.prior_layers:
            command.extend(("--prior-layers", str(args.prior_layers.resolve())))
        if args.probe_file:
            command.extend(("--probe-file", str(args.probe_file.resolve())))
        if args.max_adaptive_depth is not None:
            command.extend(("--max-adaptive-depth", str(args.max_adaptive_depth)))
        if args.continue_incomplete:
            command.append("--continue-incomplete")
        if args.split_stale_running:
            command.append("--split-stale-running")
        command.extend(("--best-trail-weight", str(args.best_trail_weight
                                                    if args.best_trail_weight is not None
                                                    else weight_start)))
        if args.reference_weight is None:
            command.append("--no-window-references")
        else:
            command.extend(("--free-window-weight", str(args.reference_weight),
                            "--anchored-window-weight", str(args.reference_weight)))

    immutable = {
        "schema_version": 2,
        "source_sha256": collect_fingerprint()["combined_sha256"], "cipher": spec.cipher, "backend": spec.backend,
        "block_size": spec.block_size, "word_size": spec.word_size,
        "rounds": args.rounds, "delta_in": list(delta_in), "delta_out": list(delta_out),
        "trail_kind": args.trail_kind,
        "delta_in_hex": normalized_input, "delta_out_hex": normalized_output,
        "method": method, "mode": args.mode,
        "window": args.window.read_text() if args.window else None,
        "window_size": args.window_size,
        # The end cap is intentionally not immutable: a completed run may be
        # extended from (say) 68-76 to 68-84 in the same durable directory.
        "weight_start": weight_start,
        "best_trail_weight": args.best_trail_weight if args.best_trail_weight is not None else weight_start,
        "prior_layers": args.prior_layers.read_text() if args.prior_layers else None,
        "reference_weight": args.reference_weight,
        "shard_round": shard_round, "initial_shard_bits": args.shard_bits,
    }
    if method == "sat-guided-window":
        # SAT-guided iterations and their strict unions depend on the complete
        # counted range.  Extending the cap therefore requires a fresh result
        # directory instead of reusing stale replay/DP stage markers.
        immutable.update({
            "guided_weight_end": weight_end,
            "samples_power": args.samples_power,
            "top_families": args.top_families,
            "max_guided_iterations": args.max_guided_iterations,
            "min_guided_improvement_bits": args.min_guided_improvement,
        })
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        old = previous.get("immutable", {})
        mismatches = [key for key, value in immutable.items() if old.get(key) != value]
        if mismatches:
            parser.error("resume manifest mismatch: " + ", ".join(mismatches))
        manifest = previous
    else:
        manifest = {"immutable": immutable, "launches": []}
    manifest["launches"].append({
        "time": dt.datetime.now(dt.timezone.utc).isoformat(),
        "weight_end_requested": weight_end,
        "workers_requested": args.workers, "timeout_per_shard": args.timeout,
        "max_shard_bits": args.max_shard_bits, "command": command,
        "dry_run": args.dry_run, "detached": args.detach,
    })
    atomic_json(manifest_path, manifest)

    print("Normalized run:", flush=True)
    print(json.dumps(immutable, indent=2), flush=True)
    print("Command:", flush=True)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    if args.detach:
        log_path = output_dir / "runner.log"
        log_handle = log_path.open("ab", buffering=0)
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                   stdout=log_handle, stderr=subprocess.STDOUT,
                                   start_new_session=True, close_fds=True)
        atomic_json(output_dir / "launcher.json", {
            "pid": process.pid, "log": str(log_path), "command": command,
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        })
        print(f"Started PID {process.pid}")
        print(f"Monitor: tail -f {shlex.quote(str(log_path))}")
        print(f"Summary: watch -n 5 cat {shlex.quote(str(output_dir / 'summary.json'))}")
        return 0
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
