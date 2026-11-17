"""Shared fixed-width windows, replay, and disjoint SAT/window accumulation.

The window rows differ between differential and linear Feistel propagation.
All indexing is centralized here; neither path kind is silently transposed.
"""
from __future__ import annotations

import concurrent.futures
import csv
import gzip
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

from Framework.aggregation.allsat import atomic_json, build_model
from Framework.aggregation.models import logaddexp2, new_solver

SEEDS = (0x12802901, 0x12802902, 0x12802903)


def native_binary(name):
    root = Path(os.environ.get("DL_FWHT_BUILD_DIR", Path(__file__).resolve().parents[1] / "build"))
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run make -C Framework native")
    return path


def row_indices(trail_kind, rnd):
    if trail_kind == "differential":
        return rnd + 1, 0 if rnd == 0 else rnd
    if trail_kind == "linear":
        return (0 if rnd == 0 else rnd), (1 if rnd == 0 else rnd + 1)
    raise ValueError(f"Unknown trail kind: {trail_kind}")


def classify(states, rows, trail_kind):
    for rnd, (left, right) in enumerate(states):
        lr, rr = row_indices(trail_kind, rnd)
        lbad = left & ~sum(1 << bit for bit in rows[lr])
        rbad = right & ~sum(1 << bit for bit in rows[rr])
        if lbad or rbad:
            return False, rnd, lbad, rbad
    return True, None, 0, 0


def window_fingerprint(rows):
    return hashlib.sha256(json.dumps([sorted(row) for row in rows]).encode()).hexdigest()


def write_window(path, rounds, width, rows):
    if len(rows) != rounds + 2 or any(len(set(row)) != width for row in rows):
        raise ValueError("Invalid fixed-width window")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{rounds} {width}\n" + "".join(
        " ".join(map(str, sorted(row))) + "\n" for row in rows))


def terminal_jobs(source, weight):
    """Use terminal leaves only: a split parent's count overlaps its children."""
    directory = Path(source) / f"W{weight:03d}"
    manifest = json.loads((directory / "manifest.json").read_text())
    config = dict(manifest["config_identity"])
    if config.get("mode", "unrestricted") != "unrestricted" or config.get("window_supports") is not None:
        raise ValueError("SAT-guided windows require unrestricted source trails")
    jobs = []
    for task in manifest["tasks"].values():
        if task.get("children"):
            continue
        result_path = directory / f"shard_{task['id']}.json"
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text())
        if result.get("status") not in ("complete", "timeout"):
            continue
        count = int(result.get("trail_count", 0))
        if count:
            jobs.append((config, {**task, "source_status": result["status"]}, count))
    layer = json.loads((directory / "layer_summary.json").read_text())
    expected = layer.get("enumerated_trail_lower_bound")
    if expected is None:
        expected = layer.get("exact_trail_count")
    if expected is None:
        expected = layer.get("completed_leaf_lower_bound")
    if expected is None or sum(job[2] for job in jobs) != int(expected):
        raise AssertionError(f"W={weight}: terminal counts disagree with layer summary")
    return jobs


def _replay_job(job):
    config, task, target, path_text = job
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    marker = path.with_suffix(".json")
    identity = {"config": config, "constraints": task["constraints"], "count": target}
    if path.exists() and marker.exists():
        previous = json.loads(marker.read_text())
        if previous.get("identity") == identity and previous.get("replayed_count") == target:
            return str(path)
    model = build_model(config, task["constraints"])
    solver = new_solver(model)
    projected = model.state_variables()
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", compresslevel=1) as handle:
        for index in range(target):
            sat, solution = solver.solve()
            if not sat:
                raise RuntimeError(f"Shard {task['id']} exhausted at {index}/{target}")
            if model.replay_weight(solution) != config["weight"]:
                raise AssertionError("SAT/replay weight mismatch")
            handle.write(json.dumps(model.decode_states(solution), separators=(",", ":")) + "\n")
            solver.add_clause([-var if solution[var] else var for var in projected])
    os.replace(temporary, path)
    atomic_json(marker, {"identity": identity, "replayed_count": target,
                         "source_status": task["source_status"]})
    return str(path)


def replay_records(source, output, start, end, workers):
    """Replay each counted leaf once; stream the same records for every window."""
    jobs, layers = [], []
    for weight in range(start, end + 1):
        layer = json.loads((Path(source) / f"W{weight:03d}" / "layer_summary.json").read_text())
        leaves = terminal_jobs(source, weight)
        layers.append({"weight": weight, "status": layer["status"],
                       "count": sum(count for _, _, count in leaves)})
        for config, task, count in leaves:
            path = Path(output) / f"W{weight:03d}" / f"{task['id']}.states.jsonl.gz"
            jobs.append((config, task, count, str(path)))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        paths = list(pool.map(_replay_job, jobs))
    exact_through = None
    for layer in layers:
        if layer["status"] != "complete":
            break
        exact_through = layer["weight"]
    summary = {"status": "complete" if all(row["status"] == "complete" for row in layers) else "partial",
               "weight_range": [start, end], "exact_through_weight": exact_through,
               "all_source_counts_replayed": True, "replayed_count": sum(row["count"] for row in layers),
               "layers": layers, "files": paths}
    atomic_json(Path(output) / "summary.json", summary)
    return summary


def iter_records(paths):
    for path_text in paths:
        path = Path(path_text)
        weight = int(path.parent.name[1:])
        with gzip.open(path, "rt") as handle:
            for line in handle:
                yield weight, json.loads(line)


def _best_trail_support(args):
    config = {"family": args.family, "trail_kind": "differential", "word_size": args.word_size,
              "rounds": args.rounds, "delta_in": args.delta_in, "delta_out": args.delta_out,
              "weight": args.best_trail_weight, "shard_round": args.shard_round,
              "window_supports": None}
    model = build_model(config, {})
    solver, projected = new_solver(model), model.state_variables()
    required = [set() for _ in range(args.rounds + 2)]
    count = 0
    while True:
        sat, solution = solver.solve()
        if not sat:
            break
        if model.replay_weight(solution) != args.best_trail_weight:
            raise AssertionError("Best-trail replay weight mismatch")
        for rnd, pair in enumerate(model.decode_states(solution)):
            for word, row in zip(pair, row_indices("differential", rnd)):
                required[row].update(bit for bit in range(args.word_size) if word >> bit & 1)
        count += 1
        solver.add_clause([-var if solution[var] else var for var in projected])
    if not count:
        raise RuntimeError("No trail at supplied best weight")
    return required, count


def initial_window(args, records, run_stage):
    """Keep the original differential Markov and linear SAT initialization."""
    size, width = args.word_size, args.window_size
    scores = [defaultdict(float) for _ in range(args.rounds + 2)]
    required = [set() for _ in scores]
    rankings, protected_count = [], None
    if args.trail_kind == "differential":
        samples = []
        for index, seed in enumerate(SEEDS, 1):
            target = args.output_dir / "sampling" / f"activation_0x{seed:x}_2p{args.samples_power}.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            run_stage(f"markov_seed{index}", [str(native_binary("markov_sampler")),
                "--family", args.family, "--word-size", str(size), "--rounds", str(args.rounds),
                "--samples-power", str(args.samples_power), "--seed", hex(seed),
                "--output", str(target), "--threads", str(args.workers),
                "--target-left", hex(args.delta_out[0]), "--target-right", hex(args.delta_out[1])], [target])
            sample = [dict() for _ in scores]
            with target.open() as handle:
                for row in csv.DictReader(handle):
                    sample[int(row["window_index"])][int(row["bit_position"])] = float(row["activation_probability"])
            samples.append(sample)
        for row in range(args.rounds + 2):
            mean = {bit: sum(sample[row][bit] for sample in samples) / len(samples) for bit in range(size)}
            rankings.append(sorted(range(size), key=lambda bit: (-mean[bit], bit)))
            for sample in samples:
                order = sorted(range(size), key=lambda bit: (-sample[row][bit], bit))
                for rank, bit in enumerate(order, 1):
                    scores[row][bit] += rank  # Higher sum of seed ranks is worse.
        required, protected_count = _best_trail_support(args)
    else:
        for weight, states in iter_records(records["files"]):
            mass = 2.0 ** (-2 * weight)
            for rnd, pair in enumerate(states):
                for word, row in zip(pair, row_indices("linear", rnd)):
                    for bit in range(size):
                        if word >> bit & 1:
                            scores[row][bit] += mass
        for pair, rnd in ((args.delta_in, 0), (args.delta_out, args.rounds)):
            for word, row in zip(pair, row_indices("linear", rnd)):
                required[row].update(bit for bit in range(size) if word >> bit & 1)
        rankings = [sorted(range(size), key=lambda bit: (-row[bit], bit)) for row in scores]
    rows = []
    for index, order in enumerate(rankings):
        if len(required[index]) > width:
            raise ValueError(f"Window row {index} needs {len(required[index])} protected bits, exceeds width {width}")
        if args.trail_kind == "differential":
            chosen = list(order[:width])
            rank = {bit: position for position, bit in enumerate(order)}
            for bit in sorted(required[index]):
                if bit not in chosen:
                    removable = [old for old in chosen if old not in required[index]]
                    chosen.remove(max(removable, key=lambda old: rank[old]))
                    chosen.append(bit)
        else:
            chosen = list(sorted(required[index]))
            for bit in order:
                if len(chosen) == width:
                    break
                if bit not in chosen:
                    chosen.append(bit)
        rows.append(sorted(chosen))
    detail = {"scores": [dict(row) for row in scores], "rankings": rankings,
              "protected": [sorted(row) for row in required], "best_trails_protected": protected_count}
    atomic_json(args.output_dir / "window_initialization.json", detail)
    return rows, scores, required


def analyze_records(records, rows, trail_kind):
    """One stream computes both overlap mass and original replacement rankings."""
    exponent = 2 if trail_kind == "linear" else 1
    inside = outside = -math.inf
    inside_count = outside_count = 0
    escapes = defaultdict(lambda: -math.inf)
    for weight, states in iter_records(records["files"]):
        ok, rnd, lbad, rbad = classify(states, rows, trail_kind)
        masslog = -exponent * weight
        if ok:
            inside, inside_count = logaddexp2(inside, masslog), inside_count + 1
            continue
        outside, outside_count = logaddexp2(outside, masslog), outside_count + 1
        if trail_kind == "differential":
            key = rnd, lbad, rbad
            escapes[key] = logaddexp2(escapes[key], masslog)
        else:
            for bad, row in zip((lbad, rbad), row_indices(trail_kind, rnd)):
                while bad:
                    bit = (bad & -bad).bit_length() - 1
                    key = rnd, row, bit
                    escapes[key] = logaddexp2(escapes[key], masslog)
                    bad &= bad - 1
    if inside_count + outside_count != records["replayed_count"]:
        raise AssertionError("Replay files changed or are incomplete")
    overlap = {"inside_count": inside_count, "outside_count": outside_count,
               "inside_log2_mass": inside, "outside_log2_mass": outside,
               "all_source_counts_replayed": True}
    return overlap, sorted(escapes.items(), key=lambda item: item[1], reverse=True)


def replacement_windows(args, rows, scores, required, escapes):
    candidates = []
    for key, masslog in escapes[:args.top_families]:
        targets = defaultdict(set)
        if args.trail_kind == "differential":
            rnd, lbad, rbad = key
            for bad, row in zip((lbad, rbad), row_indices(args.trail_kind, rnd)):
                targets[row].update(bit for bit in range(args.word_size) if bad >> bit & 1)
        else:
            _, row, bit = key
            targets[row].add(bit)
        trial, edits, valid = [list(row) for row in rows], [], True
        for row, additions in targets.items():
            for bit in sorted(additions):
                if bit in trial[row]:
                    continue
                removable = [old for old in trial[row] if old not in required[row] and old not in additions]
                if not removable:
                    valid = False
                    break
                choose = max if args.trail_kind == "differential" else min
                drop = choose(removable, key=lambda old: (scores[row][old], old))
                trial[row].remove(drop)
                trial[row].append(bit)
                trial[row].sort()
                edits.append({"row": row, "added": bit, "removed": drop})
            if not valid:
                break
        if valid:
            candidates.append({"rows": trial, "edits": edits, "escape": list(key), "escape_log2_mass": masslog})
    return candidates


def evaluate_window(args, window, label, run_stage):
    output = args.output_dir / "dp" / f"{label}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    run_stage(f"dp_{label}", [str(native_binary("window_dp")), "--trail-kind", args.trail_kind,
        "--family", args.family, "--word-size", str(args.word_size), "--window", str(window),
        "--threads", str(args.workers), "--memory-limit-gb", str(args.memory_limit_gb),
        "--input-left", hex(args.delta_in[0]), "--input-right", hex(args.delta_in[1]),
        "--output-left", hex(args.delta_out[0]), "--output-right", hex(args.delta_out[1]),
        "--output", str(output)], [output], [window])
    result = json.loads(output.read_text())
    weight = result.get("mass_weight", result.get("potential_weight", result.get("hull_weight")))
    return math.inf if weight is None else float(weight)


def strict_union(window_weight, overlap, trail_kind):
    """Window DP plus only replayed SAT trails outside it: no double counting."""
    outside = overlap["outside_log2_mass"]
    outside = -math.inf if outside is None else float(outside)
    window_weight = math.inf if window_weight is None else float(window_weight)
    union = logaddexp2(-window_weight, outside)
    result = {"semantics": "linear_hull_potential" if trail_kind == "linear" else "differential_probability",
              "window_mass_weight": window_weight, "counted_sat_outside_log2_mass": outside,
              "strict_union_mass_weight": -union,
              "all_source_counts_replayed": bool(overlap["all_source_counts_replayed"]),
              "interpretation": "lower bound: complete window DP plus disjoint counted SAT trails outside it"}
    if trail_kind == "linear":
        result.update(strict_union_potential_weight=-union, rms_correlation_weight=-0.5 * union)
    return result
