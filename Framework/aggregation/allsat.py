#!/usr/bin/env python3
"""Projected AllSAT, weight-layer sharding and resumable trail-mass accumulation.

Differential mass is sum(2**(-w)); expected linear potential is
sum(2**(-2*w)). Only exhausted terminal shards contribute exact counts."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import resource
import statistics
import time
from pathlib import Path

from Framework.aggregation.models import (
    SimonExactModel, OCPLinearModel, new_solver, logaddexp2, parse_pair, parse_window,
)
from Framework.common.path_utils import atomic_write_json


def atomic_json(path, value):
    """Write standard JSON; a null weight denotes zero probability/potential."""
    def finite(item):
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, dict):
            return {key: finite(entry) for key, entry in item.items()}
        if isinstance(item, (list, tuple)):
            return [finite(entry) for entry in item]
        return item
    atomic_write_json(path, finite(value))


def feature_name(feature, word_size, trail_kind="differential"):
    sides = ("R", "L") if trail_kind == "linear" else ("L", "R")
    return sides[int(feature) // word_size] + str(feature % word_size)


def build_model(config, constraints):
    """Build the selected model and restrict real midpoint state bits."""
    kind = config.get("trail_kind", "differential")
    if kind not in ("differential", "linear"):
        raise ValueError(f"unknown trail kind: {kind}")
    family = config.get("family", "simon")
    window = config.get("window_supports")
    if kind == "linear":
        if window is not None:
            raise ValueError("linear window restrictions are applied by the window DP")
        model = OCPLinearModel(config["word_size"], config["rounds"],
                               tuple(config["delta_in"]), tuple(config["delta_out"]),
                               exact_weight=config["weight"], family=family)
    else:
        model = SimonExactModel(config["word_size"], config["rounds"],
                                tuple(config["delta_in"]), tuple(config["delta_out"]),
                                window=window, exact_weight=config["weight"], family=family)
    for feature_text, value in constraints.items():
        if value not in (0, 1):
            raise ValueError("a midpoint bit must be 0 or 1")
        variable = model.state_bit_variable(config["shard_round"], int(feature_text))
        model.cnf.add(variable if value else -variable)
    return model


def projected_probe(config, limit, timeout, output, constraints=None):
    """Sample unique trails solely to select balanced real midpoint bits."""
    constraints = constraints or {}
    model = build_model(config, constraints)
    solver = new_solver(model)
    projected = model.state_variables()
    observations = []
    started = time.monotonic()
    stop_reason = "limit"
    while len(observations) < limit:
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            stop_reason = "timeout"
            break
        sat, solution = solver.solve(time_limit=remaining)
        if sat is None:
            stop_reason = "timeout"
            break
        if not sat:
            stop_reason = "exhausted"
            break
        states = model.decode_states(solution)
        replay_weight = model.replay_weight(solution)
        if replay_weight != config["weight"]:
            raise AssertionError("probe replay weight mismatch")
        left, right = states[config["shard_round"]]
        if config.get("trail_kind") == "linear":
            # Linear OCP features use the framework's paper-bit order.
            observations.append([(right >> bit) & 1 for bit in range(config["word_size"])] +
                                [(left >> bit) & 1 for bit in range(config["word_size"])])
        else:
            observations.append([(left >> bit) & 1 for bit in range(config["word_size"])] +
                                [(right >> bit) & 1 for bit in range(config["word_size"])])
        solver.add_clause([-v if solution[v] else v for v in projected])
    count = len(observations)
    dimensions = 2 * config["word_size"]
    if not count:
        # A probe only orders splitting bits. Even an empty or timed-out probe
        # can safely use the fixed full-state ordering; shard solving still
        # establishes whether each mutually exclusive subproblem is UNSAT.
        ranking = list(range(dimensions))
        payload = {
            "status": stop_reason, "requested": limit, "observed": 0,
            "elapsed_sec": time.monotonic() - started,
            "shard_round": config["shard_round"], "ranking": ranking,
            "ranking_names": [feature_name(x, config["word_size"], config.get("trail_kind", "differential")) for x in ranking],
            "conditioned_on": constraints,
            "selection_rule": "fixed full-state order; no observed trails",
        }
        atomic_json(output, payload)
        return payload
    means = [sum(row[i] for row in observations) / count for i in range(dimensions)]
    entropy = []
    for p in means:
        entropy.append(0.0 if p in (0.0, 1.0) else -p * math.log2(p) - (1-p) * math.log2(1-p))

    def abs_corr(i, j):
        pi, pj = means[i], means[j]
        denom = math.sqrt(pi * (1-pi) * pj * (1-pj))
        if denom == 0:
            return 1.0
        joint = sum(row[i] * row[j] for row in observations) / count
        return abs((joint - pi * pj) / denom)

    # Entropy is primary.  Correlation is a secondary balancing penalty; a
    # coefficient of one can incorrectly prefer an almost-constant bit merely
    # because it is independent of the already selected bits.
    correlation_penalty = 0.25
    ranking = []
    remaining = set(range(dimensions))
    while remaining:
        best = max(remaining, key=lambda i: (
            entropy[i] - correlation_penalty * (max((abs_corr(i, j) for j in ranking), default=0.0)),
            entropy[i], -i))
        ranking.append(best)
        remaining.remove(best)
    payload = {
        "status": "complete" if stop_reason == "limit" else stop_reason,
        "requested": limit, "observed": count,
        "elapsed_sec": time.monotonic() - started,
        "shard_round": config["shard_round"],
        "ranking": ranking,
        "ranking_names": [feature_name(x, config["word_size"], config.get("trail_kind", "differential")) for x in ranking],
        "features": [{"feature": i, "name": feature_name(i, config["word_size"], config.get("trail_kind", "differential")),
                      "p_one": means[i], "entropy": entropy[i]} for i in range(dimensions)],
        "selection_rule": "greedy entropy minus 0.25 times maximum absolute empirical phi correlation",
        "correlation_penalty": correlation_penalty,
        "conditioned_on": constraints,
        "top12_abs_phi": [[abs_corr(i, j) for j in ranking[:12]] for i in ranking[:12]],
    }
    atomic_json(output, payload)
    return payload


def shard_file(output_dir, weight, shard_id):
    return Path(output_dir) / f"W{weight:03d}" / f"shard_{shard_id}.json"


def run_shard(job):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    config, task, output_dir, timeout = job
    path = shard_file(output_dir, config["weight"], task["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    running = dict(task)
    running.update({"weight": config["weight"], "shard_round": config["shard_round"],
                    "status": "running", "pid": os.getpid(), "started_at": time.time()})
    atomic_json(path, running)
    log_path = path.with_suffix(".log")
    try:
        build_started = time.monotonic()
        model = build_model(config, task["constraints"])
        solver = new_solver(model)
        build_sec = time.monotonic() - build_started
        projected = model.state_variables()
        count = 0
        solve_started = time.monotonic()
        status = "complete"
        while True:
            remaining = timeout - (time.monotonic() - solve_started)
            if remaining <= 0:
                status = "timeout"
                break
            sat, solution = solver.solve(time_limit=remaining)
            if sat is None:
                status = "timeout"
                break
            if not sat:
                break
            replay_weight = model.replay_weight(solution)
            if replay_weight != config["weight"]:
                raise AssertionError(f"SAT/replay mismatch {config['weight']} != {replay_weight}")
            count += 1
            solver.add_clause([-v if solution[v] else v for v in projected])
        solver_sec = time.monotonic() - solve_started
        result = dict(task)
        result.update({
            "weight": config["weight"], "shard_round": config["shard_round"],
            "shard_bits": sorted(map(int, task["constraints"])),
            "shard_bit_names": [feature_name(int(x), config["word_size"], config.get("trail_kind", "differential"))
                                for x in sorted(map(int, task["constraints"]))],
            "shard_value": "".join(str(task["constraints"][str(x)])
                                   for x in sorted(map(int, task["constraints"]))),
            "status": status, "trail_count": count,
            "elapsed_sec": build_sec + solver_sec, "cnf_build_sec": build_sec,
            "solver_sec": solver_sec, "variables": model.cnf.top,
            "clauses": len(model.cnf.clauses),
            "maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "finished_at": time.time(),
        })
        atomic_json(path, result)
        log_path.write_text(json.dumps({k: result[k] for k in
            ("status", "trail_count", "elapsed_sec", "cnf_build_sec", "solver_sec")},
            sort_keys=True) + "\n")
        return result
    except BaseException as exc:
        failed = dict(task)
        failed.update({"weight": config["weight"], "shard_round": config["shard_round"],
                       "status": "failed", "error": repr(exc), "finished_at": time.time()})
        atomic_json(path, failed)
        log_path.write_text(repr(exc) + "\n")
        return failed


def task_id(constraints):
    encoded = ",".join(f"{int(k)}={int(v)}" for k, v in sorted(
        constraints.items(), key=lambda item: int(item[0])))
    return hashlib.sha256(encoded.encode()).hexdigest()[:16]


def make_task(constraints, parent=None, depth=0):
    normalized = {str(int(k)): int(v) for k, v in constraints.items()}
    return {"id": task_id(normalized), "constraints": normalized,
            "parent_shard": parent, "depth": int(depth), "children": []}


def create_roots(features):
    roots = []
    for value in range(1 << len(features)):
        constraints = {str(bit): (value >> index) & 1 for index, bit in enumerate(features)}
        roots.append(make_task(constraints))
    return roots


def split_task(task, ranking, add_bits, max_bits):
    existing = set(map(int, task["constraints"]))
    available = [x for x in ranking if x not in existing]
    take = min(add_bits, max_bits - len(existing), len(available))
    if take <= 0:
        return []
    chosen = available[:take]
    children = []
    for value in range(1 << take):
        constraints = dict(task["constraints"])
        for index, bit in enumerate(chosen):
            constraints[str(bit)] = (value >> index) & 1
        children.append(make_task(constraints, parent=task["id"], depth=task["depth"] + 1))
    return children


def load_result(output_dir, weight, task):
    path = shard_file(output_dir, weight, task["id"])
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def run_layer(args, config, ranking, manifest_path):
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        expected = dict(config)
        if manifest["config_identity"] != expected:
            raise RuntimeError("resume manifest configuration mismatch")
    else:
        initial = ranking[:args.shard_bits]
        roots = create_roots(initial)
        manifest = {
            "config_identity": dict(config),
            "initial_features": initial,
            "initial_feature_names": [feature_name(x, config["word_size"], config.get("trail_kind", "differential")) for x in initial],
            "roots": [x["id"] for x in roots],
            "tasks": {x["id"]: x for x in roots},
            "created_at": time.time(),
        }
        atomic_json(manifest_path, manifest)

    layer_started = time.monotonic()
    attempted = set()
    while True:
        runnable = []
        unresolved = []
        leaves = []
        # Conditional probes are independent SAT jobs.  Run them concurrently
        # instead of serially when many root shards time out together.
        probe_jobs = []
        if args.adaptive_probe_models:
            for task in list(manifest["tasks"].values()):
                if task.get("children") or len(task["constraints"]) >= args.max_shard_bits:
                    continue
                if (args.max_adaptive_depth is not None
                        and task["depth"] >= args.max_adaptive_depth):
                    continue
                result = load_result(args.output_dir, config["weight"], task)
                status = result.get("status") if result else None
                if status != "timeout" and not (status == "running" and args.split_stale_running):
                    continue
                probe_path = manifest_path.parent / f"adaptive_probe_{task['id']}.json"
                if not probe_path.exists():
                    probe_jobs.append((task, probe_path))
        if probe_jobs:
            print(json.dumps({"event": "probe_schedule", "weight": config["weight"],
                              "jobs": len(probe_jobs), "workers": args.workers}), flush=True)
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(projected_probe, config, args.adaptive_probe_models,
                                args.adaptive_probe_timeout, probe_path,
                                task["constraints"]): (task, probe_path)
                    for task, probe_path in probe_jobs
                }
                for future in concurrent.futures.as_completed(futures):
                    task, probe_path = futures[future]
                    try:
                        probe = future.result()
                    except RuntimeError as error:
                        # A hard shard can consume its probe budget without
                        # yielding even one model.  This is not evidence that
                        # the shard is empty, so retain it and fall back to the
                        # stable global midpoint ranking for an exhaustive,
                        # mutually-exclusive split.
                        probe = {
                            "status": "fallback",
                            "requested": args.adaptive_probe_models,
                            "observed": 0,
                            "elapsed_sec": args.adaptive_probe_timeout,
                            "shard_round": config["shard_round"],
                            "ranking": ranking,
                            "ranking_names": [feature_name(x, config["word_size"], config.get("trail_kind", "differential"))
                                              for x in ranking],
                            "conditioned_on": task["constraints"],
                            "fallback_reason": str(error),
                        }
                        atomic_json(probe_path, probe)
                    print(json.dumps({"event": "probe_done", "id": task["id"],
                                      "observed": probe["observed"],
                                      "elapsed": probe["elapsed_sec"],
                                      "status": probe["status"]}), flush=True)
        # Expand timed-out leaves before scheduling the next batch.
        for task in list(manifest["tasks"].values()):
            if task.get("children"):
                continue
            result = load_result(args.output_dir, config["weight"], task)
            status = result.get("status") if result else None
            if status == "complete":
                leaves.append((task, result))
            elif status == "timeout" or (status == "running" and args.split_stale_running):
                if (args.max_adaptive_depth is not None
                        and task["depth"] >= args.max_adaptive_depth):
                    # A new invocation may have a larger timeout even when
                    # the partition cannot grow further. Retry each terminal
                    # leaf once per invocation; never spin on the same timeout.
                    if task["id"] not in attempted:
                        runnable.append(task)
                    else:
                        unresolved.append((task, result))
                    continue
                split_ranking = ranking
                if args.adaptive_probe_models and len(task["constraints"]) < args.max_shard_bits:
                    probe_path = manifest_path.parent / f"adaptive_probe_{task['id']}.json"
                    if probe_path.exists():
                        conditional_probe = json.loads(probe_path.read_text())
                    else:
                        conditional_probe = projected_probe(
                            config, args.adaptive_probe_models, args.adaptive_probe_timeout,
                            probe_path, constraints=task["constraints"])
                    split_ranking = list(map(int, conditional_probe["ranking"]))
                    task["adaptive_probe"] = str(probe_path)
                children = split_task(task, split_ranking, 2, args.max_shard_bits)
                if not children:
                    if task["id"] not in attempted:
                        runnable.append(task)
                    else:
                        unresolved.append((task, result))
                    continue
                task["children"] = [x["id"] for x in children]
                for child in children:
                    manifest["tasks"][child["id"]] = child
                runnable.extend(children)
                atomic_json(manifest_path, manifest)
            elif status == "failed" and task["id"] in attempted:
                unresolved.append((task, result))
            else:
                runnable.append(task)
        if unresolved and not runnable:
            break
        if not runnable:
            # Re-scan: a split may have introduced new child tasks.
            terminal = [t for t in manifest["tasks"].values() if not t.get("children")]
            if all((load_result(args.output_dir, config["weight"], t) or {}).get("status") == "complete"
                   for t in terminal):
                break
            continue
        attempted.update(task["id"] for task in runnable)
        batch_started = time.monotonic()
        batch_total = len(runnable)
        batch_done = 0
        batch_trails = 0
        print(f"[weight {config['weight']}] scheduling {batch_total} shards, "
              f"concurrency {min(args.workers, batch_total)}", flush=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_shard, (config, task, str(args.output_dir),
                                               args.timeout_per_shard)): task
                       for task in runnable}
            try:
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    batch_done += 1
                    batch_trails += int(result.get("trail_count") or 0)
                    elapsed = max(time.monotonic() - batch_started, 1e-9)
                    rate = batch_done / elapsed
                    eta = ((batch_total - batch_done) / rate) if rate else math.inf
                    percent = 100.0 * batch_done / batch_total
                    print(f"[weight {config['weight']}] {batch_done}/{batch_total} "
                          f"({percent:5.1f}%) | {result['status']} | "
                          f"cumulative trails={batch_trails} | {rate:.3f} shards/s | "
                          f"ETA={eta/60:.1f} min", flush=True)
            except KeyboardInterrupt:
                for future in futures:
                    future.cancel()
                raise

    terminal = [t for t in manifest["tasks"].values() if not t.get("children")]
    results = [(t, load_result(args.output_dir, config["weight"], t)) for t in terminal]
    exact = all(result and result.get("status") == "complete" for _, result in results)
    complete_results = [r for _, r in results if r and r.get("status") == "complete"]
    partial_results = [r for _, r in results if r and r.get("status") == "timeout"]
    # Derive the original parallel wall span from durable leaf timestamps so a
    # resume/report-only invocation does not replace it with a millisecond
    # checkpoint scan time.
    timed_results = [r for _, r in results if r and r.get("finished_at") is not None
                     and r.get("elapsed_sec") is not None]
    durable_wall = (max(r["finished_at"] for r in timed_results) -
                    min(r["finished_at"] - r["elapsed_sec"] for r in timed_results)
                    if timed_results else time.monotonic() - layer_started)
    summary = {
        "weight": config["weight"], "status": "complete" if exact else "incomplete",
        "semantics": ("expected_linear_potential" if config["trail_kind"] == "linear"
                      else "differential_probability"),
        "weight_definition": ("-log2(abs(trail_correlation))" if config["trail_kind"] == "linear"
                              else "-log2(trail_probability)"),
        "trail_mass": "2**(-2*weight)" if config["trail_kind"] == "linear" else "2**(-weight)",
        "exact_trail_count": sum(x["trail_count"] for x in complete_results) if exact else None,
        "completed_leaf_lower_bound": sum(x["trail_count"] for x in complete_results),
        # A timed-out terminal leaf still contains a projected-AllSAT prefix of
        # distinct, replay-validated trails.  It is safe to include that prefix
        # in a lower bound, but never in an exact count.  Timed-out ancestors
        # are not terminal and therefore cannot be double counted here.
        "enumerated_trail_lower_bound": sum(x["trail_count"] for x in complete_results)
                                           + sum(x.get("trail_count", 0)
                                                 for x in partial_results),
        "number_of_leaf_shards": len(terminal),
        "complete_leaf_shards": len(complete_results),
        "incomplete_leaf_shards": len(terminal) - len(complete_results),
        "adaptive_split_parents": sum(bool(t.get("children")) for t in manifest["tasks"].values()),
        "max_shard_depth": max((t["depth"] for t in terminal), default=0),
        "wall_time": durable_wall,
        "leaf_elapsed_distribution": ({
            "min": min(x["elapsed_sec"] for x in complete_results),
            "median": statistics.median(x["elapsed_sec"] for x in complete_results),
            "mean": statistics.mean(x["elapsed_sec"] for x in complete_results),
            "p90": sorted(x["elapsed_sec"] for x in complete_results)[
                min(len(complete_results)-1, int(.9 * len(complete_results)))],
            "max": max(x["elapsed_sec"] for x in complete_results),
        } if complete_results else None),
        "manifest": str(manifest_path),
    }
    atomic_json(manifest_path.parent / "layer_summary.json", summary)
    return summary


def prior_counts(path, max_weight):
    counts = {}
    if path:
        with Path(path).open(newline="") as handle:
            for row in csv.DictReader(handle):
                weight = int(row["weight"])
                status = row.get("status", "complete")
                count = row.get("count", row.get("exact_trail_count"))
                if weight <= max_weight and status == "complete" and count not in (None, ""):
                    counts[weight] = int(count)
    return counts


def write_lower_bound_growth(args, exact_counts, max_weight):
    """Write a cumulative curve using every safely counted terminal trail.

    Exact layers come from ``exact_counts``.  For incomplete layers, only the
    current mutually-exclusive terminal leaves are used: completed leaves plus
    the projected models emitted by terminal timeout leaves.  Timed-out parent
    shards that have children are therefore never double counted.
    """
    layer_summaries = {}
    for path in args.output_dir.glob("W[0-9][0-9][0-9]/layer_summary.json"):
        try:
            summary = json.loads(path.read_text())
            weight = int(summary["weight"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if weight <= max_weight:
            layer_summaries[weight] = summary

    entries = {}
    for weight, count in exact_counts.items():
        if weight <= max_weight:
            entries[weight] = {"count": int(count), "status": "exact"}
    for weight, summary in layer_summaries.items():
        if summary.get("status") == "complete":
            count = summary.get("exact_trail_count")
            if count is not None:
                entries[weight] = {"count": int(count), "status": "exact"}
        else:
            count = summary.get("enumerated_trail_lower_bound",
                                summary.get("completed_leaf_lower_bound", 0))
            entries[weight] = {"count": int(count), "status": "lower_bound"}

    cumulative_log = -math.inf
    rows = []
    for weight in sorted(entries):
        entry = entries[weight]
        count = entry["count"]
        exponent = 2 if args.trail_kind == "linear" else 1
        layer_log = math.log2(count) - exponent * weight if count else -math.inf
        cumulative_log = logaddexp2(cumulative_log, layer_log)
        hull_weight = -cumulative_log
        summary = layer_summaries.get(weight, {})
        rows.append({
            "weight": weight,
            "counted_trails": count,
            "status": entry["status"],
            "log2_counted_layer_probability": layer_log,
            "cumulative_counted_probability": (2.0 ** cumulative_log
                                                if cumulative_log != -math.inf else 0.0),
            "cumulative_lower_bound_weight": hull_weight,
            "gap_to_free_window": (hull_weight - args.free_window_weight
                                   if args.free_window_weight is not None else None),
            "gap_to_anchored_window": (hull_weight - args.anchored_window_weight
                                       if args.anchored_window_weight is not None else None),
            "wall_time": summary.get("wall_time"),
            "complete_leaves": summary.get("complete_leaf_shards"),
            "incomplete_leaves": summary.get("incomplete_leaf_shards", 0)
                                   if summary else None,
        })

    path = args.output_dir / "lower_bound_growth.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    first_free = next((row["weight"] for row in rows
                       if row["gap_to_free_window"] is not None
                       and row["gap_to_free_window"] < 0), None)
    first_anchored = next((row["weight"] for row in rows
                           if row["gap_to_anchored_window"] is not None
                           and row["gap_to_anchored_window"] < 0), None)
    payload = {
        "result_kind": ("strict counted linear-hull-potential lower bound"
                        if args.trail_kind == "linear" else
                        f"strict counted {args.mode}-SAT lower bound"),
        "rows": rows,
        "first_weight_surpassing_free_window": first_free,
        "first_weight_surpassing_anchored_window": first_anchored,
        "surpassed_free_window": first_free is not None,
        "surpassed_anchored_window": first_anchored is not None,
    }
    if first_anchored is not None:
        final = next(row for row in rows if row["weight"] == first_anchored)
        payload["anchored_crossing_margin_bits"] = -final["gap_to_anchored_window"]
        payload["message"] = "STRICT CROSSING OF ANCHORED WINDOW"
        print(payload["message"], flush=True)
    atomic_json(args.output_dir / "lower_bound_summary.json", payload)
    return payload


def write_growth(args, counts, layer_summaries, total_wall):
    for summary in layer_summaries:
        if summary["status"] == "complete":
            counts[summary["weight"]] = summary["exact_trail_count"]
    cumulative_log = -math.inf
    cumulative_trails = 0
    rows = []
    previous_count = None
    for weight in sorted(counts):
        count = counts[weight]
        exponent = 2 if args.trail_kind == "linear" else 1
        layer_log = math.log2(count) - exponent * weight if count else -math.inf
        layer_probability = 2.0 ** layer_log if layer_log != -math.inf else 0.0
        before = cumulative_log
        cumulative_log = logaddexp2(cumulative_log, layer_log)
        cumulative_trails += count
        hull_weight = -cumulative_log
        summary = next((x for x in layer_summaries if x["weight"] == weight), None)
        rows.append({
            "weight": weight, "exact_trail_count": count,
            "trail_count_growth_ratio": (count / previous_count
                                         if previous_count not in (None, 0) else None),
            "layer_contribution_ratio": (count / ((2.0 ** exponent) * previous_count)
                                         if previous_count not in (None, 0) else None),
            "layer_probability": layer_probability,
            "log2_layer_probability": layer_log,
            "cumulative_trails": cumulative_trails,
            "cumulative_hull_weight": hull_weight,
            "layer_improvement_bits": (None if before == -math.inf else cumulative_log-before),
            "gain_vs_best_trail": ((2 if args.trail_kind == "linear" else 1)
                                    * args.best_trail_weight - hull_weight),
            "gap_to_free_window": (hull_weight - args.free_window_weight
                                   if args.free_window_weight is not None else None),
            "gap_to_anchored_window": (hull_weight - args.anchored_window_weight
                                       if args.anchored_window_weight is not None else None),
            "surpasses_free_window": (hull_weight < args.free_window_weight
                                      if args.free_window_weight is not None else False),
            "surpasses_anchored_window": (hull_weight < args.anchored_window_weight
                                          if args.anchored_window_weight is not None else False),
            "wall_time": summary["wall_time"] if summary else None,
            "number_of_leaf_shards": summary["number_of_leaf_shards"] if summary else None,
            "max_shard_depth": summary["max_shard_depth"] if summary else None,
        })
        previous_count = count
    path = args.output_dir / "weight_growth.csv"
    with path.open("w", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    first_free = next((x["weight"] for x in rows if x["surpasses_free_window"]), None)
    first_anchored = next((x["weight"] for x in rows if x["surpasses_anchored_window"]), None)
    incomplete = []
    for summary in layer_summaries:
        if summary["status"] == "complete":
            continue
        weight = summary["weight"]
        count_lb = summary.get("enumerated_trail_lower_bound",
                               summary["completed_leaf_lower_bound"])
        exponent = 2 if args.trail_kind == "linear" else 1
        layer_log_lb = math.log2(count_lb) - exponent * weight if count_lb else -math.inf
        # A continued run may already have completed higher weight layers.
        # A cumulative bound labelled W must never include those later layers.
        exact_prefix_log = -math.inf
        for row in rows:
            if row["weight"] <= weight:
                exact_prefix_log = logaddexp2(exact_prefix_log, row["log2_layer_probability"])
        hull_weight_ub = -logaddexp2(exact_prefix_log, layer_log_lb)
        previous_exact = next((row for row in reversed(rows) if row["weight"] < weight), None)
        previous_count = previous_exact["exact_trail_count"] if previous_exact else None
        previous_hull = previous_exact["cumulative_hull_weight"] if previous_exact else None
        incomplete.append({
            **summary,
            "trail_count_growth_ratio_lower_bound": (count_lb / previous_count
                                                        if previous_count else None),
            "layer_contribution_ratio_lower_bound": (count_lb / ((2.0 ** exponent) * previous_count)
                                                       if previous_count else None),
            "log2_layer_probability_lower_bound": layer_log_lb,
            "cumulative_hull_weight_upper_bound": hull_weight_ub,
            "layer_improvement_bits_lower_bound": (previous_hull - hull_weight_ub
                                                      if previous_hull is not None else None),
            "surpasses_free_window_strictly": (hull_weight_ub < args.free_window_weight
                                                if args.free_window_weight is not None else False),
            "surpasses_anchored_window_strictly": (hull_weight_ub < args.anchored_window_weight
                                                    if args.anchored_window_weight is not None else False),
        })
    payload = {"semantics": ("expected_linear_potential" if args.trail_kind == "linear"
                             else "differential_probability"),
               "rows": rows, "incomplete_layers": incomplete,
               "total_wall_time": total_wall,
               "surpassed_free_window": first_free is not None,
               "surpassed_anchored_window": first_anchored is not None,
               "first_weight_surpassing_free_window": first_free,
               "first_weight_surpassing_anchored_window": first_anchored}
    strict_free = next((x["weight"] for x in incomplete
                        if x["surpasses_free_window_strictly"]), None)
    strict_anchored = next((x["weight"] for x in incomplete
                            if x["surpasses_anchored_window_strictly"]), None)
    if first_free is None and strict_free is not None:
        payload["first_weight_strict_lower_bound_surpassing_free_window"] = strict_free
    if first_anchored is None and strict_anchored is not None:
        payload["first_weight_strict_lower_bound_surpassing_anchored_window"] = strict_anchored
    if payload.get("first_weight_strict_lower_bound_surpassing_free_window") is not None:
        print("unrestricted SAT lower bound has surpassed free window hull", flush=True)
    if payload["surpassed_anchored_window"]:
        payload["message"] = "unrestricted SAT lower bound has surpassed anchored window hull"
        print(payload["message"], flush=True)
    atomic_json(args.output_dir / "summary.json", payload)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trail-kind", choices=("differential", "linear"),
                        default="differential")
    parser.add_argument("--family", choices=("simon", "simeck", "speck"), default="simon")
    parser.add_argument("--word-size", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--input", dest="input_text", required=True)
    parser.add_argument("--output", dest="output_text", required=True)
    parser.add_argument("--mode", choices=("unrestricted", "window-restricted"), default="unrestricted")
    parser.add_argument("--window", type=Path)
    parser.add_argument("--weight-start", type=int, required=True)
    parser.add_argument("--weight-end", type=int, required=True)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--shard-round", type=int)
    parser.add_argument("--shard-bits", type=int, default=6)
    parser.add_argument("--max-shard-bits", type=int, default=12)
    parser.add_argument("--max-adaptive-depth", type=int,
                        help="stop splitting timeout leaves at this task depth; their emitted trails remain a lower bound")
    parser.add_argument("--timeout-per-shard", type=float, default=900)
    parser.add_argument("--probe-models", type=int, default=1000)
    parser.add_argument("--probe-timeout", type=float, default=600)
    parser.add_argument("--probe-file", type=Path,
                        help="reuse an existing midpoint-bit probe without resampling")
    parser.add_argument("--adaptive-probe-models", type=int, default=500)
    parser.add_argument("--adaptive-probe-timeout", type=float, default=300)
    parser.add_argument("--split-stale-running", action="store_true",
                        help="conditionally split interrupted running leaves instead of rerunning them")
    parser.add_argument("--continue-incomplete", action="store_true",
                        help="continue to the next weight after recording an incomplete layer lower bound")
    parser.add_argument("--stop-on-anchored-crossing", action="store_true",
                        help="stop once the strict counted lower bound exceeds the anchored window")
    parser.add_argument("--prior-layers", type=Path)
    parser.add_argument("--best-trail-weight", type=float, default=64)
    parser.add_argument("--free-window-weight", type=float, default=None)
    parser.add_argument("--anchored-window-weight", type=float, default=None)
    parser.add_argument("--no-window-references", action="store_true",
                        help="omit experiment-specific window comparisons from generic runs")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.rounds < 1 or args.word_size < 1:
        parser.error("rounds and word-size must be positive")
    if not 0 <= args.weight_start <= args.weight_end:
        parser.error("require 0 <= weight-start <= weight-end")
    if args.workers < 1 or args.probe_models < 1 or args.adaptive_probe_models < 0:
        parser.error("workers/probe-models must be positive and adaptive-probe-models nonnegative")
    if min(args.timeout_per_shard, args.probe_timeout, args.adaptive_probe_timeout) <= 0:
        parser.error("solver timeouts must be positive")
    if args.trail_kind == "linear" and args.mode != "unrestricted":
        parser.error("linear SAT supports unrestricted SPECK enumeration; "
                     "SIMON/SIMECK E2 uses fixed-key measurement")
    if args.trail_kind == "linear" and args.family in ("simon", "simeck"):
        parser.error("SIMON/SIMECK bitwise OCP linear weights are not exact whole-round "
                     "correlations; use Experiments.validation fixed-key E2 measurement")
    if args.window and args.mode != "window-restricted":
        parser.error("--window requires --mode window-restricted")
    if args.shard_round is None:
        args.shard_round = args.rounds // 2
    if args.family == "speck" and (args.trail_kind != "linear" or args.mode != "unrestricted"):
        parser.error("Speck is supported here only for unrestricted linear-hull accumulation")
    if args.no_window_references:
        args.free_window_weight = None
        args.anchored_window_weight = None
        if args.stop_on_anchored_crossing:
            parser.error("--stop-on-anchored-crossing requires a window reference")
    args.delta_in = parse_pair(args.input_text, args.word_size)
    args.delta_out = parse_pair(args.output_text, args.word_size)
    args.workers = min(args.workers, os.cpu_count() or 1)
    if not 0 < args.shard_bits <= args.max_shard_bits <= 2 * args.word_size:
        parser.error("require 0 < shard-bits <= max-shard-bits <= state bits")
    if args.max_adaptive_depth is not None and args.max_adaptive_depth < 0:
        parser.error("max-adaptive-depth must be non-negative")
    if not 0 <= args.shard_round <= args.rounds:
        parser.error("shard-round is outside the trail")
    if args.mode == "window-restricted" and not args.window:
        parser.error("window-restricted mode requires --window")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    window_supports = None
    if args.window:
        window_supports, _ = parse_window(args.window, args.rounds, args.word_size)
        window_supports = [sorted(x) for x in window_supports]
    base = {
        "family": args.family, "word_size": args.word_size, "rounds": args.rounds,
        "delta_in": list(args.delta_in), "delta_out": list(args.delta_out),
        "shard_round": args.shard_round, "mode": args.mode,
        "trail_kind": args.trail_kind,
        "window_supports": window_supports,
    }
    probe_path = args.probe_file or (args.output_dir / f"probe_W{args.weight_start:03d}.json")
    if probe_path.exists():
        probe = json.loads(probe_path.read_text())
        if int(probe["shard_round"]) != args.shard_round:
            parser.error("probe-file shard round does not match --shard-round")
    else:
        if args.probe_file:
            parser.error("--probe-file does not exist")
        probe_config = dict(base, weight=args.weight_start)
        probe = projected_probe(probe_config, args.probe_models, args.probe_timeout, probe_path)
    ranking = list(map(int, probe["ranking"]))
    if sorted(ranking) != list(range(2 * args.word_size)):
        parser.error("probe ranking must contain every state-bit index exactly once")
    counts = prior_counts(args.prior_layers, args.weight_start - 1)
    summaries = []
    overall_started = time.monotonic()
    for weight in range(args.weight_start, args.weight_end + 1):
        print(f"\n[stage] starting accumulation for weight layer W={weight} "
              f"({weight-args.weight_start+1}/{args.weight_end-args.weight_start+1})",
              flush=True)
        config = dict(base, weight=weight)
        layer_dir = args.output_dir / f"W{weight:03d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        summary = run_layer(args, config, ranking, layer_dir / "manifest.json")
        summaries.append(summary)
        write_growth(args, counts, summaries, time.monotonic() - overall_started)
        lower_bound = write_lower_bound_growth(args, counts, weight)
        print(json.dumps(summary, indent=2), flush=True)
        count_label = (summary["exact_trail_count"] if summary["status"] == "complete"
                       else summary["enumerated_trail_lower_bound"])
        print(f"[stage] W={weight} completed: status={summary['status']}, "
              f"trails={count_label}, wall={summary['wall_time']/60:.2f} min",
              flush=True)
        if args.stop_on_anchored_crossing and lower_bound["surpassed_anchored_window"]:
            break
        if summary["status"] != "complete" and not args.continue_incomplete:
            break


if __name__ == "__main__":
    main()
