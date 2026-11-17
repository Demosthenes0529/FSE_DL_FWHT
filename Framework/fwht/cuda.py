"""Shared CUDA middle-spectrum scan: coordinate selection, histogram and FWHT."""
import json
import hashlib
import math
import threading
import time
from pathlib import Path

import torch

from .distributed import (
    allocate_shards, distributed_histogram, dtype_for_samples, local_fwht,
    cross_fwht_stage, extract_top_per_hw, primitive_paper_bit_matrix,
    _random_state_and_key, _generate_z,
)
from Cipher.registry import get_cipher_spec, get_gpu_primitive
from Framework.common.gpu import validate_primitive, resolve_gpu_ids, fwht_inplace_int64
from Framework.common.path_utils import atomic_write_json, write_jsonl


def _encrypt(primitive, state, keys, rounds, round_start=0):
    if getattr(primitive, "supports_round_start", False):
        return primitive.encrypt_batch(state, keys, rounds, round_start=round_start)
    if int(round_start):
        raise ValueError(f"{primitive.config.name} does not support nonzero round_start")
    return primitive.encrypt_batch(state, keys, rounds)


PANEL_SELECTION_POLICY = "hw1-correlation-top-k-v1"
MAX_SINGLETON_BATCH = 1 << 20


def _singleton_worker(primitive, gpu, samples, rounds, difference, seed, batch_size,
                       result, progress=None, round_start=0):
    """Exactly accumulate every singleton Walsh numerator on one GPU."""
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) & ((1 << 63) - 1))
    config = primitive.config
    block_size = int(config.block_size)
    effective_batch = min(int(batch_size), MAX_SINGLETON_BATCH)
    single_acc = torch.zeros(block_size, dtype=torch.int64, device=device)
    timing = {"data_generation_sec": 0.0, "bit_unpack_sec": 0.0,
              "hw1_sum_sec": 0.0,
              "requested_batch_size": int(batch_size),
              "effective_batch_size": effective_batch}
    for offset in range(0, int(samples), effective_batch):
        current = min(effective_batch, int(samples) - offset)
        torch.cuda.synchronize(gpu)
        stage = time.perf_counter()
        state, keys = _random_state_and_key(
            torch, primitive, current, device, generator
        )
        pair_encrypt = getattr(primitive, "encrypt_pair_difference_batch", None)
        if pair_encrypt is not None:
            delta_state = pair_encrypt(
                state, keys, difference, rounds, round_start=round_start
            )
        else:
            paired = primitive.apply_difference(state, difference)
            out0 = _encrypt(primitive, state, keys, rounds, round_start)
            out1 = _encrypt(primitive, paired, keys, rounds, round_start)
            delta_state = tuple(a ^ b for a, b in zip(out0, out1))
        packed = primitive.output_to_uint(delta_state)
        torch.cuda.synchronize(gpu)
        timing["data_generation_sec"] += time.perf_counter() - stage

        stage = time.perf_counter()
        bits = primitive_paper_bit_matrix(primitive, packed)
        # float32 is intentional: every per-batch numerator has magnitude at
        # most 2^20 and is therefore an exactly representable integer.
        signs = bits.to(torch.float32).mul_(-2.0).add_(1.0)
        torch.cuda.synchronize(gpu)
        timing["bit_unpack_sec"] += time.perf_counter() - stage

        stage = time.perf_counter()
        batch_single = torch.round(signs.sum(dim=0)).to(torch.int64)
        single_acc.add_(batch_single)
        torch.cuda.synchronize(gpu)
        timing["hw1_sum_sec"] += time.perf_counter() - stage

        if progress is not None:
            progress(current)
    torch.cuda.synchronize(gpu)
    result[gpu] = {
        "single": single_acc.cpu().tolist(),
        "timing": timing,
    }


def _select_hw1_only_panel(single_rows, panel_size, rho_threshold):
    """Select the paper's stable singleton-correlation top-k coordinates."""
    selected = [int(row["paper_bit"]) for row in single_rows[:int(panel_size)]]
    if len(selected) != int(panel_size):
        raise AssertionError(f"HW1-only panel produced {len(selected)}/{panel_size} bits")
    return selected, {
        "threshold": float(rho_threshold),
        "num_hw1_above_threshold": sum(
            float(row["abs_rho"]) > float(rho_threshold) for row in single_rows),
        "fallback_bits": [],
        "coordinate_evidence": {
            str(int(row["paper_bit"])): {
                "source": "hw1-top-k", "mask": [int(row["paper_bit"])],
                "abs_rho": float(row["abs_rho"]),
            }
            for row in single_rows[:int(panel_size)]
        },
    }


def singleton_panel(primitive, gpus, samples, rounds, difference, seed, batch_size,
                    panel_path=None, resume=False, round_start=0, panel_size=32,
                    rho_threshold=0.0):
    """Select HW1 top-k coordinates, or use the full state when B == k."""
    block_size = int(primitive.config.block_size)
    panel_size = int(panel_size)
    if not 1 <= panel_size <= 32:
        raise ValueError(f"CUDA panel size must satisfy 1 <= k <= 32, got {panel_size}")
    if block_size < panel_size:
        raise ValueError(f"B={block_size}<k={panel_size} is unsupported")
    if block_size == panel_size:
        # The panel is uniquely the full state.  A singleton scan cannot
        # affect selection, so avoid an otherwise redundant second N-sample
        # pass.  Exact primitive/projection/FWHT tests remain mandatory.
        ranking = [{"rank": bit + 1, "paper_bit": bit, "Walsh": None,
                    "rho": None, "selection_role": "not-required-full-state"}
                   for bit in range(panel_size)]
        if panel_path:
            atomic_write_json(panel_path, {
                "cipher": primitive.config.name, "block_size": block_size, "k": panel_size,
                "N": 0, "seed": int(seed), "key_mode": "not-applicable",
                "rounds": int(rounds), "round_start": int(round_start),
                "selection_policy": "full-state",
                "singleton_scan": "skipped-because-B-equals-k",
                "ranking": ranking, "panel": list(range(panel_size)),
            })
        return list(range(panel_size)), ranking, 0.0
    if resume and panel_path and Path(panel_path).is_file():
        cached = json.loads(Path(panel_path).read_text())
        compatible = (
            cached.get("cipher") == primitive.config.name
            and int(cached.get("block_size", -1)) == block_size
            and int(cached.get("k", -1)) == panel_size
            and int(cached.get("N", -1)) == int(samples)
            and int(cached.get("seed", -1)) == int(seed)
            and cached.get("gpus") == list(gpus)
            and int(cached.get("batch_size", -1)) == int(batch_size)
            and int(cached.get("rounds", -1)) == int(rounds)
            and int(cached.get("round_start", -1)) == int(round_start)
            and cached.get("key_mode") == "random-per-pair"
            and cached.get("difference_words") == [int(word) for word in difference]
            and cached.get("selection_policy") == PANEL_SELECTION_POLICY
            and float(cached.get("rho_threshold", math.nan)) == float(rho_threshold)
            and len(cached.get("panel", [])) == panel_size
            and len(set(cached.get("panel", []))) == panel_size
            and all(0 <= int(bit) < block_size for bit in cached.get("panel", []))
            and len(cached.get("ranking", [])) == block_size
        )
        if compatible:
            print(f"[stage] Panel {PANEL_SELECTION_POLICY} ranking: reusing compatible cache", flush=True)
            return ([int(bit) for bit in cached["panel"]], cached["ranking"], 0.0)
    assignments = []
    for rank, gpu in enumerate(gpus):
        begin = int(samples) * rank // len(gpus)
        end = int(samples) * (rank + 1) // len(gpus)
        assignments.append((gpu, begin, end - begin))
    result = {}
    worker_errors = {}
    threads = []
    started = time.perf_counter()
    progress_lock = threading.Lock()
    progress_state = {"done": 0, "reported": 0}
    report_step = max(1, int(samples) // 16)

    def report_progress(completed):
        with progress_lock:
            progress_state["done"] += int(completed)
            done = progress_state["done"]
            if (progress_state["reported"] == 0
                    or done - progress_state["reported"] >= report_step
                    or done >= int(samples)):
                elapsed = time.perf_counter() - started
                rate = done / elapsed if elapsed else 0.0
                print(
                    f"\r[progress] Panel HW1 sampling: "
                    f"{100.0 * done / int(samples):5.1f}% ({done}/{int(samples)}), "
                    f"{rate / 1e6:.1f} M samples/s, elapsed {elapsed:.1f}s",
                    end="", flush=True,
                )
                progress_state["reported"] = done
                if done >= int(samples):
                    print(flush=True)

    print(
        f"[stage] Panel HW1 ranking started: B={block_size}, "
        f"N={int(samples)}, threshold={float(rho_threshold):.8g}, k={panel_size}",
        flush=True,
    )
    warmup_cuda = getattr(primitive, "warmup_cuda", None)
    if warmup_cuda is not None:
        # Some CUDA extensions (notably the Triton PRESENT kernel) lazily
        # initialize a process-global driver.  Do that sequentially on the
        # caller thread before the per-GPU workers start.
        for gpu, _, _ in assignments:
            warmup_cuda(gpu, rounds, round_start=round_start)
    def run_worker(gpu, count, stream_seed):
        try:
            _singleton_worker(
                primitive, gpu, count, rounds, difference, stream_seed,
                batch_size, result, report_progress, round_start,
            )
        except BaseException as exc:
            # Exceptions raised in threading.Thread do not propagate to the
            # caller.  Preserve them explicitly so a missing CUDA device or a
            # primitive failure cannot be misreported as a completed scan and
            # then surface later as an unrelated KeyError.
            with progress_lock:
                worker_errors[gpu] = exc

    for gpu, begin, count in assignments:
        stream_seed = (int(seed) + 0x6A09E667F3BCC909 * (gpu + 1) + begin) & ((1 << 63) - 1)
        thread = threading.Thread(
            target=run_worker,
            args=(gpu, count, stream_seed),
        )
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    if worker_errors:
        details = "; ".join(
            f"cuda:{gpu}: {type(exc).__name__}: {exc}"
            for gpu, exc in sorted(worker_errors.items())
        )
        raise RuntimeError(f"Panel HW1 CUDA worker failed ({details})") from next(
            iter(worker_errors.values())
        )
    print(
        f"[stage] Panel HW1 ranking completed in "
        f"{time.perf_counter() - started:.2f}s",
        flush=True,
    )
    total_single = [sum(int(result[gpu]["single"][bit]) for gpu, _, _ in assignments)
                    for bit in range(block_size)]
    rows = []
    for bit, walsh in enumerate(total_single):
        rows.append({"paper_bit": bit, "Walsh": walsh,
                     "rho": walsh / float(samples), "abs_rho": abs(walsh / float(samples))})
    rows.sort(key=lambda row: (-abs(row["Walsh"]), row["paper_bit"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    panel, selection = _select_hw1_only_panel(rows, panel_size, rho_threshold)
    timing = {key: sum(float(result[gpu]["timing"][key]) for gpu, _, _ in assignments)
              for key in ("data_generation_sec", "bit_unpack_sec", "hw1_sum_sec")}
    timing["per_gpu"] = {str(gpu): result[gpu]["timing"] for gpu, _, _ in assignments}
    if panel_path:
        atomic_write_json(panel_path, {
            "cipher": primitive.config.name, "block_size": block_size, "k": panel_size,
            "N": int(samples), "seed": int(seed), "key_mode": "random-per-pair",
            "gpus": list(gpus), "batch_size": int(batch_size),
            "difference_words": [int(word) for word in difference],
            "rounds": int(rounds), "round_start": int(round_start),
            "rho_threshold": float(rho_threshold),
            "selection_policy": PANEL_SELECTION_POLICY,
            "ranking": rows,
            "selection": selection, "timing": timing, "panel": panel,
        })
    return panel, rows, time.perf_counter() - started


def _small_k_histogram(primitive, devices, samples, rounds, difference, seed,
                       batch_size, panel, round_start=0, progress=None):
    """Accumulate one exact 2^k histogram per GPU and reduce only once."""
    k = len(panel)
    bins = 1 << k
    assignments = []
    for rank, gpu in enumerate(devices):
        begin = int(samples) * rank // len(devices)
        end = int(samples) * (rank + 1) // len(devices)
        assignments.append((gpu, begin, end - begin))
    local = {}
    errors = {}
    lock = threading.Lock()
    done = {"n": 0, "reported": 0}
    started = time.perf_counter()
    report_step = max(1, int(samples) // 16)

    def report(n):
        if progress is None:
            return
        with lock:
            done["n"] += int(n)
            if (done["reported"] == 0 or done["n"] - done["reported"] >= report_step
                    or done["n"] >= int(samples)):
                progress(done["n"], int(samples), time.perf_counter() - started)
                done["reported"] = done["n"]

    def worker(gpu, begin, count):
        try:
            torch.cuda.set_device(gpu)
            device = torch.device(f"cuda:{gpu}")
            generator = torch.Generator(device=device)
            stream_seed = (int(seed) + 0x1E3779B97F4A7C15 * (gpu + 1) + begin) & ((1 << 63) - 1)
            generator.manual_seed(stream_seed)
            hist = torch.zeros(bins, dtype=torch.int64, device=device)
            processed = 0
            reported = 0
            while processed < count:
                current = min(int(batch_size), count - processed)
                z = _generate_z(torch, gpu, current, rounds, difference, generator,
                                primitive, panel, round_start)
                # For k=24, zeroing a fresh 128-MiB minlength bincount on every
                # batch dominates the cipher.  Atomic scatter touches only the
                # sampled bins; small histograms still benefit from bincount.
                if k >= 20:
                    hist.scatter_add_(0, z, torch.ones_like(z, dtype=torch.int64))
                else:
                    hist.add_(torch.bincount(z, minlength=bins))
                processed += current
                if processed == current or processed % (16 * int(batch_size)) == 0 or processed == count:
                    torch.cuda.synchronize(gpu)
                    report(processed - reported)
                    reported = processed
            local[gpu] = hist
        except BaseException as exc:
            with lock:
                errors[gpu] = exc

    threads = []
    for gpu, begin, count in assignments:
        t = threading.Thread(target=worker, args=(gpu, begin, count))
        t.start(); threads.append(t)
    for t in threads:
        t.join()
    if errors:
        details = "; ".join(f"cuda:{g}: {type(e).__name__}: {e}" for g, e in sorted(errors.items()))
        raise RuntimeError(f"CUDA k={k} histogram worker failed ({details})") from next(iter(errors.values()))
    owner = devices[0]
    with torch.cuda.device(owner):
        histogram = local[owner]
        for gpu in devices[1:]:
            histogram.add_(local[gpu].to(owner))
        torch.cuda.synchronize(owner)
    total = int(histogram.sum(dtype=torch.int64).item())
    if total != int(samples):
        raise AssertionError(f"histogram count mismatch: {total} != {samples}")
    return histogram, {"assignments": assignments, "owner_gpu": owner,
                       "dtype": "int64", "histogram_total_sec": time.perf_counter() - started}


def run_cuda_em(delta_bits, rm, cache, n_search, seed, rho_threshold,
                    cipher, gpus, batch_size, resume=True, force=False,
                    round_start=0, panel_size=32, panel_samples=None, panel_seed=None,
                    per_hw_limit=1000, extraction_hw_min=1,
                    extraction_hw_max=None, version=None):
    """Return candidates in the normal Em contract; no E1/E2 logic lives here."""
    spec = get_cipher_spec(cipher, version)
    if min(int(rm), int(round_start)) < 0 or int(rm) + int(round_start) > spec.max_rounds:
        raise ValueError("CUDA middle-segment round interval is outside the cipher's range")
    if not math.isfinite(float(rho_threshold)) or not 0 <= float(rho_threshold) <= 1:
        raise ValueError("rho_threshold must be finite and in [0,1]")
    if not 1 <= int(panel_size) <= 32:
        raise ValueError("CUDA panel size must be in 1..32")
    if int(batch_size) <= 0 or (int(panel_size) == 32 and int(batch_size) > 1 << 24):
        raise ValueError("CUDA batch size must be positive, and at most 2^24 for k=32")
    dtype_for_samples(n_search)
    dtype_for_samples(n_search if panel_samples is None else panel_samples)
    root = Path(cache.root)
    root.mkdir(parents=True, exist_ok=True)
    rows_path = root / "cuda_candidates.jsonl"
    summary_path = root / "cuda_summary.json"
    primitive = get_gpu_primitive(cipher, version)
    # Fail before allocating 4 x 4 GiB shards when a newly registered
    # primitive has an incompatible state/configuration contract.
    validate_primitive(primitive)
    panel_size = int(panel_size)
    devices = resolve_gpu_ids(gpus, required_count=4 if panel_size == 32 else None)
    panel_samples = int(n_search if panel_samples is None else panel_samples)
    panel_seed = int(seed if panel_seed is None else panel_seed)
    selection_required = int(primitive.config.block_size) > panel_size
    expected_panel_policy = (PANEL_SELECTION_POLICY if selection_required else "full-state")
    histogram_dtype = (str(dtype_for_samples(int(n_search))).replace("torch.", "")
                       if panel_size == 32 else "int64")
    # Retain the best candidates independently in each requested HW stratum.
    # The shared candidate selector applies the optional final overall quota.
    per_hw_limit = int(per_hw_limit)
    hw_min = int(extraction_hw_min)
    hw_max = min(8, panel_size) if extraction_hw_max is None else int(extraction_hw_max)
    if per_hw_limit < 0 or not 1 <= hw_min <= hw_max <= panel_size:
        raise ValueError("per-HW extraction requires nonnegative quota and 1 <= HW min <= max <= k")
    if resume and not force and rows_path.is_file() and summary_path.is_file():
        cached_summary = json.loads(summary_path.read_text())
        compatible = (
            int(cached_summary.get("N", -1)) == int(n_search)
            and cached_summary.get("delta_m_paper_bits") == sorted(int(b) for b in delta_bits)
            and int(cached_summary.get("rounds", -1)) == int(rm)
            and int(cached_summary.get("round_start", -1)) == int(round_start)
            and int(cached_summary.get("seed", -1)) == int(seed)
            and cached_summary.get("gpus") == devices
            and int(cached_summary.get("batch_size", -1)) == int(batch_size)
            and cached_summary.get("candidates_sha256") == hashlib.sha256(rows_path.read_bytes()).hexdigest()
            and int(cached_summary.get("k", -1)) == panel_size
            and (not selection_required or (
                int(cached_summary.get("panel_N", -1)) == panel_samples
                and int(cached_summary.get("panel_seed", -1)) == panel_seed
            ))
            and cached_summary.get("histogram_dtype") == histogram_dtype
            and cached_summary.get("cipher") == primitive.config.name
            and cached_summary.get("candidate_extraction", {}).get("selection_policy")
                == "per_hw_topk_no_global_cap_v2"
            and cached_summary.get("candidate_extraction", {}).get("threshold_comparison")
                == "strictly_greater"
            and float(cached_summary.get("candidate_extraction", {}).get("rho_threshold", math.nan))
                == float(rho_threshold)
            and int(cached_summary.get("candidate_extraction", {}).get("per_hw_limit", -1)) == per_hw_limit
            and cached_summary.get("candidate_extraction", {}).get("tie_break") == "paper_bits_lexicographic"
            and cached_summary.get("candidate_extraction", {}).get("panel_bits") == cached_summary.get("panel")
            and int(cached_summary.get("candidate_extraction", {}).get("hw_min", -1)) == hw_min
            and int(cached_summary.get("candidate_extraction", {}).get("hw_max", -1)) == hw_max
            and (not selection_required or (
                 cached_summary.get("panel_selection_policy") == expected_panel_policy
                 and float(cached_summary.get("panel_rho_threshold", math.nan))
                 == float(rho_threshold)))
        )
        if compatible:
            rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
            return {"candidates": rows, "summary": cached_summary, "cached": True}
    diff = primitive.paper_bits_to_difference(delta_bits)
    started = time.perf_counter()
    panel, singleton_ranking, singleton_sec = singleton_panel(
        primitive, devices, panel_samples, int(rm), diff, panel_seed, int(batch_size),
        root / "cuda_panel.json",
        resume=bool(resume and not force), round_start=int(round_start), panel_size=panel_size,
        rho_threshold=float(rho_threshold),
    )
    panel_payload = json.loads((root / "cuda_panel.json").read_text())
    print(f"[stage] CUDA-k{panel_size} panel: B={primitive.config.block_size}, P={panel}", flush=True)
    def sample_progress(done, total, elapsed):
        rate = done / elapsed if elapsed else 0.0
        print(f"\r[progress] Em sampling/histogram: {100.0*done/total:5.1f}% "
              f"({done}/{total}), {rate/1e6:.1f} M samples/s, elapsed {elapsed:.1f}s", end="", flush=True)
        if done >= total:
            print(flush=True)
    if panel_size < 32:
        histogram, hist = _small_k_histogram(
            primitive, devices, int(n_search), int(rm), diff, int(seed), int(batch_size),
            panel, int(round_start), sample_progress,
        )
        print(f"[stage] Em-FWHT: starting {panel_size} stages", flush=True)
        def small_progress(stage, total):
            print(f"\r[progress] Em FWHT: {stage}/{total}", end="", flush=True)
            if stage == total:
                print(flush=True)
        fwht_started = time.perf_counter()
        fwht_inplace_int64(histogram, small_progress)
        fwht_sec = time.perf_counter() - fwht_started
        raw, extract_sec, extract_summary = extract_top_per_hw(
            [histogram], [devices[0]], int(n_search), per_hw_k=per_hw_limit,
            hw_min=hw_min, hw_max=hw_max, rho_threshold=float(rho_threshold),
            panel_bits=panel,
        )
        allocation = [{"gpu": int(g), "bins": 1 << panel_size, "dtype": "int64"} for g in devices]
        local = {"wall_sec": fwht_sec}
        cross31 = cross32 = 0.0
        # The unified resume gate needs the extraction threshold, not just
        # the panel threshold.  Missing metadata makes an unchanged small-k
        # run appear incompatible even when the inner engine cache is valid.
        extract_summary["mode"] = "single_gpu_after_final_histogram_reduction"
    else:
        shards, allocation = allocate_shards(devices, int(n_search))
        print(f"[stage] CUDA-k32 histogram/FWHT dtype={histogram_dtype}, "
              f"N={int(n_search)}", flush=True)
        hist = distributed_histogram(shards, devices, int(n_search), int(rm), diff, int(seed),
                                     int(batch_size), primitive=primitive, panel_bits=panel,
                                     progress=sample_progress, round_start=int(round_start))
        print("[stage] Em-FWHT: starting 30 local stages", flush=True)
        def fwht_progress(stage, gpu):
            if gpu == devices[0]:
                print(f"\r[progress] Em FWHT local stage: {stage}/30", end="", flush=True)
                if stage == 30:
                    print(flush=True)
        local = local_fwht(shards, devices, 1 << 24, progress=fwht_progress)
        print("[stage] Em-FWHT: starting cross-GPU stage 31", flush=True)
        cross31 = cross_fwht_stage(shards, devices, ((0, 1), (2, 3)), 1 << 24)
        print(f"[progress] Em FWHT cross-GPU stage: 31/32 completed in {cross31:.2f}s", flush=True)
        print("[stage] Em-FWHT: starting cross-GPU stage 32", flush=True)
        cross32 = cross_fwht_stage(shards, devices, ((0, 2), (1, 3)), 1 << 24)
        print(f"[progress] Em FWHT cross-GPU stage: 32/32 completed in {cross32:.2f}s", flush=True)
        raw, extract_sec, extract_summary = extract_top_per_hw(
            shards, devices, int(n_search), per_hw_k=per_hw_limit,
            hw_min=hw_min, hw_max=hw_max, rho_threshold=float(rho_threshold),
            panel_bits=panel,
        )
    rows = [{"panel_mask": int(row["Gamma_m"]), "Walsh": int(row["Walsh"]),
             "rho_search": float(row["rho"]), "w_Em": float(row["w_Em"])}
            for row in raw]
    summary = {"cipher": primitive.config.name, "block_size": primitive.config.block_size,
               "delta_m_paper_bits": sorted(int(b) for b in delta_bits),
               "N": int(n_search), "rounds": int(rm),
               "round_start": int(round_start), "seed": int(seed),
               "gpus": devices, "batch_size": int(batch_size),
               "panel_N": panel_samples if selection_required else 0,
               "panel_seed": panel_seed if selection_required else None,
               "panel_selection_policy": expected_panel_policy,
               "panel_rho_threshold": float(rho_threshold) if selection_required else None,
               "panel_selection": panel_payload.get("selection"),
               "panel_low_weight_timing": panel_payload.get("timing"),
               "histogram_dtype": histogram_dtype,
               "integer_safety": f"all |partial/final Walsh| <= N={int(n_search)}",
               "k": panel_size, "panel": panel, "singleton_ranking": singleton_ranking,
               "singleton_scan_sec": singleton_sec,
               "allocation": allocation, "histogram": hist, "local_fwht_sec": local["wall_sec"],
               "cross_fwht_stage_31_sec": cross31, "cross_fwht_stage_32_sec": cross32,
               "candidate_extraction_sec": extract_sec, "candidate_extraction": extract_summary,
               "total_sec": time.perf_counter() - started,
               "peak_vram_bytes_per_gpu": {str(g): int(torch.cuda.max_memory_allocated(g)) for g in devices}}
    write_jsonl(rows_path, rows)
    summary["candidates_sha256"] = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    atomic_write_json(summary_path, summary)
    return {"candidates": rows, "summary": summary, "cached": False}
