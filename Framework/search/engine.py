"""One shared scheduler for E1 + keyed Em/FWHT + E2 searches."""
import math, os, time
from dataclasses import replace
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from pathlib import Path

from .models import Panel, compose_weight
from Framework.common.path_utils import read_jsonl, atomic_write_json as write_json, write_jsonl, collect_fingerprint
from .adapter import require_version, validate_payload
from Framework.fwht.candidates import row_to_candidate, select_candidates


def _solve_e2_process(cipher,version,gamma_m,bound,config_payload):
    """Picklable worker: Python model construction and CMS both run in this process."""
    from .models import SearchConfig
    from .adapter import get_adapter
    config=SearchConfig(**config_payload);adapter=get_adapter(cipher,version)
    try:payload=validate_payload(adapter.search_e2(int(gamma_m),int(bound),config),"E2",config.r2)
    except Exception as error:payload={"status":"error","results":[],"error":repr(error)}
    payload["worker_pid"]=os.getpid();return payload


def _solve_e1_process(cipher,version,delta,config_payload):
    """Picklable E1 worker so Python trail models use real CPU parallelism."""
    from .models import SearchConfig
    from .adapter import get_adapter
    config=SearchConfig(**config_payload);adapter=get_adapter(cipher,version)
    try:payload=validate_payload(adapter.search_e1(int(delta),config),"E1",config.r1)
    except Exception as error:payload={"status":"error","results":[],"error":repr(error)}
    payload["worker_pid"]=os.getpid();return payload


def _stage(text): print(f"\n[stage] {text}", flush=True)


def validate_config(adapter,config):
    require_version(adapter, config.version)
    if min(config.r1,config.rm,config.r2,config.round_start)<0 or config.rm<=0:
        raise ValueError("round counts/start must be nonnegative and rm must be positive")
    if config.end_round>adapter.metadata.max_rounds:
        raise ValueError(f"round interval ends at {config.end_round}, beyond {adapter.metadata.max_rounds}")
    if config.backend not in {"cpu","cuda"}:
        raise ValueError("backend must be cpu or cuda")
    if adapter.metadata.name == "knot256" and config.knot_mode not in {"aead", "permutation"}:
        raise ValueError("KNOT mode must be aead or permutation")
    panel_limit = 32 if config.backend == "cuda" else 24
    if config.panel_size<1 or config.panel_size>panel_limit:
        raise ValueError(f"{config.backend} panel size must be in 1..{panel_limit}")
    if config.backend == "cuda" and config.sample_count > (1 << 34):
        raise ValueError("distributed CUDA k=32 backend currently supports N <= 2^34")
    if int(config.cuda_batch_size) < 1:
        raise ValueError("cuda_batch_size must be positive")
    if min(config.sample_count, config.scout_count) < 1:
        raise ValueError("sample counts must be positive")
    if config.scout_count>config.sample_count:
        raise ValueError("single-cache Em requires scout samples <= evaluation samples")
    if min(config.scout_samples_power,config.eval_samples_power,config.validation_samples_power)<0:
        raise ValueError("sample powers must be nonnegative")
    if min(config.parallel_workers,config.validation_threads,config.solver_threads)<1:
        raise ValueError("worker/thread counts must be positive")
    if (config.delta_candidates<0 or config.leaderboard_size<1
            or config.em_top_k<0 or config.em_top_per_hw<0):
        raise ValueError("delta/leaderboard counts must be positive and Em quotas nonnegative")
    if len(set(config.delta_cells)) != len(config.delta_cells):
        raise ValueError("delta cells must be unique")
    if any(cell < 0 or cell >= len(adapter.metadata.cells) for cell in config.delta_cells):
        raise ValueError(f"delta cells must be in 0..{len(adapter.metadata.cells)-1}")
    if not 1 <= config.mask_hw_min <= config.mask_hw_max <= adapter.metadata.block_bits:
        raise ValueError("invalid output-mask Hamming-weight interval")
    if not math.isfinite(config.rho_threshold) or config.rho_threshold < 0:
        raise ValueError("rho_threshold must be finite and nonnegative")
    if not math.isfinite(config.timeout_seconds) or config.timeout_seconds < 0:
        raise ValueError("timeout_seconds must be finite and nonnegative; zero disables it")
    if config.total_weight_bound is not None and (
            not math.isfinite(config.total_weight_bound) or config.total_weight_bound < 0):
        raise ValueError("total_weight_bound must be finite and nonnegative when supplied")
    if min(config.e1_weight_bound, config.e2_weight_bound) < 0:
        raise ValueError("outer trail weight bounds must be nonnegative")


def generate_deltas(adapter, config):
    if config.fixed_deltas:
        if any(not 0 < int(x) < (1 << adapter.metadata.block_bits) for x in config.fixed_deltas):
            raise ValueError("input differences must be nonzero and fit the cipher state")
        return list(dict.fromkeys(int(x) for x in config.fixed_deltas))
    values = []
    cells = config.delta_cells or tuple(range(len(adapter.metadata.cells)))
    for cell in cells:
        for local in range(1, 1 << adapter.metadata.cell_bits):
            value = adapter.set_cell(0, cell, local)
            if config.delta_source == "pre-linear-single-cell": value = adapter.linear(value)
            elif config.delta_source != "single-cell": raise ValueError(config.delta_source)
            values.append(value)
    return values


def _cutoff(paths, k, initial):
    weights = sorted(float(row["W_search"]) for row in paths)
    return min(float(initial) if initial is not None else math.inf,
               weights[k - 1] if len(weights) >= k else math.inf)


def _best_weight(paths):
    return min((float(row["W_search"]) for row in paths),default=None)


def _validate_local(adapter, delta, local, config, directory, index):
    if not config.validation_enabled or not local or config.validation_top_n <= 0:return 0
    selected=sorted(local,key=lambda row:float(row["W_search"]))[:config.validation_top_n]
    masks=list(dict.fromkeys(int(row["gamma_m"],16) for row in selected))
    validation_dir=directory/"validation"
    validation_config=replace(config,validation_seed=int(config.validation_seed)+index-1)
    try:
        results=adapter.validate_masks(delta,masks,validation_config,validation_dir)
    except Exception as error:
        write_json(validation_dir/"error.json",{"status":"error","error":repr(error)})
        return 0
    by_mask={row["mask"]:row for row in results};write_jsonl(validation_dir/"validation.jsonl",results)
    count=0
    for path in local:
        result=by_mask.get(path["gamma_m"])
        if result is None:continue
        rho=abs(float(result["rho_valid"]));path["rhoEm_valid"]=rho
        path["wEm_valid"]=-math.log2(rho) if rho else None
        path["W_valid"]=compose_weight(path["wE1"],path["wEm_valid"],path["wE2"]) if rho else None
        path["validation"]={**result,"seed":validation_config.validation_seed,"key_mode":"fixed","independent_from_search":True}
        count+=1
    return count


def _resume_em_panels(adapter, config, directory, em_payload):
    """Restore the coordinates and row-file names actually used by the scan."""
    stored = em_payload.get("panels")
    if stored is not None:
        if not isinstance(stored, dict):
            raise ValueError("Em cache has invalid saved panel descriptors")
        panels = {}
        for domain in ("direct",):
            descriptors = stored.get(domain)
            if not isinstance(descriptors, list) or not descriptors:
                raise ValueError(f"Em cache has no saved panel descriptors for {domain}")
            panels[domain] = []
            for descriptor in descriptors:
                try:
                    panel = Panel(
                        domain=str(descriptor["domain"]),
                        bits=tuple(int(bit) for bit in descriptor["bits"]),
                        method=str(descriptor["method"]),
                        panel_id=str(descriptor["panel_id"]),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Em cache has an invalid panel descriptor for {domain}") from exc
                if (panel.domain != domain or not panel.bits
                        or len(set(panel.bits)) != len(panel.bits)
                        or min(panel.bits) < 0 or max(panel.bits) >= adapter.metadata.block_bits
                        or not panel.panel_id or Path(panel.panel_id).name != panel.panel_id):
                    raise ValueError(f"Em cache has an invalid panel descriptor for {domain}")
                panels[domain].append(panel)
        return panels

    raise ValueError("Em cache is missing its panel descriptor; use a new output directory")


def _candidate_selection_metadata(config):
    return {
        "selection_policy": "per_hw_then_global_topk_v1",
        "hw_basis": "mapped_gamma_m",
        "per_hw_limit": int(config.em_top_per_hw),
        "global_limit": int(config.em_top_k) if config.em_top_k else None,
        "rho_threshold": float(config.rho_threshold),
        "threshold_comparison": "strictly_greater",
        "mask_hw_min": int(config.mask_hw_min), "mask_hw_max": int(config.mask_hw_max),
        "source_pool_policy": ("per_hw_topk_no_global_cap_v2" if config.backend == "cuda"
                               else "cpu_all_threshold_hits_no_global_cap"),
        "source_pool_global_limit": None,
    }


def _check_candidate_selection_cache(config, em_payload):
    expected = _candidate_selection_metadata(config)
    cached = em_payload.get("candidate_selection")
    if cached != expected:
        raise ValueError(
            "Em candidate-selection cache is missing or incompatible with the "
            "per-HW-then-global quota phase. Use a new output directory to recompute; "
            "an earlier globally capped candidate pool cannot be reused safely."
        )



def run(adapter, config):
    # Record the effective dependency paths, including environment overrides,
    # before checking cache identity. The solver itself may be installed later.
    from Framework.sat.search import backend_status
    dependency = backend_status({"cipher": adapter.gpu_name, "cms_bin": config.cms_bin,
                                 "ocp_root": config.ocp_root})["ocp"]
    config = replace(config, cms_bin=dependency["cms_bin"], ocp_root=dependency["root"])
    validate_config(adapter,config)
    started = time.monotonic(); out = Path(config.output_dir); out.mkdir(parents=True, exist_ok=True)
    identity = config.to_dict(); identity.pop("resume", None); identity["adapter_variant"] = adapter.metadata.variant
    identity["implementation_fingerprint"] = collect_fingerprint()
    # Compare the JSON representation so tuple-valued configuration fields
    # round-trip as lists without producing false resume mismatches.
    identity = __import__('json').loads(__import__('json').dumps(identity, sort_keys=True))
    identity_path = out / "run_identity.json"
    if identity_path.exists():
        existing = __import__('json').loads(identity_path.read_text()); existing.pop("resume", None)
        if existing != identity:
            changed = sorted(
                key for key in set(existing) | set(identity)
                if existing.get(key) != identity.get(key)
            )
            raise ValueError(
                "resume identity mismatch; choose a new output directory "
                f"(different fields: {', '.join(changed)})"
            )
    if not config.resume:
        # Re-running a directory must not inherit an earlier completion marker
        # after a new interrupted search. Keep old raw data as diagnostics,
        # but invalidate the caches and rankings that can affect future runs.
        stale = [out / name for name in (
            "best_path.json", "best_path_valid.json", "checkpoint.json", "summary.json",
            "hm_delta_recommendation/status.jsonl",
            "hm_delta_recommendation/selected_candidates.jsonl",
            "hm_delta_recommendation/all_candidates.jsonl")]
        stale.extend(out.glob("delta_runs/*/done.json"))
        stale.extend(out.glob("delta_runs/*/em_scan_done.json"))
        for path in stale:
            if path.is_file():
                path.unlink()
        for name in ("paths_search.jsonl", "paths_valid.jsonl"):
            write_jsonl(out / "paths" / name, [])
    write_json(identity_path, identity)
    _stage(f"{adapter.metadata.variant} unified search started: split={config.r1}+{config.rm}+{config.r2} offset={config.round_start}")

    resolved = {"ok", "sat", "no_path", "unsat", "no_path_or_pruned", "pruned_by_bound",
                "pruned_global_topk", "pruned_weight_bound", "infeasible"}
    unresolved = []
    e1_status_path = out / "hm_delta_recommendation/status.jsonl"
    e1_statuses = read_jsonl(e1_status_path)
    retry_e1 = any(str(row["status"]).lower() not in resolved for row in e1_statuses)
    selected_path = out / "hm_delta_recommendation/selected_candidates.jsonl"
    if selected_path.exists() and config.resume and not retry_e1:
        selected = read_jsonl(selected_path)
    else:
        raw = generate_deltas(adapter, config); evaluated = []; e1_statuses = []
        _stage(f"E1 started: raw_delta={len(raw)} workers={config.parallel_workers}")
        with ProcessPoolExecutor(max_workers=config.parallel_workers) as pool:
            futures = {pool.submit(_solve_e1_process,adapter.metadata.name,config.version,
                                   delta,config.to_dict()):delta for delta in raw}
            for done, future in enumerate(as_completed(futures), 1):
                delta = futures[future]; payload = future.result()
                e1_statuses.append({"delta_m":adapter.format_state(delta),"status":payload["status"],"error":payload.get("error")})
                if payload.get("results"):
                    row = payload["results"][0]
                    evaluated.append({"delta_m": adapter.format_state(delta), **row})
                if done == len(raw) or done % max(1, len(raw)//20) == 0:
                    print(f"\r[E1] {done}/{len(raw)} ({100*done/len(raw):.1f}%)", end="", flush=True)
        print(flush=True); evaluated.sort(key=lambda row: (float(row["weight"]), row["delta_m"]))
        selected = evaluated[:int(config.delta_candidates)] if config.delta_candidates else evaluated
        write_jsonl(out / "hm_delta_recommendation/all_candidates.jsonl", evaluated)
        write_jsonl(selected_path, selected)

    write_jsonl(e1_status_path, e1_statuses)
    unresolved.extend({"stage":"E1",**row} for row in e1_statuses if str(row["status"]).lower() not in resolved)
    paths = read_jsonl(out / "paths/paths_search.jsonl") if config.resume else []
    validated_paths = read_jsonl(out / "paths/paths_valid.jsonl") if config.resume else []
    selected_deltas = {row["delta_m"] for row in selected}
    paths = [row for row in paths if row.get("delta_m") in selected_deltas]
    validated_paths = [row for row in validated_paths if row.get("delta_m") in selected_deltas]
    em_audits=[]
    for index, e1 in enumerate(selected, 1):
        delta = int(e1["delta_m"], 16); directory = out / f"delta_runs/{index:03d}_{e1['delta_m']}"
        terminal = directory / "done.json"
        if terminal.exists() and config.resume:
            saved_audit = directory / "em_scan_done.json"
            if saved_audit.is_file(): em_audits.append(__import__('json').loads(saved_audit.read_text()))
            continue
        # An unfinished delta is recomputed. Its previous successful paths
        # must not count again toward the leaderboard or tighten the cutoff
        # while the missing E2 queries are retried.
        def same_search_delta(path):
            return (path.get("delta_m") == e1["delta_m"]
                    and all(int(path.get(name, -1)) == int(getattr(config, name))
                            for name in ("r1", "rm", "r2", "round_start")))
        paths = [path for path in paths if not same_search_delta(path)]
        validated_paths = [path for path in validated_paths if not same_search_delta(path)]
        candidates = []
        em_done=directory/"em_scan_done.json"
        if em_done.exists() and config.resume:
            em_payload=__import__('json').loads(em_done.read_text())
            _check_candidate_selection_cache(config, em_payload)
            em_audits.append(em_payload)
            saved_panels = _resume_em_panels(adapter, config, directory, em_payload)
            for domain in ("direct",):
                domain_dir=directory/domain
                for panel in saved_panels[domain]:
                    rows_path = domain_dir/f"{panel.panel_id}.jsonl"
                    if not rows_path.is_file():
                        raise ValueError(f"Em cache is missing saved panel rows: {rows_path}")
                    rows=read_jsonl(rows_path)
                    candidates.extend(row_to_candidate(adapter,row,panel) for row in rows)
        else:
            scan_label = "independent-panel/evaluation" if (
                config.backend == "cuda"
                and adapter.metadata.block_bits > config.panel_size
            ) else "single-cache"
            _stage(
                f"delta {index}/{len(selected)} Em {scan_label} scan: "
                "domain=direct"
            )
            # All shared CPU scanners define top-k=0 as all threshold hits.
            # Delay their global cap until after mapping, deduplication and
            # per-HW quotas, just as the CUDA per-HW extraction path does.
            scan_config = replace(config, em_top_k=0) if config.backend == "cpu" else config
            scan=adapter.scan_em(delta,scan_config,directory)
            independent_cuda_panel = (
                config.backend == "cuda"
                and adapter.metadata.block_bits > config.panel_size
                and scan.data_generation_passes == 2
                and not scan.cache_reused_for_fwht
            )
            if not independent_cuda_panel and (
                    scan.data_generation_passes != 1 or not scan.cache_reused_for_fwht):
                raise RuntimeError(
                    "scan must use one shared cache, except CUDA B>k where "
                    "independent proposer/evaluation streams require two passes"
                )
            for domain in ("direct",):
                payload=scan.domains[domain];domain_dir=directory/domain
                singleton=payload["singletons"];write_jsonl(domain_dir/"singleton.jsonl",singleton)
                for item in payload["panels"]:
                    panel=item["panel"];rows=item["rows"]
                    write_jsonl(domain_dir/f"{panel.panel_id}.jsonl",rows)
                    candidates.extend(row_to_candidate(adapter,row,panel) for row in rows)
            em_payload={"backend":scan.backend,"samples_generated":scan.samples_generated,
                "data_generation_passes":scan.data_generation_passes,
                "cache_reused_for_fwht":scan.cache_reused_for_fwht,
                "domains_derived_from_same_ciphertexts":scan.domains_derived_from_same_ciphertexts,
                "observation_domains":list(("direct",)),
                "panel_policy":"hw1-only",
                "candidate_selection":_candidate_selection_metadata(config),
                "panels":{
                    domain:[{"domain":item["panel"].domain,
                             "bits":list(item["panel"].bits),
                             "method":item["panel"].method,
                             "panel_id":item["panel"].panel_id}
                            for item in scan.domains[domain]["panels"]]
                    for domain in ("direct",)}}
            write_json(em_done,em_payload)
            em_audits.append(em_payload)
        # Quotas use the true encryption-boundary masks after domain mapping.
        candidates = select_candidates(
            candidates, adapter=adapter,
            rho_threshold=config.rho_threshold, per_hw_limit=config.em_top_per_hw,
            global_limit=config.em_top_k, mask_hw_min=config.mask_hw_min,
            mask_hw_max=config.mask_hw_max)
        write_jsonl(directory / "em_candidates.jsonl", candidates)
        local = []; statuses = []
        for candidate in candidates:
            candidate["e2_lower_bound"] = adapter.e2_lower_bound(int(candidate["gamma_m"],16),config)
            candidate["W_lower_bound"] = compose_weight(e1["weight"],candidate["wEm_search"],candidate["e2_lower_bound"])
        candidates.sort(key=lambda row:(row["W_lower_bound"],row["wEm_search"],row["gamma_m"]))
        def add_results(candidate,payload,bound,cutoff_at_launch):
            for e2 in payload.get("results", []):
                local.append({"cipher":adapter.metadata.name,"r1":config.r1,"rm":config.rm,"r2":config.r2,"round_start":config.round_start,**e1,**candidate,"gamma_out":e2["gamma_out"],"wE1":float(e1["weight"]),"wE2":float(e2["weight"]),"e2_states":e2["states"],"e2_round_weights":e2["round_weights"],"e2_path_verified":e2["path_verified"],"W_search":compose_weight(e1["weight"],candidate["wEm_search"],e2["weight"])})
            local.sort(key=lambda row:float(row["W_search"]));del local[config.leaderboard_size:]
            current_cutoff = _cutoff(paths+local,config.leaderboard_size,config.total_weight_bound)
            statuses.append({"gamma_m":candidate["gamma_m"],"status":payload.get("status"),"worker_pid":payload.get("worker_pid"),"lower_bound":candidate["e2_lower_bound"],"W_lower_bound":candidate["W_lower_bound"],"weight_bound":bound,"topk_cutoff_at_launch":cutoff_at_launch if math.isfinite(cutoff_at_launch) else None,"topk_cutoff_after":current_cutoff if math.isfinite(current_cutoff) else None,"paths_found":len(payload.get("results",[])),"error":payload.get("error")})
        _stage(f"delta {index}/{len(selected)} E2 started: candidates={len(candidates)} workers={config.parallel_workers}")
        with ProcessPoolExecutor(max_workers=config.parallel_workers) as pool:
            pending={};next_index=0;finished=0;pruned=0
            def submit_one():
                nonlocal next_index,finished,pruned
                while next_index<len(candidates):
                    candidate=candidates[next_index];next_index+=1
                    cutoff=_cutoff(paths+local,config.leaderboard_size,config.total_weight_bound)
                    if candidate["W_lower_bound"]>=cutoff-1e-12:
                        statuses.append({"gamma_m":candidate["gamma_m"],"status":"pruned_global_topk","lower_bound":candidate["e2_lower_bound"],"W_lower_bound":candidate["W_lower_bound"],"topk_cutoff":cutoff,"paths_found":0});finished+=1;pruned+=1;continue
                    bound = int(config.e2_weight_bound)
                    if math.isfinite(cutoff):
                        bound = min(bound, math.floor((cutoff-float(e1["weight"])-float(candidate["wEm_search"]))/2-1e-12))
                    if bound<candidate["e2_lower_bound"]:
                        statuses.append({"gamma_m":candidate["gamma_m"],"status":"pruned_weight_bound","lower_bound":candidate["e2_lower_bound"],"W_lower_bound":candidate["W_lower_bound"],"weight_bound":bound,"topk_cutoff":cutoff,"paths_found":0});finished+=1;pruned+=1;continue
                    future=pool.submit(_solve_e2_process,adapter.metadata.name,config.version,
                        int(candidate["gamma_m"],16),bound,config.to_dict())
                    pending[future]=(candidate,bound,cutoff);return True
                return False
            while len(pending)<max(1,config.parallel_workers) and submit_one():pass
            while pending:
                done,_=wait(set(pending),return_when=FIRST_COMPLETED)
                for future in done:
                    candidate,bound,launch=pending.pop(future)
                    try:payload=future.result()
                    except Exception as error:payload={"status":"error","results":[],"error":repr(error)}
                    add_results(candidate,payload,bound,launch);finished+=1
                    write_jsonl(directory/"e2_status.jsonl",statuses)
                while len(pending)<max(1,config.parallel_workers) and submit_one():pass
                current=paths+local;cutoff=_cutoff(current,config.leaderboard_size,config.total_weight_bound)
                best_weight=_best_weight(current)
                print(f"\r[E2] {finished}/{len(candidates)} ({100*finished/max(1,len(candidates)):.1f}%) running={len(pending)} queued={len(candidates)-next_index} pruned={pruned} top{config.leaderboard_size}={len(current)} best_W={best_weight} cutoff={cutoff if math.isfinite(cutoff) else None}",end="",flush=True)
        print(flush=True); write_jsonl(directory/"e2_status.jsonl",statuses)
        failures=[row for row in statuses if str(row["status"]).lower() not in resolved]
        unresolved.extend({"stage":"E2", "delta_m":e1["delta_m"], **row} for row in failures)
        validated=_validate_local(adapter,delta,local,config,directory,index)
        if config.validation_enabled and local and config.validation_top_n > 0 and validated == 0:
            failures.append({"stage":"validation", "status":"error"})
            unresolved.append({"stage":"validation", "delta_m":e1["delta_m"], "status":"error"})
        write_jsonl(directory/"paths.jsonl",local)
        paths.extend(local); paths.sort(key=lambda row: float(row["W_search"])); paths=paths[:config.leaderboard_size]
        write_jsonl(out/"paths/paths_search.jsonl",paths)
        validated_paths.extend(row for row in local if row.get("W_valid") is not None)
        validated_paths.sort(key=lambda row:float(row["W_valid"]));validated_paths=validated_paths[:config.leaderboard_size]
        write_jsonl(out/"paths/paths_valid.jsonl",validated_paths)
        if paths: write_json(out/"best_path.json",paths[0])
        if validated_paths:write_json(out/"best_path_valid.json",validated_paths[0])
        if paths:
            best=paths[0]
            _stage(f"delta {index}/{len(selected)} completed: global_best_W={float(best['W_search']):.6f} "
                   f"w=({float(best['wE1']):.3f}+{float(best['wEm_search']):.3f}+2*{float(best['wE2']):.3f}) "
                   f"delta_m={best['delta_m']} gamma_m={best['gamma_m']} gamma_out={best['gamma_out']}")
        else:
            _stage(f"delta {index}/{len(selected)} completed: global_best_W=None")
        write_json(out/"checkpoint.json",{"completed_delta_count":index,"total_delta_count":len(selected),"best_W_search":paths[0]["W_search"] if paths else None,"best_W_valid":validated_paths[0]["W_valid"] if validated_paths else None,"validated_paths_this_delta":validated,"updated_at":time.strftime('%F %T')})
        if not failures: write_json(terminal,{"status":"completed"})
    generation_passes = sorted(set(int(row["data_generation_passes"]) for row in em_audits))
    summary={"status":"incomplete" if unresolved else "completed","unresolved_queries":unresolved,"correlation_semantics":"three-part composition estimate; independent validation measures Em only","cipher":adapter.metadata.name,"variant":adapter.metadata.variant,"num_delta_candidates":len(selected),"num_complete_paths":len(paths),"best_W_search":paths[0]["W_search"] if paths else None,"num_validated_paths":len(validated_paths),"best_W_valid":validated_paths[0]["W_valid"] if validated_paths else None,"independent_validation_performed":any("validation" in row for row in paths),"panel_completeness":"complete_within_selected_bit_panel","observation_domains":list(("direct",)),"em_data_generation_policy":("independent_panel_and_evaluation" if generation_passes==[2] else "one_in_memory_cache_per_delta"),"em_data_generation_passes_per_delta":generation_passes[0] if len(generation_passes)==1 else generation_passes,"em_all_domains_share_ciphertexts":True,"em_backend_audits":em_audits,"runtime_seconds":time.monotonic()-started}
    write_json(out/"summary.json",summary); return summary
