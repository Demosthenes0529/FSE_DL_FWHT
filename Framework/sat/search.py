"""One OCP subprocess dispatcher for E1 and E2 across registered ciphers.

Returned linear weights are absolute trail weights. They do not provide a
fixed-key signed linear hull or a sum of path correlations.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from Cipher.registry import get_cipher_spec
from Framework.common.path_utils import normalize_bits, paper_bits_to_hex_pair, weight_to_abs_correlation
from Framework.sat.replay import correct_e1_row


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OCP_ROOT = REPO_ROOT / "OCP"


def _config(config=None):
    config = dict(config or {})
    spec = get_cipher_spec(config.get("cipher", "speck128"))
    for key, value in (("block_size", spec.block_size), ("word_size", 1 if spec.family in ("present", "knot") else spec.word_size)):
        if key in config and int(config[key]) != value:
            raise ValueError(f"{spec.name}: {key} must be {value}")
        config[key] = value
    config["cipher"] = spec.name
    config.setdefault("version", spec.block_size)
    config["ocp_root"] = str(Path(config.get("ocp_root") or os.environ.get("OCP_ROOT") or DEFAULT_OCP_ROOT).resolve())
    requested_cms = str(config.get("cms_bin") or os.environ.get("CRYPTOMINISAT_BIN") or "cryptominisat5")
    executable = shutil.which(requested_cms)
    config["cms_bin"] = str(Path(executable or requested_cms).resolve()) if executable or Path(requested_cms).is_file() else requested_cms
    backend = config.get("backend", "auto")
    if backend not in ("auto", "sat", "ocp") or config.get("table"):
        raise ValueError("Use the SAT/OCP backend; external result tables are not packaged")
    return config


def backend_status(config=None):
    config = _config(config)
    root = Path(config["ocp_root"])
    available = (root / "primitives" / "primitives.py").is_file() and Path(config["cms_bin"]).is_file()
    return {"available": available, "preferred": "ocp", "ocp": {
        "available": available, "root": str(root), "cms_bin": config["cms_bin"]}}


def e1_backend_status(r1, table=None, cipher="speck128", solver_config=None):
    config = dict(solver_config or {})
    config.setdefault("cipher", cipher)
    if table:
        config["table"] = table
    return backend_status(config)


def e2_backend_status(r2, table=None, solver_config=None):
    config = dict(solver_config or {})
    if table:
        config["table"] = table
    return backend_status(config)


def run_worker(kind, request, config=None):
    """Execute an isolated differential or linear worker with one result schema."""
    if kind not in ("differential", "linear"):
        raise ValueError("kind must be differential or linear")
    config = _config(config)
    started = time.monotonic()
    if not backend_status(config)["available"]:
        return {"status": "solver_unavailable", "error": "Bundled OCP or CryptoMiniSat is unavailable", "runtime_sec": 0.0}
    payload = {key: config[key] for key in ("cipher", "version", "block_size", "word_size", "ocp_root", "cms_bin")}
    payload.update({"threads": int(config.get("threads", 1)), "seed": int(config.get("seed", 42)),
                    "timeout_sec": float(config.get("timeout_sec", 0.0)), "max_nodes": int(config.get("max_nodes", 0))})
    for key in ("input_zero_bits", "output_zero_bits", "input_not_zero", "work_parent"):
        if key in config:
            payload[key] = config[key]
    payload.update(request)
    with tempfile.TemporaryDirectory(prefix=f"dl_{kind}_") as temporary:
        request_path, result_path = Path(temporary) / "request.json", Path(temporary) / "result.json"
        request_path.write_text(json.dumps(payload))
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), environment.get("PYTHONPATH")]))
        environment.setdefault("MPLCONFIGDIR", str(Path(temporary) / "matplotlib"))
        timeout = payload["timeout_sec"] + 10 if payload["timeout_sec"] > 0 else None
        command = [config.get("python") or sys.executable, "-m", f"Framework.sat.{kind}_worker", "--request", str(request_path), "--result", str(result_path)]
        try:
            process = subprocess.run(command, text=True, capture_output=True, timeout=timeout, env=environment, check=False)
        except subprocess.TimeoutExpired:
            return {"status": "resource_limit", "error": "OCP worker exceeded its timeout", "runtime_sec": time.monotonic() - started}
        if not result_path.is_file():
            return {"status": "solver_error", "error": (process.stderr or process.stdout)[-2000:], "runtime_sec": time.monotonic() - started}
        return json.loads(result_path.read_text())


def _e1(bits, rounds, top_k, config, fixed_input=False):
    config = _config(config)
    bits = normalize_bits(bits, config["block_size"])
    if fixed_input and not bits:
        raise ValueError("E1 input difference must be nonzero")
    request = {"input_difference" if fixed_input else "output_difference": bits,
               "r1": int(rounds), "top_k": max(1, int(top_k)), "start_round": int(config.get("e1_start", 0))}
    if config.get("max_weight") is not None:
        request["max_weight"] = int(config["max_weight"])
    payload = run_worker("differential", request, config)
    endpoint = "delta_in_paper_bits" if fixed_input else "delta_m_paper_bits"
    if payload.get("status") != "ok":
        return [{endpoint: bits, "r1": int(rounds), "solver": "ocp_cryptominisat",
                 "status": payload.get("status", "solver_error"), "warning": payload.get("error"),
                 "runtime_sec": payload.get("runtime_sec", 0.0),
                 **{key: payload[key] for key in ("model_unsat_proven", "model_weight_bound", "optimality_model")
                    if key in payload}}]
    rows = []
    for candidate in payload["results"][:max(1, int(top_k))]:
        weight = float(candidate["w_E1"])
        row = {"r1": int(rounds), "p": weight_to_abs_correlation(weight), "w_E1": weight,
               "trail": {key: candidate.get(key) for key in ("states_paper_bits", "round_weights", "model_round_weights")},
               "solver": "ocp_sat_cryptominisat", "ocp_root": config["ocp_root"], "status": "ok",
               "runtime_sec": payload.get("runtime_sec", 0.0)}
        for endpoint in ("delta_in", "delta_m"):
            row[endpoint + "_paper_bits"] = normalize_bits(candidate[endpoint + "_paper_bits"], config["block_size"])
            row[endpoint + "_hex_pair"] = paper_bits_to_hex_pair(row[endpoint + "_paper_bits"], config["word_size"], config["block_size"])
        for key in ("model_w_E1", "replay_correction", "model_optimum_proven", "optimality_model"):
            if key in candidate:
                row[key] = candidate[key]
        rows.append(correct_e1_row(row, config["cipher"], config["word_size"]))
    return rows


def search_e1_dc_to_delta(delta_m, r1, top_k=1, solver_config=None):
    return _e1(delta_m, r1, top_k, solver_config)


def search_e1_dc_from_input(input_difference, r1, solver_config=None):
    return _e1(input_difference, r1, 1, solver_config, fixed_input=True)


def search_e2_lc(input_mask, r2, best_total_weight, base_weight, top_k, solver_config=None):
    config = _config(solver_config)
    bits = normalize_bits(input_mask, config["block_size"])
    remaining = (float(best_total_weight) - float(base_weight)) / 2.0
    if remaining <= 0:
        return [{"gamma_m_paper_bits": bits, "r2": int(r2), "solver": "ocp_cryptominisat", "status": "pruned_by_bound"}]
    request = {"input_mask": bits, "r2": int(r2), "top_k": max(1, int(top_k)),
               "start_round": int(config.get("e2_start", int(config.get("r1", 0)) + int(config.get("rm", 0))))}
    if math.isfinite(remaining):
        request["max_weight"] = max(-1, int(math.ceil(remaining) - 1))
    payload = run_worker("linear", request, config)
    rows = []
    for candidate in payload.get("results", []):
        weight = float(candidate["w_E2"])
        if weight >= remaining:
            continue
        output = normalize_bits(candidate["gamma_out_paper_bits"], config["block_size"])
        magnitude = weight_to_abs_correlation(weight)
        rows.append({"gamma_m_paper_bits": bits, "gamma_out_paper_bits": output,
                     "gamma_out_hex_pair": paper_bits_to_hex_pair(output, config["word_size"], config["block_size"]),
                     "epsilon": magnitude, "abs_epsilon": magnitude, "epsilon_sign_known": False,
                     "w_E2": weight, "r2": int(r2), "linear_trail": {key: candidate.get(key) for key in ("states_paper_bits", "round_weights")},
                     "solver": "ocp_sat_cryptominisat", "status": "ok", "ocp_root": config["ocp_root"], "runtime_sec": payload.get("runtime_sec", 0.0),
                     "model_optimum_proven": bool(candidate.get("model_optimum_proven")),
                     "optimality_model": candidate.get("optimality_model")})
    if rows:
        return sorted(rows, key=lambda row: (row["w_E2"], row["gamma_out_paper_bits"]))[:max(1, int(top_k))]
    status = payload.get("status", "solver_error")
    return [{"gamma_m_paper_bits": bits, "r2": int(r2), "solver": "ocp_cryptominisat",
             "status": "no_path_or_pruned" if status in ("ok", "no_path") else status,
             "warning": payload.get("error"), "runtime_sec": payload.get("runtime_sec", 0.0),
             **{key: payload[key] for key in ("model_unsat_proven", "model_weight_bound", "optimality_model")
                if key in payload}}]


def lower_bound_e2(input_mask, r2, solver_config=None):
    """Universal admissible lower bound on the absolute correlation weight."""
    return 0.0
