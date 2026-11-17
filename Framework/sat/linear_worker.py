#!/usr/bin/env python3
"""Isolated OCP worker for fixed-input linear-trail searches.

OCP is imported in a subprocess because it uses generic top-level package names
(`attacks`, `tools`, ...).  Keeping those imports out of the main pipeline also
lets us connect OCP's SAT model generator to the locally installed
CryptoMiniSat binary without requiring python-sat.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


from Framework.sat.solver import (
    SolverLimit, _fixed_state_literals, _zero_state_literals,
    _nodes_to_paper, _round_output_layer, _make_external_solver,
)


def search(request):
    start = time.monotonic()
    ocp_root = Path(request["ocp_root"]).resolve()
    if not (ocp_root / "primitives" / "primitives.py").is_file():
        return {"status": "solver_unavailable", "error": f"invalid OCP root: {ocp_root}"}
    if not os.path.isfile(request["cms_bin"]):
        return {"status": "solver_unavailable", "error": f"CryptoMiniSat not found: {request['cms_bin']}"}

    sys.path.insert(0, str(ocp_root))
    from attacks import linear_cryptanalysis as linear
    import tools.model_constraints as model_constraints
    import tools.model_objective as model_objective
    import tools.sat_search as sat_search
    from Framework.sat.ocp_factory import make_ocp_permutation

    timeout_sec = float(request.get("timeout_sec", 0.0))
    deadline = time.monotonic() + timeout_sec if timeout_sec > 0 else None
    with tempfile.TemporaryDirectory(prefix="ocp_e2_", dir=request.get("work_parent") or None) as work_dir:
        sat_search.solving.solve_sat = _make_external_solver(request, deadline, work_dir)
        word_size = int(request.get("word_size", 64))
        block_size = int(request.get("block_size", 2 * word_size))
        if block_size % word_size:
            raise ValueError(f"block_size={block_size} is incompatible with word_size={word_size}")
        word_count = block_size // word_size
        cipher_name = request.get("cipher", "speck")
        cipher = make_ocp_permutation(
            cipher_name, int(request["r2"]), block_size,
            start_round=int(request.get("start_round", 0)),
        )
        cipher.add_copy_operators()
        config_model, config_solver = linear.parse_and_set_configs(
            cipher, "LINEARPATH_CORR", "OPTIMAL",
            {"model_type": "sat", "optimal_search_strategy_sat": "INCREASING FROM AT MOST 0"},
            {"solver": "CryptoMiniSat", "solution_number": int(request["top_k"])},
        )
        config_model["filename"] = str(Path(work_dir) / "e2.cnf")
        round_constraints, objective = model_constraints.gen_round_model_constraint_obj_fun(
            cipher, "LINEARPATH_CORR", "sat", config_model
        )
        objective_cap = sum(len(round_vars) for round_vars in model_objective.gen_obj_fun_variables(objective))
        input_mask = request.get("input_mask")
        output_mask = request.get("output_mask")
        if input_mask is None and output_mask is None:
            raise ValueError("request must contain input_mask or output_mask")
        fixed_masks = []
        if input_mask is not None:
            fixed_masks.extend(_fixed_state_literals("in", input_mask, word_size, word_count, cipher_name))
        if output_mask is not None:
            fixed_masks.extend(_fixed_state_literals("out", output_mask, word_size, word_count, cipher_name))
        fixed_masks.extend(_zero_state_literals(
            "in", request.get("input_zero_bits"), word_size, word_count, cipher_name
        ))
        fixed_masks.extend(_zero_state_literals(
            "out", request.get("output_zero_bits"), word_size, word_count, cipher_name
        ))
        try:
            max_weight = request.get("max_weight")
            if max_weight is not None and int(max_weight) < 0:
                solutions = []
            else:
                # Integer-objective linear models are optimized exactly with
                # monotone bounded-SAT queries.  First obtain a feasible upper
                # bound, then binary-search the smallest satisfiable weight.
                # This proves the same optimum as OCP's linear
                # "INCREASING FROM AT MOST 0" strategy without walking through
                # every infeasible weight one by one on wide-word instances.
                config_solver["solution_number"] = 1
                low = max(0, int(request.get("lower_bound", 0)))
                objective_cap = int(objective_cap)
                if max_weight is None:
                    # E2 suffixes in the unified search are short and their
                    # useful optima are normally far below the wide-state E1
                    # weights.  Start at 32 to avoid presenting the SAT solver
                    # with an unnecessarily loose 128-bit feasible region;
                    # geometric expansion still preserves completeness.
                    high = min(32, objective_cap)
                    solutions = []
                    while True:
                        solutions = sat_search.modeling_solving_sat(
                            f"AT MOST {high}", fixed_masks + round_constraints,
                            objective, config_model, config_solver,
                        )
                        if solutions or high >= objective_cap:
                            break
                        low = high + 1
                        high = min(2 * high, objective_cap)
                else:
                    high = min(int(max_weight), objective_cap)
                    solutions = sat_search.modeling_solving_sat(
                        f"AT MOST {high}", fixed_masks + round_constraints,
                        objective, config_model, config_solver,
                    )

                best_solutions = solutions
                if solutions:
                    high = min(high, int(round(float(solutions[0]["obj_fun_value"]))))
                    while low < high:
                        middle = (low + high) // 2
                        trial = sat_search.modeling_solving_sat(
                            f"AT MOST {middle}", fixed_masks + round_constraints,
                            objective, config_model, config_solver,
                        )
                        if trial:
                            best_solutions = trial
                            high = min(middle, int(round(float(trial[0]["obj_fun_value"]))))
                        else:
                            low = middle + 1
                    solutions = best_solutions
                if solutions and int(request["top_k"]) > 1:
                    config_solver["solution_number"] = int(request["top_k"])
                    solutions = sat_search.modeling_solving_sat(
                        f"AT MOST {low}", fixed_masks + round_constraints,
                        objective, config_model, config_solver,
                    )
        except SolverLimit as exc:
            return {"status": "resource_limit", "error": str(exc), "runtime_sec": time.monotonic() - start}

        results = []
        for solution in solutions or []:
            weight = float(solution["obj_fun_value"])
            trail = linear.extract_trail_structures(cipher, "LINEARPATH_CORR", solution)
            output_nodes = trail["outputs"]["OUT_"]
            states = [_nodes_to_paper(trail["inputs"]["IN_"], word_size, cipher_name)]
            function = trail["functions"]["PERMUTATION"]
            last_layer = cipher.functions["PERMUTATION"].nbr_layers
            round_output_layer = _round_output_layer(cipher_name, last_layer)
            for round_index in range(1, int(request["r2"]) + 1):
                states.append(_nodes_to_paper(function[round_index][round_output_layer], word_size, cipher_name))
            results.append({
                "model_optimum_proven": True,
                "optimality_model": "OCP LINEARPATH_CORR",
                "w_E2": weight,
                "gamma_out_paper_bits": _nodes_to_paper(output_nodes, word_size, cipher_name),
                "round_weights": solution.get("rounds_obj_fun_values"),
                "states_paper_bits": states,
            })
        return {
            "status": "ok" if results else "no_path",
            "model_unsat_proven": not bool(results),
            "model_weight_bound": request.get("max_weight"),
            "results": results,
            "runtime_sec": time.monotonic() - start,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    with open(args.request, "r") as handle:
        request = json.load(handle)
    try:
        result = search(request)
    except Exception as exc:
        result = {"status": "solver_error", "error": f"{type(exc).__name__}: {exc}"}
    with open(args.result, "w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
