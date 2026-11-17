#!/usr/bin/env python3
"""Isolated OCP worker for optimal fixed-input differential trails."""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from Framework.sat.solver import SolverLimit, _fixed_state_literals, _make_external_solver, _nodes_to_paper, _zero_state_literals, _round_output_layer
from Framework.sat.replay import correct_e1_candidate


def search(request):
    start = time.monotonic()
    root = Path(request["ocp_root"]).resolve()
    if not (root / "primitives" / "primitives.py").is_file():
        return {"status": "solver_unavailable", "error": f"invalid OCP root: {root}"}
    if not os.path.isfile(request["cms_bin"]):
        return {"status": "solver_unavailable", "error": f"CryptoMiniSat not found: {request['cms_bin']}"}

    sys.path.insert(0, str(root))
    from attacks import differential_cryptanalysis as differential
    import tools.model_constraints as model_constraints
    import tools.model_objective as model_objective
    import tools.sat_search as sat_search
    from Framework.sat.ocp_factory import make_ocp_permutation

    timeout_sec = float(request.get("timeout_sec", 0.0))
    deadline = time.monotonic() + timeout_sec if timeout_sec > 0 else None
    with tempfile.TemporaryDirectory(prefix="ocp_e1_", dir=request.get("work_parent") or None) as work_dir:
        sat_search.solving.solve_sat = _make_external_solver(request, deadline, work_dir)
        rounds = int(request["r1"])
        word_size = int(request["word_size"])
        block_size = int(request.get("block_size", 2 * word_size))
        if block_size % word_size:
            raise ValueError(f"block_size={block_size} is incompatible with word_size={word_size}")
        word_count = block_size // word_size
        cipher_name = request.get("cipher", "speck")
        cipher = make_ocp_permutation(
            cipher_name, rounds, block_size,
            start_round=int(request.get("start_round", 0)),
        )
        cipher.add_copy_operators()
        config_model, config_solver = differential.parse_and_set_configs(
            cipher, "DIFFERENTIALPATH_PROB", "OPTIMAL",
            {"model_type": "sat", "optimal_search_strategy_sat": "INCREASING FROM AT MOST 0"},
            {"solver": "CryptoMiniSat", "solution_number": 1},
        )
        config_model["filename"] = str(Path(work_dir) / "e1.cnf")
        round_constraints, objective = model_constraints.gen_round_model_constraint_obj_fun(
            cipher, "DIFFERENTIALPATH_PROB", "sat", config_model
        )
        objective_cap = sum(len(round_vars) for round_vars in model_objective.gen_obj_fun_variables(objective))
        input_difference = request.get("input_difference")
        output_difference = request.get("output_difference")
        free_endpoints = bool(request.get("free_endpoints", False))
        if input_difference is None and output_difference is None and not free_endpoints:
            raise ValueError("request must contain input_difference or output_difference")
        fixed_difference = []
        if input_difference is not None:
            fixed_difference.extend(_fixed_state_literals("in", input_difference, word_size, word_count, cipher_name))
        if output_difference is not None:
            fixed_difference.extend(_fixed_state_literals("out", output_difference, word_size, word_count, cipher_name))
        fixed_difference.extend(_zero_state_literals(
            "in", request.get("input_zero_bits"), word_size, word_count, cipher_name
        ))
        fixed_difference.extend(_zero_state_literals(
            "out", request.get("output_zero_bits"), word_size, word_count, cipher_name
        ))
        if bool(request.get("input_not_zero", free_endpoints)):
            fixed_difference.extend(differential.gen_input_non_zero_constraints(cipher, "DIFFERENTIALPATH_PROB", config_model))
        try:
            max_weight = request.get("max_weight")
            if max_weight is not None and int(max_weight) < 0:
                solutions = []
            else:
                # Speck/Simon/Simeck and the other integer-objective models can
                # be optimized exactly with monotone bounded-SAT queries.  A
                # feasible solution at ``objective_cap`` always exists iff the
                # unconstrained model is feasible, because the cap is the sum
                # of all binary objective variables.  Binary search therefore
                # proves the same optimum as OCP's linear
                # "INCREASING FROM AT MOST 0" strategy with far fewer solver
                # invocations on high-weight E1 instances.
                config_solver["solution_number"] = 1
                objective_cap = int(objective_cap)
                low = 0
                if max_weight is None:
                    # A weight of 128 is a much more useful first probe for the
                    # wide-word models than the loose syntactic objective cap.
                    # If it is infeasible, expand geometrically; every failed
                    # probe raises the proven lower bound, so exactness is
                    # preserved.
                    high = min(128, objective_cap)
                    solutions = []
                    while True:
                        solutions = sat_search.modeling_solving_sat(
                            f"AT MOST {high}", fixed_difference + round_constraints,
                            objective, config_model, config_solver,
                        )
                        if solutions or high >= objective_cap:
                            break
                        low = high + 1
                        high = min(2 * high, objective_cap)
                else:
                    high = min(int(max_weight), objective_cap)
                    solutions = sat_search.modeling_solving_sat(
                        f"AT MOST {high}", fixed_difference + round_constraints,
                        objective, config_model, config_solver,
                    )
                best_solutions = solutions
                if solutions:
                    high = min(high, int(round(float(solutions[0]["obj_fun_value"]))))
                    while low < high:
                        middle = (low + high) // 2
                        trial = sat_search.modeling_solving_sat(
                            f"AT MOST {middle}", fixed_difference + round_constraints,
                            objective, config_model, config_solver,
                        )
                        if trial:
                            best_solutions = trial
                            high = min(middle, int(round(float(trial[0]["obj_fun_value"]))))
                        else:
                            low = middle + 1
                    solutions = best_solutions
        except SolverLimit as exc:
            return {"status": "resource_limit", "error": str(exc), "runtime_sec": time.monotonic() - start}

        if not solutions:
            return {"status": "no_path", "results": [], "model_unsat_proven": True,
                    "model_weight_bound": request.get("max_weight"),
                    "runtime_sec": time.monotonic() - start}
        solution = min(solutions, key=lambda row: float(row["obj_fun_value"]))
        trail = differential.extract_trail_structures(cipher, "DIFFERENTIALPATH_PROB", solution)
        function = trail["functions"]["PERMUTATION"]
        last_layer = cipher.functions["PERMUTATION"].nbr_layers
        round_output_layer = _round_output_layer(cipher_name, last_layer)
        states = [_nodes_to_paper(trail["inputs"]["IN_"], word_size, cipher_name)]
        for round_index in range(1, rounds + 1):
            states.append(_nodes_to_paper(function[round_index][round_output_layer], word_size, cipher_name))
        result = {
            "model_optimum_proven": True,
            "optimality_model": "OCP DIFFERENTIALPATH_PROB",
            "w_E1": float(solution["obj_fun_value"]),
            "delta_in_paper_bits": states[0],
            "delta_m_paper_bits": states[-1],
            "round_weights": solution.get("rounds_obj_fun_values"),
            "states_paper_bits": states,
        }
        result = correct_e1_candidate(result, request.get("cipher", "speck"), word_size)
        return {"status": "ok", "results": [result], "runtime_sec": time.monotonic() - start}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text())
    try:
        result = search(request)
    except Exception as exc:
        result = {"status": "solver_error", "error": f"{type(exc).__name__}: {exc}"}
    Path(args.result).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
