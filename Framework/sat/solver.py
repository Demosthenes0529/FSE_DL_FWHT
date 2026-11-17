"""Shared state encoding and external CryptoMiniSat execution."""

import math
import re
import subprocess
import time
from pathlib import Path

class SolverLimit(RuntimeError):
    pass


def _legacy_two_word_order(cipher, word_count):
    cipher = str(cipher or "").lower()
    return int(word_count) == 2 and (
        cipher.startswith(("speck", "simon", "simeck"))
    )


def _paper_to_ocp(bit, word_size=64, word_count=2, cipher="speck"):
    """Return (word, MSB-first bit index) for one paper-indexed bit."""
    bit = int(bit)
    word_size = int(word_size)
    word_count = int(word_count)
    if bit < 0 or bit >= word_size * word_count:
        raise ValueError(f"paper bit {bit} outside [0,{word_size * word_count})")
    if _legacy_two_word_order(cipher, word_count):
        if bit >= word_size * 2:
            raise ValueError(f"paper bit {bit} outside legacy two-word state")
        if bit < word_size:
            return 1, word_size - 1 - bit
        return 0, word_size - 1 - (bit - word_size)
    word, offset = divmod(bit, word_size)
    return word, word_size - 1 - offset


def _nodes_to_paper(nodes, word_size=64, cipher="speck"):
    word_count = len(nodes)
    bits = []
    legacy = _legacy_two_word_order(cipher, word_count)
    for word, node in enumerate(nodes):
        if len(node["bin_values"]) != word_size:
            raise ValueError(
                f"OCP node width {len(node['bin_values'])} does not match configured word_size={word_size}"
            )
        for index, value in enumerate(node["bin_values"]):
            if value != "1":
                continue
            local_bit = word_size - 1 - index
            if legacy:
                bits.append(local_bit if word == 1 else word_size + local_bit)
            else:
                bits.append(word * word_size + local_bit)
    return sorted(bits)


def _fixed_state_literals(prefix, paper_bits, word_size, word_count, cipher="speck"):
    selected = {_paper_to_ocp(bit, word_size, word_count, cipher) for bit in paper_bits or []}
    literals = []
    for word in range(int(word_count)):
        for index in range(int(word_size)):
            name = f"{prefix}{word}" if int(word_size) == 1 else f"{prefix}{word}_{index}"
            literals.append(("" if (word, index) in selected else "-") + name)
    return literals


def _zero_state_literals(prefix, paper_bits, word_size, word_count, cipher="speck"):
    """Force only the selected paper-indexed state bits to zero.

    Unlike ``_fixed_state_literals``, unspecified bits remain free.  This is
    used for nonce/key and rate/capacity interface restrictions.
    """
    selected = {_paper_to_ocp(bit, word_size, word_count, cipher) for bit in paper_bits or []}
    literals = []
    for word, index in sorted(selected):
        name = f"{prefix}{word}" if int(word_size) == 1 else f"{prefix}{word}_{index}"
        literals.append("-" + name)
    return literals


def _round_output_layer(cipher_name, last_layer):
    cipher_name = str(cipher_name or "").lower()
    if cipher_name.startswith(("simon", "simeck")):
        return int(last_layer) - 1
    return int(last_layer)


def _parse_cms_solution(stdout, variable_map):
    status = next((line[2:].strip() for line in stdout.splitlines() if line.startswith("s ")), "UNKNOWN")
    if status == "UNSATISFIABLE":
        return None
    if status != "SATISFIABLE":
        raise SolverLimit(f"CryptoMiniSat returned {status}")
    literals = []
    for line in stdout.splitlines():
        if line.startswith("v "):
            literals.extend(int(value) for value in line[2:].split() if value != "0")
    positive = {value for value in literals if value > 0}
    negative = {-value for value in literals if value < 0}
    return {
        name: 1 if number in positive else 0
        for name, number in variable_map.items()
        if number in positive or number in negative
    }


def _write_blocked_cnf(source, target, blockers):
    lines = Path(source).read_text().splitlines()
    header_index = next(i for i, line in enumerate(lines) if line.startswith("p cnf "))
    _, _, num_vars, num_clauses = lines[header_index].split()
    lines[header_index] = f"p cnf {num_vars} {int(num_clauses) + len(blockers)}"
    lines.extend(" ".join(str(value) for value in clause) + " 0" for clause in blockers)
    Path(target).write_text("\n".join(lines) + "\n")


def _make_external_solver(request, deadline, work_dir):
    cms_bin = request["cms_bin"]
    threads = max(1, int(request.get("threads", 1)))
    seed = int(request.get("seed", 42))
    max_conflicts = int(request.get("max_nodes", 0))

    def solve(filename, variable_map, config_solver=None):
        solution_number = max(1, int((config_solver or {}).get("solution_number", 1)))
        output_numbers = sorted(
            number for name, number in variable_map.items()
            if re.fullmatch(r"out\d+(?:_\d+)?", name)
        )
        blockers = []
        solutions = []
        for solution_index in range(solution_number):
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise SolverLimit("OCP trail search timeout reached")
            current_cnf = filename
            if blockers:
                current_cnf = str(Path(work_dir) / f"blocked_{solution_index}.cnf")
                _write_blocked_cnf(filename, current_cnf, blockers)
            command = [
                cms_bin, "--verb", "0", "--threads", str(threads),
                "--random", str(seed + solution_index),
            ]
            if remaining is not None:
                command.extend(["--maxtime", str(max(1, math.ceil(remaining)))])
            if max_conflicts > 0:
                command.extend(["--maxconfl", str(max_conflicts)])
            command.append(current_cnf)
            try:
                process = subprocess.run(
                    command, text=True, capture_output=True,
                    timeout=None if remaining is None else max(1.0, remaining + 2.0), check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise SolverLimit("CryptoMiniSat subprocess timeout") from exc
            solution = _parse_cms_solution(process.stdout, variable_map)
            if solution is None:
                break
            solutions.append(solution)
            # Enumerate distinct E2 output masks, not auxiliary SAT assignments.
            blockers.append([
                -number if solution.get(name, 0) else number
                for name, number in variable_map.items() if number in output_numbers
            ])
        return solutions

    return solve


