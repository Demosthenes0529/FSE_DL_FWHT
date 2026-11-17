#!/usr/bin/env python3
"""Fixed-endpoint differential and linear CNF models and independent replay.

Differential weights are -log2(probability); linear weights are
-log2(abs(correlation)), so linear potential adds 2**(-2*weight).
OCP and CryptoMiniSat are loaded only when their corresponding backend runs.

The CNF follows ConstraintsDiffOneRound from the Lv--Zhang implementation.
The two cycle indices named ``a_b`` and ``2a_b`` use beta (not gamma), as
required by the theorem and verified against exhaustive SIMON evaluation.
"""

from __future__ import annotations

import itertools
import math
import os
import sys
from pathlib import Path


class CNF:
    def __init__(self):
        self.top = 0
        self.clauses = []

    def var(self):
        self.top += 1
        return self.top

    def add(self, *lits):
        self.clauses.append([int(x) for x in lits])

    def equal(self, a, b):
        self.add(a, -b)
        self.add(-a, b)

    def xor_zero(self, variables):
        variables = list(variables)
        for assignment in itertools.product((0, 1), repeat=len(variables)):
            if sum(assignment) & 1:
                self.add(*(-v if bit else v for v, bit in zip(variables, assignment)))


def at_most(cnf, literals, bound):
    """Sinz sequential counter for sum(literals) <= bound."""
    xs = list(literals)
    n, k = len(xs), int(bound)
    if k < 0:
        cnf.add()
        return
    if k >= n:
        return
    if k == 0:
        for x in xs:
            cnf.add(-x)
        return
    if n == 1:
        return
    s = [[cnf.var() for _ in range(k)] for _ in range(n - 1)]
    cnf.add(-xs[0], s[0][0])
    for j in range(1, k):
        cnf.add(-s[0][j])
    for i in range(1, n - 1):
        cnf.add(-xs[i], s[i][0])
        cnf.add(-s[i - 1][0], s[i][0])
        for j in range(1, k):
            cnf.add(-xs[i], -s[i - 1][j - 1], s[i][j])
            cnf.add(-s[i - 1][j], s[i][j])
        cnf.add(-xs[i], -s[i - 1][k - 1])
    cnf.add(-xs[-1], -s[-1][k - 1])


def _and_var(cnf, left, right):
    out = cnf.var()
    cnf.add(-left, -right, out)
    cnf.add(left, -out)
    cnf.add(right, -out)
    return out


def binary_exactly(cnf, literals, bound):
    """Exact cardinality via a binary ripple accumulator.

    Uses O(n log n) auxiliaries instead of two O(n*k) sequential counters.  It is important
    when dozens of independent shard solvers coexist in memory.
    """
    xs = list(literals)
    bound = int(bound)
    if bound < 0 or bound > len(xs):
        cnf.add()
        return
    width = max(1, (len(xs) + 1).bit_length())
    accumulator = [cnf.var() for _ in range(width)]
    for bit in accumulator:
        cnf.add(-bit)
    for value in xs:
        carry = value
        updated = []
        for old in accumulator:
            summed = cnf.var()
            cnf.xor_zero((old, carry, summed))
            next_carry = _and_var(cnf, old, carry)
            updated.append(summed)
            carry = next_carry
        # Since width represents 0..len(xs), overflow is impossible; keeping
        # this unit clause also catches an encoding mistake.
        cnf.add(-carry)
        accumulator = updated
    for index, bit in enumerate(accumulator):
        cnf.add(bit if (bound >> index) & 1 else -bit)


def rol(x, amount, word_size):
    amount %= word_size
    mask = (1 << word_size) - 1
    return ((x << amount) | (x >> (word_size - amount))) & mask


SIMON_LIKE_ROTATIONS = {
    # F(x) = (ROL_a(x) & ROL_b(x)) xor ROL_c(x).
    # Keep the distinguished alpha/beta orientation used by the theorem;
    # although AND is commutative, the CNF's i+2a-b index is not symmetric.
    "simon": (8, 1, 2),
    "simeck": (5, 0, 1),
}


def simonlike_f(x, word_size, family="simon"):
    try:
        alpha, beta, gamma = SIMON_LIKE_ROTATIONS[str(family).lower()]
    except KeyError as error:
        raise ValueError(f"unsupported SIMON-like family: {family}") from error
    return ((rol(x, alpha, word_size) & rol(x, beta, word_size)) ^
            rol(x, gamma, word_size))


def _insert_basis(basis, value):
    while value:
        pivot = value.bit_length() - 1
        if basis[pivot]:
            value ^= basis[pivot]
        else:
            basis[pivot] = value
            return True
    return False


def exact_round_weight(word_size, before, after, family="simon"):
    """Return exact -log2 transition probability, or None if impossible."""
    left, right = map(int, before)
    out_left, out_right = map(int, after)
    if out_right != left:
        return None
    affine = (simonlike_f(0, word_size, family) ^
              simonlike_f(left, word_size, family))
    columns = [
        simonlike_f(1 << i, word_size, family) ^
        simonlike_f(left ^ (1 << i), word_size, family) ^ affine
        for i in range(word_size)
    ]
    basis = [0] * word_size
    rank = sum(_insert_basis(basis, column) for column in columns)
    target = out_left ^ right ^ affine
    for pivot in range(word_size - 1, -1, -1):
        if (target >> pivot) & 1:
            if not basis[pivot]:
                return None
            target ^= basis[pivot]
    return rank


def replay_trail(word_size, states, family="simon"):
    weights = []
    for before, after in zip(states, states[1:]):
        weight = exact_round_weight(word_size, before, after, family)
        if weight is None:
            raise ValueError(f"invalid {family.upper()} transition: {before!r} -> {after!r}")
        weights.append(weight)
    return weights, sum(weights)


def pair_to_paper_bits(pair, word_size):
    left, right = map(int, pair)
    return ([i for i in range(word_size) if (right >> i) & 1] +
            [word_size + i for i in range(word_size) if (left >> i) & 1])


def paper_bits_to_pair(bits, word_size):
    left = right = 0
    for bit in bits:
        bit = int(bit)
        if bit < word_size:
            right |= 1 << bit
        else:
            left |= 1 << (bit - word_size)
    return left, right


def parse_pair(text, word_size):
    fields = str(text).split(",")
    if len(fields) != 2:
        raise ValueError("difference must be LEFT,RIGHT")
    pair = tuple(int(x, 0) for x in fields)
    if any(x < 0 or x >= (1 << word_size) for x in pair):
        raise ValueError(f"difference does not fit {word_size}-bit words")
    return pair


def parse_window(path, rounds, word_size):
    rows = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    header = [int(x) for x in rows[0].split()]
    if header[0] < rounds:
        raise ValueError(f"window depth {header[0]} is shorter than {rounds}")
    # Files contain R_0 at row -1 followed by L_0,...,L_R.
    raw = [set(map(int, row.split())) for row in rows[1:]]
    # A depth-r snapshot of a deeper output-anchored reverse window uses the
    # suffix nearest the fixed output.  E.g. depth 18 from a depth-19 file
    # drops the oldest input-side support row.
    offset = header[0] - rounds
    supports = raw[offset:offset + rounds + 2]
    if len(supports) != rounds + 2:
        raise ValueError("window file has too few support rows")
    if any(bit < 0 or bit >= word_size for row in supports for bit in row):
        raise ValueError("window bit is outside the word")
    return supports, header


class SimonExactModel:
    """Exact fixed-endpoint SIMON/SIMECK differential model."""

    def __init__(self, word_size, rounds, delta_in, delta_out, window=None,
                 exact_weight=None, max_weight=None, family="simon"):
        self.word_size = int(word_size)
        self.rounds = int(rounds)
        self.delta_in = tuple(delta_in)
        self.delta_out = tuple(delta_out)
        self.family = str(family).lower()
        if self.family not in SIMON_LIKE_ROTATIONS:
            raise ValueError(f"unsupported SIMON-like family: {family}")
        self.cnf = CNF()
        n = self.word_size
        # Arrays retain the author's MSB-first SAT convention.
        self.states = [
            ([self.cnf.var() for _ in range(n)], [self.cnf.var() for _ in range(n)])
            for _ in range(self.rounds + 1)
        ]
        self.weight_vars = []
        for rnd in range(self.rounds):
            self._add_round(rnd)
        self._fix_word(self.states[0][0], self.delta_in[0])
        self._fix_word(self.states[0][1], self.delta_in[1])
        self._fix_word(self.states[-1][0], self.delta_out[0])
        self._fix_word(self.states[-1][1], self.delta_out[1])
        if window is not None:
            self._restrict_window(window)
        flat = [x for row in self.weight_vars for x in row]
        if exact_weight is not None:
            binary_exactly(self.cnf, flat, int(exact_weight))
        elif max_weight is not None:
            at_most(self.cnf, flat, int(max_weight))

    def _add_round(self, rnd):
        n = self.word_size
        # The Lv--Zhang SAT arrays are MSB-first, so a numeric ROL_r is
        # addressed as +r.  AND rotations are interchangeable.
        alpha, beta, gamma = SIMON_LIKE_ROTATIONS[self.family]
        left, right = self.states[rnd]
        out_left, out_right = self.states[rnd + 1]
        var = [self.cnf.var() for _ in range(n)]
        double = [self.cnf.var() for _ in range(n)]
        z = [self.cnf.var() for _ in range(n)]
        b = [self.cnf.var() for _ in range(n)]
        weight = [self.cnf.var() for _ in range(n)]
        self.weight_vars.append(weight)
        # The usual var/double weight formula loses one rank on a fully
        # active rotation cycle.  For the standard SIMON/SIMECK word sizes
        # there is one cycle, so this is the all-one input difference case.
        # Handling cycles separately also covers small reference word sizes.
        cycles, cycle_full = [], {}
        unseen = set(range(n))
        while unseen:
            first = min(unseen)
            cycle, index = [], first
            while index not in cycle:
                cycle.append(index)
                unseen.remove(index)
                index = (index + alpha - beta) % n
            full = self.cnf.var()
            inputs = [left[(index + beta) % n] for index in cycle]
            for variable in inputs:
                self.cnf.add(-full, variable)
            self.cnf.add(full, *(-variable for variable in inputs))
            cycles.append((cycle, full))
            cycle_full.update((index, full) for index in cycle)
        for i in range(n):
            ia = (i + alpha) % n
            ib = (i + beta) % n
            ic = (i + gamma) % n
            # The published variable names are i_2a_b/i_a_b.  beta is
            # essential here; using gamma fails exhaustive transition tests.
            i2ab = (i + 2 * alpha - beta) % n
            iab = (i + alpha - beta) % n
            self.cnf.add(-left[ia], var[i])
            self.cnf.add(-left[ib], var[i])
            self.cnf.add(left[ia], left[ib], -var[i])
            self.cnf.add(left[ib], -double[i])
            self.cnf.add(-left[ib], left[i2ab], -double[i])
            self.cnf.add(-left[ia], -left[ib], -double[i])
            self.cnf.add(left[ia], -left[ib], -left[i2ab], double[i])
            self.cnf.add(var[i], -z[i])
            self.cnf.add(-double[i], z[i], -z[iab])
            self.cnf.add(-double[i], -z[i], z[iab])
            self.cnf.xor_zero((left[ic], b[i], z[i]))
            first_clause = len(self.cnf.clauses)
            self.cnf.xor_zero((var[i], double[i], weight[i]))
            for clause in self.cnf.clauses[first_clause:]:
                clause.append(cycle_full[i])
            self.cnf.xor_zero((b[i], right[i], out_left[i]))
            self.cnf.equal(left[i], out_right[i])
        for cycle, full in cycles:
            # D_delta(ROL_a(x)&ROL_b(x)) has parity |cycle| mod 2
            # on an active cycle and exactly 2^(|cycle|-1) outputs.
            parity = z[cycle[0]]
            for index in cycle[1:]:
                updated = self.cnf.var()
                self.cnf.xor_zero((parity, z[index], updated))
                parity = updated
            self.cnf.add(-full, parity if len(cycle) & 1 else -parity)
            for position, index in enumerate(cycle):
                self.cnf.add(-full, weight[index] if position else -weight[index])

    def _fix_word(self, variables, value):
        n = self.word_size
        for i, var in enumerate(variables):
            bit = (int(value) >> (n - 1 - i)) & 1
            self.cnf.add(var if bit else -var)

    def _restrict_word(self, variables, allowed):
        n = self.word_size
        for i, var in enumerate(variables):
            if n - 1 - i not in allowed:
                self.cnf.add(-var)

    def _restrict_window(self, supports):
        for rnd, (left, right) in enumerate(self.states):
            self._restrict_word(left, supports[rnd + 1])
            self._restrict_word(right, supports[0] if rnd == 0 else supports[rnd])

    def state_variables(self):
        return [v for state in self.states for word in state for v in word]

    def state_bit_variable(self, round_index, feature_index):
        """Return a real midpoint state variable; features are L0..L63,R0..R63."""
        n = self.word_size
        feature_index = int(feature_index)
        if not 0 <= round_index <= self.rounds or not 0 <= feature_index < 2 * n:
            raise IndexError("state bit outside model")
        side = 0 if feature_index < n else 1
        lsb_index = feature_index if side == 0 else feature_index - n
        return self.states[round_index][side][n - 1 - lsb_index]

    def decode_states(self, solution):
        result = []
        for left_vars, right_vars in self.states:
            words = []
            for variables in (left_vars, right_vars):
                value = 0
                for i, var in enumerate(variables):
                    if solution[var]:
                        value |= 1 << (self.word_size - 1 - i)
                words.append(value)
            result.append(tuple(words))
        return result

    def replay_weight(self, solution):
        return replay_trail(self.word_size, self.decode_states(solution), self.family)[1]

    def add_to_solver(self, solver):
        for clause in self.cnf.clauses:
            solver.add_clause(clause)


def logaddexp2(left, right):
    if left == -math.inf:
        return right
    if right == -math.inf:
        return left
    high, low = max(left, right), min(left, right)
    return high + math.log2(1.0 + 2.0 ** (low - high))


def _paper_to_ocp(bit: int, word_size: int) -> tuple[int, int]:
    """Legacy two-word convention used by the DL framework (MSB-first OCP)."""
    if bit < word_size:
        return 1, word_size - 1 - bit
    return 0, word_size - 1 - (bit - word_size)


def _fixed(prefix: str, bits, word_size: int):
    selected = {_paper_to_ocp(int(bit), word_size) for bit in bits}
    result = []
    for word in range(2):
        for index in range(word_size):
            name = f"{prefix}{word}_{index}"
            result.append(("" if (word, index) in selected else "-") + name)
    return result


class OCPLinearModel:
    """Numeric CNF plus projected round-boundary mask variables."""

    def __init__(self, word_size, rounds, gamma_in, gamma_out,
                 exact_weight=None, family="simon"):
        self.word_size = int(word_size)
        self.rounds = int(rounds)
        self.gamma_in = tuple(map(int, gamma_in))
        self.gamma_out = tuple(map(int, gamma_out))
        self.family = str(family).lower()
        if self.family in ("simon", "simeck"):
            raise ValueError(
                "SIMON/SIMECK bitwise OCP linear weights are not exact whole-round "
                "correlations; use Experiments.validation fixed-key E2 measurement")
        if self.family != "speck":
            raise ValueError("exact OCP linear AllSAT is supported only for SPECK")
        # Keep pure differential helpers importable without OCP dependencies.
        ocp = Path(os.environ.get("OCP_ROOT", Path(__file__).resolve().parents[2] / "OCP"))
        if str(ocp) not in sys.path:
            sys.path.insert(0, str(ocp))
        from attacks import linear_cryptanalysis as linear
        import tools.model_constraints as model_constraints
        import tools.model_objective as model_objective
        import tools.sat_search as sat_search
        from Framework.sat.ocp_factory import make_ocp_permutation

        block_size = 2 * self.word_size
        cipher_name = f"{self.family}{block_size}"
        cipher = make_ocp_permutation(cipher_name, self.rounds, block_size)
        cipher.add_copy_operators()
        config_model, _ = linear.parse_and_set_configs(
            cipher, "LINEARPATH_CORR", "OPTIMAL", {"model_type": "sat"},
            {"solver": "CryptoMiniSat", "solution_number": 1})
        round_constraints, objective = \
            model_constraints.gen_round_model_constraint_obj_fun(
                cipher, "LINEARPATH_CORR", "sat", config_model)
        fixed = (_fixed("in", pair_to_paper_bits(self.gamma_in, self.word_size),
                        self.word_size) +
                 _fixed("out", pair_to_paper_bits(self.gamma_out, self.word_size),
                        self.word_size))
        top, variable_map, clauses = sat_search.create_numerical_cnf(
            fixed + round_constraints)
        self.cnf = CNF()
        self.cnf.top = top
        self.cnf.clauses = [list(map(int, clause.split())) for clause in clauses]
        objective_names = [name for row in model_objective.gen_obj_fun_variables(objective)
                           for name in row]
        self.objective_vars = [variable_map[name] for name in objective_names]
        if exact_weight is not None:
            binary_exactly(self.cnf, self.objective_vars, int(exact_weight))

        function = cipher.functions["PERMUTATION"]
        # Round-boundary state is the layer after SIMON's final permutation.
        # OCP's extraction helper uses the preceding layer for presentation,
        # but projected trail identity and endpoint/window replay require the
        # actual external boundary variables.
        output_layer = function.nbr_layers
        nodes = [list(cipher.inputs["IN_"])]
        nodes.extend([function.vars[rnd][output_layer] for rnd in range(1, self.rounds + 1)])
        self.states = []
        for boundary in nodes:
            paper_vars = []
            for paper_bit in range(block_size):
                word, index = _paper_to_ocp(paper_bit, self.word_size)
                name = f"{boundary[word].ID}_{index}"
                paper_vars.append(variable_map[name])
            self.states.append(paper_vars)

    def add_to_solver(self, solver):
        for clause in self.cnf.clauses:
            solver.add_clause(clause)

    def state_variables(self):
        return [value for state in self.states for value in state]

    def state_bit_variable(self, round_index, feature_index):
        return self.states[int(round_index)][int(feature_index)]

    def decode_states(self, solution):
        states = []
        for variables in self.states:
            bits = [index for index, variable in enumerate(variables) if solution[variable]]
            states.append(paper_bits_to_pair(bits, self.word_size))
        return states

    def replay_weight(self, solution):
        from Framework.sat.trail_replay import round_weights

        weight = sum(bool(solution[var]) for var in self.objective_vars)
        independent = sum(round_weights(self.family, self.decode_states(solution),
                                        self.word_size, "linear"))
        if not math.isclose(independent, weight, abs_tol=1e-9):
            raise AssertionError(f"{self.family} linear SAT/replay mismatch: {weight} != {independent}")
        return weight



def new_solver(model):
    """Create one solver per shard, with only round-boundary projection exposed."""
    try:
        import pycryptosat
    except (ImportError, OSError) as error:
        raise RuntimeError("Path aggregation requires a working pycryptosat installation compatible with CryptoMiniSat") from error
    solver = pycryptosat.Solver(verbose=0, threads=1)
    model.add_to_solver(solver)
    return solver
