"""Independent word-level differential and linear trail replay.

Addition transitions use carry dynamic programming. SIMON-like differential
transitions use affine rank; linear transitions use exact quadratic Walsh sums.
No SAT auxiliary variables are consulted.
"""

import math


def rol(value, amount, width):
    amount %= width
    mask = (1 << width) - 1
    return ((value << amount) | (value >> (width - amount))) & mask


def ror(value, amount, width):
    return rol(value, -amount, width)


def _weight(count, denominator_bits):
    if not count:
        raise ValueError("infeasible round transition")
    return denominator_bits - math.log2(abs(count))


def addition_differential_weight(a, b, c, width):
    counts = {(0, 0): 1}
    for bit in range(width):
        following = {}
        aa, bb, cc = (a >> bit) & 1, (b >> bit) & 1, (c >> bit) & 1
        for (carry, paired_carry), count in counts.items():
            for x in (0, 1):
                for y in (0, 1):
                    total, paired_total = x + y + carry, (x ^ aa) + (y ^ bb) + paired_carry
                    if ((total ^ paired_total) & 1) == cc:
                        key = (total >> 1, paired_total >> 1)
                        following[key] = following.get(key, 0) + count
        counts = following
    return _weight(sum(counts.values()), 2 * width)


def addition_correlation_numerator(a, b, c, width):
    counts = [1, 0]
    for bit in range(width):
        following = [0, 0]
        aa, bb, cc = (a >> bit) & 1, (b >> bit) & 1, (c >> bit) & 1
        for carry, count in enumerate(counts):
            for x in (0, 1):
                for y in (0, 1):
                    total = x + y + carry
                    sign = -1 if ((aa & x) ^ (bb & y) ^ (cc & (total & 1))) else 1
                    following[total >> 1] += sign * count
        counts = following
    return sum(counts)


def quadratic_walsh(polynomial, width):
    """Sum (-1)^f over all inputs, with GF(2) monomials encoded as bitsets."""
    terms = set(polynomial)
    eliminated = 0
    while True:
        edge = next((term for term in terms if bin(term).count('1') == 2), None)
        if edge is None:
            if any(term for term in terms):
                return 0
            return (-1 if 0 in terms else 1) * (1 << (width - eliminated))
        i = edge & -edge
        j = edge ^ i
        a = [term ^ i for term in terms if term & i and term != edge]
        b = [term ^ j for term in terms if term & j and term != edge]
        terms = {term for term in terms if not term & edge}
        for left in a:
            for right in b:
                term = left | right
                if term in terms:
                    terms.remove(term)
                else:
                    terms.add(term)
        eliminated += 1


def simonlike_linear_weight(before, after, width, family):
    left, right = before
    out_left, out_right = after
    if right != out_left:
        raise ValueError("SIMON-like linear copy constraint failed")
    a, b, c = (1, 8, 2) if family == "simon" else (0, 5, 1)
    terms = set()
    linear = left ^ out_right ^ ror(out_left, c, width)
    for bit in range(width):
        if (linear >> bit) & 1:
            terms.symmetric_difference_update({1 << bit})
        if (out_left >> bit) & 1:
            terms.symmetric_difference_update({(1 << ((bit - a) % width)) | (1 << ((bit - b) % width))})
    return _weight(quadratic_walsh(terms, width), width)


def round_weights(family, states, width, kind, start_round=0):
    """Replay states in public word order and return each absolute trail weight."""
    result = []
    for offset, (before, after) in enumerate(zip(states, states[1:])):
        if family in ("simon", "simeck"):
            if kind == "differential":
                from Framework.aggregation.models import exact_round_weight
                weight = exact_round_weight(width, before, after, family)
                if weight is None:
                    raise ValueError("SIMON-like differential round is infeasible")
            else:
                weight = simonlike_linear_weight(before, after, width, family)
        elif family == "speck":
            alpha, beta = (7, 2) if width == 16 else (8, 3)
            left, right = before
            out_left, out_right = after
            if kind == "differential":
                if out_right != rol(right, beta, width) ^ out_left:
                    raise ValueError("SPECK differential rotation/XOR constraint failed")
                weight = addition_differential_weight(ror(left, alpha, width), right, out_left, width)
            else:
                weight = _weight(addition_correlation_numerator(ror(left, alpha, width), right ^ ror(out_right, beta, width), out_left ^ out_right, width), 2 * width)
        elif family == "cham":
            alpha, beta = (8, 1) if (start_round + offset) & 1 else (1, 8)
            if kind == "differential":
                if tuple(after[:3]) != tuple(before[1:]):
                    raise ValueError("CHAM differential word shift failed")
                weight = addition_differential_weight(before[0], rol(before[1], alpha, width), ror(after[3], beta, width), width)
            else:
                if before[2] != after[1] or before[3] != after[2]:
                    raise ValueError("CHAM linear word shift failed")
                weight = _weight(addition_correlation_numerator(before[0], rol(before[1] ^ after[0], alpha, width), ror(after[3], beta, width), width), 2 * width)
        else:
            raise ValueError(f"No word-level replay backend for {family}")
        result.append(float(weight))
    return result
