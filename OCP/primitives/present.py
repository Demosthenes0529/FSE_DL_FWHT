"""PRESENT internal permutation, extracted from OCP's primitives/present.py.

The OCP upstream license is preserved in OCP/LICENSE. Only the permutation
model required by the E1/E2 trail searches is retained; key XORs do not change
the differential probability or absolute linear correlation of a trail.
"""

from primitives.primitives import Permutation
from operators.Sbox import PRESENT_Sbox
import variables.variables as var


class PRESENT_permutation(Permutation):
    def __init__(self, name, s_input, s_output, nbr_rounds=None, represent_mode=0):
        if represent_mode != 0:
            raise ValueError("The packaged PRESENT model uses one variable per bit")
        if nbr_rounds is None:
            nbr_rounds = 31
        if not 1 <= int(nbr_rounds) <= 31:
            raise ValueError("PRESENT trail models require 1..31 rounds")
        super().__init__(name, s_input, s_output, int(nbr_rounds), [2, 64, 0, 1])
        state = self.functions["PERMUTATION"]
        # Each output bit is copied from this input index, in MSB-first order.
        permutation = [4 * (index % 16) + index // 16 for index in range(64)]
        for round_index in range(1, int(nbr_rounds) + 1):
            state.SboxLayer("SB", round_index, 0, PRESENT_Sbox,
                            index=[list(range(index, index + 4)) for index in range(0, 64, 4)])
            state.PermutationLayer("P", round_index, 1, permutation)


def PRESENT_PERMUTATION(r=None, represent_mode=0, copy_operator=False):
    inputs = [var.Variable(1, ID=f"in{index}") for index in range(64)]
    outputs = [var.Variable(1, ID=f"out{index}") for index in range(64)]
    permutation = PRESENT_permutation("PRESENT_PERM", inputs, outputs, r, represent_mode)
    permutation.post_initialization(copy_operator=copy_operator)
    return permutation
