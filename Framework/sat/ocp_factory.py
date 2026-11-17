"""OCP model construction selected through the shared Cipher registry."""


def make_ocp_permutation(cipher, rounds, block_size, start_round=0):
    from Cipher.registry import get_cipher_spec

    spec = get_cipher_spec(cipher)
    if spec.block_size != int(block_size):
        raise ValueError("Cipher name and block size disagree")
    if spec.family == "speck":
        from primitives.speck import SPECK_PERMUTATION
        return SPECK_PERMUTATION(r=int(rounds), version=spec.block_size)
    if spec.family == "simon":
        from primitives.simon import SIMON_PERMUTATION
        return SIMON_PERMUTATION(r=int(rounds), version=spec.block_size)
    if spec.family == "simeck":
        return SIMECK_PERMUTATION(r=int(rounds), version=spec.block_size)
    if spec.family == "cham":
        return CHAM_PERMUTATION(r=int(rounds), version=spec.block_size, start_round=int(start_round))
    if spec.family == "present":
        from primitives.present import PRESENT_PERMUTATION
        if int(start_round) < 0 or int(start_round) + int(rounds) > 31:
            raise ValueError("PRESENT round interval outside 0..31")
        return PRESENT_PERMUTATION(r=int(rounds))
    if spec.family == "knot":
        return KNOT256_PERMUTATION(r=int(rounds), version=spec.block_size, start_round=int(start_round))
    raise ValueError(f"OCP trail backend is unavailable for {spec.name}")


def KNOT256_PERMUTATION(r=None,version=256,copy_operator=False,start_round=0):
    """Unkeyed KNOT-256 rounds in row0||row1||row2||row3 order."""
    import operators.operators as op
    import variables.variables as var
    from operators.Sbox import KNOT_Sbox
    from primitives.primitives import Permutation,generateID
    if int(version)!=256:raise ValueError("KNOT primitive supports the 256-bit state only")
    rounds=int(r or 0);start=int(start_round)
    if start<0 or rounds<0 or start+rounds>63:raise ValueError("KNOT round interval outside 0..63")
    inputs=[var.Variable(1,ID=f"in{i}") for i in range(256)];outputs=[var.Variable(1,ID=f"out{i}") for i in range(256)]
    class KNOTPermutation(Permutation):
        def __init__(self):
            super().__init__("KNOT256_PERM",inputs,outputs,rounds,[2,256,256,1]);f=self.functions["PERMUTATION"];rot=(0,1,8,25)
            for rr in range(1,rounds+1):
                vin=f.vars[rr][0];vmid=f.vars[rr][1];vout=f.vars[rr][2]
                for bit in range(256):f.constraints[rr][0].append(op.Equal([vin[bit]],[vmid[bit]],ID=generateID("KEEP",rr,1,bit)))
                for column in range(64):
                    # KNOT's x0/row0 is the numeric S-box LSB, whereas OCP
                    # Sbox variables are passed MSB first.
                    source=[vin[row*64+column] for row in (3,2,1,0)];temporary=[vmid[256+4*column+row] for row in (3,2,1,0)]
                    f.constraints[rr][0].append(KNOT_Sbox(source,temporary,ID=generateID("SB",rr,1,column)))
                for row in range(4):
                    for column in range(64):
                        source_column=(column+rot[row])%64
                        f.constraints[rr][1].append(op.Equal([vmid[256+4*source_column+row]],[vout[row*64+column]],ID=generateID("ROT",rr,2,row*64+column)))
                for bit in range(256):f.constraints[rr][1].append(op.Equal([vmid[256+bit]],[vout[256+bit]],ID=generateID("TMP",rr,2,bit)))
    result=KNOTPermutation();result.post_initialization(copy_operator=copy_operator);return result

def CHAM_PERMUTATION(r=None, version=64, copy_operator=False, start_round=0):
    import variables.variables as var
    from operators.modular_operators import ModAdd
    from primitives.primitives import Permutation

    class ChamPermutation(Permutation):
        def __init__(self, name, version, s_input, s_output, nbr_rounds=None, start_round=0):
            p_bitsize = int(version)
            if p_bitsize not in (64, 128):
                raise ValueError("CHAM OCP primitive supports CHAM-64 and CHAM-128 states")
            if nbr_rounds is None:
                nbr_rounds = 88 if p_bitsize == 64 else 112
            word_bitsize = p_bitsize // 4
            nbr_layers, nbr_words, nbr_temp_words = 4, 4, 2
            super().__init__(name, s_input, s_output, nbr_rounds, [nbr_layers, nbr_words, nbr_temp_words, word_bitsize])
            S = self.functions["PERMUTATION"]
            for i in range(1, nbr_rounds + 1):
                absolute_round = int(start_round) + i - 1
                alpha, beta = (1, 8) if (absolute_round % 2 == 0) else (8, 1)
                S.RotationLayer("ROT_ALPHA", i, 0, ["l", alpha, 1, 4])
                S.SingleOperatorLayer("ADD", i, 1, ModAdd, [[0, 4]], [5])
                S.RotationLayer("ROT_BETA", i, 2, ["l", beta, 5, 5])
                S.PermutationLayer("PERM", i, 3, [1, 2, 3, 5, 4, 5])

    p_bitsize = int(version)
    word_size = p_bitsize // 4
    my_input = [var.Variable(word_size, ID="in" + str(i)) for i in range(4)]
    my_output = [var.Variable(word_size, ID="out" + str(i)) for i in range(4)]
    permutation = ChamPermutation(
        f"CHAM{p_bitsize}_PERM", p_bitsize, my_input, my_output,
        nbr_rounds=r, start_round=start_round,
    )
    permutation.post_initialization(copy_operator=copy_operator)
    return permutation

def SIMECK_PERMUTATION(r=None, version=64, copy_operator=False):
    import variables.variables as var
    from operators.boolean_operators import XOR, AND
    from primitives.primitives import Permutation

    class SimeckPermutation(Permutation):
        def __init__(self, name, version, s_input, s_output, nbr_rounds=None):
            p_bitsize = int(version)
            if nbr_rounds is None:
                nbr_rounds = 32 if p_bitsize == 32 else 36 if p_bitsize == 48 else 44 if p_bitsize == 64 else None
            nbr_layers, nbr_words, nbr_temp_words, word_bitsize = (5, 2, 2, p_bitsize >> 1)
            super().__init__(name, s_input, s_output, nbr_rounds, [nbr_layers, nbr_words, nbr_temp_words, word_bitsize])

            S = self.functions["PERMUTATION"]
            for i in range(1, nbr_rounds + 1):
                S.RotationLayer("ROT", i, 0, [["l", 5, 0, 2], ["l", 1, 0, 3]])
                S.SingleOperatorLayer("AND", i, 1, AND, [[0, 2]], [2])
                S.SingleOperatorLayer("XOR1", i, 2, XOR, [[1, 2]], [1])
                S.SingleOperatorLayer("XOR2", i, 3, XOR, [[1, 3]], [1])
                S.PermutationLayer("PERM", i, 4, [1, 0])

    p_bitsize, word_size = int(version), int(version) // 2
    my_input = [var.Variable(word_size, ID="in" + str(i)) for i in range(2)]
    my_output = [var.Variable(word_size, ID="out" + str(i)) for i in range(2)]
    permutation = SimeckPermutation(
        f"SIMECK{p_bitsize}_PERM",
        p_bitsize,
        my_input,
        my_output,
        nbr_rounds=r,
    )
    permutation.post_initialization(copy_operator=copy_operator)
    return permutation
