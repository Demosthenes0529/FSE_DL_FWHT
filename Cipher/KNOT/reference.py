"""Reference KNOT-256 permutation with absolute round constants."""

def int_to_hex(value):
    return f"{int(value) & ((1 << 256) - 1):064x}"


def rows_from_int(value):
    return tuple((int(value) >> (64 * (3 - row))) & ((1 << 64) - 1) for row in range(4))


def rows_to_int(rows):
    value = 0
    for row in rows:
        value = (value << 64) | (int(row) & ((1 << 64) - 1))
    return value


def get_cell(value, column):
    if not 0 <= int(column) < 64:
        raise ValueError("KNOT column must be in 0..63")
    # State row 0 is the least significant S-box bit, while each row's paper
    # coordinate 0 is its most significant bit.
    bit = 63 - int(column)
    return sum(((row >> bit) & 1) << index for index, row in enumerate(rows_from_int(value)))


def set_cell(value, column, local_value):
    if not 0 <= int(column) < 64 or not 0 <= int(local_value) < 16:
        raise ValueError("invalid KNOT cell")
    bit = 63 - int(column)
    return rows_to_int(tuple((row & ~(1 << bit)) | (((int(local_value) >> index) & 1) << bit)
                            for index, row in enumerate(rows_from_int(value))))


def linear_layer(value):
    return rows_to_int(tuple(rol64(row, shift) for row, shift in zip(rows_from_int(value), (0, 1, 8, 25))))


def inverse_linear_layer(value):
    return rows_to_int(tuple(ror64(row, shift) for row, shift in zip(rows_from_int(value), (0, 1, 8, 25))))

ROUND_CONSTANTS=(
0x01,0x02,0x04,0x08,0x10,0x21,0x03,0x06,0x0c,0x18,0x31,0x22,0x05,0x0a,0x14,0x29,
0x13,0x27,0x0f,0x1e,0x3d,0x3a,0x34,0x28,0x11,0x23,0x07,0x0e,0x1c,0x39,0x32,0x24,
0x09,0x12,0x25,0x0b,0x16,0x2d,0x1b,0x37,0x2e,0x1d,0x3b,0x36,0x2c,0x19,0x33,0x26,
0x0d,0x1a,0x35,0x2a,0x15,0x2b,0x17,0x2f,0x1f,0x3f,0x3e,0x3c,0x38,0x30,0x20)
MASK64=(1<<64)-1
def rol64(value,amount):amount%=64;return ((value<<amount)|(value>>(64-amount)))&MASK64 if amount else value&MASK64
def ror64(value,amount):return rol64(value,-amount)
def sbox_bits(a,b,c,d):
    t1=(~a)&MASK64;t2=b&t1;t3=c^t2;h=d^t3;t5=b|c;t6=d^t1;g=t5^t6
    t8=b^d;t9=t3&t6;e=t8^t9;t11=g&t8;f=t3^t11
    return e&MASK64,f&MASK64,g&MASK64,h&MASK64
SBOX=tuple(sum(((sbox_bits(x&1,(x>>1)&1,(x>>2)&1,(x>>3)&1)[row]&1)<<row) for row in range(4)) for x in range(16))
INV_SBOX=tuple(SBOX.index(x) for x in range(16))
def round_function(state,absolute_round):
    if not 0<=int(absolute_round)<len(ROUND_CONSTANTS):raise ValueError("KNOT round outside 0..62")
    x0,x1,x2,x3=rows_from_int(state);x0^=ROUND_CONSTANTS[int(absolute_round)]
    x0,x1,x2,x3=sbox_bits(x0,x1,x2,x3)
    return rows_to_int((x0,rol64(x1,1),rol64(x2,8),rol64(x3,25)))
def encrypt_segment(plaintext,key,round_start,rounds):
    # KNOT-256 is a permutation.  key is intentionally ignored so the common
    # adapter signature does not introduce a non-existent keyed round layer.
    start=int(round_start);rounds=int(rounds)
    if start<0 or rounds<0 or start+rounds>len(ROUND_CONSTANTS):raise ValueError("KNOT round interval outside 0..63")
    state=int(plaintext)
    for rnd in range(start,start+rounds):state=round_function(state,rnd)
    return state

def encrypt_batch(state, key, rounds, round_start=0):
    """NumPy row batches using the same S-box and rotation specification."""
    import numpy as np
    start, rounds = int(round_start), int(rounds)
    if len(state) != 4 or key or min(start, rounds) < 0 or start + rounds > len(ROUND_CONSTANTS):
        raise ValueError("invalid KNOT state, key, or round interval")
    rows = tuple(np.asarray(value).astype(np.uint64) for value in state)
    for index in range(start, start + rounds):
        rows = (rows[0] ^ np.uint64(ROUND_CONSTANTS[index]), *rows[1:])
        rows = sbox_bits(*rows)
        rows = tuple(((value << np.uint64(shift)) | (value >> np.uint64(64 - shift)))
                     if shift else value for value, shift in zip(rows, (0, 1, 8, 25)))
    return rows
