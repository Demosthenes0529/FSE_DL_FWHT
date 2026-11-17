#ifndef SIMON_TRANSITION_H
#define SIMON_TRANSITION_H

#include <math.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    uint64_t B;
    uint64_t basis[64];
    unsigned rank;
    double probability;
} simon_transition_space;

static inline uint64_t simon_word_mask(unsigned word_size) {
    return word_size == 64 ? UINT64_MAX : ((UINT64_C(1) << word_size) - 1);
}

static inline uint64_t simon_rotl_n(uint64_t x, unsigned n,
                                     unsigned word_size) {
    uint64_t mask = simon_word_mask(word_size);
    n %= word_size;
    x &= mask;
    return n ? ((x << n) | (x >> (word_size - n))) & mask : x;
}

static inline uint64_t simonlike_f_n(uint64_t x, unsigned word_size,
                                     unsigned and_a, unsigned and_b,
                                     unsigned xor_c) {
    return simon_rotl_n(x, xor_c, word_size) ^
           (simon_rotl_n(x, and_a, word_size) &
            simon_rotl_n(x, and_b, word_size));
}

static inline void simon_basis_add(uint64_t basis[64], uint64_t value) {
    for (int bit = 63; bit >= 0 && value; --bit) {
        if (!((value >> bit) & 1)) continue;
        if (basis[bit]) value ^= basis[bit];
        else { basis[bit] = value; return; }
    }
}

/* The affine derivative support has constant probability 2^-rank. */
static inline simon_transition_space simon_make_transition_params_n(
        uint64_t delta, unsigned word_size, unsigned and_a, unsigned and_b,
        unsigned xor_c) {
    simon_transition_space t;
    memset(&t, 0, sizeof(t));
    uint64_t mask = simon_word_mask(word_size);
    delta &= mask;
    t.B = simonlike_f_n(0, word_size, and_a, and_b, xor_c) ^
          simonlike_f_n(delta, word_size, and_a, and_b, xor_c);
    for (unsigned bit = 0; bit < word_size; ++bit) {
        uint64_t one = UINT64_C(1) << bit;
        uint64_t v = simonlike_f_n(one, word_size, and_a, and_b, xor_c) ^
                     simonlike_f_n(delta ^ one, word_size, and_a, and_b, xor_c) ^ t.B;
        simon_basis_add(t.basis, v);
    }
    for (int bit = 63; bit >= 0; --bit) {
        if (!t.basis[bit]) continue;
        ++t.rank;
        if ((t.B >> bit) & 1) t.B ^= t.basis[bit];
    }
    t.probability = ldexp(1.0, -(int)t.rank);
    return t;
}

static inline int simon_transition_contains(const simon_transition_space *t,
                                             uint64_t output) {
    uint64_t value = output ^ t->B;
    for (int bit = 63; bit >= 0 && value; --bit) {
        if (!((value >> bit) & 1)) continue;
        if (!t->basis[bit]) return 0;
        value ^= t->basis[bit];
    }
    return value == 0;
}

static inline uint64_t simon_sample_derivative_params_n(
        uint64_t delta, uint64_t x, unsigned word_size, unsigned and_a,
        unsigned and_b, unsigned xor_c) {
    uint64_t mask = simon_word_mask(word_size);
    return (simonlike_f_n(x, word_size, and_a, and_b, xor_c) ^
            simonlike_f_n(x ^ delta, word_size, and_a, and_b, xor_c)) & mask;
}

#endif
