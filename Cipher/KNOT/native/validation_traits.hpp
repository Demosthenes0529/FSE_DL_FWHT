#include "knot256_native.hpp"
#pragma once
// Cipher operations extracted from the original fixed-key validators.
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
namespace validation_cipher {
using namespace std;
using State = array<uint64_t, 4>;

struct Cipher {
    int word_size = 64, state_words = 4;
    uint64_t mask = ~0ULL;
    Cipher(int block, int key) {
        if (block != 256 || key != 0) throw runtime_error("invalid KNOT-256 permutation identity");
    }
    vector<int> key_widths() const { return {}; }
    void prepare(const vector<uint64_t>&, int count) {
        if (count > 63) throw runtime_error("KNOT-256 round interval exceeds 63");
    }
    void encrypt(State& state, int start, int rounds) const {
        state = knot256_native::encrypt_segment(state, start, rounds);
    }
};
} // namespace validation_cipher
