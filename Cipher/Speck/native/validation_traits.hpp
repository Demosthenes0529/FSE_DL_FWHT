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
uint64_t word_mask(int word_size) {
    return word_size == 64 ? numeric_limits<uint64_t>::max() : ((1ULL << word_size) - 1ULL);
}

static inline uint64_t rol(uint64_t x, int r, int word_size, uint64_t mask) {
    x &= mask;
    if (word_size == 64) return (x << r) | (x >> (64 - r));
    return ((x << r) & mask) | (x >> (word_size - r));
}

static inline uint64_t ror(uint64_t x, int r, int word_size, uint64_t mask) {
    x &= mask;
    if (word_size == 64) return (x >> r) | (x << (64 - r));
    return (x >> r) | ((x << (word_size - r)) & mask);
}

int alpha_for_word_size(int w) { return w == 16 ? 7 : 8; }
int beta_for_word_size(int w) { return w == 16 ? 2 : 3; }
static inline void enc_round(uint64_t& x, uint64_t& y, uint64_t k, int word_size, uint64_t mask) {
    x = ((ror(x, alpha_for_word_size(word_size), word_size, mask) + y) & mask) ^ k;
    y = rol(y, beta_for_word_size(word_size), word_size, mask) ^ x;
    x &= mask;
    y &= mask;
}

static inline void schedule(const vector<uint64_t>& key_words, int rounds, int word_size, uint64_t mask, uint64_t* rk) {
    // Same convention as the Python modules: first round key is the last key word.
    if (key_words.size() < 2) throw runtime_error("Speck key schedule requires at least two words");
    uint64_t k = key_words.back() & mask;
    vector<uint64_t> l;
    for (int i = static_cast<int>(key_words.size()) - 2; i >= 0; --i) {
        l.push_back(key_words[static_cast<size_t>(i)] & mask);
    }
    rk[0] = k;
    for (int i = 0; i < rounds - 1; ++i) {
        enc_round(l[static_cast<size_t>(i) % l.size()], k, static_cast<uint64_t>(i), word_size, mask);
        rk[i + 1] = k & mask;
    }
}

static inline void encrypt(uint64_t& x, uint64_t& y, int round_start, int rounds, int word_size, uint64_t mask, const uint64_t* rk) {
    for (int i = round_start; i < round_start + rounds; ++i) enc_round(x, y, rk[i], word_size, mask);
}


struct Cipher {
    int block, key_size, word_size, state_words = 2;
    uint64_t mask;
    vector<uint64_t> round_keys;
    Cipher(int b, int k): block(b), key_size(k), word_size(b / 2), mask(word_mask(word_size)) { if ((block != 32 && block != 48 && block != 64 && block != 96 && block != 128) || key_size % word_size || key_size / word_size < 2 || key_size / word_size > 4) throw runtime_error("invalid SPECK variant"); }
    vector<int> key_widths() const { return vector<int>(key_size / word_size, word_size); }
    void prepare(const vector<uint64_t>& key, int count) {
        round_keys.resize(max(1, count));
        if (count) { schedule(key, count, word_size, mask, round_keys.data()); }
    }
    void encrypt(State& state, int start, int rounds) const { validation_cipher::encrypt(state[0], state[1], start, rounds, word_size, mask, round_keys.data()); }
};
} // namespace validation_cipher
