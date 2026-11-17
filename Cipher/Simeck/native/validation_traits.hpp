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

static inline void simeck_round(uint64_t& x, uint64_t& y, uint64_t k, int word_size, uint64_t mask) {
    uint64_t z = (y ^ (x & rol(x, 5, word_size, mask)) ^
                  rol(x, 1, word_size, mask) ^ k) & mask;
    y = x & mask;
    x = z;
}

int simeck_rounds_for_variant(int block_size, int key_size) {
    if (block_size == 32 && key_size == 64) return 32;
    if (block_size == 48 && key_size == 96) return 36;
    if (block_size == 64 && key_size == 128) return 44;
    throw runtime_error("unsupported SIMECK block/key size");
}

uint64_t simeck_z_bit(int block_size, int index) {
    if (block_size == 32 || block_size == 48) return (0x9a42bb1fULL >> (index % 31)) & 1ULL;
    if (block_size == 64) {
        if (index >= 44) throw runtime_error("SIMECK64 Z1 prefix only covers the standard 44 rounds");
        return (0x938bca3083fULL >> index) & 1ULL;
    }
    throw runtime_error("unsupported SIMECK block size");
}

static inline void simeck_schedule(
    const vector<uint64_t>& key_words,
    int rounds,
    int block_size,
    int key_size,
    int word_size,
    uint64_t mask,
    uint64_t* rk
) {
    if (static_cast<int>(key_words.size()) != 4) throw runtime_error("SIMECK key schedule requires four words");
    if (key_size != 4 * word_size) throw runtime_error("wrong SIMECK key size for word size");
    if (rounds > simeck_rounds_for_variant(block_size, key_size)) {
        throw runtime_error("SIMECK schedule requested more than the standard round count");
    }
    uint64_t t0 = key_words[3] & mask;  // k0
    uint64_t t1 = key_words[2] & mask;  // t0 in the paper
    uint64_t t2 = key_words[1] & mask;  // t1 in the paper
    uint64_t t3 = key_words[0] & mask;  // t2 in the paper
    const uint64_t base_constant = mask ^ 3ULL;
    for (int i = 0; i < rounds; ++i) {
        rk[i] = t0;
        uint64_t left = t1;
        uint64_t right = t0;
        simeck_round(left, right, base_constant | simeck_z_bit(block_size, i), word_size, mask);
        t1 = left;
        t0 = right;
        uint64_t moved = t1;
        t1 = t2;
        t2 = t3;
        t3 = moved;
    }
}

static inline void simeck_encrypt_keyed(
    uint64_t& x, uint64_t& y, int round_start, int rounds, int word_size, uint64_t mask, const uint64_t* rk
) {
    for (int i = round_start; i < round_start + rounds; ++i) {
        simeck_round(x, y, rk[i], word_size, mask);
    }
}


struct Cipher {
    int block, key_size, word_size, state_words = 2;
    uint64_t mask;
    vector<uint64_t> round_keys;
    Cipher(int b, int k): block(b), key_size(k), word_size(b / 2), mask(word_mask(word_size)) { simeck_rounds_for_variant(block, key_size); }
    vector<int> key_widths() const { return vector<int>(key_size / word_size, word_size); }
    void prepare(const vector<uint64_t>& key, int count) {
        round_keys.resize(max(1, count));
        if (count) { simeck_schedule(key, count, block, key_size, word_size, mask, round_keys.data()); }
    }
    void encrypt(State& state, int start, int rounds) const { simeck_encrypt_keyed(state[0], state[1], start, rounds, word_size, mask, round_keys.data()); }
};
} // namespace validation_cipher
