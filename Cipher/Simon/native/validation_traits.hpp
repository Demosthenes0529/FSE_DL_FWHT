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

static inline void simon_round(uint64_t& x, uint64_t& y, uint64_t k, int word_size, uint64_t mask) {
    uint64_t z = (y ^ (rol(x, 1, word_size, mask) & rol(x, 8, word_size, mask)) ^
                  rol(x, 2, word_size, mask) ^ k) & mask;
    y = x & mask;
    x = z;
}

uint64_t simon_z_sequence(int block_size, int key_size) {
    if ((block_size == 32 && key_size == 64) || (block_size == 48 && key_size == 72)) return 0x19c3522fb386a45fULL;
    if (block_size == 48 && key_size == 96) return 0x16864fb8ad0c9f71ULL;
    if ((block_size == 64 && key_size == 96) || (block_size == 96 && key_size == 96) ||
        (block_size == 128 && key_size == 128)) return 0x3369f885192c0ef5ULL;
    if ((block_size == 64 && key_size == 128) || (block_size == 96 && key_size == 144) ||
        (block_size == 128 && key_size == 192)) return 0x3c2ce51207a635dbULL;
    if (block_size == 128 && key_size == 256) return 0x3dc94c3a046d678bULL;
    throw runtime_error("unsupported SIMON block/key size");
}

int simon_rounds_for_variant(int block_size, int key_size) {
    if (block_size == 32 && key_size == 64) return 32;
    if (block_size == 48 && (key_size == 72 || key_size == 96)) return 36;
    if (block_size == 64 && key_size == 96) return 42;
    if (block_size == 64 && key_size == 128) return 44;
    if (block_size == 96 && key_size == 96) return 52;
    if (block_size == 96 && key_size == 144) return 54;
    if (block_size == 128 && key_size == 128) return 68;
    if (block_size == 128 && key_size == 192) return 69;
    if (block_size == 128 && key_size == 256) return 72;
    throw runtime_error("unsupported SIMON block/key size");
}

int simon_default_key_size(int block_size) {
    if (block_size == 32) return 64;
    if (block_size == 48) return 96;
    if (block_size == 64) return 128;
    if (block_size == 96) return 144;
    if (block_size == 128) return 256;
    throw runtime_error("unsupported SIMON block size");
}

static inline void simon_schedule(
    const vector<uint64_t>& key_words,
    int rounds,
    int block_size,
    int key_size,
    int word_size,
    uint64_t mask,
    uint64_t* rk
) {
    const int m = key_size / word_size;
    if (static_cast<int>(key_words.size()) != m) throw runtime_error("wrong SIMON key word count");
    if (rounds > simon_rounds_for_variant(block_size, key_size)) {
        throw runtime_error("SIMON schedule requested more than the standard round count");
    }
    for (int i = 0; i < rounds && i < m; ++i) {
        rk[i] = key_words[static_cast<size_t>(m - 1 - i)] & mask;
    }
    const uint64_t c = mask ^ 3ULL;
    const uint64_t z = simon_z_sequence(block_size, key_size);
    for (int i = m; i < rounds; ++i) {
        uint64_t tmp = ror(rk[i - 1], 3, word_size, mask);
        if (m == 4) tmp ^= rk[i - 3];
        tmp ^= ror(tmp, 1, word_size, mask);
        rk[i] = (c ^ ((z >> ((i - m) % 62)) & 1ULL) ^ rk[i - m] ^ tmp) & mask;
    }
}

static inline void simon_encrypt_keyed(
    uint64_t& x, uint64_t& y, int round_start, int rounds, int word_size, uint64_t mask, const uint64_t* rk
) {
    for (int i = round_start; i < round_start + rounds; ++i) {
        simon_round(x, y, rk[i], word_size, mask);
    }
}


struct Cipher {
    int block, key_size, word_size, state_words = 2;
    uint64_t mask;
    vector<uint64_t> round_keys;
    Cipher(int b, int k): block(b), key_size(k), word_size(b / 2), mask(word_mask(word_size)) { simon_rounds_for_variant(block, key_size); }
    vector<int> key_widths() const { return vector<int>(key_size / word_size, word_size); }
    void prepare(const vector<uint64_t>& key, int count) {
        round_keys.resize(max(1, count));
        if (count) { simon_schedule(key, count, block, key_size, word_size, mask, round_keys.data()); }
    }
    void encrypt(State& state, int start, int rounds) const { simon_encrypt_keyed(state[0], state[1], start, rounds, word_size, mask, round_keys.data()); }
};
} // namespace validation_cipher
