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


struct CipherConfig { int word_size, key_words; uint64_t mask; };
struct PreparedKey { vector<uint64_t> cham_round_keys; };
inline void expand_cham_key(const CipherConfig& cfg, const std::vector<uint64_t>& key, PreparedKey& prepared) {
    if (static_cast<int>(key.size()) != cfg.key_words) throw std::runtime_error("invalid CHAM key word count");
    prepared.cham_round_keys.assign(static_cast<size_t>(2 * cfg.key_words), 0);
    for (int i = 0; i < cfg.key_words; ++i) {
        uint64_t w = key[static_cast<size_t>(i)] & cfg.mask;
        prepared.cham_round_keys[static_cast<size_t>(i)] =
            (w ^ rol(w, 1, cfg.word_size, cfg.mask) ^ rol(w, 8, cfg.word_size, cfg.mask)) & cfg.mask;
        int j = (i + cfg.key_words) ^ 1;
        prepared.cham_round_keys[static_cast<size_t>(j)] =
            (w ^ rol(w, 1, cfg.word_size, cfg.mask) ^ rol(w, 11, cfg.word_size, cfg.mask)) & cfg.mask;
    }
}

inline void cham_encrypt(State& state, const CipherConfig& cfg, const PreparedKey& key, int round_start, int rounds) {
    if (state.size() != 4) throw std::runtime_error("CHAM state must have 4 words");
    for (int offset = 0; offset < rounds; ++offset) {
        int round_index = round_start + offset;
        int alpha = (round_index & 1) ? 8 : 1;
        int beta = (round_index & 1) ? 1 : 8;
        uint64_t round_key = key.cham_round_keys[static_cast<size_t>(round_index % key.cham_round_keys.size())];
        uint64_t tmp = (((state[0] ^ (static_cast<uint64_t>(round_index) & cfg.mask)) +
                         (rol(state[1], alpha, cfg.word_size, cfg.mask) ^ round_key)) & cfg.mask);
        uint64_t new_word = rol(tmp, beta, cfg.word_size, cfg.mask);
        state[0] = state[1] & cfg.mask;
        state[1] = state[2] & cfg.mask;
        state[2] = state[3] & cfg.mask;
        state[3] = new_word & cfg.mask;
    }
}


struct Cipher {
    int word_size, state_words = 4;
    uint64_t mask;
    CipherConfig cfg;
    PreparedKey prepared;
    Cipher(int block, int key): word_size(block / 4), mask(word_mask(word_size)), cfg{word_size,key / word_size,mask} {
        if ((block != 64 && block != 128) || key != 128) throw runtime_error("invalid CHAM variant");
    }
    vector<int> key_widths() const { return vector<int>(cfg.key_words, word_size); }
    void prepare(const vector<uint64_t>& key, int) { expand_cham_key(cfg,key,prepared); }
    void encrypt(State& state, int start, int rounds) const { cham_encrypt(state,cfg,prepared,start,rounds); }
};
} // namespace validation_cipher
