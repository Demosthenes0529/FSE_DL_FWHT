#include "present_fast.hpp"
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
constexpr uint8_t SBOX[16] = {0xC,0x5,0x6,0xB,0x9,0x0,0xA,0xD,0x3,0xE,0xF,0x8,0x4,0x7,0x1,0x2};
void expand_key80_into(uint64_t hi16, uint64_t lo64, int rounds, vector<uint64_t>& round_keys) {
    round_keys.clear();
    round_keys.reserve(static_cast<size_t>(rounds + 1));
    uint64_t hi = hi16 & 0xFFFFULL;
    uint64_t lo = lo64;
    for (int round_counter = 1; round_counter <= rounds + 1; ++round_counter) {
        round_keys.push_back(((hi & 0xFFFFULL) << 48) | (lo >> 16));
        if (round_counter > rounds) break;
        uint64_t new_hi = (lo >> 3) & 0xFFFFULL;
        uint64_t new_lo = ((lo & 0x7ULL) << 61) | ((hi & 0xFFFFULL) << 45) | (lo >> 19);
        hi = (static_cast<uint64_t>(SBOX[(new_hi >> 12) & 0xFULL]) << 12) | (new_hi & 0x0FFFULL);
        lo = new_lo ^ (static_cast<uint64_t>(round_counter & 0x1F) << 15);
    }
}

void expand_key128_into(uint64_t hi64, uint64_t lo64, int rounds, vector<uint64_t>& round_keys) {
    round_keys.clear();
    round_keys.reserve(static_cast<size_t>(rounds + 1));
    uint64_t hi = hi64;
    uint64_t lo = lo64;
    for (int round_counter = 1; round_counter <= rounds + 1; ++round_counter) {
        round_keys.push_back(hi);
        if (round_counter > rounds) break;
        uint64_t new_hi = ((hi & 0x7ULL) << 61) | (lo >> 3);
        uint64_t new_lo = ((lo & 0x7ULL) << 61) | (hi >> 3);
        uint64_t top0 = static_cast<uint64_t>(SBOX[(new_hi >> 60) & 0xFULL]);
        uint64_t top1 = static_cast<uint64_t>(SBOX[(new_hi >> 56) & 0xFULL]);
        hi = (top0 << 60) | (top1 << 56) | (new_hi & 0x00FFFFFFFFFFFFFFULL);
        lo = new_lo ^ (static_cast<uint64_t>(round_counter & 0x3) << 62);
        hi ^= static_cast<uint64_t>((round_counter >> 2) & 0x7);
    }
}


struct Cipher {
    int word_size = 64, state_words = 1, version;
    uint64_t mask = ~0ULL;
    vector<uint64_t> round_keys;
    Cipher(int block, int key): version(key) {
        if (block != 64 || (key != 80 && key != 128)) throw runtime_error("invalid PRESENT variant");
    }
    vector<int> key_widths() const { return {version == 80 ? 16 : 64, 64}; }
    void prepare(const vector<uint64_t>& key, int count) {
        if (count > 31) throw runtime_error("PRESENT round interval exceeds 31");
        if (version == 80) expand_key80_into(key[0],key[1],count,round_keys);
        else expand_key128_into(key[0],key[1],count,round_keys);
    }
    void encrypt(State& state, int start, int rounds) const {
        state[0] = present_fast::encrypt_segment(state[0], round_keys, start, rounds);
    }
};
} // namespace validation_cipher
