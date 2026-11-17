#pragma once

#include <cstdint>
#include <immintrin.h>
#include <vector>

// A production fast path kept separate from the scalar PRESENT reference.
// State and key conventions are identical to scan_panel_fwht_present.cpp.
namespace present_fast {

constexpr std::uint64_t NIBBLE_LSB = 0x1111111111111111ULL;

// Evaluate the algebraic normal form of the PRESENT S-box on all sixteen
// nibbles in parallel.  x0 is the least-significant bit of every nibble.
inline std::uint64_t sbox_layer(std::uint64_t x) {
    const std::uint64_t x0 = x & NIBBLE_LSB;
    const std::uint64_t x1 = (x >> 1) & NIBBLE_LSB;
    const std::uint64_t x2 = (x >> 2) & NIBBLE_LSB;
    const std::uint64_t x3 = (x >> 3) & NIBBLE_LSB;
    const std::uint64_t x01 = x0 & x1;
    const std::uint64_t x02 = x0 & x2;
    const std::uint64_t x12 = x1 & x2;
    const std::uint64_t x03 = x0 & x3;
    const std::uint64_t x13 = x1 & x3;
    const std::uint64_t x23 = x2 & x3;
    const std::uint64_t x012 = x01 & x2;
    const std::uint64_t x013 = x01 & x3;
    const std::uint64_t x023 = x02 & x3;

    const std::uint64_t y0 = x0 ^ x2 ^ x12 ^ x3;
    const std::uint64_t y1 = x1 ^ x012 ^ x3 ^ x13 ^ x013 ^ x23 ^ x023;
    const std::uint64_t y2 = NIBBLE_LSB ^ x01 ^ x2 ^ x3 ^ x03 ^ x13 ^ x013 ^ x023;
    const std::uint64_t y3 = NIBBLE_LSB ^ x0 ^ x1 ^ x12 ^ x012 ^ x3 ^ x013 ^ x023;
    return y0 | (y1 << 1) | (y2 << 2) | (y3 << 3);
}

// PRESENT maps source bit i to 16*i mod 63 (and bit 63 to itself).  Gathering
// the same bit from every nibble produces exactly the four contiguous 16-bit
// output planes.  Four PEXTs replace the old 64-iteration scalar loop.
#if defined(__x86_64__) || defined(_M_X64)
__attribute__((target("bmi2")))
inline std::uint64_t p_layer(std::uint64_t x) {
    return _pext_u64(x, 0x1111111111111111ULL)
        | (_pext_u64(x, 0x2222222222222222ULL) << 16)
        | (_pext_u64(x, 0x4444444444444444ULL) << 32)
        | (_pext_u64(x, 0x8888888888888888ULL) << 48);
}
#endif

inline std::uint64_t encrypt_segment(
    std::uint64_t plaintext,
    const std::vector<std::uint64_t>& round_keys,
    int round_start,
    int rounds
) {
    std::uint64_t state = plaintext;
    for (int offset = 0; offset < rounds; ++offset) {
        state ^= round_keys[static_cast<std::size_t>(round_start + offset)];
        state = p_layer(sbox_layer(state));
    }
    return state ^ round_keys[static_cast<std::size_t>(round_start + rounds)];
}

}  // namespace present_fast
