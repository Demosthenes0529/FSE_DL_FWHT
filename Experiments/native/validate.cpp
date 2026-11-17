// Shared fixed-key DL and ordinary linear sampling. CIPHER_HEADER supplies
// only the selected family's key schedule and round function.
#include CIPHER_HEADER
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <thread>

using validation_cipher::Cipher;
using validation_cipher::State;
using namespace std;

static uint64_t splitmix(uint64_t& state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static uint64_t sample_word(uint64_t seed, uint64_t key, uint64_t sample, uint64_t stream, uint64_t mask) {
    uint64_t state = seed ^ (0xBF58476D1CE4E5B9ULL * (sample + 1)) ^
                     (0xD6E8FEB86659FD93ULL * (key + 1)) ^
                     (0x94D049BB133111EBULL * (stream + 1));
    return splitmix(state) & mask;
}

static string trim(const string& text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == string::npos) return "";
    return text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
}

static uint64_t unsigned_value(const string& text) {
    const auto value = trim(text);
    if (value.empty() || value.front() == '-' || value.front() == '+')
        throw runtime_error("invalid unsigned integer: " + text);
    size_t parsed = 0;
    const uint64_t result = stoull(value, &parsed, 0);
    if (parsed != value.size()) throw runtime_error("invalid unsigned integer: " + text);
    return result;
}

static int dimension(const string& text) {
    const uint64_t value = unsigned_value(text);
    if (value > uint64_t(numeric_limits<int>::max())) throw runtime_error("dimension too large");
    return int(value);
}

static vector<uint64_t> words(const string& text) {
    vector<uint64_t> result;
    if (trim(text).empty()) return result;  // One empty key row for KNOT.
    if (trim(text).back() == ',') throw runtime_error("empty word after comma");
    string part;
    istringstream in(text);
    while (getline(in, part, ',')) result.push_back(unsigned_value(part));
    return result;
}

static vector<vector<uint64_t>> read_keys(const string& path, int count, const vector<int>& widths) {
    ifstream input(path);
    if (!input) throw runtime_error("cannot open keys file");
    vector<vector<uint64_t>> keys;
    string line;
    while (getline(input, line)) {
        auto key = words(line);
        if (key.size() != widths.size()) throw runtime_error("wrong master-key word count");
        for (size_t i = 0; i < key.size(); ++i) {
            if (widths[i] < 1 || widths[i] > 64) throw runtime_error("invalid key word width");
            if (widths[i] < 64 && (key[i] >> widths[i])) throw runtime_error("master-key word too wide");
        }
        keys.push_back(move(key));
        if (keys.size() > size_t(count)) throw runtime_error("too many rows in keys file");
    }
    if (keys.size() != size_t(count)) throw runtime_error("keys file row count does not match --keys");
    return keys;
}

int main(int argc, char** argv) {
    try {
        const set<string> allowed = {"--measurement", "--difference", "--keys-file", "--keys", "--seed",
            "--block", "--key-bits", "--round-start", "--rounds", "--samples", "--threads", "--masks", "--output"};
        map<string, string> args;
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 >= argc) throw runtime_error("missing argument value");
            if (!allowed.count(argv[i])) throw runtime_error("unknown argument: " + string(argv[i]));
            if (!args.emplace(argv[i], argv[i + 1]).second) throw runtime_error("duplicate argument");
        }
        auto arg = [&](const char* name) {
            auto found = args.find(name);
            if (found == args.end()) throw runtime_error("missing argument: " + string(name));
            return found->second;
        };
        int block = dimension(arg("--block")), key_bits = dimension(arg("--key-bits"));
        int start = dimension(arg("--round-start")), rounds = dimension(arg("--rounds"));
        int keys = dimension(arg("--keys")), threads = dimension(arg("--threads"));
        uint64_t samples = unsigned_value(arg("--samples")), seed = unsigned_value(arg("--seed"));
        const string measurement = arg("--measurement");
        if (measurement != "dl" && measurement != "linear") throw runtime_error("measurement must be dl or linear");
        const bool linear = measurement == "linear";
        if (keys < 1 || threads < 1 || samples < 1 || samples > (1ULL << 62) ||
            start > numeric_limits<int>::max() - rounds) throw runtime_error("invalid dimensions");
        if (linear && (samples < 2 || samples % 2)) throw runtime_error("linear sampling requires an even sample count >= 2");
        Cipher identity(block, key_bits);
        const auto widths = identity.key_widths();
        identity.prepare(vector<uint64_t>(widths.size(), 0), start + rounds);
        const auto key_list = read_keys(arg("--keys-file"), keys, widths);
        const auto difference = words(arg("--difference"));
        if (difference.size() != size_t(identity.state_words)) throw runtime_error("wrong difference/input-mask word count");
        for (auto word : difference) if (word & ~identity.mask) throw runtime_error("difference/input-mask word too wide");
        vector<State> masks;
        ifstream mask_file(arg("--masks"));
        if (!mask_file) throw runtime_error("cannot open masks file");
        string line;
        while (getline(mask_file, line)) {
            if (trim(line).empty()) continue;
            auto values = words(line);
            if (values.size() != size_t(identity.state_words)) throw runtime_error("wrong mask word count");
            for (auto word : values) if (word & ~identity.mask) throw runtime_error("mask word too wide");
            State state{};
            copy(values.begin(), values.end(), state.begin());
            masks.push_back(state);
        }
        if (masks.empty()) throw runtime_error("empty masks file");
        if (samples > numeric_limits<uint64_t>::max() / uint64_t(keys)) throw runtime_error("sample count overflow");
        if (masks.size() > numeric_limits<size_t>::max() / size_t(keys)) throw runtime_error("mask count overflow");
        const uint64_t total = samples * keys;
        threads = int(min(uint64_t(threads), total));
        const size_t counts = size_t(keys) * masks.size();
        vector<vector<uint64_t>> partial(threads, vector<uint64_t>(counts));
        vector<vector<uint64_t>> first_half(linear ? threads : 0, vector<uint64_t>(linear ? counts : 0));
        vector<thread> workers;
        for (int tid = 0; tid < threads; ++tid) workers.emplace_back([&, tid] {
            Cipher cipher(block, key_bits);
            uint64_t cached_key = numeric_limits<uint64_t>::max();
            uint64_t begin = (total / threads) * tid + min(total % threads, uint64_t(tid));
            uint64_t end = (total / threads) * (tid + 1) + min(total % threads, uint64_t(tid + 1));
            for (uint64_t index = begin; index < end; ++index) {
                const uint64_t key_index = index / samples, sample = index % samples;
                if (key_index != cached_key) {
                    cipher.prepare(key_list[key_index], start + rounds);
                    cached_key = key_index;
                }
                State left{}, right{};
                for (int i = 0; i < cipher.state_words; ++i)
                    left[i] = sample_word(seed, key_index, sample, i, cipher.mask);
                uint64_t input_selected = 0;
                if (linear) {
                    for (int i = 0; i < cipher.state_words; ++i) input_selected ^= left[i] & difference[i];
                    // An empty linear suffix is the identity, including for
                    // PRESENT whose ordinary reference call includes whitening.
                    if (rounds) cipher.encrypt(left, start, rounds);
                } else {
                    right = left;
                    for (int i = 0; i < cipher.state_words; ++i) right[i] ^= difference[i];
                    cipher.encrypt(left, start, rounds);
                    cipher.encrypt(right, start, rounds);
                }
                for (size_t m = 0; m < masks.size(); ++m) {
                    uint64_t selected = input_selected;
                    for (int i = 0; i < cipher.state_words; ++i)
                        selected ^= (linear ? left[i] : left[i] ^ right[i]) & masks[m][i];
                    const unsigned odd = __builtin_parityll(selected);
                    const size_t count_index = key_index * masks.size() + m;
                    partial[tid][count_index] += odd;
                    if (linear && sample < samples / 2) first_half[tid][count_index] += odd;
                }
            }
        });
        for (auto& thread : workers) thread.join();
        ofstream out(arg("--output"));
        if (!out) throw runtime_error("cannot create output");
        out << "mask_index,key_index,odd_parity_count,Walsh,rho,Walsh_first_half,Walsh_second_half\n" << setprecision(20);
        for (int key = 0; key < keys; ++key) for (size_t m = 0; m < masks.size(); ++m) {
            uint64_t odd = 0, odd_first = 0;
            const size_t index = size_t(key) * masks.size() + m;
            for (int tid = 0; tid < threads; ++tid) {
                odd += partial[tid][index];
                if (linear) odd_first += first_half[tid][index];
            }
            const int64_t walsh = (int64_t(samples) - int64_t(odd)) - int64_t(odd);
            out << m << ',' << key << ',' << odd << ',' << walsh << ',' << double(walsh) / samples << ',';
            if (linear) {
                const uint64_t odd_second = odd - odd_first;
                out << int64_t(samples / 2) - 2 * int64_t(odd_first) << ','
                    << int64_t(samples / 2) - 2 * int64_t(odd_second);
            } else out << ',';
            out << '\n';
        }
        return 0;
    } catch (const exception& error) {
        cerr << "validation: " << error.what() << '\n';
        return 1;
    }
}
