#define _GNU_SOURCE
#include "simon_transition.h"
#include <errno.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <time.h>

/* Both window conventions form the chain (u_r,u_{r+1}):
 * differential: (R_r,L_r); linear: (L_r,R_r).
 * Their affine transition supports differ, but the restricted-window sum
 * operator, fixed-endpoint lookup and memory management are shared.
 * Linear values are sums of squared trail correlations, not signed hulls. */
typedef struct {
    int rounds, width;
    size_t n, states, bytes;
    uint64_t **values;
    int8_t (*inverse)[64];
} window_t;

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

static void *checked_calloc(size_t count, size_t size) {
    void *p = calloc(count, size);
    if (!p) { perror("calloc"); exit(3); }
    return p;
}

static void *zero_map(size_t bytes) {
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) { perror("mmap"); exit(3); }
    return p;
}

static long index_of(uint64_t value, const int8_t inverse[64]) {
    size_t index = 0;
    while (value) {
        unsigned bit = __builtin_ctzll(value);
        if (inverse[bit] < 0) return -1;
        index |= (size_t)1 << inverse[bit];
        value &= value - 1;
    }
    return (long)index;
}

static window_t read_window(const char *path, unsigned word, double limit) {
    FILE *file = fopen(path, "r");
    if (!file) { perror(path); exit(2); }
    window_t w = {0};
    if (fscanf(file, "%d%d", &w.rounds, &w.width) != 2 ||
            w.rounds < 1 || w.rounds > 4096 || w.width < 1 ||
            w.width > 20 || w.width > (int)word) {
        fprintf(stderr, "invalid window rounds/width\n"); exit(2);
    }
    w.n = (size_t)1 << w.width;
    w.states = w.n * w.n;
    w.bytes = w.states * sizeof(double);
    double estimate = (2.2 * w.bytes +
        (double)(w.rounds + 2) * w.n * sizeof(uint64_t) +
        (double)w.n * sizeof(simon_transition_space) + 16 * 1048576.0) / 1073741824.0;
    FILE *mem = fopen("/proc/meminfo", "r");
    double available = INFINITY;
    if (mem) {
        char line[256]; unsigned long long kib;
        while (fgets(line, sizeof(line), mem))
            if (sscanf(line, "MemAvailable: %llu kB", &kib) == 1)
                available = kib / 1048576.0;
        fclose(mem);
    }
    fprintf(stderr, "DP memory estimate %.6f GiB; limit %.6f GiB\n", estimate, limit);
    if (estimate > limit || estimate > 0.9 * available) {
        fprintf(stderr, "window DP exceeds memory limit or available memory\n"); exit(4);
    }
    w.values = checked_calloc(w.rounds + 2, sizeof(*w.values));
    w.inverse = checked_calloc(w.rounds + 2, sizeof(*w.inverse));
    for (int r = 0; r < w.rounds + 2; ++r) {
        int positions[20];
        memset(w.inverse[r], -1, 64);
        for (int j = 0; j < w.width; ++j) {
            if (fscanf(file, "%d", &positions[j]) != 1 ||
                    positions[j] < 0 || positions[j] >= (int)word ||
                    w.inverse[r][positions[j]] >= 0) {
                fprintf(stderr, "invalid or repeated window bit in row %d\n", r); exit(2);
            }
            w.inverse[r][positions[j]] = (int8_t)j;
        }
        w.values[r] = checked_calloc(w.n, sizeof(uint64_t));
        for (size_t i = 1; i < w.n; ++i) {
            unsigned bit = __builtin_ctzll(i);
            w.values[r][i] = w.values[r][i ^ ((size_t)1 << bit)] ^
                             (UINT64_C(1) << positions[bit]);
        }
    }
    char extra;
    if (fscanf(file, " %c", &extra) == 1) {
        fprintf(stderr, "unexpected data after window rows\n"); exit(2);
    }
    fclose(file);
    return w;
}

static uint64_t ror(uint64_t x, unsigned shift, unsigned word) {
    return simon_rotl_n(x, word - shift % word, word);
}

static simon_transition_space linear_space(uint64_t beta, unsigned word,
                                           unsigned a, unsigned b, unsigned c) {
    simon_transition_space t = {0};
    t.B = ror(beta, c, word);
    if (a % word == b % word) {
        /* When both AND rotations coincide, x & x is linear. */
        t.B ^= ror(beta, a, word);
    } else {
        /* A fully active odd cycle has quadratic parity one on its radical.
         * Its Walsh support is an affine coset, not the zero-offset row
         * space. Standard cipher word sizes have even cycles; retain the
         * correct offset for the small word sizes accepted by this program. */
        uint64_t unseen = simon_word_mask(word);
        while (unseen) {
            unsigned first = __builtin_ctzll(unseen), edge = first, length = 0;
            int full = 1;
            do {
                unseen &= ~(UINT64_C(1) << edge);
                full &= (int)((beta >> edge) & 1);
                ++length;
                edge = (edge + b - a) % word;
            } while (edge != first);
            if (full && (length & 1))
                t.B ^= UINT64_C(1) << ((first + word - a % word) % word);
        }
        for (unsigned i = 0; i < word; ++i) {
            uint64_t one = UINT64_C(1) << i;
            uint64_t value = ror((beta & simon_rotl_n(one, b-a, word)) ^
                                ror(beta & one, b-a, word), a, word);
            simon_basis_add(t.basis, value);
        }
    }
    for (int bit = (int)word - 1; bit >= 0; --bit) {
        if (!t.basis[bit]) continue;
        ++t.rank;
        if ((t.B >> bit) & 1) t.B ^= t.basis[bit];
    }
    t.probability = ldexp(1.0, -(int)t.rank);
    return t;
}

/* The transpose of the same forward restricted-window operator used in
 * both original programs. Each worker owns one predecessor second-word
 * column; no atomics or cross-thread floating-point reductions are needed. */
static void backward(const double *next, double *prev, const window_t *w, int r,
                     const simon_transition_space *spaces) {
    memset(prev, 0, w->bytes);
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t second = 0; second < w->n; ++second) {
        const simon_transition_space *t = &spaces[second];
        uint64_t compact[64]; unsigned count = 0;
        if (t->rank < (unsigned)w->width)
            for (int bit = 63; bit >= 0; --bit)
                if (t->basis[bit]) compact[count++] = t->basis[bit];
        for (size_t out = 0; out < w->n; ++out) {
            double mass = next[second * w->n + out] * t->probability;
            if (!mass) continue;
            if (t->rank < (unsigned)w->width) {
                uint64_t value = t->B ^ w->values[r+2][out], gray = 0;
                size_t combinations = (size_t)1 << t->rank;
                for (size_t k = 0; k < combinations; ++k) {
                    if (k) {
                        uint64_t new_gray = k ^ (k >> 1);
                        value ^= compact[__builtin_ctzll(gray ^ new_gray)];
                        gray = new_gray;
                    }
                    long first = index_of(value, w->inverse[r]);
                    if (first >= 0) prev[(size_t)first * w->n + second] += mass;
                }
            } else {
                for (size_t first = 0; first < w->n; ++first)
                    if (simon_transition_contains(t, w->values[r+2][out] ^ w->values[r][first]))
                        prev[first * w->n + second] += mass;
            }
        }
    }
}

static double evaluate(const window_t *w, unsigned word, unsigned a, unsigned b,
                       unsigned c, int linear, uint64_t inL, uint64_t inR,
                       uint64_t outL, uint64_t outR) {
    long input_first = index_of(linear ? inL : inR, w->inverse[0]);
    long input_second = index_of(linear ? inR : inL, w->inverse[1]);
    long output_first = index_of(linear ? outL : outR, w->inverse[w->rounds]);
    long output_second = index_of(linear ? outR : outL, w->inverse[w->rounds+1]);
    if (input_first < 0 || input_second < 0 || output_first < 0 || output_second < 0)
        return 0.0;
    double *next = zero_map(w->bytes), *prev = zero_map(w->bytes);
    next[(size_t)output_first * w->n + (size_t)output_second] = 1.0;
    simon_transition_space *spaces = checked_calloc(w->n, sizeof(*spaces));
    for (int r = w->rounds - 1; r >= 0; --r) {
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < w->n; ++i)
            spaces[i] = linear ? linear_space(w->values[r+1][i], word, a, b, c) :
                simon_make_transition_params_n(w->values[r+1][i], word, a, b, c);
        backward(next, prev, w, r, spaces);
        double *swap = next; next = prev; prev = swap;
        fprintf(stderr, "WINDOW_DP round=%d kind=%s\n", r, linear ? "linear" : "differential");
    }
    double mass = next[(size_t)input_first * w->n + (size_t)input_second];
    free(spaces); munmap(next, w->bytes); munmap(prev, w->bytes);
    return mass;
}

static uint64_t parse_word(const char *text) {
    char *end; errno = 0;
    unsigned long long value = strtoull(text, &end, 0);
    if (errno || *end || end == text || text[0] == '-') {
        fprintf(stderr, "invalid endpoint word: %s\n", text); exit(2);
    }
    return (uint64_t)value;
}

static void usage(const char *name) {
    fprintf(stderr, "usage: %s --trail-kind differential|linear --family simon|simeck "
        "--word-size N --window FILE --output JSON --input-left HEX --input-right HEX "
        "--output-left HEX --output-right HEX [--threads N] [--memory-limit-gb N]\n", name);
}

int main(int argc, char **argv) {
    const char *path = NULL, *output = NULL, *family = "simon", *kind = "differential";
    unsigned word = 0; int threads = 1, endpoints = 0;
    uint64_t inL = 0, inR = 0, outL = 0, outR = 0;
    double limit = 32;
    for (int i = 1; i < argc; ++i) {
        const char *option = argv[i];
        if (!strcmp(option, "--help")) { usage(argv[0]); return 0; }
        if (++i == argc) { usage(argv[0]); return 2; }
        const char *value = argv[i];
        if (!strcmp(option, "--window")) path = value;
        else if (!strcmp(option, "--output")) output = value;
        else if (!strcmp(option, "--family")) family = value;
        else if (!strcmp(option, "--trail-kind")) kind = value;
        else if (!strcmp(option, "--word-size")) word = (unsigned)atoi(value);
        else if (!strcmp(option, "--threads")) threads = atoi(value);
        else if (!strcmp(option, "--memory-limit-gb")) limit = atof(value);
        else if (!strcmp(option, "--input-left")) { inL = parse_word(value); endpoints |= 1; }
        else if (!strcmp(option, "--input-right")) { inR = parse_word(value); endpoints |= 2; }
        else if (!strcmp(option, "--output-left")) { outL = parse_word(value); endpoints |= 4; }
        else if (!strcmp(option, "--output-right")) { outR = parse_word(value); endpoints |= 8; }
        else { usage(argv[0]); return 2; }
    }
    if (!path || !output || word < 2 || word > 64 || threads < 1 || endpoints != 15 ||
            !isfinite(limit) || limit <= 0 ||
            (strcmp(kind, "differential") && strcmp(kind, "linear")) ||
            (strcmp(family, "simon") && strcmp(family, "simeck"))) {
        usage(argv[0]); return 2;
    }
    uint64_t mask = simon_word_mask(word);
    if ((inL | inR | outL | outR) & ~mask) {
        fprintf(stderr, "endpoint exceeds word size\n"); return 2;
    }
    unsigned a = 1, b = 8, c = 2;
    if (!strcmp(family, "simeck")) { a = 0; b = 5; c = 1; }
    int linear = !strcmp(kind, "linear");
    omp_set_num_threads(threads);
    window_t w = read_window(path, word, limit);
    double started = now();
    double mass = evaluate(&w, word, a, b, c, linear, inL, inR, outL, outR);
    char weight[64] = "null", rms[64] = "null";
    if (mass > 0) {
        snprintf(weight, sizeof(weight), "%.12f", -log2(mass));
        snprintf(rms, sizeof(rms), "%.12f", -0.5 * log2(mass));
    }
    struct rusage ru; getrusage(RUSAGE_SELF, &ru);
    FILE *file = fopen(output, "w");
    if (!file) { perror(output); return 5; }
    fprintf(file, "{\n  \"semantics\": \"%s\",\n  \"trail_kind\": \"%s\",\n"
        "  \"rounds\": %d,\n  \"width\": %d,\n  \"mass\": %.17g,\n  \"mass_weight\": %s,\n"
        "  \"%s\": %.17g,\n  \"%s_weight\": %s,\n",
        linear ? "linear_hull_potential" : "differential_probability", kind,
        w.rounds, w.width, mass, weight, linear ? "potential" : "probability", mass,
        linear ? "potential" : "probability", weight);
    if (linear) fprintf(file, "  \"rms_correlation_weight\": %s,\n", rms);
    fprintf(file, "  \"runtime_seconds\": %.6f,\n  \"peak_rss_kb\": %ld,\n"
        "  \"peak_rss_gb\": %.6f\n}\n", now() - started, ru.ru_maxrss, ru.ru_maxrss / 1048576.0);
    if (fclose(file)) { perror(output); return 5; }
    for (int r = 0; r < w.rounds + 2; ++r) free(w.values[r]);
    free(w.values); free(w.inverse);
    return 0;
}
