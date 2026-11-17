#ifndef RNG_H
#define RNG_H

#include <stdint.h>

//xoshiro256** (Blackman & Vigna), period 2^256 - 1, passes BigCrush
//splitmix64 only used to turn a 64 bit seed into the 256 bit state

typedef struct {
    uint64_t s[4];
} rng_t;

#define SM64_GOLDEN 0x9e3779b97f4a7c15ULL

//splitmix64 scramble, bijective
static inline uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

//splitmix64 next
static inline uint64_t splitmix64(uint64_t *x) {
    *x += SM64_GOLDEN;
    return mix64(*x);
}

static inline uint64_t rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

//Next raw 64 bit number
static inline uint64_t rng_next(rng_t *g) {
    uint64_t *s = g->s;
    uint64_t result = rotl64(s[1] * 5, 7) * 9;
    uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl64(s[3], 45);
    return result;
}

//Uniform double in [0, 1), top 53 bits
static inline double rng_u01(rng_t *g) {
    return (double)(rng_next(g) >> 11) * 0x1.0p-53;
}

//Jump ahead 2^128 steps
//k jumps = stream k, streams can't overlap since we use ~10^8 per stream
static inline void rng_jump(rng_t *g) {
    static const uint64_t JUMP[4] = {
        0x180ec6d33cfd0abaULL, 0xd5a61266f0c9392cULL,
        0xa9582618e03fc9aaULL, 0x39abdc4529b1661cULL
    };
    uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int i = 0; i < 4; i++) {
        for (int b = 0; b < 64; b++) {
            if (JUMP[i] & (1ULL << b)) {
                s0 ^= g->s[0];
                s1 ^= g->s[1];
                s2 ^= g->s[2];
                s3 ^= g->s[3];
            }
            rng_next(g);
        }
    }
    g->s[0] = s0;
    g->s[1] = s1;
    g->s[2] = s2;
    g->s[3] = s3;
}

//Fill 256 bit state from one 64 bit seed
static inline void rng_seed64(rng_t *g, uint64_t seed) {
    uint64_t x = seed;
    for (int i = 0; i < 4; i++) {
        g->s[i] = splitmix64(&x);
    }
}

//Fold one more field into the hash
static inline uint64_t hash_step(uint64_t h, uint64_t field) {
    return mix64((h ^ field) + SM64_GOLDEN);
}

//Seed this rank's stream for one trial
//Hash (base, method, p, n, trial) so every experiment gets its own seed
//Then jump rank times so ranks in the same trial are guaranteed disjoint
static inline void rng_seed_stream(rng_t *g, uint64_t base, uint64_t method,
                                   uint64_t p, uint64_t n, uint64_t trial, int rank) {
    uint64_t h = mix64(base + SM64_GOLDEN);
    h = hash_step(h, method);
    h = hash_step(h, p);
    h = hash_step(h, n);
    h = hash_step(h, trial);
    rng_seed64(g, h);
    for (int r = 0; r < rank; r++) {
        rng_jump(g);
    }
}

#endif