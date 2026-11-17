//Serial tests for heart.h and rng.h, no MPI
//Run from repo root: make -f Make/Makefile test
#define _POSIX_C_SOURCE 199309L
#include "heart.h"
#include "rng.h"

#include <math.h>
#include <stdio.h>
#include <time.h>

static int n_fail = 0;

static void check(int ok, const char *name) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) {
        n_fail++;
    }
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

//RNG known answers
//Expected values from the reference C code + a separate Python port
//Jump target computed separately as T^(2^128) over GF(2)
static void test_rng_known_answers(void) {
    printf("RNG known-answer tests\n");

    rng_t g = {{1, 2, 3, 4}};
    const uint64_t xo[4] = {11520ULL, 0ULL, 1509978240ULL, 1215971899390074240ULL};
    int ok = 1;
    for (int i = 0; i < 4; i++) {
        if (rng_next(&g) != xo[i]) {
            ok = 0;
        }
    }
    check(ok, "xoshiro256** from state {1,2,3,4}");

    uint64_t x = 1234567;
    const uint64_t sm[3] = {6457827717110365317ULL, 3203168211198807973ULL,
                            9817491932198370423ULL};
    ok = 1;
    for (int i = 0; i < 3; i++) {
        if (splitmix64(&x) != sm[i]) {
            ok = 0;
        }
    }
    check(ok, "splitmix64 from seed 1234567");

    rng_t j = {{1, 2, 3, 4}};
    rng_jump(&j);
    const uint64_t jt[4] = {0x8c7a153956b5f3d1ULL, 0x701f1a713401d85eULL,
                            0x6527f66a65469085ULL, 0x8386b786c4408050ULL};
    ok = 1;
    for (int i = 0; i < 4; i++) {
        if (j.s[i] != jt[i]) {
            ok = 0;
        }
    }
    check(ok, "jump() lands exactly 2^128 steps ahead");
}

//RNG stats sanity
static void test_rng_stats(void) {
    printf("RNG statistical sanity (10^7 draws)\n");
    const long N = 10000000;
    rng_t g;
    rng_seed_stream(&g, 530, 1, 16, (uint64_t)N, 0, 0);

    //Mean, var, range + 10x10 chi square on (u, v) pairs
    long bins[10][10] = {{0}};
    double sum = 0.0, sum2 = 0.0;
    int in_range = 1;
    for (long i = 0; i < N; i++) {
        double u = rng_u01(&g);
        double v = rng_u01(&g);
        if (u < 0.0 || u >= 1.0 || v < 0.0 || v >= 1.0) {
            in_range = 0;
        }
        sum += u;
        sum2 += u * u;
        bins[(int)(u * 10)][(int)(v * 10)]++;
    }
    double mean = sum / N;
    double var = sum2 / N - mean * mean;
    double exp_cnt = (double)N / 100.0;
    double chi2 = 0.0;
    for (int a = 0; a < 10; a++) {
        for (int b = 0; b < 10; b++) {
            double d = bins[a][b] - exp_cnt;
            chi2 += d * d / exp_cnt;
        }
    }
    printf("    mean = %.6f (0.5), var = %.6f (%.6f), chi2(99 df) = %.1f\n",
           mean, var, 1.0 / 12.0, chi2);
    check(in_range, "every u in [0, 1)");
    check(fabs(mean - 0.5) < 5.0 * sqrt(1.0 / 12.0 / N), "mean within 5 sigma of 1/2");
    check(fabs(var - 1.0 / 12.0) < 1e-3, "variance ~ 1/12");
    check(chi2 < 148.2, "2D chi-square below 99.9% quantile (148.2)");
}

//RNG seeding + streams
static void test_rng_streams(void) {
    printf("RNG seeding and streams\n");
    rng_t a, b;

    rng_seed_stream(&a, 530, 1, 16, 360000, 7, 3);
    rng_seed_stream(&b, 530, 1, 16, 360000, 7, 3);
    int same = 1;
    for (int i = 0; i < 1000; i++) {
        if (rng_next(&a) != rng_next(&b)) {
            same = 0;
        }
    }
    check(same, "same (seed, method, p, n, trial, rank) -> identical stream");

    //Changing any one field has to change the stream
    const uint64_t base[6][5] = {
        {530, 1, 16, 360000, 7}, //Reference
        {531, 1, 16, 360000, 7}, //Base seed
        {530, 2, 16, 360000, 7}, //Method
        {530, 1, 36, 360000, 7}, //p
        {530, 1, 16, 3600000, 7}, //n
        {530, 1, 16, 360000, 8}, //Trial
    };
    uint64_t first[7];
    for (int k = 0; k < 6; k++) {
        rng_seed_stream(&a, base[k][0], base[k][1], base[k][2], base[k][3], base[k][4], 0);
        first[k] = rng_next(&a);
    }
    rng_seed_stream(&a, 530, 1, 16, 360000, 7, 1); //Rank
    first[6] = rng_next(&a);
    int distinct = 1;
    for (int i = 0; i < 7; i++) {
        for (int k = i + 1; k < 7; k++) {
            if (first[i] == first[k]) {
                distinct = 0;
            }
        }
    }
    check(distinct, "changing any one of base/method/p/n/trial/rank changes the stream");

    //Rank 0 vs rank 1 in the same trial, corr should be ~0
    const long N = 1000000;
    rng_seed_stream(&a, 530, 4, 64, (uint64_t)N, 0, 0);
    rng_seed_stream(&b, 530, 4, 64, (uint64_t)N, 0, 1);
    double sx = 0, sy = 0, sxy = 0, sxx = 0, syy = 0;
    for (long i = 0; i < N; i++) {
        double x = rng_u01(&a);
        double y = rng_u01(&b);
        sx += x; sy += y; sxy += x * y; sxx += x * x; syy += y * y;
    }
    double cov = sxy / N - (sx / N) * (sy / N);
    double corr = cov / sqrt((sxx / N - (sx / N) * (sx / N)) * (syy / N - (sy / N) * (sy / N)));
    printf("    corr(rank 0, rank 1) = %+.2e (noise level ~%.0e)\n", corr, 1.0 / sqrt((double)N));
    check(fabs(corr) < 5.0 / sqrt((double)N), "rank streams uncorrelated (|corr| < 5/sqrt(N))");
}

//Inside test
static void test_in_heart(void) {
    printf("Inside test\n");
    //x, y, expected
    const double pts[][3] = {
        { 0.0,   0.0,  1},  //Center
        { 0.0,   0.99, 1},  //Just under the top dip (cusp at (0, 1))
        { 0.0,   1.01, 0},  //Just over the top dip
        { 0.0,  -0.99, 1},  //Just over the bottom tip (0, -1)
        { 0.0,  -1.01, 0},  //Under the tip
        { 1.13,  0.6,  1},  //Near widest point (y ~ 0.6)
        { 1.15,  0.6,  0},  //Past widest point
        { 1.13,  0.3,  0},  //Same x lower down, boundary curves in
        { 0.6,   1.2,  1},  //Upper lobe
        { 0.6,   1.25, 0},  //Over the lobe (box top is 1.23666)
        {-0.6,   1.2,  1},  //Mirror of the lobe point
        { 0.9,  -0.6,  0},  //Lower right corner
    };
    int ok = 1;
    for (size_t i = 0; i < sizeof pts / sizeof pts[0]; i++) {
        int got = in_heart(pts[i][0], pts[i][1]);
        if (got != (int)pts[i][2]) {
            printf("    mismatch at (%.2f, %.2f): got %d\n", pts[i][0], pts[i][1], got);
            ok = 0;
        }
    }
    check(ok, "hand-picked points inside/outside");

    //Should be symmetric in x
    rng_t g;
    rng_seed64(&g, 99);
    int sym = 1;
    for (int i = 0; i < 1000000; i++) {
        double x = BOX_X0 + BOX_W * rng_u01(&g);
        double y = BOX_Y0 + BOX_H * rng_u01(&g);
        if (in_heart(x, y) != in_heart(-x, y)) {
            sym = 0;
        }
    }
    check(sym, "symmetric under x -> -x");

    //Nothing on the box edges can be inside, otherwise the box cuts off part of the heart
    //Bottom edge touches the tip at x = 0, skip a tiny gap there
    const long M = 1000000;
    long hits = 0;
    for (long i = 0; i <= M; i++) {
        double s = (double)i / M;
        double x = BOX_X0 + BOX_W * s;
        double y = BOX_Y0 + BOX_H * s;
        hits += in_heart(x, BOX_Y1);
        hits += in_heart(BOX_X0, y);
        hits += in_heart(BOX_X1, y);
        if (fabs(x) > 1e-6) {
            hits += in_heart(x, BOX_Y0);
        }
    }
    check(hits == 0, "no point on the box edges is inside (box contains the heart)");
}

//Serial plain MC end to end + timing
static void test_serial_mc(void) {
    printf("Serial plain Monte Carlo (sanity + speed)\n");
    const long N = 20000000;
    rng_t g;
    rng_seed_stream(&g, 530, 1, 1, (uint64_t)N, 0, 0);

    double t0 = now_sec();
    long hits = 0;
    for (long i = 0; i < N; i++) {
        double x = BOX_X0 + BOX_W * rng_u01(&g);
        double y = BOX_Y0 + BOX_H * rng_u01(&g);
        hits += in_heart(x, y);
    }
    double dt = now_sec() - t0;

    double A_hat = BOX_B * (double)hits / N;
    double q = A_REF / BOX_B;
    double sigma = BOX_B * sqrt(q * (1.0 - q) / N);
    double z = (A_hat - A_REF) / sigma;
    printf("    A_hat = %.6f, A_ref = %.6f, rel err = %.2e, z = %+.2f\n",
           A_hat, A_REF, fabs(A_hat - A_REF) / A_REF, z);
    printf("    %.2f ns/point (= t0 estimate for this machine)\n", 1e9 * dt / N);
    check(fabs(z) < 4.0, "estimate within 4 sigma of reference area");
}

int main(void) {
    test_rng_known_answers();
    test_rng_stats();
    test_rng_streams();
    test_in_heart();
    test_serial_mc();
    printf("\n%s (%d failure%s)\n", n_fail ? "SOME TESTS FAILED" : "ALL TESTS PASSED",
           n_fail, n_fail == 1 ? "" : "s");
    return n_fail ? 1 : 0;
}