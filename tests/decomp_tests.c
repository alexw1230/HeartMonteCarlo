//Serial tests for decomp.c, no MPI
//Run from repo root: make -f Make/Makefile test
#include "decomp.h"
#include "heart.h"
#include "rng.h"

#include <math.h>
#include <stdio.h>

static int n_fail = 0;

static void check(int ok, const char *name) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) {
        n_fail++;
    }
}

static const int METHODS[N_METHODS] = {M1, M2V, M2H, M3};
static const int PS[] = {1, 4, 16, 36, 64};
#define N_PS ((int)(sizeof PS / sizeof PS[0]))

//Names + bad input
static void test_basics(void) {
    printf("Names + bad input\n");
    int ok = 1;
    for (int k = 0; k < N_METHODS; k++) {
        int m = METHODS[k];
        if (method_from_str(method_name(m)) != m) {
            ok = 0;
        }
    }
    check(ok, "method name round trip (m1, m2v, m2h, m3)");
    check(method_from_str("m4") == -1, "unknown name -> -1");

    check(isqrt_exact(16) == 4 && isqrt_exact(36) == 6 && isqrt_exact(64) == 8,
          "isqrt_exact on 16, 36, 64");
    check(isqrt_exact(32) == -1 && isqrt_exact(0) == -1, "isqrt_exact rejects 32 and 0");

    box_t b;
    check(subbox(M3, 0, 32, &b) == -1, "M3 refuses p = 32 (not a square)");
    check(subbox(M2V, 16, 16, &b) == -1 && subbox(M2V, -1, 16, &b) == -1,
          "rank out of range refused");
    check(subbox(9, 0, 16, &b) == -1, "unknown method refused");
}

//Geometry for every method and p
static void test_geometry(void) {
    printf("Sub box geometry (p = 1, 4, 16, 36, 64)\n");
    int ok_full = 1, ok_inside = 1, ok_area = 1, ok_each = 1, ok_edges = 1;

    for (int k = 0; k < N_METHODS; k++) {
        int m = METHODS[k];
        for (int pi = 0; pi < N_PS; pi++) {
            int p = PS[pi];
            double total = 0.0;
            box_t prev = {0.0, 0.0, 0.0, 0.0};
            for (int r = 0; r < p; r++) {
                box_t b;
                if (subbox(m, r, p, &b) != 0) {
                    ok_full = 0;
                    continue;
                }
                //Inside the full box
                if (b.x0 < BOX_X0 || b.x1 > BOX_X1 || b.y0 < BOX_Y0 || b.y1 > BOX_Y1) {
                    ok_inside = 0;
                }
                //Every stratum has area B/p (M1: every rank has B)
                double want = (m == M1) ? BOX_B : BOX_B / p;
                if (fabs(box_area(&b) - want) > 1e-12 * want) {
                    ok_each = 0;
                }
                total += box_area(&b);
                //Neighbors share bit identical edges
                if (r > 0 && m == M2V && b.x0 != prev.x1) {
                    ok_edges = 0;
                }
                if (r > 0 && m == M2H && b.y0 != prev.y1) {
                    ok_edges = 0;
                }
                if (m == M3) {
                    int q = isqrt_exact(p);
                    if (r % q != 0 && b.x0 != prev.x1) {
                        ok_edges = 0;
                    }
                    if (r >= q) {
                        box_t below;
                        subbox(m, r - q, p, &below);
                        if (b.y0 != below.y1) {
                            ok_edges = 0;
                        }
                    }
                }
                prev = b;
            }
            double want_total = (m == M1) ? p * BOX_B : BOX_B;
            if (fabs(total - want_total) > 1e-12 * want_total) {
                ok_area = 0;
            }
        }
    }
    check(ok_full, "subbox() succeeds for every method, p, rank");
    check(ok_inside, "every sub box inside the full box");
    check(ok_each, "every stratum has area B/p (M1: B)");
    check(ok_area, "areas add up to B (M1: p*B)");
    check(ok_edges, "neighbors share bit identical edges");
}

//Random points in the full box land in exactly one stratum -> no gaps, no overlaps
static void test_tiling(void) {
    printf("Tiling (10^5 random points + box corners, every stratified method and p)\n");
    int ok = 1;
    rng_t g;
    rng_seed64(&g, 7);
    for (int k = 1; k < N_METHODS; k++) { //Skip M1, its boxes overlap on purpose
        int m = METHODS[k];
        for (int pi = 0; pi < N_PS; pi++) {
            int p = PS[pi];
            if (m == M3 && isqrt_exact(p) < 0) {
                continue;
            }
            box_t boxes[64];
            for (int r = 0; r < p; r++) {
                subbox(m, r, p, &boxes[r]);
            }
            for (int i = 0; i < 100004; i++) {
                double x, y;
                if (i < 4) {
                    //Corners, incl. the outer upper edges
                    x = (i & 1) ? BOX_X1 : BOX_X0;
                    y = (i & 2) ? BOX_Y1 : BOX_Y0;
                } else {
                    x = BOX_X0 + BOX_W * rng_u01(&g);
                    y = BOX_Y0 + BOX_H * rng_u01(&g);
                }
                int count = 0;
                for (int r = 0; r < p; r++) {
                    count += box_contains(&boxes[r], x, y);
                }
                if (count != 1) {
                    printf("    %s p=%d: (%.6f, %.6f) in %d strata\n", method_name(m), p, x, y, count);
                    ok = 0;
                }
            }
        }
    }
    check(ok, "every point in exactly one stratum");
}

//Mapping u in [0,1) into a sub box always lands inside it
static void test_mapping(void) {
    printf("Mapping uniforms into sub boxes\n");
    int ok = 1;
    rng_t g;
    rng_seed64(&g, 11);
    for (int r = 0; r < 64; r++) {
        box_t b;
        subbox(M3, r, 64, &b);
        for (int i = 0; i < 10000; i++) {
            double x = b.x0 + (b.x1 - b.x0) * rng_u01(&g);
            double y = b.y0 + (b.y1 - b.y0) * rng_u01(&g);
            if (x < b.x0 || x > b.x1 || y < b.y0 || y > b.y1) {
                ok = 0;
            }
        }
    }
    check(ok, "mapped points stay in their sub box (M3, p = 64)");
}

//Serial stand in for a p rank run: loop over ranks instead of MPI, same seeding as the real thing
static void test_serial_run(void) {
    printf("Serial stand in for a parallel run (p = 16, n = 10^6 per rank)\n");
    const int p = 16;
    const long n = 1000000;
    const double q = A_REF / BOX_B;
    const double sig_plain = BOX_B * sqrt(q * (1.0 - q) / ((double)p * n)); //Plain MC std

    int ok = 1;
    for (int k = 0; k < N_METHODS; k++) {
        int m = METHODS[k];
        double A_hat = 0.0;
        double var_hat = 0.0;
        for (int r = 0; r < p; r++) {
            box_t b;
            subbox(m, r, p, &b);
            rng_t g;
            rng_seed_stream(&g, 530, (uint64_t)m, (uint64_t)p, (uint64_t)n, 0, r);
            long hits = 0;
            for (long i = 0; i < n; i++) {
                double x = b.x0 + (b.x1 - b.x0) * rng_u01(&g);
                double y = b.y0 + (b.y1 - b.y0) * rng_u01(&g);
                hits += in_heart(x, y);
            }
            //Each rank's estimate weighted by B/p (M1: average of p, same thing)
            double ph = (double)hits / n;
            A_hat += (BOX_B / p) * ph;
            var_hat += (BOX_B / p) * (BOX_B / p) * ph * (1.0 - ph) / (n - 1);
        }
        double sig_hat = sqrt(var_hat);
        double z = (A_hat - A_REF) / sig_hat;
        printf("    %-3s A_hat = %.6f  rel err = %.2e  predicted sigma/A = %.2e (plain %.2e)  z = %+.2f\n",
               method_name(m), A_hat, fabs(A_hat - A_REF) / A_REF, sig_hat / A_REF,
               sig_plain / A_REF, z);
        if (fabs(z) > 4.0 || sig_hat > 1.05 * sig_plain) {
            ok = 0;
        }
    }
    check(ok, "every method within 4 predicted sigma, no method worse than plain");
}

int main(void) {
    test_basics();
    test_geometry();
    test_tiling();
    test_mapping();
    test_serial_run();
    printf("\n%s (%d failure%s)\n", n_fail ? "SOME TESTS FAILED" : "ALL TESTS PASSED",
           n_fail, n_fail == 1 ? "" : "s");
    return n_fail ? 1 : 0;
}