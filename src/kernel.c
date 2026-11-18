#include "kernel.h"
#include "heart.h"

#include <math.h>

long long count_hits(const box_t *b, long n, rng_t *g) {
    //Pull the box into locals so the loop only touches registers
    const double x0 = b->x0, wx = b->x1 - b->x0;
    const double y0 = b->y0, wy = b->y1 - b->y0;
    long long hits = 0;
    for (long i = 0; i < n; i++) {
        double x = x0 + wx * rng_u01(g);
        double y = y0 + wy * rng_u01(g);
        hits += in_heart(x, y);
    }
    return hits;
}

void combine(const long long *hits, int p, long n, est_t *e) {
    const double w = BOX_B / p; //Stratum weight
    double A = 0.0, var = 0.0;
    int mixed = 0;
    for (int i = 0; i < p; i++) {
        double ph = (double)hits[i] / n;
        A += w * ph;
        var += w * w * ph * (1.0 - ph) / (n - 1);
        if (hits[i] > 0 && hits[i] < n) {
            mixed++;
        }
    }
    e->A_hat = A;
    e->sig_hat = sqrt(var);
    e->n_mixed = mixed;
}