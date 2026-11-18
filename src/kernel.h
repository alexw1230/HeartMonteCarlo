#ifndef KERNEL_H
#define KERNEL_H

#include "decomp.h"
#include "rng.h"

//Flops per sample: 2 mul (u01 scale) + 2 x (mul + add) to map into the box + 9 in_heart
#define FLOPS_PER_POINT 15

//Result of one trial, built on rank 0 from everyone's hit counts
typedef struct {
    double A_hat;   //Area estimate
    double sig_hat; //Predicted std of A_hat from this run's own sample variance
    int n_mixed;    //Strata with 0 < hits < n (cut by the heart boundary)
} est_t;

//Draw n points uniformly in b, count how many land in the heart
long long count_hits(const box_t *b, long n, rng_t *g);

//Combine p gathered hit counts (n points each, every stratum weight B/p)
//A_hat = sum (B/p) h_i / n
//sig_hat^2 = sum (B/p)^2 ph_i (1 - ph_i) / (n - 1)
//Same formula for every method, M1 just has every "stratum" be the whole box
void combine(const long long *hits, int p, long n, est_t *e);

#endif