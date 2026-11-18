//Monte Carlo area of the heart, plain vs stratified, MPI
//Usage: mpirun -np P ./heart -n N_PER_CORE [-m m1|m2v|m2h|m3|all] [-t TRIALS] [-s SEED] [-o CSV]
#include "decomp.h"
#include "heart.h"
#include "kernel.h"
#include "rng.h"

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    long n;          //Points per core
    int trials;
    int method;      //0 = all
    long seed;
    const char *csv;
} opts_t;

static int parse_args(int argc, char **argv, opts_t *o) {
    o->n = 0; o->trials = 30; o->method = 0; o->seed = 530;
    o->csv = "results/results.csv";

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        int has_val = (i + 1 < argc);
        if (!strcmp(a, "-n") && has_val) {
            o->n = atol(argv[++i]);
        } else if (!strcmp(a, "-t") && has_val) {
            o->trials = atoi(argv[++i]);
        } else if (!strcmp(a, "-s") && has_val) {
            o->seed = atol(argv[++i]);
        } else if (!strcmp(a, "-o") && has_val) {
            o->csv = argv[++i];
        } else if (!strcmp(a, "-m") && has_val) {
            const char *m = argv[++i];
            o->method = strcmp(m, "all") ? method_from_str(m) : 0;
            if (o->method < 0) {
                return -1;
            }
        } else {
            return -1;
        }
    }
    //n >= 2 since sigma uses n - 1
    if (o->n < 2 || o->trials < 1) {
        return -1;
    }
    return 0;
}

//Running sums for the per method summary printed at the end
typedef struct {
    double err, err2, sig, t_total;
    int count;
} summary_t;

//One method, all trials. Returns 0, or -1 if the method can't run on this p
static int run_method(int method, const opts_t *o, int rank, int p,
                      long long *all_hits, FILE *csv, summary_t *sum) {
    box_t b;
    if (subbox(method, rank, p, &b) != 0) {
        return -1; //Same answer on every rank, so nobody gets stuck in a collective
    }
    const long long N = (long long)p * o->n;

    for (int t = 0; t < o->trials; t++) {
        rng_t g;
        rng_seed_stream(&g, (uint64_t)o->seed, (uint64_t)method, (uint64_t)p,
                        (uint64_t)o->n, (uint64_t)t, rank); //Not timed

        MPI_Barrier(MPI_COMM_WORLD); //Everyone starts together
        double ta = MPI_Wtime();
        long long hits = count_hits(&b, o->n, &g);
        double tb = MPI_Wtime();

        MPI_Barrier(MPI_COMM_WORLD); //So the gather time is just comm, not waiting on the slowest rank
        double tc = MPI_Wtime();
        MPI_Gather(&hits, 1, MPI_LONG_LONG, all_hits, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
        double td = MPI_Wtime();

        //Worst + best compute, worst gather, worst total (outside the timed part)
        double t_max_in[3] = {tb - ta, td - tc, td - ta};
        double t_max[3];
        double t_comp_min;
        MPI_Reduce(t_max_in, t_max, 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&t_max_in[0], &t_comp_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

        //Only rank 0 has the counts
        if (rank == 0) {
            est_t e;
            combine(all_hits, p, o->n, &e);
            double rel_err = fabs(e.A_hat - A_REF) / A_REF;
            double sig_rel = e.sig_hat / A_REF;

            fprintf(csv, "%s,%d,%ld,%lld,%d,%.12f,%.6e,%.6e,%d,%.6e,%.6e,%.6e,%.6e\n",
                    method_name(method), p, o->n, N, t, e.A_hat, rel_err, sig_rel,
                    e.n_mixed, t_max[0], t_comp_min, t_max[1], t_max[2]);

            sum->err += rel_err;
            sum->err2 += rel_err * rel_err;
            sum->sig += sig_rel;
            sum->t_total += t_max[2];
            sum->count++;
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    opts_t o;
    if (parse_args(argc, argv, &o) != 0) {
        if (rank == 0) {
            fprintf(stderr, "usage: %s -n N_PER_CORE [-m m1|m2v|m2h|m3|all] [-t TRIALS] [-s SEED] [-o CSV]\n",
                    argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    //Rank 0 holds the gathered counts + writes the csv
    long long *all_hits = NULL;
    FILE *csv = NULL;
    if (rank == 0) {
        all_hits = malloc((size_t)p * sizeof *all_hits);
        csv = fopen(o.csv, "a");
        if (!all_hits || !csv) {
            fprintf(stderr, "can't open %s (does the folder exist?)\n", o.csv);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        //Header only if the file is new
        fseek(csv, 0, SEEK_END);
        if (ftell(csv) == 0) {
            fprintf(csv, "method,p,n,N,trial,A_hat,rel_err,sig_rel,n_mixed,"
                         "t_comp_max,t_comp_min,t_gather,t_total\n");
        }
        printf("p = %d, n = %ld per core, N = %lld, %d trials, seed %ld -> %s\n",
               p, o.n, (long long)p * o.n, o.trials, o.seed, o.csv);
        printf("%-4s %12s %12s %14s %12s\n", "meth", "mean err", "sd err", "mean pred sig", "avg time s");
    }

    for (int m = 1; m <= N_METHODS; m++) {
        if (o.method != 0 && o.method != m) {
            continue;
        }
        summary_t sum = {0};
        if (run_method(m, &o, rank, p, all_hits, csv, &sum) != 0) {
            if (rank == 0) {
                printf("%-4s skipped (p = %d not a perfect square)\n", method_name(m), p);
            }
            continue;
        }
        if (rank == 0) {
            double mean = sum.err / sum.count;
            double sd = sum.count > 1
                ? sqrt((sum.err2 - sum.count * mean * mean) / (sum.count - 1)) : 0.0;
            printf("%-4s %12.4e %12.4e %14.4e %12.4f\n", method_name(m), mean, sd,
                   sum.sig / sum.count, sum.t_total / sum.count);
            fflush(csv);
        }
    }

    if (rank == 0) {
        fclose(csv);
        free(all_hits);
    }
    MPI_Finalize();
    return 0;
}