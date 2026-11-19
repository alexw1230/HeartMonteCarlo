//Monte Carlo area of the heart, plain vs stratified, MPI
//Usage: mpirun -np P ./heart -n N_PER_CORE [-m m1|m2v|m2h|m3|all] [-t TRIALS] [-s SEED] [-o CSV] [-r RANK_CSV] [-w WARMUP_SEC]
#define _GNU_SOURCE //For sched_getcpu()
#include "decomp.h"
#include "heart.h"
#include "kernel.h"
#include "rng.h"

#include <math.h>
#include <mpi.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    long n;          //Points per core
    int trials;
    int method;      //0 = all
    long seed;
    const char *csv;
    const char *rank_csv; //NULL = don't write per rank times
    double warm;          //Seconds of untimed sampling before the trials
} opts_t;

static int parse_args(int argc, char **argv, opts_t *o) {
    o->n = 0; o->trials = 30; o->method = 0; o->seed = 530;
    o->csv = "results/results.csv";
    o->rank_csv = NULL;
    o->warm = 0.0;

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
        } else if (!strcmp(a, "-r") && has_val) {
            o->rank_csv = argv[++i];
        } else if (!strcmp(a, "-w") && has_val) {
            o->warm = atof(argv[++i]);
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

//Current clock of a core in MHz from sysfs, -1 if the node doesn't expose it
static int read_mhz(int cpu) {
    char path[96];
    snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", cpu);
    FILE *f = fopen(path, "r");
    if (!f) {
        return -1;
    }
    long khz = -1;
    if (fscanf(f, "%ld", &khz) != 1) {
        khz = -1;
    }
    fclose(f);
    return khz > 0 ? (int)(khz / 1000) : -1;
}

//Run the sampling loop for a few seconds so every core is at full clock before timing
//Own rng stream, so the trial results are the same with or without it
static void warm_up(double seconds, int rank) {
    if (seconds <= 0.0) {
        return;
    }
    box_t b;
    subbox(M1, 0, 1, &b);
    rng_t g;
    rng_seed64(&g, 0xC0FFEEULL + (uint64_t)rank);
    volatile long long sink = 0; //Keep the compiler from dropping the work
    double t0 = MPI_Wtime();
    while (MPI_Wtime() - t0 < seconds) {
        sink += count_hits(&b, 100000, &g);
    }
    (void)sink;
    MPI_Barrier(MPI_COMM_WORLD);
}

//Running sums for the per method summary printed at the end
typedef struct {
    double err, err2, sig, t_total;
    int count;
} summary_t;

//Everything a run needs besides the options
typedef struct {
    int rank, p;
    long long *all_hits; //Rank 0 only, p entries
    double *all_tcomp;   //Rank 0 only, p entries
    int *all_cpu;        //Rank 0 only, p entries
    int *all_mhz;        //Rank 0 only, p entries
    FILE *csv;           //Rank 0 only
    FILE *rank_csv;      //Rank 0 only, NULL if not asked for
} ctx_t;

//One method, all trials. Returns 0, or -1 if the method can't run on this p
static int run_method(int method, const opts_t *o, const ctx_t *c, summary_t *sum) {
    const int rank = c->rank, p = c->p;
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
        MPI_Gather(&hits, 1, MPI_LONG_LONG, c->all_hits, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
        double td = MPI_Wtime();
        int cpu = sched_getcpu(); //Core this rank ran on
        int mhz = o->rank_csv ? read_mhz(cpu) : -1; //Clock right after the trial

        //Worst + best compute, worst gather, worst total (outside the timed part)
        double t_max_in[3] = {tb - ta, td - tc, td - ta};
        double t_max[3];
        double t_comp_min;
        MPI_Reduce(t_max_in, t_max, 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&t_max_in[0], &t_comp_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        if (o->rank_csv) {
            MPI_Gather(&t_max_in[0], 1, MPI_DOUBLE, c->all_tcomp, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gather(&cpu, 1, MPI_INT, c->all_cpu, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Gather(&mhz, 1, MPI_INT, c->all_mhz, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }

        //Only rank 0 has the counts
        if (rank == 0) {
            est_t e;
            combine(c->all_hits, p, o->n, &e);
            double rel_err = fabs(e.A_hat - A_REF) / A_REF;
            double sig_rel = e.sig_hat / A_REF;

            fprintf(c->csv, "%s,%d,%ld,%lld,%d,%.12f,%.6e,%.6e,%d,%.6e,%.6e,%.6e,%.6e\n",
                    method_name(method), p, o->n, N, t, e.A_hat, rel_err, sig_rel,
                    e.n_mixed, t_max[0], t_comp_min, t_max[1], t_max[2]);
            if (c->rank_csv) {
                for (int r = 0; r < p; r++) {
                    fprintf(c->rank_csv, "%s,%d,%ld,%d,%d,%d,%lld,%.6e,%d\n", method_name(method), p,
                            o->n, t, r, c->all_cpu[r], c->all_hits[r], c->all_tcomp[r],
                            c->all_mhz[r]);
                }
            }

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
            fprintf(stderr, "usage: %s -n N_PER_CORE [-m m1|m2v|m2h|m3|all] [-t TRIALS] [-s SEED] [-o CSV] [-r RANK_CSV] [-w WARMUP_SEC]\n",
                    argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    //Rank 0 holds the gathered values + writes the csvs
    ctx_t c = {rank, p, NULL, NULL, NULL, NULL, NULL, NULL};
    if (rank == 0) {
        c.all_hits = malloc((size_t)p * sizeof *c.all_hits);
        c.all_tcomp = malloc((size_t)p * sizeof *c.all_tcomp);
        c.all_cpu = malloc((size_t)p * sizeof *c.all_cpu);
        c.all_mhz = malloc((size_t)p * sizeof *c.all_mhz);
        c.csv = fopen(o.csv, "a");
        if (!c.all_hits || !c.all_tcomp || !c.all_cpu || !c.all_mhz || !c.csv) {
            fprintf(stderr, "can't open %s (does the folder exist?)\n", o.csv);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        //Header only if the file is new
        fseek(c.csv, 0, SEEK_END);
        if (ftell(c.csv) == 0) {
            fprintf(c.csv, "method,p,n,N,trial,A_hat,rel_err,sig_rel,n_mixed,"
                           "t_comp_max,t_comp_min,t_gather,t_total\n");
        }
        if (o.rank_csv) {
            c.rank_csv = fopen(o.rank_csv, "a");
            if (!c.rank_csv) {
                fprintf(stderr, "can't open %s (does the folder exist?)\n", o.rank_csv);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            fseek(c.rank_csv, 0, SEEK_END);
            if (ftell(c.rank_csv) == 0) {
                fprintf(c.rank_csv, "method,p,n,trial,rank,cpu,hits,t_comp,mhz\n");
            }
        }
        printf("p = %d, n = %ld per core, N = %lld, %d trials, seed %ld, warm up %.1f s -> %s\n",
               p, o.n, (long long)p * o.n, o.trials, o.seed, o.warm, o.csv);
        printf("%-4s %12s %12s %14s %12s\n", "meth", "mean err", "sd err", "mean pred sig", "avg time s");
    }

    //Warm up the gather once, the first call sets up connections and takes ~100x longer
    long long dummy = 0;
    MPI_Gather(&dummy, 1, MPI_LONG_LONG, c.all_hits, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    warm_up(o.warm, rank);

    for (int m = 1; m <= N_METHODS; m++) {
        if (o.method != 0 && o.method != m) {
            continue;
        }
        summary_t sum = {0};
        if (run_method(m, &o, &c, &sum) != 0) {
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
            fflush(c.csv);
            if (c.rank_csv) {
                fflush(c.rank_csv);
            }
        }
    }

    if (rank == 0) {
        fclose(c.csv);
        if (c.rank_csv) {
            fclose(c.rank_csv);
        }
        free(c.all_hits);
        free(c.all_tcomp);
        free(c.all_cpu);
        free(c.all_mhz);
    }
    MPI_Finalize();
    return 0;
}