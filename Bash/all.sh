#!/bin/bash
#SBATCH --job-name=heart_all
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH -p short-96core
#SBATCH --output=run_all_%j.out

#Full experiment grid for Problem 4.2 (~6-8 min)
#Tasks A1-C3: p = 16, 36, 64, total N fixed at 5.76e6 / 5.76e7 / 5.76e8, n = N / p per core
#Plus p = 1 at the same N for t0 + speedup (m1 only, every method costs the same)

module purge
module load gcc/13.2.0
module load openmpi/gcc13.2/4.1.6

set -euo pipefail

#Don't even try to open hcoll (not installed on the nodes, just spams warnings)
export OMPI_MCA_coll=^hcoll

cd "$SLURM_SUBMIT_DIR"

make -f Make/Makefile clean && make -f Make/Makefile heart

RES=results/final_${SLURM_JOB_ID}
mkdir -p "$RES"

#Clock histogram of every core from /proc/cpuinfo (count, MHz)
mhz_snapshot() {
    awk -F: '/cpu MHz/ {printf "%d\n", int($2 / 100) * 100}' /proc/cpuinfo | sort -n | uniq -c
}

#Run info for the README
{
    echo "job $SLURM_JOB_ID on $(hostname), $(date)"
    echo "commit $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
    lscpu | grep -E "Model name|Socket\(s\)|Core\(s\) per socket|Thread\(s\) per core|CPU max MHz"
    echo "governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)"
    echo "boost $(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo n/a)"
    module list 2>&1
} | tee "$RES/run_info.txt"

MPIRUN="mpirun --bind-to core --map-by core"
WARM=5
TRIALS=30
NS=(5760000 57600000 576000000)

#1: Tasks A (p = 16), B (p = 36), C (p = 64, bonus), every method
for P in 16 36 64; do
    for N in "${NS[@]}"; do
        n=$((N / P))
        echo "== p = $P, n = $n, N = $N"
        $MPIRUN -np $P ./heart -n $n -t $TRIALS -w $WARM \
            -o "$RES/results.csv" -r "$RES/ranks.csv"
    done
done

#2: One core baseline, same N
for N in "${NS[@]}"; do
    echo "== p = 1, n = N = $N (m1 only)"
    $MPIRUN -np 1 ./heart -n $N -t $TRIALS -w $WARM -m m1 \
        -o "$RES/results.csv" -r "$RES/ranks.csv"
done

echo "== clocks right after (count, MHz)"
mhz_snapshot

#Every (method, p, n) should have exactly 30 rows: 4 methods x 9 tasks + 3 baselines = 39 groups
echo "== row check"
awk -F, 'NR > 1 {c[$1 "," $2 "," $3]++} END {
    bad = 0
    for (k in c) {
        if (c[k] != 30) {
            print "  " k ": " c[k] " rows"
            bad++
        }
    }
    print "  " length(c) " groups, " bad " with the wrong row count"
}' "$RES/results.csv"

echo "done, csvs in $RES"