#!/bin/bash
#SBATCH --job-name=heart_diag2
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH -p short-96core
#SBATCH --output=diag2_%j.out

#Clock scaling check (~1 min): same runs with and without a 5 s warm up
#Each config gets its own csv pair so they're easy to compare

module purge
module load gcc/13.2.0
module load openmpi/gcc13.2/4.1.6

set -euo pipefail

#Don't even try to open hcoll (not installed on the nodes, just spams warnings)
export OMPI_MCA_coll=^hcoll

cd "$SLURM_SUBMIT_DIR"

make -f Make/Makefile clean && make -f Make/Makefile heart

RES=results/diag2_${SLURM_JOB_ID}
mkdir -p "$RES"

#Clock histogram of every core from /proc/cpuinfo (count, MHz)
mhz_snapshot() {
    awk -F: '/cpu MHz/ {printf "%d\n", int($2 / 100) * 100}' /proc/cpuinfo | sort -n | uniq -c
}

echo "== node"
hostname
lscpu | grep -E "Model name|Thread\(s\) per core|CPU max MHz|CPU min MHz"
CF=/sys/devices/system/cpu/cpu0/cpufreq
for f in scaling_driver scaling_governor scaling_min_freq scaling_max_freq cpuinfo_max_freq; do
    echo "$f: $(cat $CF/$f 2>/dev/null || echo n/a)"
done
echo "boost: $(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo n/a)"
echo "== idle clocks (count, MHz)"
mhz_snapshot

MPIRUN="mpirun --bind-to core --map-by core"

#1: one core, no warm up vs 5 s warm up
for W in 0 5; do
    echo "== p = 1, n = 3.6*10^7, 5 trials, warm up $W s"
    $MPIRUN -np 1 ./heart -n 36000000 -t 5 -w $W \
        -o "$RES/p1_w$W.csv" -r "$RES/p1_w${W}_ranks.csv"
done

#2: 64 cores, smoke test size, no warm up vs 5 s warm up
#Grab a clock snapshot 3 s in, while the warm up is running
for W in 0 5; do
    echo "== p = 64, n = 360000, 30 trials, warm up $W s"
    if [ "$W" -gt 0 ]; then
        (sleep 3; mhz_snapshot > "$RES/mhz_during_warmup.txt") &
    fi
    $MPIRUN -np 64 ./heart -n 360000 -t 30 -w $W \
        -o "$RES/p64_w$W.csv" -r "$RES/p64_w${W}_ranks.csv"
    wait
done
if [ -f "$RES/mhz_during_warmup.txt" ]; then
    echo "== clocks 3 s into the p = 64 warm up (count, MHz)"
    cat "$RES/mhz_during_warmup.txt"
fi

#3: 64 cores, 10x longer trials, warmed up
echo "== p = 64, n = 3.6*10^6, 10 trials, warm up 5 s"
$MPIRUN -np 64 ./heart -n 3600000 -t 10 -w 5 \
    -o "$RES/p64_big_w5.csv" -r "$RES/p64_big_w5_ranks.csv"

echo "== idle clocks after (count, MHz)"
mhz_snapshot

echo "done, csvs in $RES"