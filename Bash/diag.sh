#!/bin/bash
#SBATCH --job-name=heart_diag
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH -p short-96core
#SBATCH --output=diag_%j.out

#Timing diagnostics on a whole node to ourselves (~2 min)
#Checks whether the 4.4 vs 5.9 ns/point split from the smoke test goes away with --exclusive

module purge
module load gcc/13.2.0
module load openmpi/gcc13.2/4.1.6

set -euo pipefail

#Don't even try to open hcoll (not installed on the nodes, just spams warnings)
export OMPI_MCA_coll=^hcoll

cd "$SLURM_SUBMIT_DIR"

make -f Make/Makefile clean && make -f Make/Makefile heart

RES=results/diag_${SLURM_JOB_ID}
mkdir -p "$RES"

#Node info for the performance section
echo "== node"
hostname
lscpu | grep -E "Model name|Socket\(s\)|Core\(s\) per socket|Thread\(s\) per core|NUMA node\(s\)|CPU max MHz|L3"
echo "cpus slurm gave us: $SLURM_CPUS_ON_NODE"

#Bindings get printed to the log, one line per rank
MPIRUN="mpirun --bind-to core --map-by core --report-bindings"

#1: t0 on one core of a compute node
echo "== 1: p = 1, n = 3.6*10^7, 5 trials (t0)"
$MPIRUN -np 1 ./heart -n 36000000 -t 5 -o "$RES/diag.csv" -r "$RES/ranks.csv"

#2-4: same size as the smoke test, now with the node to ourselves
for P in 4 16 64; do
    echo "== 2: p = $P, n = 360000, 30 trials"
    $MPIRUN -np $P ./heart -n 360000 -t 30 -o "$RES/diag.csv" -r "$RES/ranks.csv"
done

#5: longer compute per trial at full size
echo "== 3: p = 64, n = 3.6*10^6, 10 trials"
$MPIRUN -np 64 ./heart -n 3600000 -t 10 -o "$RES/diag.csv" -r "$RES/ranks.csv"

echo "done, csvs in $RES"
