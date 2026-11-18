#!/bin/bash
#SBATCH --job-name=heart_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=00:10:00
#SBATCH -p short-96core
#SBATCH --output=smoke_%j.out

#Quick checks of the MPI build before the real runs (~1 min)

module purge
module load gcc/13.2.0
module load openmpi/gcc13.2/4.1.6

set -euo pipefail

export OMPI_MCA_coll_hcoll_enable=0

cd "$SLURM_SUBMIT_DIR"

make -f Make/Makefile clean && make -f Make/Makefile heart

RES=results/smoke_${SLURM_JOB_ID}
mkdir -p "$RES"

MPIRUN="mpirun --bind-to core --map-by core"

#1: Same seeds as the serial dry run in decomp_tests -> A_hat has to match it exactly
#Expected rel err: m1 2.54e-05, m2v 1.76e-04, m2h 1.88e-04, m3 1.83e-04
echo "== 1: match serial dry run (p = 16, n = 10^6, 1 trial)"
$MPIRUN -np 16 ./heart -n 1000000 -t 1 -o "$RES/match.csv"

#2: Small version of the real runs, p = 4 and 16, every method
for P in 4 16; do
    echo "== 2: p = $P, n = 360000, 30 trials"
    $MPIRUN -np $P ./heart -n 360000 -t 30 -o "$RES/smoke.csv"
done

#3: Same run twice -> same estimates (timing columns will differ)
echo "== 3: repeatability"
$MPIRUN -np 16 ./heart -n 100000 -t 5 -o "$RES/rep1.csv" > /dev/null
$MPIRUN -np 16 ./heart -n 100000 -t 5 -o "$RES/rep2.csv" > /dev/null
if diff <(cut -d, -f1-9 "$RES/rep1.csv") <(cut -d, -f1-9 "$RES/rep2.csv") > /dev/null; then
    echo "repeat: identical"
else
    echo "repeat: DIFFERENT"
fi

#4: M3 on a p that isn't a square should skip, not hang
echo "== 4: m3 on p = 8"
$MPIRUN -np 8 ./heart -n 1000 -t 1 -m m3 -o "$RES/bad.csv"

echo "done, csvs in $RES"