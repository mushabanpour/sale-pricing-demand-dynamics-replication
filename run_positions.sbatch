#!/bin/bash
#SBATCH --job-name=lognormal_pos
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=positions_%j.log
# Table 5A column (iv), lognormal price coefficient, on the other three position samples
# (data/positions/pos2..pos4: every fourth store in store-code order starting from the 2nd,
# 3rd and 4th store; rules otherwise as pos1).  Each runs in positions/posN/ with its own
# copy of code/; 
#     sbatch run_positions.sbatch      or     nohup bash run_positions.sbatch > positions.log 2>&1 &
module load anaconda3/2024.06 2>/dev/null || true
set -u
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
for d in "$PWD" "${SLURM_SUBMIT_DIR:-}" "$(dirname "$0")"; do
  if [ -n "$d" ] && [ -f "$d/code/f71_baseline_table.py" ]; then cd "$d"; break; fi
done
[ -f code/f71_baseline_table.py ] || { echo "run this from the replication package folder"; exit 1; }
P="$PWD"
step() { echo "[$(date '+%H:%M:%S')] $*"; }
for i in 2 3 4; do
  r=positions/pos$i
  mkdir -p $r/code $r/data $r/logs $r/estimation
  cp code/*.py $r/code/
  ln -sf "$P/data/acs_zip3_demographics_2016_2019.csv" $r/data/
  ln -sf "$P/data/positions/pos$i/regression ready.parquet" $r/data/
  (cd $r && python3 code/prepare_data.py) > $r/logs/prepare.log 2>&1 || { step "prepare_data failed in $r"; exit 1; }
done
step "start: column (iv), lognormal, positions 2-4"
for i in 2 3 4; do
  ( cd positions/pos$i && if python3 code/f71_baseline_table.py --col iv --tag long > logs/col_iv.log 2>&1
    then step "pos$i done"; else step "pos$i FAILED: see positions/pos$i/logs/col_iv.log"; fi ) &
done
wait
step "ALL DONE -- positions/pos2..pos4/estimation/"
