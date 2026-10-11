#!/bin/bash
#SBATCH --job-name=lognormal_main
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=main_%j.log
# Every estimate behind Tables 2-12 and Figures 1-3 on the pos1 sample with the lognormal
# price coefficient.
#     sbatch run_main.sbatch                       # batch job, or inside a session:
#     nohup bash run_main.sbatch > main.log 2>&1 &
# To finish an interrupted run (completed steps are skipped, their results kept):
#     RESUME=1 sbatch run_main.sbatch
module load anaconda3/2024.06 2>/dev/null || true
set -u
export PYTHONUNBUFFERED=1
for d in "$PWD" "${SLURM_SUBMIT_DIR:-}" "$(dirname "$0")"; do
  if [ -n "$d" ] && [ -f "$d/code/f71_baseline_table.py" ]; then cd "$d"; break; fi
done
[ -f code/f71_baseline_table.py ] || { echo "run this from the replication package folder"; exit 1; }

RESUME=${RESUME:-0}
shopt -s nullglob; prev=(estimation/* results/* logs/*); shopt -u nullglob
if [ "$RESUME" != 1 ] && [ ${#prev[@]} -gt 0 ]; then
  old="previous_run_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$old"
  for x in estimation results logs; do [ -d $x ] && mv $x "$old/"; done
  echo "earlier outputs moved to $old/"
fi
mkdir -p logs estimation results
step() { echo "[$(date '+%H:%M:%S')] $*"; }
run() { local n=$1 lg=$2; shift 2
  if [ "$RESUME" = 1 ] && [ "$(cat "logs/$lg.status" 2>/dev/null)" = ok ]; then
    step "$n already done"; return 0; fi
  if "$@" > "logs/$lg.log" 2>&1; then step "$n done"; echo ok > "logs/$lg.status"
  else step "FAILED $n: see logs/$lg.log"; echo failed > "logs/$lg.status"; return 1; fi; }
# NumPy threads per process: three demand columns run at once on 24 cores
th() { local n=$1; shift; ( export OMP_NUM_THREADS=$n MKL_NUM_THREADS=$n OPENBLAS_NUM_THREADS=$n; "$@" ); }

run "prepare data" prepare python3 code/prepare_data.py || exit 1
read rows nst ndma <<< "$(python3 -c 'import pandas as pd; d = pd.read_parquet("data_work/f70_baseline_long.parquet", columns=["store_code_uc", "dma_code"]); print(len(d), d.store_code_uc.nunique(), d.dma_code.nunique())')"
step "start: $rows rows, $nst stores, $ndma DMAs"

# Each step starts as soon as its inputs exist:
#   reduced forms, column (i)          from the start
#   columns (ii), (iii), (iv)          from the start, in parallel; elasticities after each
#   column (v), supply -> counterfactuals -> {firm table, sales value (iv and v regimes),
#   rivals}   after (iv)
th 2 run "scanner reduced forms" rf_rms python3 code/RF_RMS.py &
th 1 run "household reduced forms" data_hms python3 code/Data_HMS.py &
th 1 run "column (i)" col_i python3 code/f71_baseline_table.py --col i --tag long &
for c in ii iii; do
  ( th 7 run "column ($c)" col_$c python3 code/f71_baseline_table.py --col $c --tag long &&
    th 7 run "elasticities ($c)" elast_$c python3 code/f85_alpha_by_rung.py $c ) &
done
( th 8 run "column (iv)" col_iv python3 code/f71_baseline_table.py --col iv --tag long &&
  { th 4 run "elasticities (iv)" elast_iv python3 code/f85_alpha_by_rung.py iv &
    th 4 run "brand elasticities (iv)" brand_iv python3 code/f86_brand_elast.py iv &
    th 4 run "column (v)" col_v python3 code/f71_fixtheta_spfe.py --tag long &
    ( th 8 run "supply" supply python3 code/f90_supply_mw.py iv &&
      th 8 run "counterfactuals" counter python3 code/f92_counterfactual.py iv &&
      { th 1 run "firm counterfactuals" counter_firm python3 code/f92_by_firm.py &
        th 6 run "sales at observed prices" sales python3 code/f98_sales_value.py &
        th 8 run "rivals re-optimizing" rivals python3 code/f99_solo_rivals.py &
        th 6 run "sales at observed prices, column (v) regimes" sales_lamv \
          python3 code/f98_sales_value.py --lam v &
        wait; } ) &
    wait; } ) &
wait

expected="prepare rf_rms data_hms col_i col_ii col_iii col_iv col_v elast_ii elast_iii elast_iv brand_iv supply counter counter_firm sales rivals sales_lamv"
missing=""; for x in $expected; do [ "$(cat logs/$x.status 2>/dev/null)" = ok ] || missing="$missing $x"; done
if [ -n "$missing" ]; then step "NOT COMPLETE, failed or not run:$missing"; exit 1; fi
step "ALL DONE -- estimation/ and results/ hold the estimates"
