#!/bin/bash
# Where column (iv)'s Hansen J comes from (by DMA, by instrument, by eigen-direction), at the
# lognormal estimates of subsample 1 (estimation/) and 2-4 (positions/posN/estimation/).
# One contraction per subsample, no optimization; about 20 minutes, the four in parallel.
#     nohup bash run_jdiag.sh > logs/jdiag_all.log 2>&1 &
module load anaconda3/2024.06 2>/dev/null || true
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
cd "$(cd "$(dirname "$0")" && pwd)"
mkdir -p logs
python3 code/f71_jdiag.py --tag long > logs/jdiag.log 2>&1 &
for i in 2 3 4; do
  cp code/f71_jdiag.py positions/pos$i/code/
  mkdir -p positions/pos$i/logs
  (cd positions/pos$i && python3 code/f71_jdiag.py --tag long > logs/jdiag.log 2>&1) &
done
wait
for f in logs/jdiag.log positions/pos{2,3,4}/logs/jdiag.log; do
  echo "===== $f"; grep -v "^  states\|^  reg3\|^  demographics" "$f"
done
