#!/usr/bin/env python3
"""Four samples, one per position in rule 5: every fourth store in store-code order, starting
from the 1st, 2nd, 3rd or 4th store (U[0::4] ... U[3::4]).  Rules 1-11 as listed in the README;
market size = all malt-beverage servings in the store-week (the paper's definition).

    python3 build/build_positions.py      # -> build/data/pos1/ ... build/data/pos4/
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_foodall as B

NW = B.NW


def main():
    z = np.load(B.WORK / "storeweek_totals.npz", allow_pickle=True)
    U = B.universe(z)                                   # rules 3-4, store-code order
    full = pq.read_table(B.WORK / "frame_foodall.parquet")
    u_all = pc.divide(full.column("market_ids"), NW)
    for o in range(4):
        q = U[o::4]
        B.log(f"--- position {o + 1}: rule 5 keeps {len(q):,} stores")
        frame = B.WORK / f"frame_pos{o + 1}.parquet"
        t = full.filter(pc.is_in(u_all, value_set=pa.array(q.astype(np.int64))))
        pq.write_table(t, frame)
        B.log(f"rules 6-10: {len(np.unique(t.column('market_ids').to_numpy() // NW)):,} stores")
        B.FRAME = frame
        B.OUT = HERE / "data" / f"pos{o + 1}" / "regression ready.parquet"
        B.PER_DMA, B.DMA_ORDER = 5, "first"
        B.sample()


if __name__ == "__main__":
    main()
