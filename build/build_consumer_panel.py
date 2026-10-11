#!/usr/bin/env python3
"""The household sample (Table 4, Figure 3): Homescan beer purchases of the fourteen brands'
6- to 36-packs of 12-ounce cans and bottles, in the scanner sample's window.

Input: purchases_with_trips_2016_2019.dta, the Homescan purchases of every beer UPC merged
with their trips, written by build/HMScleaningdo.do from the raw Homescan files and the UPC
lookup of build/legacy/build_exact_pack_container_beer_panel.py.

    python3 build/build_consumer_panel.py /path/to/purchases_with_trips_2016_2019.dta
        -> data/consumer panel.parquet
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNGS = (6, 12, 18, 24, 30, 36)
COLS = ["household_code", "purchase_date", "panel_year", "n_cans", "oz_per_pack", "quantity",
        "deal_flag_uc", "is_focal_brand", "in_target_pack", "known_container",
        "total_price_paid", "brand_descr", "store_code_uc"]


def window():
    """The calendar span of the scanner sample (pos1), scanner week 65 = 2017-04-01."""
    w = pd.read_parquet(ROOT / "data" / "regression ready.parquet", columns=["w"]).w
    d0 = pd.Timestamp("2017-04-01")
    return d0 + pd.Timedelta(weeks=int(w.min()) - 65), d0 + pd.Timedelta(weeks=int(w.max()) - 65)


def main(src):
    w0, w1 = window()
    d = pd.read_stata(src, columns=COLS)
    n0 = len(d)
    d["purchase_date"] = pd.to_datetime(d.purchase_date)
    d = d[(d.purchase_date >= w0) & (d.purchase_date <= w1)]
    n1 = len(d)
    d = d[(d.in_target_pack == 1) & (d.known_container == 1) & d.n_cans.isin(RUNGS)]
    n2 = len(d)
    d = d[d.is_focal_brand == 1]
    d["oz"] = d.oz_per_pack * d.quantity
    d["deal"] = (d.deal_flag_uc == 1).astype(float)
    d["month"] = d.purchase_date.values.astype("datetime64[M]")
    d = d.sort_values(["household_code", "purchase_date"]).reset_index(drop=True)
    print(f"Homescan purchases {n0:,} -> window {w0.date()} to {w1.date()} {n1:,} -> "
          f"pack ladder {n2:,} -> fourteen brands {len(d):,}; "
          f"{d.household_code.nunique():,} households")
    out = ROOT / "data" / "consumer panel.parquet"
    d.to_parquet(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1])
