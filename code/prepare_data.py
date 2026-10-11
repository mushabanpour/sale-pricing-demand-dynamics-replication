#!/usr/bin/env python3
"""Splits data/regression ready.parquet into the working files the scripts read.

    -> data_work/f70_baseline_long.parquet   the store-week-product panel
       data_work/f38_market_size.parquet     market size by store-week
       data_work/store_zip3_all.parquet      each store's zip3 (ACS demographics)
       data_work/f91_store_chain.parquet     each store's parent code (Table 8)
"""
from _lib import DATA, DW
import pandas as pd

d = pd.read_parquet(DATA / "regression ready.parquet")
d.drop(columns=["mkt_size", "zip3", "parent"]).to_parquet(
    DW / "f70_baseline_long.parquet", index=False)
d[["market_ids", "mkt_size"]].drop_duplicates("market_ids").to_parquet(
    DW / "f38_market_size.parquet", index=False)
s = d[["store_code_uc", "zip3", "parent"]].drop_duplicates("store_code_uc")
s[["store_code_uc", "zip3"]].to_parquet(DW / "store_zip3_all.parquet", index=False)
s[["store_code_uc", "parent"]].to_parquet(DW / "f91_store_chain.parquet", index=False)
print(f"{len(d):,} rows, {d.market_ids.nunique():,} store-weeks, {len(s)} stores, "
      f"{s.parent.nunique()} chains -> {DW}")
