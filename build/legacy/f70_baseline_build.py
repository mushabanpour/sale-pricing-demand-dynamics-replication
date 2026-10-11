#!/usr/bin/env python3
"""Build the estimation-ready dataset for baselines 1-3.  ONE script, one output file.

    market      store x week
    reference   median of the prior k=4 OBSERVED weeks, per (store, alternative)
    sale        (p_reg - p) / p_reg > 0.05
    baseline1   regime = <t-1><t>        4 states, NN omitted   -- backward pair
    baseline2   regime = <t><t+1>        4 states, NN omitted   -- forward pair
    baseline3   regime = <t-1><t><t+1>   8 states, NNN omitted  -- both neighbors

SAMPLE.  Default tag is `est`, the HOLDOUT draw: 580,071 rows against the dev sample's
244,172, and disjoint in stores (f3_draw.py --exclude).  That is where the extra
observations come from -- baseline1's thinnest regime x rung cell goes 114 -> 160 and the
count of cells under 300 rows falls from 3 to 1 of 24.  COST, stated plainly: 57 DMA
clusters rather than 83, because 26 DMAs carry <= 2 stores in the frame and excluding the
dev stores empties them (f3_draw.py:172-178).  Clustered inference is on 57 groups here.

WHAT THE EXTRA ROWS DO NOT FIX.  baseline3 at regime x rung is 8 x 6 cells, and rung 36
is rare in the frame itself: SSS x 36 holds 16 rows on `est` and 28 on `dev`.  No sample
size available here populates it, so that cell is not interpretable at any significance
level.  Eight of 48 baseline3 cells sit under 300 rows.

INSTRUMENTS are computed ONCE, on the full shares>0 & s0_inside>0 frame, BEFORE any
regime filtering, so all four columns carry identical instrument values.  f61 computed
them after filtering, which made them depend on the column's own sample; that is fine for
a single column and wrong for a table meant to be read across columns.

    python3 code/f70_baseline_build.py            # est (holdout), the default
    python3 code/f70_baseline_build.py --tag dev
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
K, CUT = 4, 0.05
KEEP = ["market_ids", "alt", "store_code_uc", "w", "prices", "shares", "s0_inside",
        "n_cans", "log_n_eq", "is_can", "brand_id", "year_month", "dma_code", "z1",
        "income"]


def prior_median(P, O, k):
    S = np.stack([np.where(np.pad(O[:, :-l], ((0, 0), (l, 0)), constant_values=False),
                           np.pad(P[:, :-l], ((0, 0), (l, 0)), constant_values=np.nan), np.nan)
                  for l in range(1, k + 1)], axis=2)
    with np.errstate(all="ignore"):
        return np.nanmedian(S, axis=2)


def main(tag="est"):
    d = pd.read_parquet(ROOT / "data_work" / f"f3_{tag}.parquet", columns=KEEP)
    d = d[(d.shares > 0) & (d.s0_inside > 0)].copy()
    print(f"f3_{tag}: {len(d):,} rows  {d.market_ids.nunique():,} markets  "
          f"{d.dma_code.nunique()} DMAs  {d.store_code_uc.nunique()} stores")

    # ---- sale flag and the three regimes, all off one price panel --------------------
    piv = d.pivot_table(index=["store_code_uc", "alt"], columns="w", values="prices")
    P = piv.to_numpy(np.float64); O = np.isfinite(P)
    ref = prior_median(P, O, K)
    ok = O & np.isfinite(ref)
    with np.errstate(all="ignore"):
        sale = np.where(ok, (ref - P) / np.where(ref > 0, ref, np.nan) > CUT, False)
    lag_o = np.pad(ok[:, :-1], ((0, 0), (1, 0)), constant_values=False)
    lag_s = np.pad(sale[:, :-1], ((0, 0), (1, 0)), constant_values=False)
    led_o = np.pad(ok[:, 1:], ((0, 0), (0, 1)), constant_values=False)
    led_s = np.pad(sale[:, 1:], ((0, 0), (0, 1)), constant_values=False)
    L = lambda X: np.where(X, "S", "N").astype("U1")
    defs = {
        "reg1": (np.char.add(L(lag_s), L(sale)),                  ok & lag_o),
        "reg2": (np.char.add(L(sale), L(led_s)),                  ok & led_o),
        "reg3": (np.char.add(np.char.add(L(lag_s), L(sale)), L(led_s)), ok & lag_o & led_o),
    }
    for nm, (arr, msk) in defs.items():
        R = (pd.DataFrame(np.where(msk, arr, ""), index=piv.index, columns=piv.columns)
               .stack().rename(nm).reset_index())
        d = d.merge(R, on=["store_code_uc", "alt", "w"], how="left")
        d[nm] = d[nm].fillna("")
    for nm, arr in (("sale", np.where(ok, sale, np.nan)),
                    ("p_reg", np.where(ok, ref, np.nan))):
        T = (pd.DataFrame(arr, index=piv.index, columns=piv.columns)
               .stack(dropna=False).rename(nm).reset_index())
        d = d.merge(T, on=["store_code_uc", "alt", "w"], how="left")
    # depth = (p_reg - p) / p_reg, the fraction off the reference; 0 when at or above it
    d["depth"] = np.where(d.p_reg > 0, (d.p_reg - d.prices) / d.p_reg, np.nan)

    # ---- instruments, ONCE, on the common frame --------------------------------------
    g = d.groupby("market_ids")
    d["sq_dev_lsz"] = g.log_n_eq.transform(
        lambda x: ((x.values[:, None] - x.values[None, :]) ** 2).sum(1))
    d["sum_lsz"] = g.log_n_eq.transform("sum") - d.log_n_eq
    d["sum_can"] = g.is_can.transform("sum").astype(float) - d.is_can
    aw = d.groupby(["alt", "w"]).prices.agg(asum="sum", an="size").reset_index()
    dw = d.groupby(["alt", "w", "dma_code"]).prices.agg(dsum="sum", dn="size").reset_index()
    d = d.merge(aw, on=["alt", "w"], how="left").merge(
        dw, on=["alt", "w", "dma_code"], how="left")
    num, den = d.asum - d.dsum, d.an - d.dn
    zh = np.where(den > 0, num / den.replace(0, np.nan), np.nan)
    fb = np.where(d.an > 1, (d.asum - d.prices) / (d.an - 1).replace(0, np.nan), np.nan)
    d["z_haus"] = pd.Series(zh).fillna(pd.Series(fb)).fillna(d.asum / d.an).to_numpy()
    d = d.drop(columns=["asum", "an", "dsum", "dn"])

    # ---- ALTERNATIVE instruments -----------------------------------------------------
    # sum_lsz and sum_can are sums over the REST of the choice set, so they are count-like
    # in exactly the sense entry 38 flagged: s_0 is other beer in the same store, so
    # anything counting products in the market correlates with xi.  Their moments are the
    # ones that fail -- sum_can t -2.84 on dev and -2.09 on est, sum_lsz -2.45 on est.
    # Replacements, both already sanctioned by the project:
    #   z1_x_inc, zh_x_inc  instrument x demeaned zip3 income, the parent's configuration
    #                       E, recorded in entry 39 as "not counts"
    #   sq_dev_can          the Gandhi-Houde differentiation IV on the can dummy, built
    #                       like sq_dev_lsz, which is the one choice-set moment that holds
    d["inc_d"] = d.income - d.income.mean()
    d["z1_x_inc"] = d.z1 * d.inc_d
    d["zh_x_inc"] = d.z_haus * d.inc_d
    d["sq_dev_can"] = d.groupby("market_ids").is_can.transform(
        lambda x: ((x.values[:, None] - x.values[None, :]) ** 2).sum(1)).astype(float)

    # ---- nested-logit within-nest share: MW's single inside nest ----------------------
    d["sjg"] = d.shares / d.groupby("market_ids").shares.transform("sum")
    d["clustering_ids"] = d.dma_code
    d = d.sort_values(["market_ids", "alt"]).reset_index(drop=True)

    for nm, lab in (("reg1", "baseline1 (t-1,t)"), ("reg2", "baseline2 (t,t+1)"),
                    ("reg3", "baseline3 (t-1,t,t+1)")):
        u = d[d[nm] != ""]
        t = pd.crosstab(u[nm], u.n_cans)
        print(f"\n{lab}: usable {len(u):,} rows, {u.market_ids.nunique():,} markets, "
              f"{u.dma_code.nunique()} DMAs")
        print("  " + t.to_string().replace("\n", "\n  "))
        print(f"  thinnest cell {int(t.values.min()):,}   "
              f"cells < 300 rows: {int((t.values < 300).sum())} of {t.size}")
    out = ROOT / "data_work" / f"f70_baseline_{tag}.parquet"
    d.to_parquet(out, index=False)
    print(f"\nwrote {out.name}  ({len(d):,} rows, {len(d.columns)} cols)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="est")
    main(**vars(ap.parse_args()))
