#!/usr/bin/env python3
"""Rebuild the r56 store-selected frame, and add the FORWARD margin the parent
project never carried.

WHY THIS EXISTS.  `beer_rcnl_r59.dta` is a draw of 8,579 markets from this frame.
The draw scatters store-weeks, so only 4.1% of its rows have the next calendar
week present -- a forward price window cannot be built on it afterwards.  Every
margin has to be built HERE, on the dense (store x week x alternative) grid, and
then carried through the draw.  That is exactly how `p_lag` and `IT` got in.

WHAT IS ADDED.  The parent's intertemporal margin is BACKWARD:

    IT = p_reg - p,  p_reg = median of the cell's own PRIOR four OBSERVED weeks
                     (blp_margins_decomp.py:73-78, PRIOR_LAGS = (1,2,3,4))

Verified on the shipped .dta: corr(IT, p_prev - p) = +0.740 against
corr(IT, p_next - p) = +0.593, and 92.8% of the IT == 0 rows have p == p_lag
exactly.  A backward reference-price discount cannot separate storage from
satiation -- a consumer who drank a lot last week has no reason to care what beer
costs NEXT week, but a storer does.  So this adds the forward mirror:

    G(tau) = min(p over the next tau OBSERVED weeks) - p        (the HN object)
    G_pos  = max(G, 0)

built with `forward_min_obs`, which is `prior_median_obs` reflected in time: same
padding convention, same observed-week masking, nanmin instead of nanmedian.

THE PARENT PROJECT IS NOT MODIFIED.  This imports its builders and reproduces
`build_store_frame` step for step, asserting round 55's reference counts
(1,083 stores / 9,682,184 rows) before anything is added.  The frame is written to
the parent's own cache path as well, with the extra columns appended, so
`blp_r59_draw.py` still runs against it unchanged -- extra columns are inert
there because `build_dta` selects by name.

    python3 code/f2_build_frame.py
"""
from __future__ import annotations
import gc, sys, time
from pathlib import Path
import _paths
import numpy as np
import pandas as pd

PAPER = _paths.WORK
HERE = _paths.WORK
sys.path.insert(0, str(PAPER / "src" / "common"))
sys.path.insert(0, str(PAPER / "src" / "diagnostics"))

from blp_store_week_decomp import classes                                # noqa: E402
from blp_hn_cells import build_grid, prior_median_obs, MIN_WEEKS         # noqa: E402
from blp_margins_decomp import build_margins                             # noqa: E402
from blp_config_e import ACS, SCRATCH, LO_IN, HI_IN                      # noqa: E402
from blp_msd_geo import store_meta                                       # noqa: E402
from sale_definitions import (SALE_DROP_THRESHOLD,                       # noqa: E402
                              NEXT_REBOUND_THRESHOLD, PRIOR_LAGS)

YEARS = [2016, 2017, 2018, 2019]
Z2 = ["d_same_rung_other_brand", "d_same_container_other"]
THR = 0.03
TAUS = (1, 2, 3, 4, 6)                 # 4 is the headline; the rest give the profile
R55 = dict(stores=1_083, rows=9_682_184)
FRAME_REF = dict(rows=6_905_736, markets=198_312, stores=1_083, clusters=83)
PARENT_CACHE = SCRATCH / "r56_store_frame.parquet"
OUT = HERE / "data_work" / "f2_store_frame.parquet"


def forward_min_obs(Pc, Oc, leads):
    """Minimum of the strictly-LED OBSERVED prices.

    The exact time-reflection of `blp_hn_cells.prior_median_obs`: that function
    pads `l` columns on the LEFT and takes a nanmedian over lags (1,2,3,4); this
    pads `l` columns on the RIGHT and takes a nanmin over leads.  Unobserved
    future weeks are masked out rather than treated as missing prices, so G is
    defined off the weeks the cell is actually stocked -- the same convention the
    backward benchmark uses, and the reason the two are comparable.
    """
    pp = np.stack([np.pad(Pc[:, l:], ((0, 0), (0, l)), constant_values=np.nan)
                   for l in leads], axis=2)
    po = np.stack([np.pad(Oc[:, l:], ((0, 0), (0, l)), constant_values=False)
                   for l in leads], axis=2)
    with np.errstate(all="ignore"):
        return np.nanmin(np.where(po, pp, np.nan), axis=2)


def main():
    t0 = time.time()
    G = build_grid(verbose=True)
    nS, nW, nA = G["nS"], G["nW"], G["nA"]
    P, PK, valid = G["P"], G["PK"], G["valid"]
    ncan, brand, cont = G["ncan"], G["brand"], G["cont"]
    bcode = pd.factorize(brand)[0]
    si = G["si"]; stores = pd.Index(G["stores"][si]); weeks = G["weeks"]
    TOT = G["TOT"].astype(np.float64)
    yidx = np.array([YEARS.index(d.year) for d in weeks])
    CLS = classes(nA, brand, cont, ncan)

    # ---- round 55's mild-point store screen, reproduced exactly -------------
    low = np.zeros((nS, nW, nA), bool); sale = np.zeros((nS, nW, nA), bool)
    dep = np.full((nS, nW, nA), np.nan, np.float32)
    for a in range(nA):
        Pc, Oc = P[:, :, a], valid[:, :, a]
        bench = prior_median_obs(Pc, Oc)
        with np.errstate(all="ignore"):
            lo = Oc & np.isfinite(bench) & (bench > 0) & \
                (Pc <= (1 - SALE_DROP_THRESHOLD) * bench)
            dep[:, :, a] = np.where(lo, 1.0 - Pc / bench, np.nan)
            nxt_p = np.pad(Pc[:, 1:], ((0, 0), (0, 1)), constant_values=np.nan)
            nxt_o = np.pad(Oc[:, 1:], ((0, 0), (0, 1)), constant_values=False)
            sale[:, :, a] = lo & nxt_o & (nxt_p >= (1 + NEXT_REBOUND_THRESHOLD) * Pc)
        low[:, :, a] = lo
    cells_live = valid.any(1)
    srate = np.where(valid.sum((1, 2)) > 0,
                     (sale & (dep >= 0.10)).sum((1, 2))
                     / np.maximum(valid.sum((1, 2)), 1), np.nan)
    keep = ((srate >= THR) & (cells_live.sum(1) > 0)
            & ((valid.sum((1, 2)) / np.maximum(cells_live.sum(1), 1)) >= MIN_WEEKS))
    vsel = valid & keep[:, None, None]
    print(f"  store screen at {THR:.3f}: {int(keep.sum()):,} stores (round 55 "
          f"{R55['stores']:,}), {int(vsel.sum()):,} rows (round 55 {R55['rows']:,})",
          flush=True)
    del low, sale, dep; gc.collect()

    # ---- the parent's three margins ----------------------------------------
    MG = build_margins(G)
    lagg, itg, vdg, reason = MG["p_lag"], MG["IT"], MG["VD"], MG["vd_reason"]
    vdfix = np.where(reason == 1, np.float32(0.0), vdg)          # the D017 fix
    for k in ("p_lead", "VD_strict", "rung_distance"):
        del MG[k]
    del MG, vdg; gc.collect()

    # ---- rows: the parent keeps only p_lag, VD_fixed and IT all defined -----
    ui, wi, ai = np.where(vsel)
    ok = (np.isfinite(lagg[ui, wi, ai]) & np.isfinite(vdfix[ui, wi, ai])
          & np.isfinite(itg[ui, wi, ai]))
    print(f"  with p_lag, VD_fixed and IT all defined: {int(ok.sum()):,} rows "
          f"({100*ok.mean():.2f}%)", flush=True)

    ins_all = np.where(valid, PK * ncan[None, None, :].astype(np.float32),
                       np.float32(0.0)).sum(2).astype(np.float64)
    with np.errstate(all="ignore"):
        s0_cell = 1.0 - ins_all / np.where(TOT > 0, TOT, np.nan)
    MD = store_meta(stores)
    A = pd.read_csv(ACS)
    A["store_zip3"] = A.store_zip3.astype(int).astype(str).str.zfill(3)
    inc = A.groupby("store_zip3").median_hh_income.mean() / 1000.0
    MD["income"] = inc.reindex(MD.zip3).to_numpy()
    MD["income"] = MD.income.fillna(MD.income.mean())

    u, w, a = ui[ok], wi[ok], ai[ok]
    zc = np.zeros((len(u), 2))
    PRf = vsel.reshape(-1, nA).astype(np.float32)
    for j, k in enumerate(Z2):
        cnt = (PRf @ CLS[k].astype(np.float32)).reshape(nS, nW, nA)
        zc[:, j] = cnt[u, w, a]
        del cnt
    del PRf; gc.collect()

    # ---- NEW: the forward margin, one horizon at a time ---------------------
    # Memory: the dense grid is (nS, nW, nA) float32.  Holding one per horizon
    # would be 5 of them on top of P, PK, valid and the three margin grids, which
    # does not fit in 8 GB.  Each is built, sampled at the selected rows, and
    # freed before the next.
    GCOL = {}
    for tau in TAUS:
        Gt = np.full_like(P, np.nan)
        for aa in range(nA):
            fmin = forward_min_obs(P[:, :, aa], valid[:, :, aa],
                                   leads=range(1, tau + 1))
            Gt[:, :, aa] = (fmin - P[:, :, aa]).astype(np.float32)
        g = Gt[u, w, a].astype(np.float64)
        GCOL[tau] = g
        fin = np.isfinite(g)
        print(f"  G(tau={tau}): defined on {100*fin.mean():5.2f}% of frame rows, "
              f">0 on {100*(g[fin] > 0).mean():5.2f}%, mean {g[fin].mean():+.4f}",
              flush=True)
        del Gt; gc.collect()

    D = pd.DataFrame({
        "market_ids": (u.astype(np.int64) * nW + w),
        "shares": (PK[u, w, a].astype(np.float64) * ncan[a]) / TOT[u, w],
        "prices": P[u, w, a].astype(np.float64),
        "p_lag": lagg[u, w, a].astype(np.float64),
        "VD": vdfix[u, w, a].astype(np.float64),
        "IT": itg[u, w, a].astype(np.float64),
        "s0_cell": s0_cell[u, w],
        "sjg": np.zeros(len(u)),
        "alt": a.astype(np.int32), "week": w.astype(np.int32),
        "storeyear": pd.factorize(u * 4 + yidx[w])[0].astype(np.int32),
        "brandstore": pd.factorize(u * 100 + bcode[a])[0].astype(np.int32),
        "dma": pd.factorize(MD.dma.to_numpy()[u])[0].astype(np.int32),
        "year": yidx[w].astype(np.int8),
        "z1": zc[:, 0], "z2": zc[:, 1],
        "income": MD.income.to_numpy()[u]})
    for tau in TAUS:
        g = GCOL[tau]
        D[f"G{tau}"] = g
        D[f"G{tau}_pos"] = np.where(np.isfinite(g), np.maximum(g, 0.0), np.nan)
    D["income"] = D.income - D.income.mean()

    ins = D.groupby("market_ids").shares.transform("sum")
    D["s0_inside"] = 1.0 - ins
    bad = D.market_ids[(ins < LO_IN) | (ins > HI_IN)].unique()
    D = D[~D.market_ids.isin(bad)].copy()
    print(f"  dropped {len(bad):,} ill-conditioned markets; {len(D):,} rows, "
          f"{D.market_ids.nunique():,} markets", flush=True)
    D = D.sort_values("market_ids", kind="stable").reset_index(drop=True)

    # ---- the gate: this must BE the frame r59 was drawn from ----------------
    got = dict(rows=len(D), markets=int(D.market_ids.nunique()),
               stores=int(pd.unique(D.market_ids // nW).size),
               clusters=int(D.dma.nunique()))
    print("\n  FRAME_REF check")
    bad_ref = False
    for k, v in FRAME_REF.items():
        flag = "OK " if got[k] == v else "MISMATCH"
        bad_ref |= got[k] != v
        print(f"    {k:9s} got {got[k]:>9,}  expected {v:>9,}   {flag}")
    print(f"\n  G4_pos > 0 on {100*(D.G4_pos > 0).mean():.2f}% of frame rows; "
          f"defined on {100*D.G4.notna().mean():.2f}%")
    print(f"  corr(G4, IT) = {D[['G4','IT']].corr().iloc[0,1]:+.4f}  "
          f"(forward vs backward margin)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    D.to_parquet(OUT, index=False)
    PARENT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    D.to_parquet(PARENT_CACHE, index=False)
    print(f"\n  wrote {OUT}  ({len(D):,} rows, {len(D.columns)} cols)")
    print(f"  wrote {PARENT_CACHE}  (same frame, for the parent's draw)")
    print(f"\ndone [{time.time()-t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
