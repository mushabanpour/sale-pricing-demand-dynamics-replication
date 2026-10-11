#!/usr/bin/env python3
"""Round 59 Parts 1-3 and 6: the finalised estimation sample, and the deliverable.

One sample of about 300,000 rows drawn from the STORE-selected frame (>= 3% rebound-sale
rate; 1,083 stores, 6,905,736 rows, 198,312 markets), on which the RCNL is estimable, the
`VD` counterfactual runnable and the HMS micro moments formable.

**The draw is at the MARKET level, never the store level.**  Drawing stores would take the
DMA cluster count with it -- round 39 got F = 3.2e17 on 3 clusters from parent 184 -- while
markets are store-weeks, so every store in the frame keeps a positive probability of
appearing and the 83 DMAs survive.  The rule: a uniform random permutation of the 198,312
market ids, then the SHORTEST PREFIX whose cumulative row count reaches 300,000.

**Alternatives per market are held, not left to fall out of the row target.**  Every
alternative observed in a drawn market is kept -- the market's full choice set, 34.8 of
them on average.  D184 cell C is the reason this is not a free lever: restricting the store
panel to the cell panel's 76 alternatives moved `alpha` the wrong way and collapsed
sigma_IT to exactly 0.0000.  The row target is met by the MARKET COUNT alone.

**No seed survives into the deliverable.**  The draw is seeded here, but the drawn market
ids are carried in the `.dta` and in a CSV, so reading them back needs no RNG (D175).

    python3 paper/src/diagnostics/blp_r59_draw.py
    python3 paper/src/diagnostics/blp_r59_draw.py --target 200000 --no-stata
"""
from __future__ import annotations
import argparse, gc, hashlib, json, subprocess, sys, time
from datetime import datetime
from pathlib import Path
import _paths
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve()
PAPER = _paths.WORK
sys.path.insert(0, str(PAPER / "src" / "common"))
sys.path.insert(0, str(HERE.parent))
from run_manifest import RunManifest                                  # noqa: E402
import blp_first_stage_strength as M                                  # noqa: E402
from blp_hn_cells import store_set                                    # noqa: E402
from blp_config_e import ACS, SCRATCH                                 # noqa: E402
from blp_msd_geo import resid, store_meta                             # noqa: E402

OUT = PAPER / "outputs" / "blp_r59"
DTA = PAPER / "data" / "stata"
STATA = Path("/Applications/Stata/StataMP.app/Contents/MacOS/stata-mp")
FRAME = SCRATCH / "r56_store_frame.parquet"
NPZ = SCRATCH / "storeweek_totals.npz"
YEARS = [2016, 2017, 2018, 2019]
NW = 209
RUNGS = [6, 12, 18, 24, 30, 36]
IMPORTS = {"CORONA EXTRA", "CORONA LIGHT", "HEINEKEN", "MODELO ESPECIAL", "MODELO NEGRA"}
TARGET = 300_000
# a FIXED header timestamp: Stata's .dta header carries a creation time, so without this
# two identical builds differ in their file hash while carrying identical data
STAMP = datetime(2026, 9, 9, 0, 0, 0)
CLUSTER_FLOOR = 40
DRAWS = [1, 2, 3, 4]

# the frame, asserted before anything is drawn
FRAME_REF = dict(rows=6_905_736, markets=198_312, stores=1_083, clusters=83)
# round 55 D172's per-144-oz ladder on the 9,682,184-row mild point, servings-weighted
LADDER_REF = {6: 16.3332, 12: 12.1192, 18: 10.7211, 24: 9.0967, 30: 7.9998, 36: 7.5225}
# the cell-panel linear baseline the deliverable is measured against (round 49/51, D120)
REF = dict(alpha=-1.1103, se=0.1643, alpha_ols=-0.2847, swF=23.35, Jp=0.0724,
           epsM=-6.0582, pbar=10.1389, s0=0.5382, z1=-0.052097)


def content_sha256(F):
    """A hash of the DATA, invariant to the .dta header.  This is the one a replicator
    should check: it does not move when the file is rebuilt on another day."""
    h = hashlib.sha256()
    for c in F.columns:
        h.update(c.encode())
        s = F[c]
        if pd.api.types.is_numeric_dtype(s):
            h.update(np.ascontiguousarray(s.to_numpy(np.float64)).tobytes())
        else:
            h.update("\x1f".join(s.astype(str)).encode())
    return h.hexdigest()


def sha256(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def labels():
    """Store codes, week dates and the 116-alternative grid -- from the npz, no dense load."""
    z = np.load(NPZ, allow_pickle=True)
    si, _, _ = store_set(z)
    cells = z["cells"].astype(str)
    return dict(si=si, stores=z["stores"].astype(str)[si],
                weeks=pd.DatetimeIndex(pd.to_datetime(z["weeks"].astype(str))),
                cells=cells, TOT=z["tot_sv"][si].astype(np.float64),
                brand=np.array([c.split("|")[0] for c in cells]),
                cont=np.array([c.split("|")[1] for c in cells]),
                ncan=np.array([int(c.split("|")[2]) for c in cells]))


def load_frame(verbose=True):
    D = pd.read_parquet(FRAME)
    n, m = len(D), D.market_ids.nunique()
    u = D.market_ids.to_numpy() // NW
    if verbose:
        print(f"  store frame: {n:,} rows, {m:,} markets, {len(np.unique(u)):,} stores, "
              f"{D.dma.nunique()} DMA clusters", flush=True)
    assert n == FRAME_REF["rows"], n
    assert m == FRAME_REF["markets"], m
    assert len(np.unique(u)) == FRAME_REF["stores"]
    assert D.dma.nunique() == FRAME_REF["clusters"]
    return D


def draw_markets(counts, seed, target):
    """A uniform permutation of the markets, then the shortest prefix reaching `target`."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(counts.index.to_numpy())
    cum = counts.reindex(order).to_numpy().cumsum()
    k = int(np.searchsorted(cum, target) + 1)
    return np.sort(order[:k]), int(cum[k - 1])


def sample_report(S, L, tag):
    """Rows, markets, stores, DMAs, alternatives per market, rungs and the ladder."""
    u = S.market_ids.to_numpy() // NW
    a = S.alt.to_numpy()
    ncan, TOT = L["ncan"], L["TOT"]
    w = S.market_ids.to_numpy() % NW
    sv = S.shares.to_numpy() * TOT[u, w]
    rung = ncan[a]
    grid_alts = pd.Series(ncan).value_counts()
    lad, comp = {}, []
    for r in RUNGS:
        m = rung == r
        lad[r] = float(np.average(S.prices.to_numpy()[m], weights=sv[m])) if m.any() else None
        comp.append(dict(rung=r, rows=int(m.sum()),
                         row_share=float(m.mean()),
                         alts_present=int(pd.unique(a[m]).size),
                         alts_in_grid=int(grid_alts.get(r, 0)),
                         servings_weighted_price=lad[r],
                         ladder_ref=LADDER_REF[r],
                         ladder_gap_pct=(None if lad[r] is None else
                                         100 * (lad[r] - LADDER_REF[r]) / LADDER_REF[r])))
    rep = dict(tag=tag, rows=int(len(S)), markets=int(S.market_ids.nunique()),
               stores=int(np.unique(u).size), dmas=int(S.dma.nunique()),
               clusters=int(S.dma.nunique()),
               alts_per_market=float(len(S) / S.market_ids.nunique()),
               alts_present=int(S.alt.nunique()), alts_in_grid=len(ncan),
               weeks=int(pd.unique(w).size), rungs_present=int(pd.unique(rung).size),
               all_six_rungs=bool(pd.unique(rung).size == 6),
               sixpack_rows=int((rung == 6).sum()),
               sixpack_share=float((rung == 6).mean()),
               meets_cluster_floor=bool(S.dma.nunique() >= CLUSTER_FLOOR),
               composition=comp)
    return rep


def print_report(rep):
    print(f"  rows {rep['rows']:,}  markets {rep['markets']:,}  stores {rep['stores']:,}  "
          f"DMAs/clusters {rep['clusters']}  alts/market {rep['alts_per_market']:.2f}  "
          f"alts present {rep['alts_present']}/{rep['alts_in_grid']}", flush=True)
    print(f"  six rungs present: **{rep['all_six_rungs']}**   six-packs "
          f"{rep['sixpack_rows']:,} rows ({100*rep['sixpack_share']:.2f}%)   cluster floor "
          f"({CLUSTER_FLOOR}) met: **{rep['meets_cluster_floor']}**", flush=True)
    print(f"  rung   rows      share   alts present/grid   $/144oz   round-55 ref    gap",
          flush=True)
    for c in rep["composition"]:
        p = c["servings_weighted_price"]
        print(f"  {c['rung']:4d}  {c['rows']:8,d}  {100*c['row_share']:6.2f}%   "
              f"{c['alts_present']:3d} / {c['alts_in_grid']:3d}        "
              f"{p:8.4f}   {c['ladder_ref']:8.4f}  {c['ladder_gap_pct']:+6.2f}%",
              flush=True)


def linear(S, spec="baseline", reading="cell"):
    """The linear baseline.  `baseline` = price only on the two rival counts, the
    specification round 55's `estimate.do` reproduces; `configE` = the four-instrument,
    three-margin specification the nonlinear estimation actually uses."""
    s0 = (S.s0_cell if reading == "cell" else S.s0_inside).to_numpy(np.float64)
    sj = S.shares.to_numpy(np.float64)
    y = np.log(sj) - np.log(s0)
    pr = S.prices.to_numpy(np.float64)
    z1 = S.z1.to_numpy(np.float64); z2 = S.z2.to_numpy(np.float64)
    dd = S.income.to_numpy(np.float64)
    gr = [pd.factorize(S.alt)[0], pd.factorize(S.week)[0],
          pd.factorize(S.storeyear)[0], pd.factorize(S.brandstore)[0]]
    cl = pd.factorize(S.dma)[0]
    if spec == "baseline":
        A, mx = resid(np.column_stack([y, pr, z1, z2]), gr)
        yy, XX, ZZ = A[:, 0], A[:, [1]], A[:, 2:4]
        zn = ["z1_same_rung_other_brand", "z2_same_container_other"]
    else:
        A, mx = resid(np.column_stack([y, pr, S.p_lag, S.VD, S.IT,
                                       z1, z2, z1 * dd, z2 * dd]), gr)
        yy = A[:, 0]; XX = A[:, 1:5]
        ZZ = np.column_stack([A[:, 5:9], A[:, 2:5]])
        zn = ["z1", "z2", "z1_x_income", "z2_x_income"]
    b, V, u_, Gc = M.tsls(yy, XX, ZZ, cl)
    bo, Vo, _, _ = M.ols(yy, XX, cl)
    J, Jdf, Jp, _ = M.hansen_j(yy, XX, ZZ, cl)
    if spec == "baseline":
        f1 = M.sw_stats(XX[:, 0], np.empty((len(S), 0)), ZZ, cl)["sw_F"]
        fsX = XX[:, [0]]
        excl = ZZ
    else:
        # purge the three predetermined margins from price and from the excluded set
        W = XX[:, 1:]
        Pw = np.linalg.lstsq(W, np.column_stack([XX[:, [0]], A[:, 5:9]]), rcond=None)[0]
        Rz = np.column_stack([XX[:, [0]], A[:, 5:9]]) - W @ Pw
        f1 = M.sw_stats(Rz[:, 0], np.empty((len(S), 0)), Rz[:, 1:], cl)["sw_F"]
        fsX = Rz[:, [0]]
        excl = Rz[:, 1:]
    fs = []
    for j in range(excl.shape[1]):
        bb, VV, _, _ = M.ols(fsX[:, 0], excl[:, [j]], cl)
        fs.append(dict(instrument=zn[j], coef=float(bb[0]), se=float(np.sqrt(VV[0, 0])),
                       sign="+" if bb[0] > 0 else "-"))
    pbar = float(np.average(pr, weights=sj))
    s0med = float(np.median(s0))
    out = dict(spec=spec, reading=reading, rows=int(len(S)),
               markets=int(S.market_ids.nunique()), clusters=int(Gc),
               alpha=float(b[0]), se=float(np.sqrt(V[0, 0])), alpha_ols=float(bo[0]),
               swF=float(f1), J=float(J), Jdf=int(Jdf), Jp=float(Jp),
               pbar_rows=pbar, s0_median_rows=s0med, s0_mean_rows=float(np.mean(s0)),
               own_elast=float(np.median(b[0] * pr * (1 - sj))),
               eps_M_rows=float(b[0] * pbar * s0med),
               first_stages=fs, absorb_check=float(mx))
    if spec == "configE":
        for k, nm in enumerate(["p_lag", "VD_fixed", "IT"]):
            out[f"b_{nm}"] = float(b[1 + k]); out[f"se_{nm}"] = float(np.sqrt(V[1 + k, 1 + k]))
    del A; gc.collect()
    return out


def print_linear(w):
    print(f"  [{w['spec']:8s}/{w['reading']:6s}] **alpha {w['alpha']:+.6f} ({w['se']:.6f})** "
          f" OLS {w['alpha_ols']:+.6f}  **SW F {w['swF']:.4f}**  **J {w['J']:.4f}/{w['Jdf']} "
          f"p {w['Jp']:.4f}**  clusters {w['clusters']}", flush=True)
    print(f"             s_0 {w['s0_median_rows']:.4f}  p-bar {w['pbar_rows']:.4f}  own el "
          f"{w['own_elast']:+.3f}  **eps_M(rows) {w['eps_M_rows']:+.4f}**", flush=True)
    for f in w["first_stages"]:
        print(f"             first stage {f['instrument']:26s} {f['coef']:+.6f} "
              f"({f['se']:.6f})  sign **{f['sign']}**", flush=True)


def build_dta(S, L, py_base, py_cfg, target, out_dta):
    """The rectangular deliverable: every variable the estimation uses, sorted, no seed."""
    u = S.market_ids.to_numpy() // NW
    w = S.market_ids.to_numpy() % NW
    a = S.alt.to_numpy()
    stores, weeks, TOT = L["stores"], L["weeks"], L["TOT"]
    ncan, brand, cont, cells = L["ncan"], L["brand"], L["cont"], L["cells"]
    MD = store_meta(pd.Index(stores))
    A = pd.read_csv(ACS)
    A["store_zip3"] = A.store_zip3.astype(int).astype(str).str.zfill(3)
    inc = A.groupby("store_zip3").median_hh_income.mean() / 1000.0
    MD["income"] = inc.reindex(MD.zip3).to_numpy()
    MD["income"] = MD.income.fillna(MD.income.mean())
    yidx = np.array([YEARS.index(d.year) for d in weeks])
    servings = S.shares.to_numpy(np.float64) * TOT[u, w]
    F = pd.DataFrame({
        # ---- identifiers
        "market_id": (u.astype(np.int64) * NW + w + 1),
        "alt_id": (a + 1).astype(np.int16),
        "store_code_uc": stores[u],
        "week_end": np.array([d.strftime("%Y%m%d") for d in weeks])[w],
        "year": np.array([d.year for d in weeks], np.int16)[w],
        "month": np.array([d.month for d in weeks], np.int8)[w],
        "quarter": np.array([d.quarter for d in weeks], np.int8)[w],
        "dma_code": MD.dma.to_numpy()[u],
        "county_fips": MD.county.to_numpy()[u],
        "parent_code": MD.parent.to_numpy()[u],
        "retailer_code": MD.retailer.to_numpy()[u],
        "state": MD.state.to_numpy()[u],
        "zip3": MD.zip3.to_numpy()[u],
        # ---- product
        "brand": brand[a], "container": cont[a],
        "n_cans": ncan[a].astype(np.int16),
        "n_eq144": (ncan[a] / 12.0).astype(np.float32),
        "log_n_eq": np.log(ncan[a] / 12.0).astype(np.float32),
        "is_can": (cont[a] == "CAN").astype(np.int8),
        "is_import": np.isin(brand[a], list(IMPORTS)).astype(np.int8),
        # ---- quantities, prices, shares
        "servings": servings.astype(np.float32),
        "packs": (servings / ncan[a]).astype(np.float32),
        "qty_eq144": (servings / 12.0).astype(np.float32),
        "p_eq144": S.prices.to_numpy(np.float64).astype(np.float32),
        "p_pack": (S.prices.to_numpy() * ncan[a] / 12.0).astype(np.float32),
        "revenue": (S.prices.to_numpy() * servings / 12.0).astype(np.float32),
        "mkt_size_allbeer": TOT[u, w].astype(np.float32),
        "s_j": S.shares.to_numpy(np.float64),
        "s_0": S.s0_cell.to_numpy(np.float64),
        "s_0_inside": S.s0_inside.to_numpy(np.float64),
        # ---- margins (VD carries the D017 fix, D173)
        "p_lag": S.p_lag.to_numpy(np.float64),
        "VD_fixed": S.VD.to_numpy(np.float64),
        "IT": S.IT.to_numpy(np.float64),
        "p_ref4": (S.prices.to_numpy() + S.IT.to_numpy()).astype(np.float32),
        "p_smaller_rung": (S.prices.to_numpy() + S.VD.to_numpy()).astype(np.float32),
        # ---- instruments and demographics
        "z1_same_rung_other_brand": S.z1.to_numpy(np.float64),
        "z2_same_container_other": S.z2.to_numpy(np.float64),
        "income_demeaned": S.income.to_numpy(np.float64),
        "income_zip3": MD.income.to_numpy()[u].astype(np.float32),
    })
    F["z1_x_income"] = F.z1_same_rung_other_brand * F.income_demeaned
    F["z2_x_income"] = F.z2_same_container_other * F.income_demeaned
    F["J_rt"] = F.groupby("market_id").alt_id.transform("size").astype(np.int16)
    # ---- fixed-effect and cluster ids: exactly the groups PyBLP absorbs
    bcode = pd.factorize(brand)[0]
    F["fe_alt"] = (a + 1).astype(np.int32)
    F["fe_week"] = (w + 1).astype(np.int32)
    F["fe_store_year"] = (pd.factorize(u * 4 + yidx[w])[0] + 1).astype(np.int32)
    F["fe_brand_store"] = (pd.factorize(u * 100 + bcode[a])[0] + 1).astype(np.int32)
    F["fe_dma"] = (pd.factorize(MD.dma.to_numpy()[u])[0] + 1).astype(np.int32)
    F["fe_store"] = (pd.factorize(u)[0] + 1).astype(np.int32)
    F["logsj_over_s0"] = (np.log(F.s_j) - np.log(F.s_0)).astype(np.float64)
    F["logsj_over_s0_inside"] = (np.log(F.s_j) - np.log(F.s_0_inside)).astype(np.float64)
    F = F.sort_values(["market_id", "alt_id"], kind="stable").reset_index(drop=True)
    F.to_stata(out_dta, write_index=False, version=118, time_stamp=STAMP,
               data_label="round 59 estimation sample: store-selected, market-level draw")
    return F


DO_TEMPLATE = r"""*==============================================================================
* estimate_r59.do -- the linear baseline on the round-59 estimation sample
*
* {ROWS:,} rows, {MARKETS:,} markets, {STORES:,} stores, {CLUSTERS} DMA clusters, drawn at
* the MARKET level from the store-selected frame (6,905,736 rows / 198,312 markets).
* The file carries the drawn market ids AS DATA: no seed, no RNG, nothing to re-draw.
*
* Reproduces, to four significant figures, the Python numbers printed at the foot of
* this file.  Requires reghdfe, ivreghdfe, ivreg2, ftools.
* Run: stata-mp -b -q do estimate_r59.do
*==============================================================================
clear all
set more off
set linesize 200

use "{DTA}", clear
di as text "{{hline 78}}"
di as result "rows = " %12.0fc _N
qui levelsof market_id, local(mk)
di as result "markets = " %12.0fc `: word count `mk''
qui levelsof fe_dma, local(dm)
di as result "DMA clusters = " %6.0f `: word count `dm''
qui levelsof store_code_uc, local(st)
di as result "stores = " %8.0f `: word count `st''
di as text "{{hline 78}}"

gen double y = ln(s_j) - ln(s_0)
qui gen double _d = abs(y - logsj_over_s0)
qui summ _d
di as result "check |y(Stata) - logsj_over_s0(file)|: max " %12.3e r(max)
drop _d

* 1. the OLS benchmark
qui reghdfe y p_eq144, absorb(fe_alt fe_week fe_store_year fe_brand_store) ///
    vce(cluster fe_dma)
di as result "OLS   alpha_ols = " %12.6f _b[p_eq144] "  (" %12.6f _se[p_eq144] ")"
scalar b_ols = _b[p_eq144]
scalar se_ols = _se[p_eq144]

* 2. the two first stages, one instrument at a time
qui reghdfe p_eq144 z1_same_rung_other_brand, ///
    absorb(fe_alt fe_week fe_store_year fe_brand_store) vce(cluster fe_dma)
di as result "first stage z1_same_rung_other_brand = " %12.6f ///
    _b[z1_same_rung_other_brand] "  (" %12.6f _se[z1_same_rung_other_brand] ")"
scalar b_z1 = _b[z1_same_rung_other_brand]
scalar se_z1 = _se[z1_same_rung_other_brand]
qui reghdfe p_eq144 z2_same_container_other, ///
    absorb(fe_alt fe_week fe_store_year fe_brand_store) vce(cluster fe_dma)
di as result "first stage z2_same_container_other = " %12.6f ///
    _b[z2_same_container_other] "  (" %12.6f _se[z2_same_container_other] ")"
scalar b_z2 = _b[z2_same_container_other]
scalar se_z2 = _se[z2_same_container_other]

* 3. THE BASELINE: 2SLS, price on the two rival counts, four absorbed effect sets
qui ivreghdfe y (p_eq144 = z1_same_rung_other_brand z2_same_container_other), ///
    absorb(fe_alt fe_week fe_store_year fe_brand_store) cluster(fe_dma)
di as text "{{hline 78}}"
di as result "2SLS  alpha     = " %12.6f _b[p_eq144] "  (" %12.6f _se[p_eq144] ")"
di as result "Hansen J        = " %12.6f e(j) "   df = " %4.0f e(jdf) "   p = " %12.6f e(jp)
di as result "KP rk Wald F    = " %12.6f e(widstat)
scalar alpha = _b[p_eq144]
scalar se_alpha = _se[p_eq144]
scalar j_stat = e(j)
scalar j_p = e(jp)
scalar kp_f = e(widstat)

* 3b. what ivreghdfe actually estimated on, and the small-sample factor it applied.
*     reghdfe DROPS SINGLETON observations -- rows that are the only member of an
*     absorbed group, perfectly fitted by their own effect and contributing exactly
*     zero to the within moment -- so e(N) < _N by construction and `alpha` is
*     unaffected.  K = e(df_a) + e(rankxx) is what enters (N-1)/(N-K).
di as result "e(N) = " %12.0f e(N) "   e(df_a) = " %6.0f e(df_a) ///
    "   e(rankxx) = " %4.0f e(rankxx) "   e(N_clust) = " %5.0f e(N_clust)

* 4. the market elasticity, over ROWS (D120 as amended in round 56)
qui summ p_eq144 [aw=s_j]
scalar pbar = r(mean)
qui summ s_0, detail
scalar s0med = r(p50)
di as result "p-bar (share-weighted, over rows) = " %10.4f pbar
di as result "s_0 (median, over rows)           = " %10.4f s0med
di as result "eps_M = alpha x p-bar x s_0       = " %10.4f (alpha * pbar * s0med)

* 5. the specification the NONLINEAR estimation uses: three predetermined margins,
*    four excluded instruments.  `alpha` here is the linear counterpart of the RCNL's.
qui ivreghdfe y p_lag VD_fixed IT (p_eq144 = z1_same_rung_other_brand ///
    z2_same_container_other z1_x_income z2_x_income), ///
    absorb(fe_alt fe_week fe_store_year fe_brand_store) cluster(fe_dma)
di as text "{{hline 78}}"
di as result "configE alpha   = " %12.6f _b[p_eq144] "  (" %12.6f _se[p_eq144] ")"
di as result "        p_lag   = " %12.6f _b[p_lag] "  (" %12.6f _se[p_lag] ")"
di as result "        VD_fixed= " %12.6f _b[VD_fixed] "  (" %12.6f _se[VD_fixed] ")"
di as result "        IT      = " %12.6f _b[IT] "  (" %12.6f _se[IT] ")"
di as result "        Hansen J= " %12.6f e(j) "   df = " %4.0f e(jdf) "   p = " %12.6f e(jp)
scalar c_alpha = _b[p_eq144]
scalar c_se = _se[p_eq144]
scalar c_j = e(j)

* 6. the inside-good reading (D134)
gen double y_inside = ln(s_j) - ln(s_0_inside)
qui ivreghdfe y_inside (p_eq144 = z1_same_rung_other_brand z2_same_container_other), ///
    absorb(fe_alt fe_week fe_store_year fe_brand_store) cluster(fe_dma)
di as result "D134 inside reading: alpha = " %12.6f _b[p_eq144] "  (" %12.6f _se[p_eq144] ")"
scalar i_alpha = _b[p_eq144]

* 7. the same statistics at full precision, for the Python comparison.  The %12.6f
*    lines above are for reading; these are the ones the comparison parses.
di as text "{{hline 78}}"
di as result "FULLPREC alpha       " %20.12f alpha
di as result "FULLPREC se_alpha    " %20.12f se_alpha
di as result "FULLPREC alpha_ols   " %20.12f b_ols
di as result "FULLPREC se_alpha_ols" %20.12f se_ols
di as result "FULLPREC z1          " %20.12f b_z1
di as result "FULLPREC se_z1       " %20.12f se_z1
di as result "FULLPREC z2          " %20.12f b_z2
di as result "FULLPREC se_z2       " %20.12f se_z2
di as result "FULLPREC hansen_j    " %20.12f j_stat
di as result "FULLPREC hansen_p    " %20.12f j_p
di as result "FULLPREC kp_f        " %20.12f kp_f
di as result "FULLPREC pbar        " %20.12f pbar
di as result "FULLPREC s0_median   " %20.12f s0med
di as result "FULLPREC eps_M       " %20.12f (alpha * pbar * s0med)
di as result "FULLPREC configE_a   " %20.12f c_alpha
di as result "FULLPREC configE_se  " %20.12f c_se
di as result "FULLPREC configE_j   " %20.12f c_j
di as result "FULLPREC inside_alpha" %20.12f i_alpha

di as text "{{hline 78}}"
di as result "PYTHON REFERENCE (this sample)"
di as result "  baseline alpha {PY_ALPHA:.6f} ({PY_SE:.6f})   OLS {PY_OLS:.6f}"
di as result "  SW F {PY_F:.4f}   Hansen J {PY_J:.4f} on {PY_JDF:.0f} df, p {PY_JP:.6f}"
di as result "  first stages {PY_Z1:.6f} ({PY_Z1SE:.6f})   {PY_Z2:.6f} ({PY_Z2SE:.6f})"
di as result "  p-bar {PY_PBAR:.4f}  s_0 {PY_S0:.4f}  eps_M {PY_EPSM:.4f}"
di as result "  configE alpha {PY_CALPHA:.6f} ({PY_CSE:.6f})  J {PY_CJ:.4f}/{PY_CJDF:.0f} p {PY_CJP:.6f}"
di as text "{{hline 78}}"
di as result "done"
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=TARGET)
    ap.add_argument("--no-stata", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with RunManifest(OUT, args=vars(a), inputs=[FRAME, NPZ, ACS],
                     tag=f"round 59 Parts 1-3: the {a.target:,}-row estimation sample "
                         f"and the deliverable") as rm:
        rm.add_note("Market-level draw, full choice sets, drawn ids carried as data.")
        t0 = time.time()
        L = labels()
        D = load_frame()
        counts = D.groupby("market_ids").size()

        print(f"\n=== 1  THE DRAW: market-level, {a.target:,} rows, four independent "
              f"draws [{time.time()-t0:.0f}s] ===", flush=True)
        print(f"  rule: uniform permutation of the {len(counts):,} markets, shortest "
              f"prefix reaching {a.target:,} rows; every alternative in a drawn market "
              f"is kept", flush=True)
        draws, reports = {}, {}
        for s in DRAWS:
            ids, nrows = draw_markets(counts, s, a.target)
            draws[s] = ids
            S = D[D.market_ids.isin(ids)]
            rep = sample_report(S, L, f"draw {s}")
            reports[s] = rep
            print(f"\n  --- draw {s} ---", flush=True)
            print_report(rep)
            del S; gc.collect()
        # the frame itself, for the composition comparison
        rep_frame = sample_report(D, L, "store frame (parent)")
        print(f"\n  --- the parent frame, for comparison ---", flush=True)
        print_report(rep_frame)

        S1 = D[D.market_ids.isin(draws[1])].copy().reset_index(drop=True)
        del D; gc.collect()

        print(f"\n=== 2  THE LINEAR BASELINE on draw 1 [{time.time()-t0:.0f}s] ===",
              flush=True)
        lin = {}
        for spec in ("baseline", "configE"):
            for reading in ("cell", "inside"):
                w = linear(S1, spec, reading)
                lin[f"{spec}_{reading}"] = w
                print_linear(w)
        print(f"\n  the cell-panel reference (round 49/51, D120 as amended): alpha "
              f"{REF['alpha']} ({REF['se']}), OLS {REF['alpha_ols']}, SW F {REF['swF']}, "
              f"J p {REF['Jp']}, eps_M {REF['epsM']} (p-bar {REF['pbar']}, s_0 "
              f"{REF['s0']})", flush=True)

        print(f"\n  the same baseline on the other three draws, for stability:",
              flush=True)
        lin_draws = {1: lin["baseline_cell"]}
        for s in DRAWS[1:]:
            SS = pd.read_parquet(FRAME)
            SS = SS[SS.market_ids.isin(draws[s])].copy().reset_index(drop=True)
            lin_draws[s] = linear(SS, "baseline", "cell")
            w = lin_draws[s]
            print(f"    draw {s}: alpha {w['alpha']:+.6f} ({w['se']:.6f})  SW F "
                  f"{w['swF']:7.4f}  J p {w['Jp']:.4f}  clusters {w['clusters']}",
                  flush=True)
            del SS; gc.collect()
        al = np.array([lin_draws[s]["alpha"] for s in DRAWS])
        se = np.array([lin_draws[s]["se"] for s in DRAWS])
        print(f"    spread {np.ptp(al):.4f} = {np.ptp(al)/se.mean():.2f} average standard "
              f"errors", flush=True)

        print(f"\n=== 3  THE DELIVERABLE [{time.time()-t0:.0f}s] ===", flush=True)
        dta = DTA / "beer_rcnl_r59.dta"
        F = build_dta(S1, L, lin["baseline_cell"], lin["configE_cell"], a.target, dta)
        sz = dta.stat().st_size
        print(f"  wrote {dta.name}: {sz/1e6:.1f} MB, {len(F):,} rows x {F.shape[1]} "
              f"columns, sorted on market_id alt_id", flush=True)
        ids_csv = OUT / "p1_drawn_market_ids.csv"
        pd.DataFrame({"market_id": np.sort(F.market_id.unique())}).to_csv(ids_csv,
                                                                         index=False)
        # the round trip
        RT = pd.read_stata(dta)
        worst, wcol = 0.0, ""
        for c in F.columns:
            if pd.api.types.is_numeric_dtype(F[c]):
                x = F[c].to_numpy(np.float64); z = RT[c].to_numpy(np.float64)
                d = float(np.nanmax(np.abs(np.nan_to_num(x) - np.nan_to_num(z))))
            else:
                d = float((F[c].astype(str).to_numpy() != RT[c].astype(str).to_numpy()).mean())
            if d > worst:
                worst, wcol = d, c
        print(f"  round trip: shapes {'match' if RT.shape == F.shape else 'DIFFER'}; "
              f"worst column '{wcol}' at {worst:.3e}", flush=True)
        del RT; gc.collect()
        h = sha256(dta)
        hc = content_sha256(F)
        print(f"  SHA256 (file)    {h}", flush=True)
        print(f"  SHA256 (content) {hc}", flush=True)

        # ---- the .do file
        b, c = lin["baseline_cell"], lin["configE_cell"]
        do = DO_TEMPLATE.format(
            ROWS=len(F), MARKETS=int(F.market_id.nunique()),
            STORES=int(F.store_code_uc.nunique()), CLUSTERS=int(F.fe_dma.nunique()),
            DTA=dta.name, PY_ALPHA=b["alpha"], PY_SE=b["se"], PY_OLS=b["alpha_ols"],
            PY_F=b["swF"], PY_J=b["J"], PY_JDF=b["Jdf"], PY_JP=b["Jp"],
            PY_Z1=b["first_stages"][0]["coef"], PY_Z1SE=b["first_stages"][0]["se"],
            PY_Z2=b["first_stages"][1]["coef"], PY_Z2SE=b["first_stages"][1]["se"],
            PY_PBAR=b["pbar_rows"], PY_S0=b["s0_median_rows"], PY_EPSM=b["eps_M_rows"],
            PY_CALPHA=c["alpha"], PY_CSE=c["se"], PY_CJ=c["J"], PY_CJDF=c["Jdf"],
            PY_CJP=c["Jp"])
        dofile = DTA / "estimate_r59.do"
        dofile.write_text(do)
        print(f"  wrote {dofile.name}", flush=True)

        man = dict(target=a.target, frame=FRAME_REF, draws={
            str(s): dict(markets=int(len(draws[s])), report=reports[s],
                         linear_baseline=lin_draws[s]) for s in DRAWS},
            parent=rep_frame, linear=lin, dta=dict(
                path=str(dta), sha256=h, size_bytes=int(sz), rows=int(len(F)),
                cols=int(F.shape[1]), columns=list(F.columns),
                roundtrip_worst_column=wcol, roundtrip_worst=worst),
            reference=REF, ladder_reference=LADDER_REF)
        json.dump(man, open(OUT / "p1_p3_sample.json", "w"), indent=1, default=float)
        pd.DataFrame([dict(draw=s, **{k: v for k, v in reports[s].items()
                                      if k != "composition"}) for s in DRAWS]
                     ).to_csv(OUT / "p1_draws.csv", index=False)
        pd.DataFrame(reports[1]["composition"]).to_csv(OUT / "p2_rung_composition.csv",
                                                       index=False)
        pd.DataFrame([lin[k] for k in lin]).drop(columns=["first_stages"]).to_csv(
            OUT / "p3_linear.csv", index=False)
        rm.add_outputs([OUT / "p1_p3_sample.json", OUT / "p1_draws.csv",
                        OUT / "p2_rung_composition.csv", OUT / "p3_linear.csv",
                        ids_csv, dta, dofile])

        # ---- run Stata and compare
        if not a.no_stata and STATA.exists():
            print(f"\n=== 3.4  Stata reproduction [{time.time()-t0:.0f}s] ===", flush=True)
            r = subprocess.run([str(STATA), "-b", "-q", "do", dofile.name],
                               cwd=str(DTA), capture_output=True, text=True, timeout=7200)
            log = (DTA / "estimate_r59.log").read_text()
            (OUT / "p3_stata.log").write_text(log)
            print(log[-3000:], flush=True)
            print(f"  stata exit {r.returncode}", flush=True)
        print(f"\ndone [{time.time()-t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
