#!/usr/bin/env python3
"""Round 52 Parts 1 and 2: configuration E, the first margin-heterogeneity estimate.

Round 45 ran an under-identified specification (K1 = 1, K2 = 2, MD = 2) and round 50 caught
configuration C one moment short.  **The count is confirmed here before anything is estimated
and the run aborts if it does not come out at three parameters against four moments.**

The specification: own price endogenous; `p_lag`, `VD` and `IT` predetermined; **sigma on one
margin**; **one Pi interaction** of that margin with zip3 median household income, demeaned;
`p_lead` dropped (D148); rho = 0 (round 22); `alternative + week + store x year + brand x store`
absorbed through pyhdfe; instruments z1, z2, z1 x d, z2 x d.

**Two design facts stated before the numbers.**  (i) A store-week market is one store, one
location and one income, so `IT x d` is a MARKET-LEVEL shift of the mean coefficient, not
Miller-Weinberg's within-market income draw (round 44); its purpose here is the two moments it
brings.  (ii) PyBLP computes the outside share as `1 - sum(shares)` over the rows supplied, so a
CELL-restricted panel becomes an INSIDE-GOOD restriction inside PyBLP by construction (D134).
The linear 2SLS is therefore reported under both readings on the identical sample.

    python3 paper/src/diagnostics/blp_config_e.py --spec IT
    python3 paper/src/diagnostics/blp_config_e.py --spec VD
"""
from __future__ import annotations
import argparse, gc, json, sys, time
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
from blp_frequency_grid import R3                                     # noqa: E402
from blp_rho_sweep import bracket                                     # noqa: E402
import blp_store_week_demand as SWD                                   # noqa: E402
import blp_chain_by_week as CBW                                       # noqa: E402
from blp_store_week_demand import absorb                              # noqa: E402
from blp_store_week_decomp import classes, frame                      # noqa: E402
from blp_hn_cells import build_grid, store_set                        # noqa: E402
from blp_margins_decomp import build_margins                          # noqa: E402

RMS = _paths.RMS
ACS = _paths.ACS
OUT = PAPER / "outputs" / "blp_confige"
VD = PAPER / "outputs" / "blp_vdcov"
SCRATCH = _paths.WORK
SWD.SCRATCH = SCRATCH
CBW.SCRATCH = SCRATCH
YEARS = [2016, 2017, 2018, 2019]
Z2 = ["d_same_rung_other_brand", "d_same_container_other"]
MIN_WEEKS = 52
LO_IN, HI_IN = 0.01, 0.95
GRID = [0.0, 0.05, 0.10, 0.25, 0.50]
STARTS = [0.05, 0.25, 0.50, 1.00]
FRAC_LADDER = [1.0, 0.5, 0.25, 0.10, 0.05]
BUDGET_S = 3000.0
EVALS_ASSUMED = 300
REF = dict(cells=17_512, rows=3_220_839, markets=259_639, alpha=-1.1103, se=0.1643,
           alpha_ols=-0.2847, swF=23.35, J=3.23, Jp=0.0724, fs_z1=-0.052097,
           el=-12.60, s0=0.5572, epsM=-7.0814, div=54.64,
           vd_cov=0.7859, it_cov=0.9933, common=2_309_466)


CACHE = SCRATCH / "config_e_frame.parquet"


def build_frame(verbose=True):
    """The adopted panel restricted to rows where p, p_lag, VD and IT are all defined.

    Cached, because the grid, the margins and the frame together take six minutes and every
    specification in this round uses the identical frame.
    """
    if CACHE.exists():
        D = pd.read_parquet(CACHE)
        if verbose:
            print(f"  frame read from cache: {len(D):,} rows, "
                  f"{D.market_ids.nunique():,} markets ({CACHE.name})", flush=True)
        return D
    z = np.load(SCRATCH / "storeweek_totals.npz", allow_pickle=True)
    stores = pd.Index(z["stores"].astype(str))
    weeks = pd.DatetimeIndex(pd.to_datetime(z["weeks"].astype(str)))
    cells_lab = z["cells"].astype(str)
    nW, nA = len(weeks), len(cells_lab)
    si, unb, present = store_set(z)
    nS = len(si)
    tot_sv = z["tot_sv"]; dma_y = z["dma"]
    yidx = np.array([YEARS.index(y) for y in weeks.year])
    brand = np.array([c.split("|")[0] for c in cells_lab])
    cont = np.array([c.split("|")[1] for c in cells_lab])
    ncan = np.array([int(c.split("|")[2]) for c in cells_lab])
    CLS = classes(nA, brand, cont, ncan)
    bcode = pd.factorize(brand)[0]
    dma_last = np.array([""] * len(stores), dtype=object)
    for k in range(len(YEARS)):
        h = dma_y[:, k] != ""
        dma_last[h] = dma_y[h, k]
    ST = []
    for y in YEARS:
        ST.append(pd.read_csv(RMS / f"{y}/Annual_Files/stores_{y}.tsv", sep="\t",
                              dtype={"store_code_uc": "string", "store_zip3": "string"},
                              usecols=["store_code_uc", "year", "store_zip3"],
                              encoding="latin-1", low_memory=False))
    ST = pd.concat(ST, ignore_index=True)
    ST["store_code_uc"] = ST.store_code_uc.str.strip()
    lastz = ST.sort_values("year").drop_duplicates("store_code_uc", keep="last") \
              .set_index("store_code_uc")
    zip3 = lastz.store_zip3.reindex(stores).astype("string").str.zfill(3).to_numpy()
    del ST, lastz; gc.collect()
    # the margins, from the same builder rounds 47-51 used
    G = build_grid(verbose=verbose)
    MG = build_margins(G)
    for k in ("VD_strict", "rung_distance", "vd_reason", "p_lead"):
        del MG[k]
    gc.collect()
    valid = G["valid"]
    lag = MG["p_lag"][valid].astype(np.float64)
    vd = MG["VD"][valid].astype(np.float64)
    it = MG["IT"][valid].astype(np.float64)
    pgrid = G["P"][valid].astype(np.float64)
    del MG, G; gc.collect()
    PK, RV, kept = SWD.load_dense(si, nW, nA)
    XPK, XRV, nex = CBW.load_excluded(si, nW, nA)
    PK = np.maximum(PK - XPK, 0.0); RV = np.maximum(RV - XRV, 0.0)
    del XPK, XRV; gc.collect()
    d, Z, sj, sjg, peq = frame(PK, RV, tot_sv[si], CLS, ncan.astype(np.float32),
                               yidx, "allbeer")
    del PK, RV; gc.collect()
    n = len(sj)
    ui = d["un"].astype(np.int64); wi = d["wk"].astype(np.int64)
    ai = d["al"].astype(np.int64)
    # the frame and the grid must be in the same (store, week, alternative) order for the
    # margins to line up with the shares; asserted rather than assumed
    dmax = float(np.abs(peq.astype(np.float64) - pgrid).max())
    if verbose:
        print(f"  row-order check: max |frame price - grid price| = {dmax:.3g} over "
              f"{n:,} rows", flush=True)
    assert dmax < 1e-6, dmax
    del pgrid; gc.collect()
    mk = pd.factorize(ui * 300 + wi)[0]
    RB = pd.read_parquet(VD / "p2_rebound_cells.parquet")
    m = ((RB.spells_rebound >= 12) & (RB.depth_rebound >= 0.10)
         & (RB.n_obs >= MIN_WEEKS)).to_numpy()
    keep = np.zeros(nS * nA, bool); keep[RB.cell.to_numpy()[m]] = True
    inrow = keep[(ui * nA + ai)]
    assert int(m.sum()) == REF["cells"], (int(m.sum()), REF["cells"])
    assert int(inrow.sum()) == REF["rows"], (int(inrow.sum()), REF["rows"])
    ok = inrow & np.isfinite(lag) & np.isfinite(vd) & np.isfinite(it)
    if verbose:
        print(f"  adopted panel {int(inrow.sum()):,} rows; with `p_lag`, `VD` and `IT` all "
              f"defined **{int(ok.sum()):,}** ({100*ok.mean()/inrow.mean():.2f}% of it; "
              f"round 51's common sample {REF['common']:,})", flush=True)
    i2 = [R3.index(c) for c in Z2]
    D = pd.DataFrame({
        "market_ids": mk[ok].astype(np.int64),
        "shares": sj[ok].astype(np.float64),
        "prices": peq[ok].astype(np.float64),
        "p_lag": lag[ok], "VD": vd[ok], "IT": it[ok],
        "s0_cell": d["s0"][ok].astype(np.float64),
        "sjg": sjg[ok].astype(np.float64),
        "alt": ai[ok].astype(np.int32), "week": wi[ok].astype(np.int32),
        "storeyear": pd.factorize(ui[ok] * 4 + d["yr"][ok])[0].astype(np.int32),
        "brandstore": pd.factorize(ui[ok] * 100 + bcode[ai[ok]])[0].astype(np.int32),
        "dma": pd.factorize(dma_last[si][ui[ok]])[0].astype(np.int32),
        "zip3": zip3[si][ui[ok]],
        "year": d["yr"][ok].astype(np.int8),
        "z1": Z[i2[0]][ok].astype(np.float64),
        "z2": Z[i2[1]][ok].astype(np.float64)})
    del Z, sj, sjg, peq, d, lag, vd, it; gc.collect()
    # zip3-year median household income, in $1,000s, demeaned over the estimation sample
    A = pd.read_csv(ACS)
    A["store_zip3"] = A.store_zip3.astype(int).astype(str).str.zfill(3)
    A["yr"] = A.year.map({y: i for i, y in enumerate(YEARS)})
    D = D.merge(A[["store_zip3", "yr", "median_hh_income"]],
                left_on=["zip3", "year"], right_on=["store_zip3", "yr"], how="left")
    miss = float(D.median_hh_income.isna().mean())
    D["income"] = D.median_hh_income.fillna(D.median_hh_income.mean()) / 1000.0
    D["income"] = D.income - D.income.mean()
    D = D.drop(columns=["store_zip3", "yr", "median_hh_income", "zip3"])
    if verbose:
        print(f"  income: zip3-year median household income in $1,000s, demeaned; "
              f"{100*(1-miss):.2f}% matched, sd {D.income.std():.3f}, range "
              f"[{D.income.min():.2f}, {D.income.max():.2f}]", flush=True)
    # PyBLP's outside share is 1 - sum(shares) over the rows supplied: the INSIDE-GOOD
    # reading (D134).  Both readings are carried so the 2SLS can be run under each.
    ins = D.groupby("market_ids").shares.transform("sum")
    D["s0_inside"] = 1.0 - ins
    bad = D.market_ids[(ins < LO_IN) | (ins > HI_IN)].unique()
    D = D[~D.market_ids.isin(bad)].copy()
    if verbose:
        print(f"  dropped {len(bad):,} ill-conditioned markets (inside share outside "
              f"[{LO_IN}, {HI_IN}]); {len(D):,} rows, {D.market_ids.nunique():,} markets",
              flush=True)
        print(f"  implied `s_0`: CELL reading median {np.median(D.s0_cell):.4f} "
              f"(round 51 {REF['s0']}); INSIDE-GOOD reading median "
              f"{np.median(D.s0_inside):.4f}", flush=True)
    D = D.sort_values("market_ids", kind="stable").reset_index(drop=True)
    D.to_parquet(CACHE, index=False)
    return D


def tsls_on(D, reading="cell"):
    """2SLS on the identical regressor set: p endogenous, the three margins included."""
    s0 = D.s0_cell.to_numpy() if reading == "cell" else D.s0_inside.to_numpy()
    y = np.log(D.shares.to_numpy()) - np.log(s0)
    X = np.column_stack([D.prices, D.p_lag, D.VD, D.IT]).astype(np.float64)
    z1 = D.z1.to_numpy(); z2 = D.z2.to_numpy(); dd = D.income.to_numpy()
    Zx = np.column_stack([z1, z2, z1 * dd, z2 * dd])
    gr = [pd.factorize(D.alt)[0], pd.factorize(D.week)[0],
          pd.factorize(D.storeyear)[0], pd.factorize(D.brandstore)[0]]
    A, it, mx = absorb(np.column_stack([y, X, Zx]), gr)
    cl = pd.factorize(D.dma)[0]
    n = len(D)
    yy = A[:, 0]; XX = A[:, 1:5]; ZZ = np.column_stack([A[:, 5:9], A[:, 2:5]])
    b, V, u, G = M.tsls(yy, XX, ZZ, cl)
    bo, Vo, _, _ = M.ols(yy, XX, cl)
    swf = M.sw_stats(XX[:, 0], XX[:, 1:], ZZ, cl)      # treats the margins as endogenous
    # the statistic this design needs: purge the three PREDETERMINED margins from price and
    # from the four excluded instruments, then the F on the excluded set
    W = XX[:, 1:]
    Pw = np.linalg.lstsq(W, np.column_stack([XX[:, [0]], A[:, 5:9]]), rcond=None)[0]
    Rz = np.column_stack([XX[:, [0]], A[:, 5:9]]) - W @ Pw
    swf_net = M.sw_stats(Rz[:, 0], np.empty((n, 0)), Rz[:, 1:], cl)
    J, Jdf, Jp, _ = M.hansen_j(yy, XX, ZZ, cl)
    br = bracket(0.0, D.sjg.to_numpy(), D.shares.to_numpy())
    pbar = float(np.average(D.prices, weights=D.shares))
    out = dict(reading=reading, n=int(n), markets=int(D.market_ids.nunique()),
               alpha=float(b[0]), se=float(np.sqrt(V[0, 0])),
               b_p_lag=float(b[1]), se_p_lag=float(np.sqrt(V[1, 1])),
               b_VD=float(b[2]), se_VD=float(np.sqrt(V[2, 2])),
               b_IT=float(b[3]), se_IT=float(np.sqrt(V[3, 3])),
               alpha_ols=float(bo[0]), swF=float(swf["sw_F"]),
               swF_net=float(swf_net["sw_F"]), J=float(J), Jdf=int(Jdf),
               Jp=float(Jp), clusters=int(G),
               own_elast=float(np.median(b[0] * D.prices.to_numpy() * br)),
               market_elast=float(b[0] * pbar * np.median(s0)),
               diversion=float(100 * np.median(s0 / br)), s0_median=float(np.median(s0)))
    # the C-statistic for each excluded moment, on one weight matrix
    order = np.argsort(cl, kind="stable")
    Zs, us, cs = ZZ[order], u[order], cl[order]
    bnd = np.flatnonzero(np.diff(cs)) + 1
    S = np.zeros((ZZ.shape[1], ZZ.shape[1]))
    for a_, b_ in zip(np.r_[0, bnd], np.r_[bnd, len(cs)]):
        g = Zs[a_:b_].T @ us[a_:b_]
        S += np.outer(g, g)
    S /= n

    def Jof(idx):
        Zi = ZZ[:, idx]
        Si = np.linalg.pinv(S[np.ix_(idx, idx)])
        Ai = np.linalg.inv(XX.T @ Zi @ Si @ Zi.T @ XX)
        bb = Ai @ (XX.T @ Zi @ Si @ Zi.T @ yy)
        uu = yy - XX @ bb
        g = Zi.T @ uu / n
        return float(n * g @ Si @ g), float(bb[0])

    full = list(range(ZZ.shape[1]))
    Jf, _ = Jof(full)
    cst = []
    for k, nm in enumerate(["z1", "z2", "z1_x_income", "z2_x_income"]):
        sub = [j for j in full if j != k]
        Js, bs = Jof(sub)
        cst.append(dict(moment=nm, C=Jf - Js,
                        C_p=float(stats.chi2.sf(max(Jf - Js, 0.0), 1)),
                        alpha_without=bs))
    out["cstats"] = cst
    del A; gc.collect()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", choices=["IT", "VD"], default="IT")
    ap.add_argument("--frac", type=float, default=0.0, help="0 = choose by the probe")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--maxiter", type=int, default=50)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    key = f"E_{a.spec}"
    with RunManifest(OUT, args=vars(a),
                     inputs=[SCRATCH / "focal_rows.bin", ACS,
                             VD / "p2_rebound_cells.parquet"],
                     tag=f"configuration E, sigma on {a.spec} (round 52)") as rm:
        t0 = time.time()
        import pyblp
        pyblp.options.verbose = False
        print(f"pyblp {pyblp.__version__}", flush=True)

        # ---------------- 1.1 the gate
        print(f"\n=== 1.1  THE ARITHMETIC GATE, before anything is estimated ===",
              flush=True)
        endog, nonlin_sigma, nonlin_pi = 1, 1, 1
        excluded = 4
        params = endog + nonlin_sigma + nonlin_pi
        print(f"  endogenous regressors                     {endog}  (own price)")
        print(f"  predetermined regressors                  3  (p_lag, VD, IT)")
        print(f"  nonlinear parameters                      {nonlin_sigma + nonlin_pi}  "
              f"(sigma_{a.spec}, pi_{a.spec}xincome)")
        print(f"  linear parameters needing a moment        {endog}")
        print(f"  EXCLUDED moments                          {excluded}  (z1, z2, "
              f"z1 x income, z2 x income)")
        print(f"  parameters requiring moments              **{params}**")
        print(f"  degrees of freedom                        **{excluded - params}**")
        if params != 3 or excluded != 4:
            print(f"  **GATE FAILED: the count is not three against four. STOPPING.**")
            return 1
        print(f"  **GATE PASSES: three parameters against four moments, over-identified "
              f"by one.**", flush=True)
        print(f"  The Pi is a MARKET-LEVEL interaction: one store, one location, one "
              f"income (round 44), so `{a.spec} x income` shifts the mean coefficient "
              f"across markets and generates no within-market heterogeneity. Its purpose "
              f"here is the two moments it brings.", flush=True)

        print(f"\n=== building the frame [{time.time()-t0:.0f}s] ===", flush=True)
        D = build_frame()
        gc.collect()
        DFULL = D
        if 0 < a.frac < 1.0:
            rng = np.random.default_rng(a.seed)
            allm = D.market_ids.unique()
            kp = rng.choice(allm, size=int(round(a.frac * len(allm))), replace=False)
            D = D[D.market_ids.isin(kp)].copy().reset_index(drop=True)
            print(f"  **market subsample {a.frac:.2f} (seed {a.seed}) applied BEFORE the "
                  f"Problem: {len(D):,} rows, {D.market_ids.nunique():,} markets of "
                  f"{DFULL.market_ids.nunique():,}** -- PyBLP's per-market Python cost "
                  f"binds at 235,751 markets (round 44's finding), so the nonlinear run "
                  f"uses a market subsample as rounds 44 and 45 did", flush=True)

        # ---------------- 1.4 the linear 2SLS first, both readings
        print(f"\n=== 1.4  the linear 2SLS on the IDENTICAL regressor set, before any "
              f"nonlinear parameter [{time.time()-t0:.0f}s] ===", flush=True)
        TS = {}
        if 0 < a.frac < 1.0:
            print(f"  --- on the FULL common sample ({len(DFULL):,} rows) ---", flush=True)
            for reading in ("cell", "inside"):
                Tf = tsls_on(DFULL, reading)
                TS[f"full_{reading}"] = Tf
                print(f"  [{reading:6s}, full ] alpha {Tf['alpha']:+.4f} ({Tf['se']:.4f})"
                      f"  F(net) {Tf['swF_net']:.2f}  J {Tf['J']:.2f}/{Tf['Jdf']} p "
                      f"{Tf['Jp']:.4f}  own el {Tf['own_elast']:+.2f}  div "
                      f"{Tf['diversion']:.2f}%", flush=True)
            print(f"  --- on the estimation subsample ({len(D):,} rows) ---", flush=True)
        for reading in ("cell", "inside"):
            T = tsls_on(D, reading)
            TS[reading] = T
            print(f"  [{reading:6s} reading] alpha {T['alpha']:+.4f} ({T['se']:.4f})  "
                  f"OLS {T['alpha_ols']:+.4f}  **F(price net of the margins) "
                  f"{T['swF_net']:.2f}**  [SW-with-margins-endogenous {T['swF']:.2f}]  "
                  f"J {T['J']:.2f} on {T['Jdf']} df p {T['Jp']:.4f}  s0 "
                  f"{T['s0_median']:.4f}  own el {T['own_elast']:+.2f}  epsM "
                  f"{T['market_elast']:+.3f}  div {T['diversion']:.2f}%")
            print(f"  {'':17s} p_lag {T['b_p_lag']:+.5f} ({T['se_p_lag']:.5f})  "
                  f"VD {T['b_VD']:+.5f} ({T['se_VD']:.5f})  IT {T['b_IT']:+.5f} "
                  f"({T['se_IT']:.5f})")
            for c in T["cstats"]:
                print(f"  {'':17s} C dropping {c['moment']:14s} {c['C']:7.3f} "
                      f"(p {c['C_p']:.4f})  alpha without {c['alpha_without']:+.4f}")
        print(f"  round 51 two-count baseline (cell reading, no margins in the model): "
              f"alpha {REF['alpha']:+.4f} ({REF['se']:.4f}) F {REF['swF']:.2f} J p "
              f"{REF['Jp']:.4f} el {REF['el']} epsM {REF['epsM']} div {REF['div']}%",
              flush=True)
        json.dump(TS, open(OUT / f"p1_tsls_{key}.json", "w"), indent=1, default=float)

        # ---------------- the problem
        X1 = pyblp.Formulation("0 + prices + p_lag + VD + IT",
                               absorb="C(alt) + C(week) + C(storeyear) + C(brandstore)")
        X2 = pyblp.Formulation(f"0 + {a.spec}")
        agent_form = pyblp.Formulation("0 + income")
        P = D.rename(columns={"z1": "demand_instruments0", "z2": "demand_instruments1"})
        P["demand_instruments2"] = P.demand_instruments0 * P.income
        P["demand_instruments3"] = P.demand_instruments1 * P.income
        NODES = 7
        def make_agents(Dx):
            mi = Dx.groupby("market_ids", as_index=False).income.first()
            return pd.DataFrame({
                "market_ids": np.repeat(mi.market_ids.to_numpy(), NODES),
                "income": np.repeat(mi.income.to_numpy(), NODES)})
        agents = make_agents(D)
        prod = P.drop(columns=["s0_cell", "s0_inside", "sjg", "income", "dma", "year"])
        integ = pyblp.Integration("product", NODES)
        print(f"\n=== the pyblp Problem [{time.time()-t0:.0f}s] ===", flush=True)
        prob = pyblp.Problem((X1, X2), prod, agent_formulation=agent_form,
                             agent_data=agents, integration=integ)
        print(prob, flush=True)
        print(f"  MD {prob.MD}  K1 {prob.K1}  K2 {prob.K2}  D {prob.D}  "
              f"-> J df = MD - K1 - K2 - D = {prob.MD - prob.K1 - prob.K2 - prob.D}",
              flush=True)
        it_ = pyblp.Iteration("squarem", {"atol": 1e-13})
        opt = pyblp.Optimization("l-bfgs-b", {"gtol": 1e-4,
                                                "maxiter": a.maxiter})

        def solve(s0v, ret=False, pi0=0.0, fix=False):
            s = np.array([[s0v]])
            kw = dict(sigma=s, pi=np.array([[pi0]]),
                      iteration=it_, method="1s",
                      optimization=pyblp.Optimization("return") if ret else opt)
            if fix or ret:
                kw["sigma_bounds"] = (s, s)
            else:
                kw["sigma_bounds"] = (np.array([[0.0]]), np.array([[10.0]]))
            kw["pi_bounds"] = (np.array([[-0.5]]), np.array([[0.5]]))
            return prob.solve(**kw)

        # ---------------- 1.3 the probe and the time budget
        print(f"\n=== 1.3  timing probe: one objective evaluation ===", flush=True)
        tp = time.time()
        r0 = solve(0.10, ret=True)
        sec = time.time() - tp
        proj = sec * EVALS_ASSUMED
        print(f"  one evaluation {sec:.1f}s on {len(D):,} rows and "
              f"{D.market_ids.nunique():,} markets; {EVALS_ASSUMED} evaluations projects "
              f"{proj/3600:.2f} h", flush=True)
        print(f"  contraction: {int(np.sum(r0.fp_iterations))} total iterations, mean "
              f"{np.asarray(r0.fp_iterations).mean():.2f} per market, "
              f"{int(np.sum(~np.asarray(r0.fp_converged).astype(bool)))} markets failing",
              flush=True)
        frac = 1.0 if a.frac > 0 else 0.0
        if frac <= 0:
            frac = 1.0
            for f in FRAC_LADDER:
                if proj * f <= BUDGET_S:
                    frac = f
                    break
            else:
                frac = FRAC_LADDER[-1]
        if frac < 1.0:
            rng = np.random.default_rng(a.seed + 100)
            allm = D.market_ids.unique()
            kp = rng.choice(allm, size=int(round(frac * len(allm))), replace=False)
            D = D[D.market_ids.isin(kp)].copy()
            P = P[P.market_ids.isin(kp)].copy()
            agents = make_agents(D)
            prod = P.drop(columns=["s0_cell", "s0_inside", "sjg", "income", "dma", "year"])
            prob = pyblp.Problem((X1, X2), prod, agent_formulation=agent_form,
                                 agent_data=agents, integration=integ)
            print(f"  **market subsample {frac:.2f} chosen against a {BUDGET_S/3600:.2f} h "
                  f"budget: {len(D):,} rows, {D.market_ids.nunique():,} markets**",
                  flush=True)
        else:
            print(f"  **the full adopted common sample fits the budget; no subsample**",
                  flush=True)

        # ---------------- 1.5 the objective profile (run BEFORE the starts: when the
        # objective is flat the optimiser crawls, and the profile is the decisive object)
        print(f"\n=== 1.5  the objective profile at fixed sigma_{a.spec} "
              f"[{time.time()-t0:.0f}s] ===", flush=True)
        prof, base = [], None
        for v in GRID:
            rr = solve(v, fix=True, pi0=0.0)
            ob = float(np.asarray(rr.objective).ravel()[0])
            if base is None:
                base = ob
            prof.append(dict(sigma=v, objective=ob, delta=ob - base))
            pv_ = float(np.asarray(rr.pi).ravel()[0])
            prof[-1]["pi"] = pv_
            print(f"  sigma_{a.spec} {v:5.2f}: objective {ob:16.6f}  ({ob-base:+14.6f} "
                  f"vs sigma = 0)  pi at that sigma {pv_:+.6f}", flush=True)
        print(f"  round 45's standard: sigma_price cost 102 objective units at 0.10 and "
              f"sigma_constant cost 0.28 -- the asymmetry that separated an estimate from "
              f"a corner", flush=True)
        pd.DataFrame(prof).to_csv(OUT / f"p1_profile_{key}.csv", index=False)

        # ---------------- 1.7 four starts
        print(f"\n=== 1.7  four starting values [{time.time()-t0:.0f}s] ===", flush=True)
        tried, best, best_obj = [], None, np.inf
        for s0v in STARTS:
            rr = solve(s0v)
            ob = float(np.asarray(rr.objective).ravel()[0])
            sg = float(np.asarray(rr.sigma).ravel()[0])
            pi_ = float(np.asarray(rr.pi).ravel()[0])
            bb = np.asarray(rr.beta).ravel()
            tried.append(dict(start=s0v, objective=ob, sigma=sg, pi=pi_,
                              alpha=float(bb[0]), converged=bool(rr.converged),
                              evaluations=int(rr.objective_evaluations)))
            print(f"  start {s0v:5.2f} -> objective {ob:.6e}  sigma_{a.spec} {sg:.6f}  "
                  f"pi {pi_:+.6f}  alpha {bb[0]:+.4f}  converged {rr.converged}  "
                  f"({rr.objective_evaluations} evals)", flush=True)
            if ob < best_obj:
                best, best_obj = rr, ob
        pd.DataFrame(tried).to_csv(OUT / f"p1_starts_{key}.csv", index=False)
        r = best
        el = time.time() - t0

        # ---------------- 1.3 convergence, before any parameter
        fpi = np.asarray(r.fp_iterations).ravel()
        fpc = np.asarray(r.fp_converged).ravel().astype(bool)
        diag = dict(key=key, spec=a.spec, frac=frac, rows=int(len(D)),
                    markets=int(D.market_ids.nunique()),
                    converged=bool(r.converged),
                    objective=float(np.asarray(r.objective).ravel()[0]),
                    gradient_norm=float(np.asarray(r.projected_gradient_norm).ravel()[0]),
                    gradient_max=float(np.abs(np.asarray(r.gradient)).max()),
                    hessian_min_eig=float(np.asarray(
                        r.reduced_hessian_eigenvalues).min()),
                    hessian_max_eig=float(np.asarray(
                        r.reduced_hessian_eigenvalues).max()),
                    clipped_shares=int(np.sum(r.clipped_shares)),
                    objective_evaluations=int(r.objective_evaluations),
                    optimization_iterations=int(r.optimization_iterations),
                    fp_iterations_mean=float(fpi.mean()), fp_iterations_max=float(fpi.max()),
                    fp_failures=int(np.sum(~fpc)), fp_converged_all=bool(np.all(fpc)),
                    wall_seconds=el, sec_per_eval=sec,
                    cumulative_optimization_time=float(r.cumulative_optimization_time))
        print(f"\n=== 1.3  convergence [{el:.0f}s = {el/3600:.2f} h] ===")
        for k in ("converged", "objective", "gradient_norm", "gradient_max",
                  "hessian_min_eig", "hessian_max_eig", "objective_evaluations",
                  "optimization_iterations", "fp_iterations_mean", "fp_iterations_max",
                  "fp_failures", "fp_converged_all", "clipped_shares"):
            print(f"  {k:28s} {diag[k]}")

        # ---------------- 1.4 the parameters
        sig = float(np.asarray(r.sigma).ravel()[0])
        sig_se = float(np.asarray(r.sigma_se).ravel()[0])
        pi_ = float(np.asarray(r.pi).ravel()[0])
        pi_se = float(np.asarray(r.pi_se).ravel()[0])
        bb = np.asarray(r.beta).ravel(); bse = np.asarray(r.beta_se).ravel()
        par = dict(sigma=sig, sigma_se=sig_se, pi=pi_, pi_se=pi_se,
                   alpha=float(bb[0]), alpha_se=float(bse[0]),
                   b_p_lag=float(bb[1]), se_p_lag=float(bse[1]),
                   b_VD=float(bb[2]), se_VD=float(bse[2]),
                   b_IT=float(bb[3]), se_IT=float(bse[3]))
        print(f"\n=== 1.4  parameters ===")
        print(f"  **sigma_{a.spec}      {sig:+.6f} ({sig_se:.6f})**")
        print(f"  **pi_{a.spec}xincome  {pi_:+.6f} ({pi_se:.6f})** per $1,000 of zip3 "
              f"median household income -- a MARKET-LEVEL shift of the mean "
              f"{a.spec} coefficient, not within-market heterogeneity")
        print(f"  alpha (price)   {par['alpha']:+.4f} ({par['alpha_se']:.4f})   2SLS on "
              f"the same sample, inside reading {TS['inside']['alpha']:+.4f} "
              f"({TS['inside']['se']:.4f})")
        print(f"  p_lag           {par['b_p_lag']:+.5f} ({par['se_p_lag']:.5f})")
        print(f"  VD              {par['b_VD']:+.5f} ({par['se_VD']:.5f})")
        print(f"  IT              {par['b_IT']:+.5f} ({par['se_IT']:.5f})", flush=True)

        # ---------------- 1.6 the J and 1.8 the diversion
        Jval = float(np.asarray(r.objective).ravel()[0])
        Jdf = int(prob.MD - prob.K1 - prob.K2 - prob.D)
        print(f"\n=== 1.6  the Hansen J ===")
        print(f"  GMM objective at the optimum {Jval:.6f}; J df = MD - K1 - K2 - D = "
              f"{prob.MD} - {prob.K1} - {prob.K2} - {prob.D} = **{Jdf}**")
        print(f"  p-value {stats.chi2.sf(Jval, Jdf) if Jdf > 0 else float('nan'):.4f} "
              f"(the linear 2SLS on the same moments gave J {TS['inside']['J']:.2f} on "
              f"{TS['inside']['Jdf']} df, p {TS['inside']['Jp']:.4f})", flush=True)
        print(f"\n=== 1.8  diversion and elasticities [{time.time()-t0:.0f}s] ===",
              flush=True)
        MAXM = 40_000
        if D.market_ids.nunique() > MAXM:
            rng2 = np.random.default_rng(7)
            keepm = rng2.choice(D.market_ids.unique(), size=MAXM, replace=False)
            Dsub = D[D.market_ids.isin(keepm)].copy()
            print(f"  elasticities computed on a {MAXM:,}-market random subset of "
                  f"{D.market_ids.nunique():,} for memory", flush=True)
            rsub = r
            E = np.asarray(rsub.compute_elasticities(name="prices",
                                                     market_ids=keepm))
            Dv = np.asarray(rsub.compute_diversion_ratios(name="prices",
                                                          market_ids=keepm))
            Dit = Dsub
        else:
            E = np.asarray(r.compute_elasticities(name="prices"))
            Dv = np.asarray(r.compute_diversion_ratios(name="prices"))
            Dit = D
        own, out_div, pos = [], [], 0
        for _m, g in Dit.groupby("market_ids", sort=False):
            k = len(g)
            own.append(np.diag(E[pos:pos + k, :k]))
            bd = Dv[pos:pos + k, :k]
            out_div.append(1.0 - (bd.sum(axis=1) - np.diag(bd)))
            pos += k
        own = np.concatenate(own); out_div = np.concatenate(out_div)
        eM = float(np.asarray(r.compute_aggregate_elasticities(
            name="prices", market_ids=(keepm if D.market_ids.nunique() > MAXM
                                       else None))).mean())
        res = dict(own_elast_median=float(np.median(own)),
                   diversion_outside_median=float(np.median(out_div)),
                   aggregate_semi_elasticity=eM, J=Jval, J_df=Jdf,
                   J_p=float(stats.chi2.sf(Jval, Jdf)) if Jdf > 0 else np.nan)
        print(f"  median own-price elasticity {res['own_elast_median']:+.3f}  (2SLS "
              f"inside reading {TS['inside']['own_elast']:+.2f}; round 51 two-count "
              f"{REF['el']})")
        print(f"  median outside diversion **{100*res['diversion_outside_median']:.2f}%**"
              f"  (2SLS inside reading {TS['inside']['diversion']:.2f}%; round 51 "
              f"two-count cell reading {REF['div']}%; MW 12-17%)")
        print(f"  PyBLP aggregate elasticity {eM:+.4f} -- a SEMI-elasticity, never quote "
              f"as an elasticity (D120)", flush=True)
        json.dump(dict(gate=dict(endogenous=endog, nonlinear=2, excluded=excluded,
                                 parameters=params, df=excluded - params),
                       diagnostics=diag, parameters=par, profile=prof, results=res,
                       tsls=TS, starts=tried),
                  open(OUT / f"p1_blp_{key}.json", "w"), indent=1, default=float)
        rm.add_outputs([OUT / f"p1_blp_{key}.json", OUT / f"p1_tsls_{key}.json",
                        OUT / f"p1_starts_{key}.csv", OUT / f"p1_profile_{key}.csv"])
        print(f"\ndone [{time.time()-t0:.0f}s = {(time.time()-t0)/3600:.2f} h]",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
