#!/usr/bin/env python3
"""Round 37 Part 1: why does `alpha` jump eight-fold with brand-by-store effects?

Round 36's store-week ladder gives `alpha` = +0.3238, +0.0179, -0.0196, **-1.1280** across
alternative+week, +store, +store x year, +brand x store, with the SW F running 5.35, 20.61,
21.40, 15.04 -- non-monotone, so instrument strength does not track the coefficient and the
simple weak-instrument reading is out.

This reports, at every rung: the full first stage (each differentiation class's coefficient on
residualised price with cluster-robust standard errors, its partial R-squared and its own
residualised standard deviation), the reduced form separately, the 2SLS projection's numerator
and denominator per observation, and two direct tests of the candidate explanations -- an
absorbed assortment-quality term, and a broken collinearity between the classes and the unit
effects.

The dense grids are freed before the first absorption, because at 52.8 million rows the frame
plus a six-column float64 design matrix plus the grids exceeds this machine's memory.

    python3 paper/src/diagnostics/blp_sw_first_stage_ladder.py
"""
from __future__ import annotations
import gc, json, sys, time
from pathlib import Path
import _paths
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
PAPER = _paths.WORK
sys.path.insert(0, str(PAPER / "src" / "common"))
sys.path.insert(0, str(HERE.parent))
from run_manifest import RunManifest                                  # noqa: E402
import blp_first_stage_strength as M                                  # noqa: E402
from blp_frequency_grid import R3                                     # noqa: E402
from blp_rho_sweep import bracket, MW_RHO_NL                          # noqa: E402
from blp_store_week_demand import absorb, load_dense                  # noqa: E402
from blp_store_week_decomp import classes, frame                      # noqa: E402

RMS = _paths.RMS
OUT = PAPER / "outputs" / "blp_swdiag"
SCRATCH = _paths.WORK
YEARS = [2016, 2017, 2018, 2019]
R36 = dict(stores=5208, markets=1_087_636, n=52_780_237, alpha=-1.1280, se=0.2966,
           swF=15.04, Jp=0.0484, s0=0.5581, el=-13.86, epsM=-6.783,
           ladder=[0.3238, 0.0179, -0.0196, -1.1280],
           ladder_F=[5.35, 20.61, 21.40, 15.04])


def ols_cluster(y, X, cl):
    """OLS with cluster-robust variance; X already residualised, no intercept."""
    XtX = X.T @ X
    b = np.linalg.solve(XtX, X.T @ y)
    u = y - X @ b
    k = X.shape[1]
    G = int(cl.max()) + 1
    meat = np.zeros((k, k))
    order = np.argsort(cl, kind="stable")
    cs = cl[order]
    bounds = np.searchsorted(cs, np.arange(G + 1))
    Xo = X[order]; uo = u[order]
    for g in range(G):
        a, z = bounds[g], bounds[g + 1]
        if z > a:
            s = Xo[a:z].T @ uo[a:z]
            meat += np.outer(s, s)
    Xi = np.linalg.inv(XtX)
    V = Xi @ meat @ Xi * (G / max(G - 1, 1))
    r2 = 1.0 - float(u @ u) / float(y @ y)
    return b, np.sqrt(np.diag(V)), r2, u


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with RunManifest(OUT, args={}, inputs=[SCRATCH / "focal_rows.bin", RMS],
                     tag="store-week first-stage ladder (round 37 Part 1)") as rm:
        rm.add_note("Full first stage and reduced form at every rung of the store-week "
                    "fixed-effect ladder, with the covariance decomposition.")
        t0 = time.time()
        z = np.load(SCRATCH / "storeweek_totals.npz", allow_pickle=True)
        sv = np.load(SCRATCH / "survivors.npz")
        stores = pd.Index(z["stores"].astype(str))
        weeks = pd.DatetimeIndex(pd.to_datetime(z["weeks"].astype(str)))
        cells = z["cells"].astype(str)
        nW, nA = len(weeks), len(cells)
        surv_idx = np.where(sv["surv"])[0]; nS = len(surv_idx)
        assert nS == R36["stores"], (nS, R36["stores"])
        yidx = np.array([YEARS.index(y) for y in weeks.year])
        brand = np.array([c.split("|")[0] for c in cells])
        cont = np.array([c.split("|")[1] for c in cells])
        ncan = np.array([int(c.split("|")[2]) for c in cells])
        ncan32 = ncan.astype(np.float32)
        CLS = classes(nA, brand, cont, ncan)
        bcode = pd.factorize(brand)[0]
        PK, RV, kept = load_dense(surv_idx, nW, nA)
        TOT = z["tot_sv"][surv_idx]
        d, Z, sj, sjg, peq = frame(PK, RV, TOT, CLS, ncan32, yidx, "allbeer")
        del PK, RV, TOT; gc.collect()
        n = len(sj)
        assert n == R36["n"], (n, R36["n"])
        nMk = int(d["mk"].max()) + 1
        assert nMk == R36["markets"], (nMk, R36["markets"])
        print(f"ASSERTED round 36: {nS:,} stores, {nMk:,} markets, {n:,} rows", flush=True)
        ST = []
        for yv in YEARS:
            ST.append(pd.read_csv(RMS / f"{yv}/Annual_Files/stores_{yv}.tsv", sep="\t",
                                  dtype={"store_code_uc": "string", "dma_code": "string"},
                                  usecols=["store_code_uc", "dma_code"],
                                  encoding="latin-1", low_memory=False))
        ST = pd.concat(ST, ignore_index=True)
        ST["store_code_uc"] = ST.store_code_uc.str.strip()
        s2d = ST.drop_duplicates("store_code_uc").set_index("store_code_uc").dma_code
        dcode = pd.factorize(s2d.reindex(stores[surv_idx]).fillna("NA").to_numpy())[0]
        cl = pd.factorize(dcode[d["un"]])[0]
        del ST, s2d; gc.collect()

        gal = d["al"].astype(np.int64); gwk = d["wk"].astype(np.int64)
        gst = d["un"].astype(np.int64)
        gsy = pd.factorize(gst * 4 + d["yr"])[0]
        gbs = pd.factorize(gst * 100 + bcode[gal])[0]
        LAD = [("alternative + week", [gal, gwk]),
               ("+ store", [gal, gwk, gst]),
               ("+ store x year", [gal, gwk, gsy]),
               ("+ brand x store", [gal, gwk, gsy, gbs])]

        # ---------------- 1.3 the two candidate tests, on the RAW classes
        print(f"\n=== 1.3  the two candidates, measured on the raw classes "
              f"[{time.time()-t0:.0f}s] ===", flush=True)
        yraw = np.log(sj.astype(np.float64)) - np.log(d["s0"].astype(np.float64))
        praw = peq.astype(np.float64)
        cand = []
        for gname, g in (("store", gst), ("brand x store", gbs)):
            gp = np.bincount(g, weights=praw) / np.bincount(g)
            gy = np.bincount(g, weights=yraw) / np.bincount(g)
            mp, my = gp[g], gy[g]
            for k, c in enumerate(R3):
                zk = Z[k].astype(np.float64)
                gz = np.bincount(g, weights=zk) / np.bincount(g)
                mz = gz[g]
                vb = float(np.var(mz)); vt = float(np.var(zk))
                cand.append(dict(group=gname, cls=c,
                                 corr_z_groupmean_price=float(np.corrcoef(zk, mp)[0, 1]),
                                 corr_z_groupmean_y=float(np.corrcoef(zk, my)[0, 1]),
                                 var_between=vb, var_total=vt,
                                 pct_between=100 * vb / vt))
                r = cand[-1]
                print(f"  {gname:14s} {c:26s} corr(z, group mean p) "
                      f"{r['corr_z_groupmean_price']:+.4f}  corr(z, group mean y) "
                      f"{r['corr_z_groupmean_y']:+.4f}  z variance between-group "
                      f"{r['pct_between']:5.1f}%", flush=True)
                del zk, gz, mz
            del gp, gy, mp, my
        pd.DataFrame(cand).to_csv(OUT / "p1_candidates.csv", index=False)
        print(f"  corr(price, y) raw: {np.corrcoef(praw, yraw)[0,1]:+.4f}")
        del yraw, praw; gc.collect()

        # ---------------- 1.1, 1.2, 1.4, 1.5 the ladder
        print(f"\n=== 1.1/1.2/1.4  the ladder [{time.time()-t0:.0f}s] ===", flush=True)
        rows, fsrows = [], []
        for lab, gr in LAD:
            X = np.empty((n, 6), np.float64, order="F")
            X[:, 0] = np.log(sj.astype(np.float64)) - np.log(d["s0"].astype(np.float64))
            X[:, 1] = peq
            X[:, 2] = np.log(peq.astype(np.float64))
            for j in range(3):
                X[:, 3 + j] = Z[j]
            A, it, mx = absorb(X, gr); del X; gc.collect()
            yd = A[:, 0]; Zd = A[:, 3:]
            for spec, colx, isl in (("level", 1, False), ("log", 2, True)):
                pd_ = A[:, colx]
                bfs, sefs, r2fs, ufs = ols_cluster(pd_, Zd, cl)
                brf, serf, r2rf, _ = ols_cluster(yd, Zd, cl)
                ph = pd_ - ufs
                b, V, _, G = M.tsls(yd, A[:, [colx]], Zd, cl)
                bo, Vo, _, _ = M.ols(yd, A[:, [colx]], cl)
                swf = M.sw_stats(pd_, np.empty((n, 0)), Zd, cl)
                J, Jdf, Jp, _ = M.hansen_j(yd, A[:, [colx]], Zd, cl)
                r = dict(spec=lab, price=spec, n=n, alpha_iv=float(b[0]),
                         se_iv=float(np.sqrt(V[0, 0])), alpha_ols=float(bo[0]),
                         sw_F=swf["sw_F"], hansen_J=J, hansen_p=Jp,
                         fs_partial_R2=r2fs, rf_partial_R2=r2rf,
                         num_phat_y=float(ph @ yd / n), den_phat_p=float(ph @ pd_ / n),
                         sd_y=float(np.std(yd)), sd_p=float(np.std(pd_)),
                         absorb_iters=it)
                for k, c in enumerate(R3):
                    r[f"fs_b_{c}"] = float(bfs[k]); r[f"fs_se_{c}"] = float(sefs[k])
                    r[f"fs_t_{c}"] = float(bfs[k] / sefs[k])
                    r[f"rf_b_{c}"] = float(brf[k]); r[f"rf_se_{c}"] = float(serf[k])
                    r[f"rf_t_{c}"] = float(brf[k] / serf[k])
                    r[f"sd_z_{c}"] = float(np.std(Zd[:, k]))
                    r[f"cov_p_{c}"] = float(pd_ @ Zd[:, k] / n)
                    r[f"cov_y_{c}"] = float(yd @ Zd[:, k] / n)
                rows.append(r)
                print(f"  {lab:20s} {spec:5s} a={b[0]:+9.4f}({np.sqrt(V[0,0]):.4f}) "
                      f"F={swf['sw_F']:7.2f} Jp={Jp:.4f} num={r['num_phat_y']:+.6f} "
                      f"den={r['den_phat_p']:+.6f} fsR2={r2fs:.5f} rfR2={r2rf:.5f}",
                      flush=True)
                print(f"      first stage  " + "  ".join(
                    f"{c.replace('d_','')}={bfs[k]:+8.4f}({sefs[k]:.4f}) t={bfs[k]/sefs[k]:+6.2f}"
                    for k, c in enumerate(R3)), flush=True)
                print(f"      reduced form " + "  ".join(
                    f"{c.replace('d_','')}={brf[k]:+8.4f}({serf[k]:.4f}) t={brf[k]/serf[k]:+6.2f}"
                    for k, c in enumerate(R3)), flush=True)
                print(f"      residual sd  " + "  ".join(
                    f"{c.replace('d_','')}={np.std(Zd[:,k]):8.4f}" for c in R3
                    for k in [R3.index(c)]), flush=True)
                del pd_, ufs, ph
            del A, yd, Zd; gc.collect()
        RW = pd.DataFrame(rows); RW.to_csv(OUT / "p1_ladder.csv", index=False)
        # consecutive-rung percentage changes in the projection pieces
        print(f"\n=== 1.2  consecutive-rung changes in the 2SLS projection ===", flush=True)
        chg = []
        for spec in ("level", "log"):
            g = RW.loc[RW.price == spec].reset_index(drop=True)
            for i in range(1, len(g)):
                a, bb = g.loc[i - 1], g.loc[i]
                dn = 100 * (bb.num_phat_y / a.num_phat_y - 1)
                dd = 100 * (bb.den_phat_p / a.den_phat_p - 1)
                pred = (1 + dn / 100) / (1 + dd / 100)
                obs = bb.alpha_iv / a.alpha_iv
                chg.append(dict(price=spec, step=f"{a.spec} -> {bb.spec}",
                                d_pct_num=dn, d_pct_den=dd, predicted_ratio=pred,
                                observed_ratio=obs, alpha_from=a.alpha_iv,
                                alpha_to=bb.alpha_iv))
                print(f"  {spec:5s} {a.spec:20s} -> {bb.spec:20s} num {dn:+10.2f}%  "
                      f"den {dd:+10.2f}%  ratio predicted {pred:+9.4f} observed "
                      f"{obs:+9.4f}  alpha {a.alpha_iv:+8.4f} -> {bb.alpha_iv:+8.4f}",
                      flush=True)
        pd.DataFrame(chg).to_csv(OUT / "p1_projection_changes.csv", index=False)
        json.dump(dict(n=int(n), markets=int(nMk), stores=int(nS)),
                  open(OUT / "p1_summary.json", "w"), indent=1)
        rm.add_outputs([OUT / f for f in ("p1_ladder.csv", "p1_projection_changes.csv",
                                          "p1_candidates.csv", "p1_summary.json")])
        print(f"\ndone [{time.time()-t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
