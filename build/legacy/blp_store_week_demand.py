#!/usr/bin/env python3
"""Round 36 Part 3: demand at the store-week with an all-beer outside good.

Markets are store-weeks.  The outside good is all RMS beer in the same store-week that is
not one of the model's fourteen brands, so `s_0` is a measured share of observed beer
volume rather than one minus a fixed fraction of an annual peak.  Quantities are 12-ounce
servings; for every model alternative size1_amount is 12, so a serving is a bottle there,
and the two diverge only inside the outside good.

Runs off the row file `blp_store_week_scan.py` wrote and the survivor mask
`blp_store_week_screens.py` wrote.  No movement pass.

    python3 paper/src/diagnostics/blp_store_week_demand.py
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
from blp_frequency_grid import R3, MULT                               # noqa: E402
from blp_rho_sweep import bracket, MW_RHO_NL                          # noqa: E402

RMS = _paths.RMS
OUT = PAPER / "outputs" / "blp_storeweek"
SCRATCH = _paths.WORK
ROW_DTYPE = np.dtype([("s", "i4"), ("w", "i2"), ("c", "i2"),
                      ("pk", "f4"), ("rv", "f4")])
YEARS = [2016, 2017, 2018, 2019]
CHUNK_M = 40_000                      # markets per frame-building chunk
REF = dict(alpha=-2.2878, se=0.2527, swF=27.27, el=-28.539, epsM=-16.752,
           div=62.59, s0=0.6243, pbar=11.779, n=2_595_065, markets=39_863)
MW = dict(market_nl=-1.10, market_rcnl=(-0.72, -0.60), own_nl=-3.81, div_nl=0.2980)


def absorb(X, groups, iters=150, tol=1e-9):
    """Alternating projections, column at a time and in place.

    `blp_pre_rcnl_diagnostics.absorbN` materialises an n x k group-mean expansion, which
    at store-week granularity is several gigabytes.  This does the same arithmetic one
    column at a time so the largest temporary is one column.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    gs = [(g, int(g.max()) + 1) for g in groups]
    cs = [np.maximum(np.bincount(g, minlength=n).astype(float), 1.0) for g, n in gs]
    it = 0
    for it in range(iters):
        mx = 0.0
        for (g, n), c in zip(gs, cs):
            for k in range(X.shape[1]):
                col = X[:, k]
                m = np.bincount(g, weights=col, minlength=n) / c
                col -= m[g]
                mx = max(mx, float(np.abs(m).max()))
        if mx < tol:
            break
    return X, it + 1, mx


def load_dense(surv_idx, nW, nA):
    """Row file -> dense (store, week, alternative) packs and revenue for survivors."""
    f = SCRATCH / "focal_rows.bin"
    nS = len(surv_idx)
    remap = np.full(int(surv_idx.max()) + 1, -1, np.int32)
    remap[surv_idx] = np.arange(nS, dtype=np.int32)
    pk = np.zeros(nS * nW * nA, np.float32)
    rv = np.zeros(nS * nW * nA, np.float32)
    total = f.stat().st_size // ROW_DTYPE.itemsize
    step = 20_000_000
    kept = 0
    for off in range(0, total, step):
        a = np.fromfile(f, dtype=ROW_DTYPE, count=min(step, total - off), offset=off * ROW_DTYPE.itemsize)
        s = a["s"]
        ok = (s <= remap.shape[0] - 1)
        si = np.where(ok, remap[np.clip(s, 0, remap.shape[0] - 1)], -1)
        m = si >= 0
        if m.any():
            flat = (si[m].astype(np.int64) * nW + a["w"][m]) * nA + a["c"][m]
            np.add.at(pk, flat, a["pk"][m])
            np.add.at(rv, flat, a["rv"][m])
            kept += int(m.sum())
        del a, s, si, m
    print(f"  aggregated {kept:,} of {total:,} focal rows into survivor stores",
          flush=True)
    return pk.reshape(nS, nW, nA), rv.reshape(nS, nW, nA), kept


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with RunManifest(OUT, args={}, inputs=[SCRATCH / "focal_rows.bin", RMS],
                     tag="store-week demand (round 36 Part 3)") as rm:
        rm.add_note("Store-week markets with an all-beer outside good. Linear IV only.")
        t0 = time.time()
        z = np.load(SCRATCH / "storeweek_totals.npz", allow_pickle=True)
        sv = np.load(SCRATCH / "survivors.npz")
        surv = sv["surv"]
        stores = pd.Index(z["stores"].astype(str))
        weeks = pd.DatetimeIndex(pd.to_datetime(z["weeks"].astype(str)))
        cells = z["cells"].astype(str)
        tot_sv = z["tot_sv"]
        nW, nA = len(weeks), len(cells)
        surv_idx = np.where(surv)[0]
        nS = len(surv_idx)
        yr_w = weeks.year.to_numpy()
        yidx = np.array([YEARS.index(y) for y in yr_w])
        print(f"survivor stores {nS:,}  weeks {nW}  alternatives {nA}", flush=True)
        brand = np.array([c.split("|")[0] for c in cells])
        cont = np.array([c.split("|")[1] for c in cells])
        ncan = np.array([int(c.split("|")[2]) for c in cells])
        sb = brand[:, None] == brand[None, :]; sc = cont[:, None] == cont[None, :]
        sn = ncan[:, None] == ncan[None, :]; eye = np.eye(nA, dtype=bool)
        CLS = {"d_same_brand_other_rung": (sb & sc & ~sn & ~eye).astype(np.float32),
               "d_same_rung_other_brand": (sc & sn & ~sb & ~eye).astype(np.float32),
               "d_same_container_other": (sc & ~sn & ~sb & ~eye).astype(np.float32)}

        PK, RV, kept = load_dense(surv_idx, nW, nA)
        TOT = tot_sv[surv_idx]                                  # (nS, nW) all-beer servings
        pres = PK > 0
        nrows = int(pres.sum())
        print(f"  store-alternative-weeks with positive volume: {nrows:,} "
              f"({nrows/max(int((PK.sum(2)>0).sum()),1):.1f} alternatives per market)",
              flush=True)
        ncan32 = ncan.astype(np.float32)
        ins = np.tensordot(PK, ncan32, axes=([2], [0]))         # focal servings
        with np.errstate(all="ignore"):
            S0 = 1.0 - ins / np.where(TOT > 0, TOT, np.nan)
        good_mkt = (TOT > 0) & np.isfinite(S0) & (S0 > 0) & (ins > 0)
        print(f"  markets: {int(good_mkt.sum()):,} of {nS*nW:,} store-weeks "
              f"({int(((TOT>0)&~good_mkt).sum()):,} dropped for s_0 <= 0 or no inside "
              f"volume)", flush=True)

        # -------- build the flat estimation frame in market chunks
        mflat = np.where(good_mkt.ravel())[0]
        nM = len(mflat)
        cols = {k: [] for k in ("mk", "st", "wk", "al", "yr", "serv", "rev", "s0")}
        zs = {k: [] for k in R3}
        for a in range(0, nM, CHUNK_M):
            idx = mflat[a:a + CHUNK_M]
            si, wi = idx // nW, idx % nW
            P = pres[si, wi, :]                                  # (m, nA)
            Sv = PK[si, wi, :] * ncan32[None, :]; Rv = RV[si, wi, :]
            zc = {k: P.astype(np.float32) @ v for k, v in CLS.items()}
            r, c = np.where(P)
            cols["mk"].append((a + r).astype(np.int64))
            cols["st"].append(si[r].astype(np.int32))
            cols["wk"].append(wi[r].astype(np.int16))
            cols["al"].append(c.astype(np.int16))
            cols["yr"].append(yidx[wi[r]].astype(np.int8))
            cols["serv"].append(Sv[r, c].astype(np.float64))
            cols["rev"].append(Rv[r, c].astype(np.float64))
            cols["s0"].append(S0[si[r], wi[r]].astype(np.float64))
            for k in R3:
                zs[k].append(zc[k][r, c].astype(np.float32))
            del P, Sv, Rv, zc
        del PK, RV, pres; gc.collect()
        d = {k: np.concatenate(v) for k, v in cols.items()}
        Z = np.column_stack([np.concatenate(zs[k]).astype(np.float64) for k in R3])
        del cols, zs; gc.collect()
        n = len(d["mk"])
        print(f"  estimation frame: {n:,} rows over {nM:,} markets "
              f"[{time.time()-t0:.0f}s]", flush=True)
        tot_m = np.bincount(d["mk"], weights=d["serv"])[d["mk"]]
        sjg = d["serv"] / tot_m
        # s_j = focal servings / all-beer servings = s_jg * (1 - s0)
        sj = sjg * (1.0 - d["s0"])
        peq = d["rev"] / (d["serv"] / 12.0)
        ok = (sj > 0) & (sjg > 0) & (d["s0"] > 0) & np.isfinite(peq) & (peq > 0)
        print(f"  usable rows {int(ok.sum()):,} ({100*ok.mean():.3f}%)", flush=True)
        for k in d:
            d[k] = d[k][ok]
        Z = Z[ok]; sj = sj[ok]; sjg = sjg[ok]; peq = peq[ok]
        n = len(sj)
        y = np.log(sj) - np.log(d["s0"])
        # market-level pieces for the elasticities
        mk2 = pd.factorize(d["mk"])[0]
        nMk = mk2.max() + 1
        wsum = np.bincount(mk2, weights=sj * peq)
        ssum = np.bincount(mk2, weights=sj)
        pbar = wsum / ssum
        s0m = np.zeros(nMk); s0m[mk2] = d["s0"]
        print(f"  markets retained {nMk:,}; s_0 median {np.median(s0m):.4f}; "
              f"pbar median {np.median(pbar):.4f}", flush=True)

        # DMA and store labels for clustering and composition
        ST = []
        for yv in YEARS:
            s = pd.read_csv(RMS / f"{yv}/Annual_Files/stores_{yv}.tsv", sep="\t",
                            dtype={"store_code_uc": "string", "dma_code": "string"},
                            usecols=["store_code_uc", "dma_code", "parent_code"],
                            encoding="latin-1", low_memory=False)
            ST.append(s)
        ST = pd.concat(ST, ignore_index=True)
        ST["store_code_uc"] = ST.store_code_uc.str.strip()
        s2d = ST.drop_duplicates("store_code_uc").set_index("store_code_uc").dma_code
        dma_of_store = s2d.reindex(stores[surv_idx]).fillna("NA").to_numpy()
        cl = pd.factorize(dma_of_store[d["st"]])[0]
        print(f"  clusters (DMA): {cl.max()+1}", flush=True)

        # -------- the fixed-effect ladder
        gal = d["al"].astype(np.int64)
        gwk = d["wk"].astype(np.int64)
        gst = d["st"].astype(np.int64)
        gsy = (gst * len(YEARS) + d["yr"]).astype(np.int64)
        gsy = pd.factorize(gsy)[0]
        gbs = pd.factorize(gst * 1000 + np.array([np.where(np.unique(brand) == b)[0][0]
                                                  for b in brand])[gal])[0]
        LAD = [("alternative + week", [gal, gwk]),
               ("+ store", [gal, gwk, gst]),
               ("+ store x year", [gal, gwk, gsy]),
               ("+ brand x store", [gal, gwk, gsy, gbs])]
        print(f"\n=== 3.1  the fixed-effect ladder: variation before coefficients ===",
              flush=True)
        var_rows = []
        sd_p_raw = float(np.std(peq))
        for lab, gr in LAD:
            gc_ = [int(g.max()) + 1 for g in gr]
            sing = [int((np.bincount(g) == 1).sum()) for g in gr]
            Pd, it, mx = absorb(peq.reshape(-1, 1).copy(), gr)
            sdp = float(np.std(Pd[:, 0]))
            var_rows.append(dict(spec=lab, groups=sum(gc_), group_counts=str(gc_),
                                 singletons=sum(sing), sd_price_raw=sd_p_raw,
                                 sd_price_absorbed=sdp, pct_price_var_left=100*sdp/sd_p_raw,
                                 iters=it, tol=mx))
            print(f"  {lab:22s} groups {str(gc_):28s} singletons {sum(sing):>7,}  "
                  f"price sd {sd_p_raw:.4f} -> {sdp:.4f} ({100*sdp/sd_p_raw:5.1f}% left)"
                  f"  [{it} iters, {mx:.1e}]", flush=True)
            del Pd
        pd.DataFrame(var_rows).to_csv(OUT / "p3_variation.csv", index=False)

        # -------- estimation
        print(f"\n=== 3.2  estimates [{time.time()-t0:.0f}s] ===", flush=True)
        rows = []
        lsjg = np.log(sjg)
        for lab, gr in LAD:
            X = np.column_stack([y, peq, np.log(peq), lsjg, Z])
            A, it, mx = absorb(X, gr)
            del X; gc.collect()
            for spec, colx, isl in (("level", 1, False), ("log", 2, True)):
                b, V, u_, G = M.tsls(A[:, 0], A[:, [colx]], A[:, 4:], cl)
                bo, Vo, _, _ = M.ols(A[:, 0], A[:, [colx]], cl)
                swf = M.sw_stats(A[:, colx], np.empty((n, 0)), A[:, 4:], cl)
                J, Jdf, Jp, _ = M.hansen_j(A[:, 0], A[:, [colx]], A[:, 4:], cl)
                r = dict(spec=lab, price=spec, n=n, markets=nMk, clusters=G,
                         alpha_iv=float(b[0]), se_iv=float(np.sqrt(V[0, 0])),
                         alpha_ols=float(bo[0]), ols_to_iv=float(b[0] / bo[0]),
                         sw_F=swf["sw_F"], hansen_J=J, hansen_df=Jdf, hansen_p=Jp,
                         pbar_p50=float(np.median(pbar)), s0_p50=float(np.median(s0m)))
                for rho in (0.0, MW_RHO_NL):
                    br = bracket(rho, sjg, sj)
                    a_rho = b[0] if rho == 0.0 else b[0] * (1 - rho)
                    el = (a_rho if isl else a_rho * peq) * br
                    epsM = a_rho * s0m * (1.0 if isl else pbar)
                    r[f"elast_p50_rho{rho:g}"] = float(np.median(el))
                    r[f"market_elast_p50_rho{rho:g}"] = float(np.median(epsM))
                    r[f"diversion_p50_rho{rho:g}"] = float(100*np.median(d["s0"]/br))
                rows.append(r)
                print(f"  {lab:22s} {spec:5s} a={b[0]:+9.4f}({np.sqrt(V[0,0]):.4f}) "
                      f"OLS{bo[0]:+8.4f} r={b[0]/bo[0]:6.2f} F={swf['sw_F']:8.2f} "
                      f"J={J:6.2f} p={Jp:.3f} el={r['elast_p50_rho0']:+8.2f} "
                      f"eM={r['market_elast_p50_rho0']:+7.3f} "
                      f"div={r['diversion_p50_rho0']:5.2f}%", flush=True)
            del A; gc.collect()
        RW = pd.DataFrame(rows); RW.to_csv(OUT / "p3_cells.csv", index=False)
        json.dump(dict(n=int(n), markets=int(nMk), stores=int(nS),
                       s0_p50=float(np.median(s0m)), pbar_p50=float(np.median(pbar))),
                  open(OUT / "p3_summary.json", "w"), indent=1)
        rm.add_outputs([OUT / f for f in ("p3_variation.csv", "p3_cells.csv",
                                          "p3_summary.json")])
        print(f"\ndone [{time.time()-t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
