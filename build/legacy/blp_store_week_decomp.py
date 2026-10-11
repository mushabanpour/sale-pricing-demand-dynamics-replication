#!/usr/bin/env python3
"""Round 36 Parts 3.4 and 3.5: what the store-week redesign actually changed.

Part 3 moved two things at once -- the market from the DMA-week to the store-week, and
the outside good from `1.5 x annual peak` to all RMS beer outside the model's brands --
and `alpha` went from -2.2878 to -1.1280 with the analogous fixed effects, and to about
zero without brand-by-unit effects.  Without separating them the change cannot be
attributed, so this runs the 2x2 on one fixed sample, the 5,208 surviving stores.

The store-week cell with the all-beer denominator is Part 3's `+ brand x store` rung and
is not recomputed here.  Part 3.5 as the prompt specifies it is the DMA-week peak-rule
cell; it is reported with the servings quantity for comparability with the rest of the
2x2 and again with the standing frame's own PACK numerator, since D076 established the
convention is worth 7% of the market elasticity.

Everything is kept in float32 until the absorption, because at 52.8 million rows a
float64 frame plus a float64 design matrix exceeds this machine's memory and the run
pages instead of computing.

    python3 paper/src/diagnostics/blp_store_week_decomp.py
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
from blp_store_week_demand import absorb, load_dense, ROW_DTYPE       # noqa: E402

RMS = _paths.RMS
OUT = PAPER / "outputs" / "blp_storeweek"
SCRATCH = _paths.WORK
YEARS = [2016, 2017, 2018, 2019]
CHUNK_M = 25_000
BANDS = [(0.0, 0.5), (0.5, 5.0), (5.0, 95.0), (95.0, 99.5), (99.5, 100.0)]
PACKS = [6, 12, 18, 24, 30, 36]
REF = dict(alpha=-2.2878, se=0.2527, swF=27.27, el=-28.539, epsM=-16.752, div=62.59)


def classes(nA, brand, cont, ncan):
    sb = brand[:, None] == brand[None, :]; sc = cont[:, None] == cont[None, :]
    sn = ncan[:, None] == ncan[None, :]; eye = np.eye(nA, dtype=bool)
    return {"d_same_brand_other_rung": (sb & sc & ~sn & ~eye).astype(np.float32),
            "d_same_rung_other_brand": (sc & sn & ~sb & ~eye).astype(np.float32),
            "d_same_container_other": (sc & ~sn & ~sb & ~eye).astype(np.float32)}


def frame(PK, RV, TOT, CLS, ncan32, yidx, rule, qty="servings", mask=None):
    """(unit, week, alternative) grid -> a float32 estimation frame."""
    nU, nW, nA = PK.shape
    keep = (PK > 0) if mask is None else ((PK > 0) & mask)
    ins = np.zeros((nU, nW), np.float32); rvm = np.zeros((nU, nW), np.float32)
    for i in range(nU):
        q = PK[i] * ncan32[None, :] if qty == "servings" else PK[i]
        ins[i] = np.where(keep[i], q, 0.0).sum(1)
    if rule == "allbeer":
        with np.errstate(all="ignore"):
            S0 = 1.0 - ins / np.where(TOT > 0, TOT, np.nan)
    else:
        Mry = np.zeros_like(ins)
        for k in range(len(YEARS)):
            s = yidx == k
            Mry[:, s] = (MULT * ins[:, s].max(1))[:, None]
        with np.errstate(all="ignore"):
            S0 = 1.0 - ins / np.where(Mry > 0, Mry, np.nan)
    good = np.isfinite(S0) & (S0 > 0) & (ins > 0)
    del ins, rvm; gc.collect()
    mflat = np.where(good.ravel())[0]
    out = {k: [] for k in ("mk", "un", "wk", "al", "yr", "q", "rev", "s0")}
    zz = [[] for _ in R3]
    for a in range(0, len(mflat), CHUNK_M):
        idx = mflat[a:a + CHUNK_M]
        ui, wi = idx // nW, idx % nW
        P = keep[ui, wi, :]
        Qv = PK[ui, wi, :] * (ncan32[None, :] if qty == "servings" else 1.0)
        Rv = RV[ui, wi, :]
        zc = [P.astype(np.float32) @ CLS[k] for k in R3]
        r, c = np.where(P)
        out["mk"].append((a + r).astype(np.int32)); out["un"].append(ui[r].astype(np.int32))
        out["wk"].append(wi[r].astype(np.int16)); out["al"].append(c.astype(np.int16))
        out["yr"].append(yidx[wi[r]].astype(np.int8))
        out["q"].append(Qv[r, c].astype(np.float32))
        out["rev"].append(Rv[r, c].astype(np.float32))
        out["s0"].append(S0[ui[r], wi[r]].astype(np.float32))
        for j in range(len(R3)):
            zz[j].append(zc[j][r, c].astype(np.float32))
        del P, Qv, Rv, zc
    d = {k: np.concatenate(v) for k, v in out.items()}
    Z = [np.concatenate(v) for v in zz]
    del out, zz, keep, good, S0; gc.collect()
    tot_m = np.bincount(d["mk"], weights=d["q"].astype(np.float64))
    sjg = (d["q"] / tot_m[d["mk"]]).astype(np.float32)
    sj = (sjg * (1.0 - d["s0"])).astype(np.float32)
    eq = d["q"] / 12.0 if qty == "servings" else d["q"] * ncan32[d["al"]] / 12.0
    peq = (d["rev"] / eq).astype(np.float32)
    del tot_m, eq
    ok = (sj > 0) & (sjg > 0) & (d["s0"] > 0) & np.isfinite(peq) & (peq > 0)
    if not ok.all():
        for k in d:
            d[k] = d[k][ok]
        Z = [v[ok] for v in Z]; sj = sj[ok]; sjg = sjg[ok]; peq = peq[ok]
    del d["q"], d["rev"]; gc.collect()
    return d, Z, sj, sjg, peq


def est(d, Z, sj, sjg, peq, groups, cl, tag, rows, extra=None):
    n = len(sj)
    X = np.empty((n, 6), np.float64, order="F")
    X[:, 0] = np.log(sj.astype(np.float64)) - np.log(d["s0"].astype(np.float64))
    X[:, 1] = peq
    X[:, 2] = np.log(peq.astype(np.float64))
    for j in range(3):
        X[:, 3 + j] = Z[j]
    A, it, mx = absorb(X, groups); del X; gc.collect()
    mk = pd.factorize(d["mk"])[0]
    sjd = sj.astype(np.float64); pqd = peq.astype(np.float64)
    pbar = np.bincount(mk, weights=sjd * pqd) / np.bincount(mk, weights=sjd)
    s0m = np.zeros(mk.max() + 1); s0m[mk] = d["s0"]
    del sjd, pqd
    out = {}
    for spec, colx, isl in (("level", 1, False), ("log", 2, True)):
        b, V, _, G = M.tsls(A[:, 0], A[:, [colx]], A[:, 3:], cl)
        bo, Vo, _, _ = M.ols(A[:, 0], A[:, [colx]], cl)
        swf = M.sw_stats(A[:, colx], np.empty((n, 0)), A[:, 3:], cl)
        J, Jdf, Jp, _ = M.hansen_j(A[:, 0], A[:, [colx]], A[:, 3:], cl)
        r = dict(cell=tag, price=spec, n=n, markets=int(mk.max() + 1), clusters=G,
                 alpha_iv=float(b[0]), se_iv=float(np.sqrt(V[0, 0])),
                 alpha_ols=float(bo[0]), ols_to_iv=float(b[0] / bo[0]),
                 sw_F=swf["sw_F"], hansen_J=J, hansen_df=Jdf, hansen_p=Jp,
                 absorb_iters=it, absorb_tol=mx,
                 pbar_p50=float(np.median(pbar)), s0_p50=float(np.median(s0m)))
        for rho in (0.0, MW_RHO_NL):
            br = bracket(rho, sjg.astype(np.float64), sj.astype(np.float64))
            a_rho = b[0] if rho == 0.0 else b[0] * (1 - rho)
            r[f"elast_p50_rho{rho:g}"] = float(np.median((a_rho if isl else a_rho*peq)*br))
            r[f"market_elast_p50_rho{rho:g}"] = float(np.median(a_rho * s0m
                                                               * (1.0 if isl else pbar)))
            r[f"diversion_p50_rho{rho:g}"] = float(100 * np.median(d["s0"] / br))
            del br
        if extra:
            r.update(extra)
        rows.append(r); out[spec] = r
        print(f"  {tag:42s} {spec:5s} n={n:>10,} a={b[0]:+9.4f}({np.sqrt(V[0,0]):.4f}) "
              f"OLS{bo[0]:+8.4f} F={swf['sw_F']:8.2f} Jp={Jp:.4f} "
              f"s0={r['s0_p50']:.4f} pbar={r['pbar_p50']:7.3f} "
              f"el={r['elast_p50_rho0']:+9.3f} eM={r['market_elast_p50_rho0']:+8.3f}",
              flush=True)
    del A; gc.collect()
    return out


def fs_pieces(d, Z, sj, sjg, peq, groups, cl):
    n = len(sj)
    X = np.empty((n, 5), np.float64, order="F")
    X[:, 0] = np.log(sj.astype(np.float64)) - np.log(d["s0"].astype(np.float64))
    X[:, 1] = peq
    for j in range(3):
        X[:, 2 + j] = Z[j]
    A, _, _ = absorb(X, groups); del X; gc.collect()
    yd, pd_, Zd = A[:, 0], A[:, 1], A[:, 2:]
    ph = Zd @ np.linalg.lstsq(Zd, pd_, rcond=None)[0]
    b, V, _, G = M.tsls(yd, A[:, [1]], Zd, cl)
    sw = M.sw_stats(pd_, np.empty((n, 0)), Zd, cl)
    r = dict(n=n, alpha=float(b[0]), se=float(np.sqrt(V[0, 0])), sw_F=sw["sw_F"],
             num=float(ph @ yd / n), den=float(ph @ pd_ / n))
    del A, ph; gc.collect()
    return r


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with RunManifest(OUT, args={}, inputs=[SCRATCH / "focal_rows.bin", RMS],
                     tag="store-week decomposition (round 36 Parts 3.4-3.5)") as rm:
        rm.add_note("Separates the market redefinition from the denominator change on one "
                    "fixed sample, then the first-stage band decomposition.")
        t0 = time.time()
        z = np.load(SCRATCH / "storeweek_totals.npz", allow_pickle=True)
        sv = np.load(SCRATCH / "survivors.npz")
        surv = sv["surv"]
        stores = pd.Index(z["stores"].astype(str))
        weeks = pd.DatetimeIndex(pd.to_datetime(z["weeks"].astype(str)))
        cells = z["cells"].astype(str)
        tot_all = z["tot_sv"]
        nW, nA = len(weeks), len(cells)
        surv_idx = np.where(surv)[0]; nS = len(surv_idx)
        yidx = np.array([YEARS.index(y) for y in weeks.year])
        brand = np.array([c.split("|")[0] for c in cells])
        cont = np.array([c.split("|")[1] for c in cells])
        ncan = np.array([int(c.split("|")[2]) for c in cells])
        ncan32 = ncan.astype(np.float32)
        CLS = classes(nA, brand, cont, ncan)
        bcode = pd.factorize(brand)[0]
        ST = []
        for yv in YEARS:
            ST.append(pd.read_csv(RMS / f"{yv}/Annual_Files/stores_{yv}.tsv", sep="\t",
                                  dtype={"store_code_uc": "string", "dma_code": "string"},
                                  usecols=["store_code_uc", "dma_code"],
                                  encoding="latin-1", low_memory=False))
        ST = pd.concat(ST, ignore_index=True)
        ST["store_code_uc"] = ST.store_code_uc.str.strip()
        s2d = ST.drop_duplicates("store_code_uc").set_index("store_code_uc").dma_code
        dma_s = s2d.reindex(stores[surv_idx]).fillna("NA").to_numpy()
        dcode, dlab = pd.factorize(dma_s); nD = len(dlab)
        del ST, s2d; gc.collect()
        rows = []

        # ---------------- DMA-week cells first: cheap, and Part 3.5 lives here
        PK, RV, kept = load_dense(surv_idx, nW, nA)
        TOT = tot_all[surv_idx]
        PKd = np.zeros((nD, nW, nA), np.float32); RVd = np.zeros_like(PKd)
        TOTd = np.zeros((nD, nW), np.float32)
        for i in range(nS):
            PKd[dcode[i]] += PK[i]; RVd[dcode[i]] += RV[i]; TOTd[dcode[i]] += TOT[i]
        del PK, RV, TOT; gc.collect()
        print(f"\n=== DMA-week cells on the 5,208 surviving stores, {nD} DMAs "
              f"[{time.time()-t0:.0f}s] ===", flush=True)
        for rule, qty, lab in (("allbeer", "servings", "DMA-week, all-beer outside good"),
                               ("peak", "servings", "DMA-week, 1.5x peak (servings)"),
                               ("peak", "packs", "DMA-week, 1.5x peak (PACKS = standing)")):
            d, Z, sj, sjg, peq = frame(PKd, RVd, TOTd, CLS, ncan32, yidx, rule, qty=qty)
            gdy = pd.factorize(d["un"].astype(np.int64) * 4 + d["yr"])[0]
            gbd = pd.factorize(d["un"].astype(np.int64) * 100 + bcode[d["al"]])[0]
            gr = [d["al"].astype(np.int64), d["wk"].astype(np.int64), gdy, gbd]
            cl = pd.factorize(d["un"])[0]
            est(d, Z, sj, sjg, peq, gr, cl, lab, rows,
                extra=dict(rule=rule, market="DMA-week", qty=qty))
            del d, Z, sj, sjg, peq, gr, gdy, gbd, cl; gc.collect()
        del PKd, RVd, TOTd; gc.collect()
        pd.DataFrame(rows).to_csv(OUT / "p3_decomposition.csv", index=False)

        # ---------------- the store-week peak-rule cell
        PK, RV, kept = load_dense(surv_idx, nW, nA)
        TOT = tot_all[surv_idx]
        print(f"\n=== the store-week peak-rule cell [{time.time()-t0:.0f}s] ===",
              flush=True)
        d, Z, sj, sjg, peq = frame(PK, RV, TOT, CLS, ncan32, yidx, "peak")
        gsy = pd.factorize(d["un"].astype(np.int64) * 4 + d["yr"])[0]
        gbs = pd.factorize(d["un"].astype(np.int64) * 100 + bcode[d["al"]])[0]
        gr = [d["al"].astype(np.int64), d["wk"].astype(np.int64), gsy, gbs]
        cl = pd.factorize(dcode[d["un"]])[0]
        est(d, Z, sj, sjg, peq, gr, cl, "STORE-week, 1.5x peak (servings)", rows,
            extra=dict(rule="peak", market="store-week", qty="servings"))
        del d, Z, sj, sjg, peq, gr, gsy, gbs, cl; gc.collect()
        pd.DataFrame(rows).to_csv(OUT / "p3_decomposition.csv", index=False)

        # ---------------- 3.4 the band decomposition at store-week
        print(f"\n=== 3.4  first-stage contribution by within-rung price band "
              f"[{time.time()-t0:.0f}s] ===", flush=True)
        nzp = PK > 0
        cuts = {}
        vol_tot = 0.0
        for p in PACKS:
            sel = np.where(ncan == p)[0]
            if not len(sel):
                continue
            col = (RV[:, :, sel] / (PK[:, :, sel] * p))[nzp[:, :, sel]]
            col = col[np.isfinite(col)]
            for q in sorted({x for bd in BANDS for x in bd}):
                cuts[(p, q)] = float(np.percentile(col, q))
            vol_tot += float((PK[:, :, sel] * p)[nzp[:, :, sel]].sum())
            del col
        gc.collect()
        p3 = []
        for lab, band in [("baseline", None)] + [(f"drop p{a}-p{b}", (a, b))
                                                 for a, b in BANDS]:
            if band is None:
                mask = None; dv = 0.0
            else:
                lo, hi = band
                mask = np.ones_like(nzp)
                dvol = 0.0
                for p in PACKS:
                    sel = np.where(ncan == p)[0]
                    if not len(sel):
                        continue
                    with np.errstate(all="ignore"):
                        sp = RV[:, :, sel] / (PK[:, :, sel] * p)
                    hb = cuts[(p, hi)] if hi < 100 else np.inf
                    inb = nzp[:, :, sel] & (sp <= hb)
                    if lo > 0:
                        inb &= sp > cuts[(p, lo)]
                    mask[:, :, sel] = ~inb
                    dvol += float((PK[:, :, sel] * p)[inb].sum())
                    del sp, inb
                dv = 100 * dvol / vol_tot
                gc.collect()
            d, Z, sj, sjg, peq = frame(PK, RV, TOT, CLS, ncan32, yidx, "allbeer",
                                       mask=mask)
            gsy = pd.factorize(d["un"].astype(np.int64) * 4 + d["yr"])[0]
            gbs = pd.factorize(d["un"].astype(np.int64) * 100 + bcode[d["al"]])[0]
            cl = pd.factorize(dcode[d["un"]])[0]
            r = fs_pieces(d, Z, sj, sjg, peq,
                          [d["al"].astype(np.int64), d["wk"].astype(np.int64), gsy, gbs],
                          cl)
            r.update(variant=lab, pct_volume=dv)
            if p3:
                base = p3[0]
                r.update(d_pct_den=100 * (r["den"] / base["den"] - 1),
                         d_pct_num=100 * (r["num"] / base["num"] - 1),
                         d_pct_F=100 * (r["sw_F"] / base["sw_F"] - 1))
            p3.append(r)
            print(f"  {lab:20s} vol {dv:6.3f}%  n={r['n']:>10,} alpha{r['alpha']:+9.4f} "
                  f"F={r['sw_F']:8.2f} den={r['den']:+.6f} num={r['num']:+.6f}"
                  + (f"  den {r['d_pct_den']:+8.2f}% num {r['d_pct_num']:+8.2f}%"
                     if "d_pct_den" in r else ""), flush=True)
            del d, Z, sj, sjg, peq, gsy, gbs, cl, mask; gc.collect()
            pd.DataFrame(p3).to_csv(OUT / "p3_bands.csv", index=False)
        rm.add_outputs([OUT / "p3_decomposition.csv", OUT / "p3_bands.csv"])
        print(f"\ndone [{time.time()-t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
