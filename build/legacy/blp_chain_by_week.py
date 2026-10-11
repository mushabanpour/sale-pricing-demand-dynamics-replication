#!/usr/bin/env python3
"""Round 39 Parts 1 and 2: chain-by-week effects, and the corrected focal set.

D092 found the pooled first stage diluted rather than cancelled -- the pooled coefficients
are 13% and 30% LARGER than their row-weighted component means while the pooled SW F is
0.476 of the component mean of 31.62 -- and named a parent-by-week effect as the direct
remedy, because brand-by-store absorbs a chain's price LEVEL but not the interaction of its
pricing strategy with week and alternative.  This tests it.

Part 2 applies D090's and D095's string rules by subtracting the 29 excluded UPCs'
store-cell-week contributions from the grid.  The excluded units stay in the data: they are
RMS beer, so they move from the inside goods into the all-beer outside good, which needs no
change to the denominator.

Run on the unbalanced grocery-and-liquor panel with the volume floor on the median week --
D093's headline sample -- at the one-in-four store subsample the reference cell uses.

    python3 paper/src/diagnostics/blp_chain_by_week.py
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
from blp_store_week_demand import absorb, load_dense, ROW_DTYPE       # noqa: E402
from blp_store_week_decomp import classes, frame                      # noqa: E402
from blp_sw_first_stage_ladder import ols_cluster                     # noqa: E402

RMS = _paths.RMS
SW = PAPER / "outputs" / "blp_storeweek"
OUT = PAPER / "outputs" / "blp_chainweek"
SCRATCH = _paths.WORK
YEARS = [2016, 2017, 2018, 2019]
BOTTLE_FLOOR = 2000
EVERY = 4
REF = dict(alpha=-1.0528, se=0.2031, swF=24.81, Jp=0.017, s0=0.5581, el=-13.08,
           epsM=-6.346, stores=2716, n=20_524_883, clusters=184,
           component_F=31.62)


def load_excluded(surv_idx, nW, nA):
    f = SCRATCH / "excluded_rows.bin"
    nS = len(surv_idx)
    remap = np.full(int(surv_idx.max()) + 1, -1, np.int32)
    remap[surv_idx] = np.arange(nS, dtype=np.int32)
    pk = np.zeros(nS * nW * nA, np.float32); rv = np.zeros_like(pk)
    total = f.stat().st_size // ROW_DTYPE.itemsize
    a = np.fromfile(f, dtype=ROW_DTYPE, count=total)
    s = np.clip(a["s"], 0, remap.shape[0] - 1)
    m = (a["s"] <= remap.shape[0] - 1) & (remap[s] >= 0)
    flat = (remap[s[m]].astype(np.int64) * nW + a["w"][m]) * nA + a["c"][m]
    np.add.at(pk, flat, a["pk"][m]); np.add.at(rv, flat, a["rv"][m])
    print(f"  excluded rows in this store set: {int(m.sum()):,} of {total:,}", flush=True)
    return pk.reshape(nS, nW, nA), rv.reshape(nS, nW, nA), int(m.sum())


def est(d, Z, sj, sjg, peq, groups, cl, tag, rows, extra):
    n = len(sj)
    sd_raw = [float(np.std(v)) for v in Z]
    X = np.empty((n, 6), np.float64, order="F")
    X[:, 0] = np.log(sj.astype(np.float64)) - np.log(d["s0"].astype(np.float64))
    X[:, 1] = peq; X[:, 2] = np.log(peq.astype(np.float64))
    for j in range(3):
        X[:, 3 + j] = Z[j]
    A, it, mx = absorb(X, groups); del X; gc.collect()
    gc_ = [int(g.max()) + 1 for g in groups]
    sing = [int((np.bincount(g) == 1).sum()) for g in groups]
    u5 = [int((np.bincount(g) < 5).sum()) for g in groups]
    sd_abs = [float(np.std(A[:, 3 + j])) for j in range(3)]
    print(f"  {tag}", flush=True)
    print(f"    groups {gc_} singletons {sing} under-5 {u5}")
    print(f"    instrument sd raw -> absorbed: " + "  ".join(
        f"{c.replace('d_same_','')}: {sd_raw[j]:.4f} -> {sd_abs[j]:.4f} "
        f"({100*sd_abs[j]/sd_raw[j]:.1f}% left)" for j, c in enumerate(R3)), flush=True)
    out = {}
    for spec, colx, isl in (("level", 1, False), ("log", 2, True)):
        pdv, Zd = A[:, colx], A[:, 3:]
        bfs, sefs, r2fs, _ = ols_cluster(pdv, Zd, cl)
        b, V, _, G = M.tsls(A[:, 0], A[:, [colx]], Zd, cl)
        bo, _, _, _ = M.ols(A[:, 0], A[:, [colx]], cl)
        swf = M.sw_stats(pdv, np.empty((n, 0)), Zd, cl)
        J, Jdf, Jp, _ = M.hansen_j(A[:, 0], A[:, [colx]], Zd, cl)
        br = bracket(0.0, sjg.astype(np.float64), sj.astype(np.float64))
        el = (b[0] if isl else b[0] * peq.astype(np.float64)) * br
        s0m = d["s0"].astype(np.float64)
        pbar = float(np.average(peq.astype(np.float64), weights=sj.astype(np.float64)))
        r = dict(cell=tag, price=spec, n=n, clusters=G, groups=str(gc_),
                 singletons=sum(sing), alpha_iv=float(b[0]),
                 se_iv=float(np.sqrt(V[0, 0])), t_iv=float(b[0] / np.sqrt(V[0, 0])),
                 alpha_ols=float(bo[0]), ols_to_iv=float(b[0] / bo[0]),
                 sw_F=swf["sw_F"], hansen_J=J, hansen_df=Jdf, hansen_p=Jp,
                 fs_partial_R2=r2fs, s0_p50=float(np.median(s0m)), pbar=pbar,
                 elast_p50=float(np.median(el)),
                 market_elast=float(b[0] * (1.0 if isl else pbar) * np.median(s0m)),
                 **extra)
        for j, c in enumerate(R3):
            r[f"fs_b_{c}"] = float(bfs[j]); r[f"fs_se_{c}"] = float(sefs[j])
            r[f"fs_t_{c}"] = float(bfs[j] / sefs[j])
            r[f"sd_z_raw_{c}"] = sd_raw[j]; r[f"sd_z_abs_{c}"] = sd_abs[j]
        rows.append(r); out[spec] = r
        del br, el
    o = out["level"]
    print(f"    LVL a={o['alpha_iv']:+9.4f}({o['se_iv']:.4f}) t={o['t_iv']:+7.2f} "
          f"OLS{o['alpha_ols']:+8.4f} r={o['ols_to_iv']:6.2f} **F={o['sw_F']:8.2f}** "
          f"Jp={o['hansen_p']:.4f} s0={o['s0_p50']:.4f} el={o['elast_p50']:+8.2f} "
          f"eM={o['market_elast']:+7.3f} | fs " + " ".join(
              f"{o[f'fs_b_{c}']:+7.4f}({o[f'fs_t_{c}']:+5.1f})" for c in R3)
          + f" | LOG a={out['log']['alpha_iv']:+10.4f} F={out['log']['sw_F']:8.2f}",
          flush=True)
    del A; gc.collect()
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with RunManifest(OUT, args={"every": EVERY}, inputs=[SCRATCH / "focal_rows.bin", RMS],
                     tag="chain-by-week and the corrected set (round 39 Parts 1-2)") as rm:
        rm.add_note("Parent-by-week and store-by-week effects on the unbalanced panel, and "
                    "the corrected focal set. Nothing under existing outputs modified.")
        t0 = time.time()
        z = np.load(SCRATCH / "storeweek_totals.npz", allow_pickle=True)
        sv = np.load(SCRATCH / "survivors.npz")
        stores = pd.Index(z["stores"].astype(str))
        weeks = pd.DatetimeIndex(pd.to_datetime(z["weeks"].astype(str)))
        cells = z["cells"].astype(str)
        nW, nA = len(weeks), len(cells)
        tot_bt, tot_sv_all = z["tot_bt"], z["tot_sv"]
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
                                  dtype={"store_code_uc": "string",
                                         "parent_code": "string", "dma_code": "string"},
                                  usecols=["store_code_uc", "year", "parent_code",
                                           "dma_code"], encoding="latin-1",
                                  low_memory=False))
        ST = pd.concat(ST, ignore_index=True)
        ST["store_code_uc"] = ST.store_code_uc.str.strip()
        last = ST.sort_values("year").drop_duplicates("store_code_uc", keep="last") \
                 .set_index("store_code_uc")
        par_all = last.parent_code.reindex(stores).fillna("NA").to_numpy()
        dma_all = last.dma_code.reindex(stores).fillna("NA").to_numpy()
        del ST, last; gc.collect()

        pres = tot_bt > 0
        univ = sv["univ"]
        med = np.full(len(stores), np.nan)
        for i in np.where(univ)[0]:
            v = tot_bt[i][pres[i]]
            if len(v):
                med[i] = np.median(v)
        unb = univ & (np.nan_to_num(med) >= BOTTLE_FLOOR)
        si = np.where(unb)[0][::EVERY]
        print(f"unbalanced universe {int(unb.sum()):,}; one-in-four {len(si):,} stores; "
              f"parents {len(set(par_all[si]))}", flush=True)
        assert len(si) == REF["stores"], (len(si), REF["stores"])

        PK, RV, kept = load_dense(si, nW, nA)
        TOT = tot_sv_all[si]
        pcode = pd.factorize(par_all[si])[0]
        dcode = pd.factorize(dma_all[si])[0]
        rows = []

        # ---------------- Part 1
        d, Z, sj, sjg, peq = frame(PK, RV, TOT, CLS, ncan32, yidx, "allbeer")
        n = len(sj)
        print(f"\n=== Part 1  the fixed-effect variants [{time.time()-t0:.0f}s] ===",
              flush=True)
        print(f"  frame {n:,} rows (reference {REF['n']:,}), "
              f"{int(d['mk'].max())+1:,} markets", flush=True)
        assert n == REF["n"], (n, REF["n"])
        gst = d["un"].astype(np.int64)
        gal = d["al"].astype(np.int64); gwk = d["wk"].astype(np.int64)
        gsy = pd.factorize(gst * 4 + d["yr"])[0]
        gbs = pd.factorize(gst * 100 + bcode[d["al"]])[0]
        gpw = pd.factorize(pcode[d["un"]].astype(np.int64) * 1000 + d["wk"])[0]
        gswk = pd.factorize(gst * 1000 + d["wk"])[0]
        cl = pd.factorize(dcode[d["un"]])[0]
        base = dict(kind="uncorrected", stores=len(si))
        est(d, Z, sj, sjg, peq, [gal, gwk, gsy, gbs], cl,
            "REFERENCE alt + week + store x year + brand x store", rows,
            dict(fe="reference", **base))
        est(d, Z, sj, sjg, peq, [gal, gwk, gsy, gbs, gpw], cl,
            "+ PARENT x WEEK", rows, dict(fe="parent x week", **base))
        est(d, Z, sj, sjg, peq, [gal, gsy, gbs, gswk], cl,
            "+ STORE x WEEK (the market effect; week is nested in it)", rows,
            dict(fe="store x week", **base))
        pd.DataFrame(rows).to_csv(OUT / "p1_fe_variants.csv", index=False)

        # ---------------- Part 2: the corrected set
        print(f"\n=== Part 2  the corrected focal set [{time.time()-t0:.0f}s] ===",
              flush=True)
        XPK, XRV, nex = load_excluded(si, nW, nA)
        # per-alternative effect of the correction, before any estimate
        eff = []
        for a in range(nA):
            p0 = PK[:, :, a].sum(); r0 = RV[:, :, a].sum()
            pe = XPK[:, :, a].sum(); re = XRV[:, :, a].sum()
            if pe <= 0:
                continue
            eff.append(dict(cell=cells[a], packs=float(p0), packs_removed=float(pe),
                            pct_packs=100 * pe / p0,
                            price_before=float(r0 / p0) if p0 else np.nan,
                            price_after=float((r0 - re) / (p0 - pe)) if p0 > pe else np.nan,
                            price_removed=float(re / pe)))
        E = pd.DataFrame(eff).sort_values("pct_packs", ascending=False)
        E["pct_price_change"] = 100 * (E.price_after / E.price_before - 1)
        print(f"  alternatives affected: {len(E)} of {nA}")
        print(E.round(4).to_string(index=False), flush=True)
        E.to_csv(OUT / "p2_affected_alternatives.csv", index=False)
        PK2 = np.maximum(PK - XPK, 0.0); RV2 = np.maximum(RV - XRV, 0.0)
        del PK, RV, XPK, XRV, d, Z, sj, sjg, peq; gc.collect()
        d, Z, sj, sjg, peq = frame(PK2, RV2, TOT, CLS, ncan32, yidx, "allbeer")
        del PK2, RV2; gc.collect()
        n2 = len(sj)
        print(f"\n  corrected frame {n2:,} rows ({n2-n:+,} against the uncorrected), "
              f"{int(d['mk'].max())+1:,} markets", flush=True)
        gst = d["un"].astype(np.int64)
        gal = d["al"].astype(np.int64); gwk = d["wk"].astype(np.int64)
        gsy = pd.factorize(gst * 4 + d["yr"])[0]
        gbs = pd.factorize(gst * 100 + bcode[d["al"]])[0]
        gpw = pd.factorize(pcode[d["un"]].astype(np.int64) * 1000 + d["wk"])[0]
        cl = pd.factorize(dcode[d["un"]])[0]
        cbase = dict(kind="corrected", stores=len(si))
        est(d, Z, sj, sjg, peq, [gal, gwk, gsy, gbs], cl,
            "CORRECTED, reference effects", rows, dict(fe="reference", **cbase))
        est(d, Z, sj, sjg, peq, [gal, gwk, gsy, gbs, gpw], cl,
            "CORRECTED + PARENT x WEEK", rows, dict(fe="parent x week", **cbase))
        RW = pd.DataFrame(rows); RW.to_csv(OUT / "p1_fe_variants.csv", index=False)

        lv = RW.loc[RW.price == "level"]
        print(f"\n=== 1.3  does the F rise toward the component mean of "
              f"{REF['component_F']}? ===", flush=True)
        for _, r in lv.iterrows():
            print(f"  {r.cell:56s} F = {r.sw_F:8.2f}  "
                  f"({100*r.sw_F/REF['component_F']:5.1f}% of the component mean)")
        json.dump(dict(n=int(n), n_corrected=int(n2), excluded_rows=int(nex),
                       component_F=REF["component_F"]),
                  open(OUT / "p1_summary.json", "w"), indent=1)
        rm.add_outputs([OUT / f for f in ("p1_fe_variants.csv", "p1_summary.json",
                                          "p2_affected_alternatives.csv")])
        print(f"\ndone [{time.time()-t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
