#!/usr/bin/env python3
"""Round 47 Part 2: what variation the four margins have WITHOUT week effects.

Round 46 ran this decomposition on `s_0` and found it is a location property -- 12.68% of its
variance survives `alternative + brand x store`.  The objects that matter for an intertemporal
model are not location properties: lagged price, lead price, the volume-discount margin and the
own-history margin vary within product-store over weeks, which is what location effects leave
alone and week effects destroy.  This measures each one against the same ladder, with the WEEK
rung reported last and separately so the cost of the effect the design criticism is about is
visible as its own line.

Definitions, all in levels and per 144-ounce equivalent, one definition site each:
  `p_lag`, `p_lead`  the cell's own price at t-1 and t+1, requiring the cell to be observed
                     in that week (a gap is a missing value, not an interpolation)
  `VD`               `storage_margins.py`: the per-144oz price of the nearest SMALLER rung
                     stocked that week, same brand, container, store and week, minus own
                     price, with rung distance recorded (D016 -- the fallback rule, NOT D040
                     which is the great-circle-distance decision).  Six-packs carry no VD
                     (D017).  The STRICT adjacent-rung variant is reported beside it to
                     measure what strictness costs.
  `IT`               `p_reg - p` with `p_reg` the median of the cell's own prior four OBSERVED
                     weeks (`sale_definitions.prior_median`).  Any c(tau) = tau/(1+tau)
                     rescales this column and changes no correlation, variance share or
                     condition number reported here (storage_margins.py, D014).

DESCRIPTIVE ONLY.

    python3 paper/src/diagnostics/blp_margins_decomp.py
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
from blp_store_week_demand import absorb                              # noqa: E402
from blp_hn_cells import build_grid, prior_median_obs, OUT as HNOUT   # noqa: E402

OUT = PAPER / "outputs" / "blp_hncells"
SCRATCH = _paths.WORK
YEARS = [2016, 2017, 2018, 2019]
MIN_WEEKS = 52
GRID_POINTS = [(4, 0.15), (8, 0.15), (12, 0.20)]
REF = dict(n=20_493_943, s0_brand_store=0.1268, s0_between_week=0.0372,
           vd_it_corr_dma=0.406, r39_distinct=2.20, r39_never=0.1972)


def vshare(v, g):
    """Between-group share of the variance of v.  Round 45's function, unchanged."""
    v = np.asarray(v, float)
    n = np.bincount(g, minlength=int(g.max()) + 1).astype(float)
    m = np.bincount(g, weights=v, minlength=len(n)) / np.maximum(n, 1)
    return float((n * (m - v.mean()) ** 2).sum() / len(v)) / float(v.var())


def build_margins(G):
    """The four margin grids, plus the strict-VD variant and the rung distance."""
    P = G["P"]
    nS, nW, nA = G["nS"], G["nW"], G["nA"]
    brand, cont, ncan = G["brand"], G["cont"], G["ncan"]
    out = {}
    out["p_lag"] = np.concatenate([np.full((nS, 1, nA), np.nan, np.float32),
                                   P[:, :-1, :]], axis=1)
    out["p_lead"] = np.concatenate([P[:, 1:, :],
                                    np.full((nS, 1, nA), np.nan, np.float32)], axis=1)
    # IT: p_reg - p, p_reg = median of the prior four OBSERVED weeks, per cell
    IT = np.full_like(P, np.nan)
    for a in range(nA):
        Pc = P[:, :, a]
        Oc = np.isfinite(Pc)
        bench = prior_median_obs(Pc, Oc)
        IT[:, :, a] = (bench - Pc).astype(np.float32)
    out["IT"] = IT
    # VD: nearest smaller stocked rung of the same brand-container (D016), and strict (adjacent)
    VD = np.full_like(P, np.nan)
    VDs = np.full_like(P, np.nan)
    DIST = np.zeros_like(P)
    reason = np.zeros((nS, nW, nA), np.int8)     # 0 defined, 1 no rung, 2 store never, 3 not stocked
    for a in range(nA):
        cand = np.where((brand == brand[a]) & (cont == cont[a]) & (ncan < ncan[a]))[0]
        if not len(cand):
            reason[:, :, a] = 1
            continue
        cand = cand[np.argsort(-ncan[cand])]     # descending pack size = ascending rung distance
        filled = np.zeros((nS, nW), bool)
        for k, c in enumerate(cand):
            av = np.isfinite(P[:, :, c])
            take = av & ~filled
            if take.any():
                VD[:, :, a] = np.where(take, P[:, :, c] - P[:, :, a], VD[:, :, a])
                DIST[:, :, a] = np.where(take, k + 1, DIST[:, :, a])
                filled |= take
            if k == 0:
                VDs[:, :, a] = np.where(av, P[:, :, c] - P[:, :, a], np.nan)
        ever = np.isfinite(P[:, :, cand]).any(axis=(1, 2))
        reason[:, :, a] = np.where(filled, 0, np.where(ever[:, None], 3, 2))
    out["VD"] = VD
    out["VD_strict"] = VDs
    out["rung_distance"] = DIST
    out["vd_reason"] = reason
    return out


def cellvar(Xc, dec=2):
    """Per-cell distinct values, changes and never-changes -- round 39's format."""
    nC, nW = Xc.shape
    R = np.where(np.isfinite(Xc), np.round(Xc.astype(np.float64), dec), np.nan)
    obs = np.isfinite(R)
    nobs = obs.sum(1)
    srt = np.sort(R, axis=1)
    d = np.diff(srt, axis=1)
    isnew = np.zeros_like(srt, dtype=bool)
    isnew[:, 0] = np.isfinite(srt[:, 0])
    isnew[:, 1:] = np.isfinite(srt[:, 1:]) & (d != 0)
    ndist = isnew.sum(1)
    # changes between CONSECUTIVE observed weeks
    prev = np.full(nC, np.nan)
    chg = np.zeros(nC, np.int64)
    for t in range(nW):
        v = R[:, t]
        ok = np.isfinite(v)
        moved = ok & np.isfinite(prev) & (v != prev)
        chg += moved
        prev = np.where(ok, v, prev)
    return dict(n_obs=nobs, n_distinct=ndist, n_changes=chg)


def qt(x, qs=(10, 25, 50, 75, 90)):
    x = np.asarray(x, float); m = np.isfinite(x)
    return {f"p{q}": float(np.percentile(x[m], q)) for q in qs} | \
        dict(mean=float(x[m].mean()), n=int(m.sum()))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with RunManifest(OUT, args={}, inputs=[SCRATCH / "focal_rows.bin"],
                     tag="the four margins without week effects (round 47 Part 2)") as rm:
        rm.add_note("Descriptive. The week rung is reported last and separately.")
        t0 = time.time()
        print("=== 0  the reference assertion ===", flush=True)
        G = build_grid()
        nS, nW, nA = G["nS"], G["nW"], G["nA"]
        brand, ncan = G["brand"], G["ncan"]
        valid = G["valid"]
        del G["PK"]; gc.collect()          # the margins need prices, not quantities
        bcode = pd.factorize(brand)[0]
        print(f"\n=== building the four margins [{time.time()-t0:.0f}s] ===", flush=True)
        res = {}
        MG = build_margins(G)
        print(f"  built [{time.time()-t0:.0f}s]", flush=True)
        # ---------------- VD coverage diagnostics
        print(f"\n=== 2.x  `VD` coverage and the three missing reasons (D016/D017) ===",
              flush=True)
        rs = MG["vd_reason"][valid]
        vdf = np.isfinite(MG["VD"][valid]); vds = np.isfinite(MG["VD_strict"][valid])
        dist = MG["rung_distance"][valid]
        labs = {0: "defined", 1: "no smaller rung exists (six-packs, D017)",
                2: "rung exists but the store never carries it",
                3: "store carries it but it is not stocked that week"}
        vdd = {}
        for k, lab in labs.items():
            vdd[lab] = float((rs == k).mean())
            print(f"  {lab:52s} {100*vdd[lab]:6.2f}% of rows")
        print(f"  fallback coverage {100*vdf.mean():.2f}% against strict adjacency "
              f"{100*vds.mean():.2f}% -- strictness costs "
              f"{100*(vdf.mean()-vds.mean()):.2f} points (D016)")
        dd = dist[vdf]
        print(f"  rung distance among defined rows: " + "  ".join(
            f"{int(k)}:{100*np.mean(dd == k):.1f}%" for k in np.unique(dd)), flush=True)
        res["vd_coverage"] = dict(reasons=vdd, fallback=float(vdf.mean()),
                                  strict=float(vds.mean()),
                                  rung_dist={str(int(k)): float(np.mean(dd == k))
                                             for k in np.unique(dd)})
        del MG["VD_strict"], MG["rung_distance"], MG["vd_reason"]
        gc.collect()

        # row-level group indices
        si_, wi_, ai_ = np.where(valid)
        n = len(si_)
        assert n == REF["n"], (n, REF["n"])
        si_ = si_.astype(np.int16); wi_ = wi_.astype(np.int16); ai_ = ai_.astype(np.int16)
        gc.collect()
        yr = np.array([YEARS.index(y) for y in G["weeks"].year])[wi_].astype(np.int8)
        gal = ai_.astype(np.int32)
        gst = si_.astype(np.int32)
        gwk = wi_.astype(np.int32)
        gsy = pd.factorize(gst.astype(np.int32) * 4 + yr)[0].astype(np.int32)
        gbs = pd.factorize(gst.astype(np.int32) * 200 + bcode[ai_])[0].astype(np.int32)
        rowcell = (gst.astype(np.int32) * nA + gal).astype(np.int32)
        del yr; gc.collect()
        RUNGS = {"alternative": ["gal"], "+ store": ["gal", "gst"],
                 "+ brand x store": ["gal", "gst", "gbs"],
                 "+ store x year": ["gal", "gst", "gbs", "gsy"],
                 "+ week": ["gal", "gst", "gbs", "gsy", "gwk"]}
        OBJ = ["p", "p_lag", "p_lead", "VD", "IT"]
        rows = {}
        res["n"] = int(n)

        def flat(name):
            if name == "p":
                return G["P"][valid].astype(np.float64)
            return MG[name][valid].astype(np.float64)

        # ---------------- 2.1 / 2.2 / 2.3
        print(f"\n=== 2.1  the fixed-effect ladder, each object on its own defined sample "
              f"===", flush=True)
        lad_rows = []
        for nm in OBJ:
            v = flat(nm)
            m = np.isfinite(v)
            cov = float(m.mean())
            vv = v[m]
            raw = float(vv.var()); sd0 = float(vv.std())
            gmap = {k: pd.factorize(g[m])[0].astype(np.int32) for k, g in
                    (("gal", gal), ("gst", gst), ("gbs", gbs), ("gsy", gsy),
                     ("gwk", gwk))}
            print(f"\n  {nm}: coverage {100*cov:6.2f}% ({int(m.sum()):,} rows), "
                  f"mean {vv.mean():+.4f}, raw sd {sd0:.4f}", flush=True)
            lad_rows.append(dict(object=nm, rung="raw", coverage=cov, n=int(m.sum()),
                                 mean=float(vv.mean()), resid_sd=sd0, share=1.0))
            for lab, key in RUNGS.items():
                A, it, mx = absorb(vv.reshape(-1, 1).copy(),
                                   [gmap[k] for k in key])
                sd = float(A[:, 0].std()); sh = sd ** 2 / raw
                del A; gc.collect()
                lad_rows.append(dict(object=nm, rung=lab, coverage=cov, n=int(m.sum()),
                                     resid_sd=sd, share=sh))
                print(f"    {lab:20s} residual sd {sd:8.4f} = {100*sh:6.2f}% of raw "
                      f"variance  [{it} iters]", flush=True)
            # the headline: no time effects at all
            A, it, mx = absorb(vv.reshape(-1, 1).copy(),
                               [gmap["gal"], gmap["gbs"]])
            sd_nw = float(A[:, 0].std()); sh_nw = sd_nw ** 2 / raw
            del A; gc.collect()
            bw = vshare(vv, gmap["gwk"])
            lad_rows.append(dict(object=nm, rung="alternative + brand x store (NO time)",
                                 coverage=cov, n=int(m.sum()), resid_sd=sd_nw,
                                 share=sh_nw, between_week=bw))
            print(f"    {'NO TIME EFFECTS':20s} residual sd {sd_nw:8.4f} = "
                  f"**{100*sh_nw:6.2f}%** of raw variance   (round 46: `s_0` retains "
                  f"{100*REF['s0_brand_store']:.2f}%)")
            print(f"    between-week share of variance: **{100*bw:6.2f}%**   (`s_0` "
                  f"{100*REF['s0_between_week']:.2f}% at store-week)", flush=True)
            rows[nm] = dict(coverage=cov, raw_sd=sd0, noweek_sd=sd_nw, noweek_share=sh_nw,
                            between_week=bw)
            del v, vv, m, gmap; gc.collect()
        pd.DataFrame(lad_rows).to_csv(OUT / "p2_ladder.csv", index=False)
        res["ladder"] = lad_rows
        res["headline"] = rows

        # ---------------- 2.4 collinearity on the common sample
        print(f"\n=== 2.4  collinearity after `alternative + brand x store`, common "
              f"sample [{time.time()-t0:.0f}s] ===", flush=True)
        cols = ["p_lag", "p_lead", "VD", "IT"]
        M = np.column_stack([flat(c) for c in cols])
        cm = np.isfinite(M).all(1)
        print(f"  common sample {int(cm.sum()):,} of {n:,} rows ({100*cm.mean():.2f}%)",
              flush=True)
        A, it, mx = absorb(M[cm].copy(), [pd.factorize(gal[cm])[0],
                                          pd.factorize(gbs[cm])[0]])
        del M; gc.collect()
        Cm = np.corrcoef(A, rowvar=False)
        print("  correlation matrix (after the effects, NO week effects):")
        print("            " + "".join(f"{c:>10s}" for c in cols))
        for i, c in enumerate(cols):
            print(f"  {c:>10s}" + "".join(f"{Cm[i, j]:>10.4f}" for j in range(len(cols))))
        vif = []
        for i in range(len(cols)):
            oth = [j for j in range(len(cols)) if j != i]
            b, *_ = np.linalg.lstsq(A[:, oth], A[:, i], rcond=None)
            r2 = 1 - float(((A[:, i] - A[:, oth] @ b) ** 2).sum()
                           / (A[:, i] ** 2).sum())
            vif.append(1.0 / max(1 - r2, 1e-12))
        sv = np.linalg.svd(A, compute_uv=False)
        print(f"  VIFs: " + "  ".join(f"{c}={v:.3f}" for c, v in zip(cols, vif)))
        print(f"  singular values {np.array2string(sv, precision=1)}; condition number "
              f"**{sv[0]/sv[-1]:.3f}**")
        print(f"  round 41 measured corr(VD, IT) = {REF['vd_it_corr_dma']} pooled at "
              f"DMA-week WITH week effects; here it is "
              f"**{Cm[cols.index('VD'), cols.index('IT')]:+.4f}**", flush=True)
        res["collinearity"] = dict(cols=cols, corr=Cm.tolist(), vif=vif,
                                   singular=sv.tolist(),
                                   condition=float(sv[0] / sv[-1]),
                                   common_rows=int(cm.sum()))
        del A; gc.collect()

        # ---------------- 2.5 within-cell variation
        print(f"\n=== 2.5  within-cell variation, round 39's format "
              f"[{time.time()-t0:.0f}s] ===", flush=True)
        cv_rows = []
        for nm in OBJ:
            X = (G["P"] if nm == "p" else MG[nm])
            Xc = np.ascontiguousarray(X.transpose(0, 2, 1)).reshape(nS * nA, nW)
            keep = np.isfinite(Xc).any(1)
            d = cellvar(Xc[keep])
            del Xc; gc.collect()
            nd, ch, nb = d["n_distinct"], d["n_changes"], d["n_obs"]
            r = dict(object=nm, cells=int(keep.sum()),
                     distinct_mean=float(nd.mean()),
                     distinct_p10=float(np.percentile(nd, 10)),
                     distinct_p50=float(np.percentile(nd, 50)),
                     distinct_p90=float(np.percentile(nd, 90)),
                     changes_mean=float(ch.mean()),
                     changes_p10=float(np.percentile(ch, 10)),
                     changes_p50=float(np.percentile(ch, 50)),
                     changes_p90=float(np.percentile(ch, 90)),
                     never_change=float((ch == 0).mean()),
                     never_change_1obs=float(((ch == 0) & (nb > 1)).mean()),
                     weeks_mean=float(nb.mean()))
            cv_rows.append(r)
            print(f"  {nm:8s} cells {r['cells']:>7,}  distinct values mean "
                  f"{r['distinct_mean']:6.2f} (p10/p50/p90 {r['distinct_p10']:.0f}/"
                  f"{r['distinct_p50']:.0f}/{r['distinct_p90']:.0f})  changes mean "
                  f"{r['changes_mean']:7.2f} (p50 {r['changes_p50']:.0f})  never change "
                  f"**{100*r['never_change']:5.2f}%**", flush=True)
        print(f"  round 39's leading rival count, for comparison: {REF['r39_distinct']} "
              f"distinct values, {100*REF['r39_never']:.2f}% never changing (5,208-store "
              f"balanced panel, store-ALTERNATIVE series)", flush=True)
        pd.DataFrame(cv_rows).to_csv(OUT / "p2_within_cell.csv", index=False)
        res["within_cell"] = cv_rows

        # ---------------- 2.6 the selected sub-panels
        print(f"\n=== 2.6  the same on the Part 1 selected sub-panels "
              f"[{time.time()-t0:.0f}s] ===", flush=True)
        C = pd.read_parquet(HNOUT / "p1_cell_table.parquet")
        sub_rows = []
        for ns, dp in GRID_POINTS:
            sel = ((C.n_spells >= ns) & (C.median_depth >= dp)
                   & (C.n_obs >= MIN_WEEKS)).to_numpy()
            cell_id = C.cell.to_numpy()[sel]
            keep_cell = np.zeros(nS * nA, bool); keep_cell[cell_id] = True
            inrow = keep_cell[rowcell]
            print(f"\n  >= {ns} spells, depth >= {100*dp:.0f}%: {int(sel.sum()):,} cells, "
                  f"{int(inrow.sum()):,} rows", flush=True)
            for nm in OBJ:
                v = flat(nm)
                m = inrow & np.isfinite(v)
                if m.sum() < 1000:
                    continue
                vv = v[m]; raw = float(vv.var())
                A, it, mx = absorb(vv.reshape(-1, 1).copy(),
                                   [pd.factorize(gal[m])[0], pd.factorize(gbs[m])[0]])
                sd_nw = float(A[:, 0].std()); del A
                bw = vshare(vv, pd.factorize(gwk[m])[0])
                A, it, mx = absorb(vv.reshape(-1, 1).copy(),
                                   [pd.factorize(gal[m])[0], pd.factorize(gst[m])[0],
                                    pd.factorize(gbs[m])[0], pd.factorize(gsy[m])[0],
                                    pd.factorize(gwk[m])[0]])
                sd_full = float(A[:, 0].std()); del A; gc.collect()
                r = dict(min_spells=ns, min_depth=dp, object=nm, n=int(m.sum()),
                         coverage=float(m.sum() / max(int(inrow.sum()), 1)),
                         raw_sd=float(vv.std()), noweek_sd=sd_nw,
                         noweek_share=sd_nw ** 2 / raw, full_sd=sd_full,
                         full_share=sd_full ** 2 / raw, between_week=bw)
                sub_rows.append(r)
                print(f"    {nm:8s} coverage {100*r['coverage']:5.1f}%  raw sd "
                      f"{r['raw_sd']:7.4f}  NO-TIME sd {sd_nw:7.4f} "
                      f"(**{100*r['noweek_share']:5.2f}%**)  with week {sd_full:7.4f} "
                      f"({100*r['full_share']:5.2f}%)  between-week "
                      f"{100*bw:5.2f}%", flush=True)
                del v, vv, m; gc.collect()
        pd.DataFrame(sub_rows).to_csv(OUT / "p2_subpanel_margins.csv", index=False)
        res["subpanels"] = sub_rows
        json.dump(res, open(OUT / "p2_summary.json", "w"), indent=1, default=float)
        rm.add_outputs([OUT / f for f in ("p2_summary.json", "p2_ladder.csv",
                                          "p2_within_cell.csv",
                                          "p2_subpanel_margins.csv")])
        print(f"\ndone [{time.time()-t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
