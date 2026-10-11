#!/usr/bin/env python3
"""Parts 1, 2 and 4: instrument-subset stability, the fixed-effect ladder, and the
21 partial-coverage DMAs. Linear IV only.

Reuses the estimators validated in round 13 section 3.1 (blp_first_stage_strength).

    python3 paper/src/diagnostics/blp_pre_rcnl_diagnostics.py
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import _paths
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
PAPER = _paths.WORK
sys.path.insert(0, str(PAPER / "src" / "common"))
sys.path.insert(0, str(HERE.parent))
from run_manifest import RunManifest                    # noqa: E402
import blp_first_stage_strength as M                    # noqa: E402

OUT = PAPER / "outputs" / "blp_pre_rcnl"
DIFF = M.DIFF                       # 4 differentiation classes
NEAR = ["d_same_brand_other_rung", "d_same_rung_other_brand"]
FAR = ["d_same_container_other", "d_other_container"]
IVSET = DIFF + ["z_haus"]           # D028: cost shifters removed
SEED = 20260902


def absorbN(X, groups, iters=600, tol=1e-10):
    """Alternating projections over an arbitrary list of group-code arrays."""
    X = np.array(X, dtype=np.float64, copy=True)
    gs = [(g, int(g.max()) + 1) for g in groups]
    cs = [np.maximum(np.bincount(g, minlength=n).astype(float), 1.0) for g, n in gs]
    for _ in range(iters):
        mx = 0.0
        for (g, n), c in zip(gs, cs):
            m = np.zeros((n, X.shape[1]))
            for k in range(X.shape[1]):
                m[:, k] = np.bincount(g, weights=X[:, k], minlength=n) / c
            X -= m[g]
            mx = max(mx, float(np.abs(m).max()))
        if mx < tol:
            break
    return X


def fit(e, znames, groups, tag, rows, label=""):
    """rho = 0 specification: y = alpha*p + FE, price endogenous."""
    col = "share_inside_year"
    y = (np.log(e[col]) - np.log(e["s0_" + col])).to_numpy()
    cl = pd.factorize(e.dmac)[0]
    X = np.column_stack([y, e.price.to_numpy(float), e[znames].to_numpy(float)])
    A = absorbN(X, groups)
    yd, p_, Z = A[:, 0], A[:, [1]], A[:, 2:]
    L = Z.shape[1]
    b, V, _, G = M.tsls(yd, p_, Z, cl)
    sw = M.sw_stats(p_[:, 0], np.empty((len(yd), 0)), Z, cl)
    if L > 1:
        J, Jdf, Jp, _ = M.hansen_j(yd, p_, Z, cl)
    else:
        J = Jdf = Jp = np.nan
    sj = e[col].to_numpy(float); pr = e.price.to_numpy(float)
    el = b[0] * pr * (1 - sj)
    rows.append(dict(cell=tag, instruments=label or "+".join(znames), n_iv=L,
                     n=len(yd), clusters=G, alpha=b[0], se=float(np.sqrt(V[0, 0])),
                     t=b[0] / float(np.sqrt(V[0, 0])),
                     elast_p50=float(np.median(el)), elast_mean=float(el.mean()),
                     sw_F_price=sw["sw_F"], hansen_J=J, hansen_df=Jdf, hansen_p=Jp))
    print(f"  {tag:26s} {rows[-1]['instruments'][:38]:38s} L={L} "
          f"alpha={b[0]:+.4f} el50={rows[-1]['elast_p50']:+.3f} "
          f"SWF={sw['sw_F']:8.1f} J={J if np.isfinite(J) else float('nan'):.1f}", flush=True)
    return rows[-1]


def main() -> int:
    out = Path(OUT); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    with RunManifest(out, args={}, inputs=[PAPER / "outputs" / "blp_panel" /
                                           "market_panel_dma_week.parquet"],
                     tag="pre-RCNL diagnostics (round 14)") as rm:
        rm.add_note("Linear IV only. D027 cell: DMA-week, within-year market size, DMA-by-year "
                    "effects. D028: cost shifters excluded. Estimators from round 13, validated "
                    "in round 13 section 3.1.")
        d = M.build("dma_week")
        d["year"] = pd.to_datetime(d.period.astype(str)).dt.year
        dt = pd.to_datetime(d.period.astype(str))
        d["quarter"] = dt.dt.year * 4 + dt.dt.quarter
        d["month"] = dt.dt.year * 12 + dt.dt.month
        e = d.loc[(d.share_inside_year > 0) & (d.s0_share_inside_year > 0) & (d.s_jg > 0)].copy()
        ga = pd.factorize(e.alt)[0]; gt = pd.factorize(e.per)[0]
        gd = pd.factorize(e.dmac)[0]
        gdy = pd.factorize(list(zip(e.dmac, e.year)))[0]
        gdq = pd.factorize(list(zip(e.dmac, e.quarter)))[0]
        gdm = pd.factorize(list(zip(e.dmac, e.month)))[0]
        print(f"estimation sample {len(e):,} rows", flush=True)

        # ---------------- Part 1: instrument subsets, in the D027 cell
        p1 = []
        base = [ga, gt, gdy]
        SETS = [("all five", IVSET), ("four differentiation classes", DIFF),
                ("Hausman only (just identified)", ["z_haus"]),
                ("two near classes", NEAR), ("two far classes", FAR)]
        for lab, zs in SETS:
            fit(e, zs, base, "P1 alt+time+dma_x_year", p1, lab)
        for z in IVSET:
            fit(e, [z], base, "P1 alt+time+dma_x_year", p1, f"single: {z}")
        pd.DataFrame(p1).to_csv(out / "part1_instrument_subsets.csv", index=False)

        # ---------------- Part 1b: J scaling in the sample size
        p1b = []
        mk = e.mkt.to_numpy()
        umk = np.unique(mk)
        for frac in (0.01, 0.10, 1.00):
            keep = umk if frac == 1.0 else rng.choice(
                umk, size=max(int(len(umk) * frac), 2), replace=False)
            sub = e.loc[np.isin(mk, keep)].copy()
            ga2 = pd.factorize(sub.alt)[0]; gt2 = pd.factorize(sub.per)[0]
            gdy2 = pd.factorize(list(zip(sub.dmac, sub.year)))[0]
            for lab, zs in (("all five", IVSET), ("four differentiation classes", DIFF)):
                r = fit(sub, zs, [ga2, gt2, gdy2], f"P1b frac={frac:g}", p1b, lab)
                r["frac"] = frac
        pd.DataFrame(p1b).to_csv(out / "part1b_j_scaling.csv", index=False)

        # ---------------- Part 2: the fixed-effect ladder
        p2 = []
        LADDER = [("alt+time", [ga, gt]), ("+dma", [ga, gt, gd]),
                  ("+dma_x_year", [ga, gt, gdy]), ("+dma_x_quarter", [ga, gt, gdq]),
                  ("+dma_x_month", [ga, gt, gdm])]
        for lab, gr in LADDER:
            r = fit(e, IVSET, gr, f"P2 {lab}", p2, "all five")
            gg = gr[-1]
            sizes = np.bincount(gg)
            sizes = sizes[sizes > 0]
            r.update(dict(fe=lab, n_groups_last=len(sizes),
                          total_absorbed_params=sum(int(g.max()) + 1 for g in gr),
                          singleton_groups=int((sizes == 1).sum()),
                          groups_lt10=int((sizes < 10).sum()),
                          mean_obs_per_group=float(sizes.mean())))
        pd.DataFrame(p2).to_csv(out / "part2_fe_ladder.csv", index=False)

        # ---------------- Part 4: the partial-coverage DMAs
        yrs = e.groupby("dmac")["year"].nunique()
        partial = yrs[yrs < 4].index
        tot = e.packs.sum()
        info = (e.loc[e.dmac.isin(partial)].groupby(["dmac", "year"])
                 .agg(packs=("packs", "sum"), rows=("packs", "size")).reset_index())
        share = e.loc[e.dmac.isin(partial)].groupby("dmac")["packs"].sum() / tot
        p4 = []
        for dc in partial:
            have = sorted(e.loc[e.dmac == dc, "year"].unique())
            p4.append(dict(dmac=int(dc), dma=str(e.loc[e.dmac == dc, "dma"].iloc[0]),
                           years_present=",".join(str(y) for y in have),
                           years_missing=",".join(str(y) for y in
                                                  sorted(set([2016, 2017, 2018, 2019]) - set(have))),
                           packs_share_pct=float(100 * share.loc[dc]),
                           rows=int((e.dmac == dc).sum())))
        p4 = pd.DataFrame(p4).sort_values("packs_share_pct", ascending=False)
        p4.to_csv(out / "part4_partial_dmas.csv", index=False)
        print(f"\npartial-coverage DMAs: {len(p4)}; combined pack share "
              f"{p4.packs_share_pct.sum():.4f}%", flush=True)
        e2 = e.loc[~e.dmac.isin(partial)].copy()
        ga3 = pd.factorize(e2.alt)[0]; gt3 = pd.factorize(e2.per)[0]
        gdy3 = pd.factorize(list(zip(e2.dmac, e2.year)))[0]
        p4r = []
        fit(e, IVSET, base, "P4 all DMAs", p4r, "all five")
        fit(e2, IVSET, [ga3, gt3, gdy3], "P4 excluding partial DMAs", p4r, "all five")
        pd.DataFrame(p4r).to_csv(out / "part4_exclusion.csv", index=False)

        pd.set_option("display.width", 250)
        print("\n--- Part 1 ---"); print(pd.DataFrame(p1).to_string(index=False))
        print("\n--- Part 1b ---"); print(pd.DataFrame(p1b).to_string(index=False))
        print("\n--- Part 2 ---"); print(pd.DataFrame(p2).to_string(index=False))
        print("\n--- Part 4 ---"); print(p4.to_string(index=False))
        print(pd.DataFrame(p4r).to_string(index=False))
        rm.add_outputs(sorted(out.glob("*.csv")))
    return 0


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    sys.exit(main())
