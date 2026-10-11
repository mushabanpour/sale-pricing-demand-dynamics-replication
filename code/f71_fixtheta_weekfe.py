#!/usr/bin/env python3
"""Column (iv)'s regime cells with week effects, holding the nonlinear parameters at column
(iv)'s estimates: does holiday demand load onto the sale regime terms?

Sale weeks cluster on holidays (July 4, Memorial Day, Labor Day, the Super Bowl), and the
isolated sales of holiday weeks fall disproportionately on 24- and 30-packs.  Year-month
effects leave that variation in.  As in f71_fixtheta_spfe.py, (pi_p, sigma_p, sigma_n) are
fixed at estimation/f71_col_iv_<tag>.json, delta comes from one contraction, and the linear
parameters are the least-squares projection of delta on the regime x pack cells, absorbing

    base           brand + year-month, demographics x ln n   (column (iv); checked against
                   pyblp's one-step beta)
    week           brand + week, demographics x ln n         (every national weekly shock,
                   holidays included wherever they fall)
    week_pack      brand + week x pack size, demographics    (also a holiday shift toward
                   particular pack sizes)
    spfe_week_pack store-product + week x pack size          (column (v) with week x pack
                   size in place of year-month)

Standard errors are clustered on DMA and conditional on the fixed nonlinear parameters.

    python3 code/f71_fixtheta_weekfe.py            # -> results/f71_fixtheta_weekfe_iv_long.csv
    BLP_DRAWS=50 python3 code/f71_fixtheta_weekfe.py --stores 5     # smoke test
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f71_baseline_table as f71
from f71_fixtheta_spfe import project

log = f71.log
VARIANTS = {"base": (["brand_id", "year_month"], True),
            "week": (["brand_id", "w"], True),
            "week_pack": (["brand_id", "week_pack"], True),
            "spfe_week_pack": (["store_product", "week_pack"], False)}
# the contrasts Section 4.3 quotes, as (cell, minus cell or None for NNN)
CONTRASTS = ([(f"NSN_{k}", None) for k in f71.RUNGS]
             + [(f"{r}_{k}", None) for r in ("NNS", "SNN") for k in (12, 30)]
             + [(f"{r}_36", "NSN_36") for r in ("NSS", "SSN", "SSS")]
             + [("SSS_24", "NSN_24")])


def main(tag, stores):
    import pyblp
    pyblp.options.verbose = False
    t0 = time.time()
    j = json.load(open(f71.E / f"f71_col_iv_{tag}.json"))
    d, terms = f71.load(tag, "iv")
    if stores:
        keep = sorted(d.store_code_uc.unique())[:stores]
        d = d[d.store_code_uc.isin(keep)].reset_index(drop=True)
    d["week_pack"] = d.w.astype(str) + "|" + d.n_cans.astype(str)
    log(f"theta fixed at column (iv): pi_p {j['pi_p']:.5f}  sigma_p {j['sigma_p']:.5f}  "
        f"sigma_lsz {j['sigma_lsz']:.5f};  {len(d):,} rows, {d.w.nunique()} weeks, "
        f"{d.week_pack.nunique():,} week x pack cells")
    r = f71.lognormal_problem(d, terms).solve(
        sigma=np.diag([j["sigma_p"], j["sigma_lsz"]]), pi=np.array([[j["pi_p"]], [0.0]]),
        optimization=pyblp.Optimization("return"),
        iteration=pyblp.Iteration("squarem", {"atol": 1e-12}), method="1s")
    delta = np.asarray(r.delta).ravel()
    log(f"  delta from one contraction  [{time.time() - t0:.0f}s]")
    cl = pd.factorize(d.clustering_ids)[0]
    reg = [t for t in terms if not t.startswith("dx_")]
    B, SE, V = {}, {}, {}
    for v, (fe, demo) in VARIANTS.items():
        x = terms if demo else reg
        b, se, Vv = project(d, delta, x, fe, cl)
        k = [x.index(t) for t in reg]
        B[v], SE[v], V[v] = b[k], se[k], Vv[np.ix_(k, k)]
        if v == "base":
            gap = np.max(np.abs(b - np.asarray(r.beta).ravel()))
            log(f"  check: base reproduces pyblp's one-step beta at this theta to {gap:.1e}")
            assert gap < 1e-6, gap
        log(f"  {v}: {' + '.join(fe)}{' + demographics' if demo else ''}  "
            f"[{time.time() - t0:.0f}s]")
    T = pd.concat({v: pd.DataFrame({"b": B[v], "se": SE[v]}, index=reg) for v in VARIANTS},
                  axis=1)
    T.columns = [f"{v}_{c}" for v, c in T.columns]
    out = f71.ROOT / "results" / f"f71_fixtheta_weekfe_iv_{tag}{'_smoke' if stores else ''}.csv"
    T.to_csv(out)

    star = lambda b, s: "*" if abs(b / s) >= 1.96 else " "
    log(f"\n  contrasts quoted in Section 4.3 (estimate, se; * |t| >= 1.96)")
    log(f"  {'':16s}" + "".join(f"{v:>20s}" for v in VARIANTS))
    for c, m in CONTRASTS:
        row = f"  {c + (' - ' + m if m else ''):16s}"
        for v in VARIANTS:
            g = np.zeros(len(reg)); g[reg.index(c)] = 1
            if m:
                g[reg.index(m)] = -1
            b, s = g @ B[v], np.sqrt(g @ V[v] @ g)
            row += f"{b:>11.3f} ({s:.3f}){star(b, s)}"
        log(row)
    log(f"\n  all {len(reg)} cells against base:")
    for v in list(VARIANTS)[1:]:
        sig = int((np.abs(B[v] / SE[v]) >= 1.96).sum())
        same = int((np.sign(B[v]) == np.sign(B["base"])).sum())
        log(f"  {v:15s} significant {sig:2d}  same sign as base {same:2d}  correlation "
            f"{np.corrcoef(B[v], B['base'])[0, 1]:.3f}  max |difference| "
            f"{np.max(np.abs(B[v] - B['base'])):.3f}")
    log(f"\n  wrote {out.relative_to(f71.ROOT)}")
    log(f"[{time.time() - t0:.0f}s] done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="long")
    ap.add_argument("--stores", type=int, default=0, help="first k stores (smoke test)")
    a = ap.parse_args()
    main(a.tag, a.stores)
