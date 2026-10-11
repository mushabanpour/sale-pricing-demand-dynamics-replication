#!/usr/bin/env python3
"""Column (iv)'s regime cells with store-product effects, holding the nonlinear parameters at
column (iv)'s estimates.

(pi_p, sigma_p, sigma_n) are fixed at estimation/f71_col_iv_<tag>.json, so the mean
utilities delta are column (iv)'s (one contraction, no optimization).  Price is not in X1, so
the linear parameters given delta are a least-squares projection: delta on the regime x pack
cells, absorbing
    base   brand + year-month effects, with the demographics x ln n terms (column (iv) at
           one-step weights; must equal pyblp's one-step beta, checked), and
    spfe   store-product + year-month effects, without the demographics (fixed within a
           store, so absorbed).
Standard errors are clustered on DMA and conditional on the fixed nonlinear parameters.
Elasticities, markups and shares are column (iv)'s: only the split of delta changes.

    python3 code/f71_fixtheta_spfe.py --tag long   # -> results/f71_fixtheta_spfe_iv_<tag>.csv,
                                                   #    estimation/f71_col_v_<tag>.json (column (v))
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f71_baseline_table as f71

log = f71.log


def absorb(A, fe, tol=1e-12, it=10000):
    """A with the effects fe (list of id columns) partialled out, by alternating demeaning."""
    A = A.copy()
    for _ in range(it):
        A0 = A.copy()
        for g in fe:
            n = np.bincount(g)
            A -= np.column_stack([np.bincount(g, A[:, k]) / n for k in range(A.shape[1])])[g]
        if np.max(np.abs(A - A0)) < tol:
            return A
    raise RuntimeError("absorption did not converge")


def project(d, delta, terms, fe, cl):
    """OLS of delta on terms with the effects fe absorbed, clustered on cl."""
    ids = [pd.factorize(d[c])[0] for c in fe]
    A = absorb(np.column_stack([delta, d[terms].to_numpy(float)]), ids)
    b, se, V, *_ = f71.clustered_2sls(A[:, :1], A[:, 1:], A[:, 1:], cl)
    return b, se, V


def main(tag):
    import pyblp
    pyblp.options.verbose = False
    t0 = time.time()
    j = json.load(open(f71.E / f"f71_col_iv_{tag}.json"))
    d, terms = f71.load(tag, "iv")
    log(f"theta fixed at column (iv): pi_p {j['pi_p']:.5f}  sigma_p {j['sigma_p']:.5f}  "
        f"sigma_lsz {j['sigma_lsz']:.5f}  (median alpha {j['median_alpha']:.4f})")
    r = f71.lognormal_problem(d, terms).solve(
        sigma=np.diag([j["sigma_p"], j["sigma_lsz"]]), pi=np.array([[j["pi_p"]], [0.0]]),
        optimization=pyblp.Optimization("return"),
        iteration=pyblp.Iteration("squarem", {"atol": 1e-12}), method="1s")
    delta = np.asarray(r.delta).ravel()
    log(f"  delta from one contraction  [{time.time() - t0:.0f}s]")
    cl = pd.factorize(d.clustering_ids)[0]
    reg = [t for t in terms if not t.startswith("dx_")]
    bb, sb, _ = project(d, delta, terms, ["brand_id", "year_month"], cl)
    bs, ss, Vs = project(d, delta, reg, ["store_product", "year_month"], cl)
    T = pd.DataFrame({"col_iv": [j["beta"][t] for t in reg], "col_iv_se": [j["se"][t] for t in reg],
                      "base": bb[[terms.index(t) for t in reg]],
                      "base_se": sb[[terms.index(t) for t in reg]],
                      "spfe": bs, "spfe_se": ss}, index=reg)
    gap = np.max(np.abs(bb - np.asarray(r.beta).ravel()))
    log(f"  check: brand-effects projection reproduces pyblp's one-step beta at this theta "
        f"to {gap:.1e}")
    assert gap < 1e-6, gap
    star = lambda v, s: "*" if abs(v / s) >= 1.96 else " "
    log(f"\n  {'cell':8s}{'col (iv)':>18s}{'store-product FE':>22s}")
    for t, x in T.iterrows():
        log(f"  {t:8s}{x.col_iv:>9.3f} ({x.col_iv_se:.3f}){star(x.col_iv, x.col_iv_se)}"
            f"{x.spfe:>12.3f} ({x.spfe_se:.3f}){star(x.spfe, x.spfe_se)}")
    sig = lambda b, s: int((np.abs(b / s) >= 1.96).sum())
    same = int((np.sign(T.spfe) == np.sign(T.col_iv)).sum())
    log(f"\n  significant: col (iv) {sig(T.col_iv, T.col_iv_se)} of {len(T)}, "
        f"store-product FE {sig(T.spfe, T.spfe_se)} of {len(T)};  same sign {same} of {len(T)};  "
        f"correlation {np.corrcoef(T.col_iv, T.spfe)[0, 1]:.3f}")
    T.to_csv(f71.ROOT / "results" / f"f71_fixtheta_spfe_iv_{tag}.csv")
    # column (v) of Table 5A/5B, in the format of the estimated columns: theta and the
    # statistics that depend only on it (alpha, elasticities) are column (iv)'s; no J
    keep = ("pi_p", "sigma_p", "sigma_lsz", "median_alpha", "by_rung", "eps_median",
            "n", "markets", "clusters")
    json.dump(dict(col="v", kind="fixed_theta_spfe", form=j.get("form"), tag=tag,
                   regime=j["regime"], theta_from="iv", fe="store_product + year_month",
                   J=None, df=None, p=None, beta=dict(zip(reg, bs.tolist())),
                   se=dict(zip(reg, ss.tolist())),
                   n_sig=sig(T.spfe, T.spfe_se), seconds=time.time() - t0)
              | {k: j[k] for k in keep},
              open(f71.E / f"f71_col_v_{tag}.json", "w"), indent=1)
    np.save(f71.E / f"f71_vcov_v_{tag}.npy", Vs)
    json.dump(reg, open(f71.E / f"f71_vcov_v_{tag}_labels.json", "w"))
    log(f"  wrote estimation/f71_col_v_{tag}.json and its covariance")
    log(f"[{time.time() - t0:.0f}s] done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="long")
    main(ap.parse_args().tag)
