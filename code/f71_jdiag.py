#!/usr/bin/env python3
"""Where the column (iv) Hansen J comes from: by DMA, by instrument, by eigen-direction.

Lognormal price coefficient (f71.lognormal_problem).  Holds (pi_p, sigma) and beta at the estimates in estimation/f71_col_iv_<tag>.json, recovers xi
with one contraction (no optimization), and recomputes J exactly as f71 does:
J = N g' S^+ g, g = mean of Z_D xi, S = clustered (DMA) covariance of the moments.  The
recomputed J must equal the J in the json; the script says so.  Then, with the parameters
held fixed (a diagnostic, not a re-estimation):

  by DMA         each DMA's additive share of J (the shares sum to J), and J without it
  by instrument  the z-score of each excluded instrument's moment, and J without it
  by direction   S's eigenvalues and each eigen-direction's share of J

    python3 code/f71_jdiag.py --tag long      # -> logs to stdout, estimation/f71_jdiag_iv_<tag>.json
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f71_baseline_table as f71

log = f71.log


def jstat(m, cl, G, keep=None):
    """f71's J from the N x MD moment contributions m, clustered on cl."""
    if keep is not None:
        m = m[:, keep]
    M = np.zeros((G, m.shape[1])); np.add.at(M, cl, m)
    S = M.T @ M / len(m); gb = m.mean(0)[:, None]
    return len(m) * float(np.squeeze(gb.T @ np.linalg.pinv(S) @ gb)), M, S, gb


def main(tag, markets=0):
    import pyblp
    pyblp.options.verbose = False
    t0 = time.time()
    d, terms = f71.load(tag, "iv")
    if markets:                                   # smoke test: parameters from a 1-step fit
        keep = np.random.default_rng(0).choice(d.market_ids.unique(), markets, replace=False)
        d = d[d.market_ids.isin(keep)].reset_index(drop=True)
        terms = [t for t in terms if (d[t] != 0).sum() >= 200]
        est = dict(pi_p=-0.493, sigma_p=0.353, sigma_lsz=1.080, beta=None, J=np.nan)
    else:
        est = json.load(open(f71.E / f"f71_col_iv_{tag}.json"))
    for k, c in enumerate(f71.ZCOLS):
        d[f"demand_instruments{k}"] = d[c].astype(float)
    prob = f71.lognormal_problem(d, terms)
    K = prob.K1 + prob.K2 + 1                       # + pi_p
    log(f"(iv) J diagnostic  rows {len(d):,}  markets {prob.T:,}  MD {prob.MD}  df {prob.MD - K}")
    sig = np.diag([est["sigma_p"], est["sigma_lsz"]])
    r = prob.solve(sigma=sig, pi=np.array([[est["pi_p"]], [0.0]]),
                   optimization=pyblp.Optimization("return"),
                   iteration=pyblp.Iteration("squarem", {"atol": 1e-12}),
                   method="1s", se_type="unadjusted", check_optimality="gradient")
    log(f"  contraction at pi_p {est['pi_p']:.5f}  sigma_p {est['sigma_p']:.5f}  sigma_lsz {est['sigma_lsz']:.5f}  "
        f"[{time.time() - t0:.0f}s]")
    # xi = absorbed delta - absorbed X1 beta, with the estimates' beta (two-step W)
    lab = list(terms)
    dl, _ = prob._absorb_demand_ids(np.asarray(r.delta))
    X = np.asarray(prob.products.X1)
    b1 = np.asarray(r.beta).ravel()
    gap = np.max(np.abs(dl.ravel() - X @ b1 - np.asarray(r.xi).ravel()))
    log(f"  check: xi rebuilt from delta and X1 matches pyblp's xi to {gap:.1e}")
    assert gap < 1e-8, gap
    b = b1 if est["beta"] is None else np.array([est["beta"][k] for k in lab])
    xi = dl.ravel() - X @ b
    Z = np.asarray(prob.products.ZD)
    zc, _ = prob._absorb_demand_ids(d[f71.ZCOLS].to_numpy(float))
    assert np.allclose(Z[:, :len(f71.ZCOLS)], zc), "Z_D does not start with the excluded instruments"
    m = Z * xi[:, None]
    dma = d.clustering_ids.astype(str).to_numpy()
    cl, names = pd.factorize(dma); G = len(names)
    J, M, S, gb = jstat(m, cl, G)
    df = prob.MD - K
    log(f"\n  HANSEN J {J:.3f}  df {df}  p {1 - stats.chi2.cdf(J, df):.4f}  clusters {G}"
        + (f"   (estimation log: {est['J']:.3f}"
           + ("  MATCH)" if abs(J - est["J"]) < 1e-3 * max(1, est["J"]) else "  MISMATCH)")
           if not markets else ""))
    Sp = np.linalg.pinv(S)
    N = len(m)

    # ---- by DMA: J = sum_c M_c' S^+ g, so each DMA's term is additive
    share = M @ Sp @ gb.ravel()
    loo = np.array([jstat(m[cl != c], pd.factorize(cl[cl != c])[0], G - 1)[0] for c in range(G)])
    st = d.groupby("clustering_ids").store_code_uc.nunique()
    ch = pd.read_parquet(f71.ROOT / "data_work" / "f91_store_chain.parquet")
    chain = dict(zip(ch.store_code_uc.astype(int), ch.parent.astype(str)))
    chs = d.drop_duplicates("store_code_uc").assign(
        p=lambda x: x.store_code_uc.astype(int).map(chain)) \
        .groupby("clustering_ids").p.agg(lambda s: ",".join(sorted(set(map(str, s)))))
    T = pd.DataFrame({"dma": names, "share": share, "J_without": loo,
                      "rows": np.bincount(cl), "stores": st.reindex(names).to_numpy(),
                      "chains": chs.reindex(names).to_numpy()}) \
        .sort_values("share", ascending=False).reset_index(drop=True)
    log(f"\n  BY DMA: additive shares sum to {T.share.sum():.3f}; "
        f"top 5 carry {T.share.head(5).sum():.3f}, top 10 {T.share.head(10).sum():.3f}")
    log(f"  {'dma':>6s}{'share':>9s}{'J w/o':>9s}{'rows':>9s}{'stores':>8s}  chains")
    for _, x in T.head(12).iterrows():
        log(f"  {x.dma:>6s}{x.share:>9.3f}{x.J_without:>9.3f}{x.rows:>9,}{x.stores:>8}  {x.chains}")
    log(f"  ... lowest J without one DMA: {T.J_without.min():.3f} (DMA {T.dma[T.J_without.idxmin()]})")

    # ---- by instrument: the excluded instruments are the first len(ZCOLS) columns of Z_D
    nz = len(f71.ZCOLS)
    zs = np.sqrt(N) * gb.ravel() / np.sqrt(np.diag(S))
    log(f"\n  BY INSTRUMENT (excluded; z = sqrt(N) g_k / sqrt(S_kk), J without it has df {df - 1})")
    inst = []
    for k, c in enumerate(f71.ZCOLS):
        keep = [i for i in range(prob.MD) if i != k]
        jk = jstat(m, cl, G, keep)[0]
        inst.append(dict(instrument=c, z=float(zs[k]), J_without=jk))
        log(f"  {c:12s} z {zs[k]:+7.3f}   J without {jk:8.3f}")
    log(f"  largest |z| among the {prob.MD - nz} exogenous X1 moments: "
        f"{np.max(np.abs(zs[nz:])):.3f}")

    # ---- by direction: J = sum_i N (v_i' g)^2 / lambda_i over S's eigenpairs
    lam, V = np.linalg.eigh(S)
    tol = lam.max() * len(lam) * np.finfo(float).eps            # pinv's cutoff
    part = np.where(lam > tol, N * (V.T @ gb.ravel()) ** 2 / np.where(lam > tol, lam, 1), 0)
    o = np.argsort(lam)
    log(f"\n  BY DIRECTION: condition number of S {lam.max() / lam[lam > tol].min():.2e}; "
        f"{int((lam <= tol).sum())} directions below pinv's cutoff")
    log(f"  smallest 5 eigenvalues carry {part[o[:5]].sum():.3f} of J; largest 5 carry "
        f"{part[o[-5:]].sum():.3f}")
    for i in o[:5]:
        top = np.argsort(-np.abs(V[:, i]))[:3]
        log(f"    lambda {lam[i]:.2e}  share {part[i]:7.3f}   loads on "
            + ", ".join(f"{(f71.ZCOLS + lab)[j]}({V[j, i]:+.2f})" for j in top))
    if not markets:
        json.dump(dict(tag=tag, J=J, J_json=est["J"], df=int(df), clusters=G,
                       by_dma=T.to_dict("records"), by_instrument=inst,
                       eig=lam.tolist(), eig_share=part.tolist(), seconds=time.time() - t0),
                  open(f71.E / f"f71_jdiag_iv_{tag}.json", "w"), indent=1)
    log(f"\n[{time.time() - t0:.0f}s] done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="long")
    ap.add_argument("--markets", type=int, default=0, help="smoke test on this many markets")
    a = ap.parse_args()
    main(a.tag, a.markets)
