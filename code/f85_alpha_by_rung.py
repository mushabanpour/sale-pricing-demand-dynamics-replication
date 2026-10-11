"""Demand-weighted mean of the household price coefficient, by pack size.

alpha_i = beta_p - exp(sigma_p nu_i) does not depend on the pack size a household
buys.  What varies across the ladder is WHO buys: the linear random coefficient on
log_n_eq sorts households across pack sizes, and because the draws on the two
dimensions are independent under the estimated sigma, any gradient in

    E[alpha_i | i buys a pack of n units]
      = sum_t sum_{j: n(j)=n} sum_i w_i P_ijt alpha_i / sum_t sum_j sum_i w_i P_ijt

comes through the choice probabilities alone.  Delta is recovered by inverting
observed shares at the estimated sigma, which needs no beta and no weighting matrix,
so a single objective evaluation reproduces the estimated model exactly.

    python3 code/f85_alpha_by_rung.py ii iii iv
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import f71_baseline_table as f71                                   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
E, RES = ROOT / "estimation", ROOT / "results"
TAG = "long"


def MA_NAME(t):
    return "f85_alpha_by_rung.csv" if t == "dev" else f"f85_alpha_by_rung_{t}.csv"


def run(col):
    import pyblp
    pyblp.options.verbose = False
    t0 = time.time()
    jf = E / f"f71_col_{col}_{TAG}.json"
    j = json.load(open(jf))
    sp, sl = j["sigma_p"], j["sigma_lsz"]
    bp, pi = f71.price_draw(j)                         # alpha_i = bp - exp(pi + sigma_p nu)

    d, terms = f71.load(TAG, col)
    for k, c in enumerate(f71.ZCOLS):
        d[f"demand_instruments{k}"] = d[c].astype(float)
    prob, delta = f71.demand_at(d, terms, j)          # lognormal, at the estimates

    nodes = np.asarray(prob.agents.nodes)
    w = np.asarray(prob.agents.weights).ravel()
    pm = np.asarray(prob.products.market_ids).ravel()
    am = np.asarray(prob.agents.market_ids).ravel()
    price = d.prices.to_numpy(float)
    lnz = d.log_n_eq.to_numpy(float)
    rung = d.n_cans.to_numpy(int)

    pstart = {m: s for m, s in zip(*np.unique(pm, return_index=True))}
    astart = {m: s for m, s in zip(*np.unique(am, return_index=True))}
    pcount = dict(zip(*np.unique(pm, return_counts=True)))
    acount = dict(zip(*np.unique(am, return_counts=True)))

    num = {k: 0.0 for k in f71.RUNGS}
    den = {k: 0.0 for k in f71.RUNGS}
    eps = np.full(len(d), np.nan)                                  # own-price, per row
    # Purchase weight each agent draw earns from each rung, kept at draw level so the
    # conditional MEDIAN of alpha_i can be taken later.  pyblp continues the Halton
    # sequence across markets, so every one of the T x I draws is distinct.
    AW = np.zeros((len(f71.RUNGS), len(nodes)))
    # Mean elasticity matrix across pack sizes, in the manner of Miller and Weinberg's
    # Table V: cell (n, m) is the percentage change in the combined quantity of
    # n-unit packs from a one percent change in the price of every m-unit pack, and
    # the last row is the outside good.  Averaged over markets, as they do.
    NR = len(f71.RUNGS)
    ES = np.zeros((NR + 1, NR)); EN = np.zeros((NR + 1, NR))
    abar = np.full(len(d), np.nan)                                 # alpha of j's buyers
    curv = np.full(len(d), np.nan)                                 # the (1 - c_j) term
    for m in pstart:
        ps, pn = pstart[m], pcount[m]
        as_, an = astart[m], acount[m]
        pj = slice(ps, ps + pn)
        ai = slice(as_, as_ + an)
        nu = nodes[ai]
        alpha = bp - np.exp(pi + sp * nu[:, 0])                         # (I,)
        u = (delta[pj][:, None]
             + np.outer(-price[pj], np.exp(pi + sp * nu[:, 0]))
             + np.outer(lnz[pj], sl * nu[:, 1]))
        mx = np.maximum(0.0, u.max(0))                             # stable softmax
        ex = np.exp(u - mx)
        P = ex / (np.exp(-mx) + ex.sum(0))                         # (J, I)
        wi = w[ai]
        pw = P * wi                                                # w_i P_ijt
        num_j = pw @ alpha
        den_j = pw.sum(1)
        # ds_j/dp_j = sum_i w_i alpha_i P_ij (1 - P_ij);  eps_jj = (p_j / s_j) ds/dp
        ds = (P * (1.0 - P) * (wi * alpha)).sum(1)
        eps[pj] = ds * price[pj] / den_j
        # eps_j = p_j * alphabar_j * (1 - c_j) exactly, with c_j = sum w a P^2 / sum w a P
        abar[pj] = num_j / den_j
        curv[pj] = ds / num_j
        A = wi * alpha
        Q = np.zeros((NR, an)); R = np.zeros((NR, an)); sn = np.zeros(NR)
        for q, k in enumerate(f71.RUNGS):
            msk = rung[pj] == k
            if msk.any():
                Q[q] = P[msk].sum(0)
                R[q] = (P[msk] * price[pj][msk, None]).sum(0)
                sn[q] = float(wi @ Q[q])
        T = P.sum(0)
        s0 = 1.0 - float(wi @ T)
        for m_ in range(NR):
            if sn[m_] <= 0:
                continue                       # no such pack in this store's choice set
            for n_ in range(NR):
                if sn[n_] <= 0:
                    continue
                nm_ = A @ ((R[n_] if n_ == m_ else 0.0) - Q[n_] * R[m_])
                ES[n_, m_] += nm_ / sn[n_]; EN[n_, m_] += 1
            if s0 > 0:
                ES[NR, m_] += float(-(A @ (R[m_] * (1.0 - T)))) / s0
                EN[NR, m_] += 1

        for q, k in enumerate(f71.RUNGS):
            sel = rung[pj] == k
            if sel.any():
                num[k] += float(num_j[sel].sum())
                den[k] += float(den_j[sel].sum())
                AW[q, ai] = pw[sel].sum(0)

    # alpha_i = bp - exp(pi + sigma_p nu) is strictly decreasing in nu, so the
    # purchase-weighted median of alpha is that transform of the weighted median of nu.
    nu0 = nodes[:, 0]
    order = np.argsort(nu0)
    nus = nu0[order]
    med_a = {}
    for q, k in enumerate(f71.RUNGS):
        cw = np.cumsum(AW[q, order])
        if cw[-1] > 0:
            med_a[k] = float(bp - np.exp(pi + sp * nus[np.searchsorted(cw, cw[-1] / 2)]))
    cw = np.cumsum(AW[:, order].sum(0))
    med_pool = float(bp - np.exp(pi + sp * nus[np.searchsorted(cw, cw[-1] / 2)]))
    del AW

    out = {k: num[k] / den[k] for k in f71.RUNGS if den[k] > 0}
    pooled = sum(num.values()) / sum(den.values())
    uncond = bp - np.exp(pi + sp ** 2 / 2)                              # E[alpha_i]
    print(f"\ncolumn ({col})   pi_p {pi:+.4f}   sigma_p {sp:.4f}   "
          f"median alpha_i {j["median_alpha"]:+.4f}   unconditional mean {uncond:+.4f}")
    for k in f71.RUNGS:
        if k in out:
            print(f"   {k:2d}-unit packs   mean alpha | buy  {out[k]:+.4f}")
    print(f"   all packs        mean alpha | buy  {pooled:+.4f}")
    print(f"\ncolumn ({col})   MEDIAN alpha_i among buyers, by pack size")
    for k in f71.RUNGS:
        if k in med_a:
            print(f"   {k:2d}-unit packs   median alpha | buy  {med_a[k]:+.4f}")
    print(f"   all packs        median alpha | buy  {med_pool:+.4f}"
          f"   (unconditional {j["median_alpha"]:+.4f})")
    np.save(RES / f"f85_eps_{col}_{TAG}.npy", eps)
    dec = pd.DataFrame(dict(rung=rung, p=price, abar=abar, curv=curv, eps=eps))
    g = dec.groupby("rung").mean()
    print(f"\ncolumn ({col})   eps_jj = p_j * alphabar_j * (1 - c_j), rung means")
    print(f"   {'':14s}{'p_j':>9s}{'alphabar':>10s}{'1-c_j':>8s}"
          f"{'product':>10s}{'mean eps':>10s}{'max err':>10s}")
    for k in f71.RUNGS:
        r_ = g.loc[k]; e_ = dec[dec.rung == k]
        err = float((e_.p * e_.abar * e_.curv - e_.eps).abs().max())
        print(f"   {k:2d}-unit packs{r_.p:>9.2f}{r_.abar:>10.4f}{r_.curv:>8.4f}"
              f"{r_.p * r_.abar * r_.curv:>10.2f}{r_.eps:>10.2f}{err:>10.1e}")
    dec.to_csv(RES / f"f85_eps_decomp_{col}_{TAG}.csv", index=False)
    M = np.where(EN > 0, ES / np.maximum(EN, 1), np.nan)
    lab = [f"{k}" for k in f71.RUNGS] + ["outside"]
    mat = pd.DataFrame(M, index=lab, columns=[str(k) for k in f71.RUNGS])
    mat.to_csv(RES / f"f85_elast_matrix_{col}_{TAG}.csv")
    pd.DataFrame(EN, index=lab, columns=[str(k) for k in f71.RUNGS]).to_csv(
        RES / f"f85_elast_matrix_{col}_{TAG}_n.csv")
    print(f"\ncolumn ({col})   mean elasticity matrix (row = quantity, col = price)")
    print(mat.round(4).to_string())
    print("   markets per cell:")
    print(pd.DataFrame(EN.astype(int), index=lab,
                       columns=[str(k) for k in f71.RUNGS]).to_string())
    ref = dict(json.load(open(jf))["by_rung"])
    print(f"\ncolumn ({col})   own-price elasticity by pack size")
    print(f"   {'':14s}{'mean':>9s}{'median':>9s}{'p25':>9s}{'p75':>9s}{'rows':>9s}"
          f"   {'f71 mean':>9s}")
    erows = {}
    for k in f71.RUNGS:
        e = eps[rung == k]
        erows[k] = (float(e.mean()), float(np.median(e)))
        print(f"   {k:2d}-unit packs{e.mean():>9.2f}{np.median(e):>9.2f}"
              f"{np.percentile(e, 25):>9.2f}{np.percentile(e, 75):>9.2f}{len(e):>9,d}"
              f"   {ref[k]:>9.2f}")
    print(f"   all packs    {eps.mean():>9.2f}{np.median(eps):>9.2f}"
          f"{np.percentile(eps, 25):>9.2f}{np.percentile(eps, 75):>9.2f}{len(eps):>9,d}")
    chk = max(abs(erows[k][0] - ref[k]) for k in f71.RUNGS)
    print(f"   max |mine - f71| across rungs  {chk:.2e}   [{time.time() - t0:.0f}s]")
    return dict(col=col, beta_p=bp, sigma_p=sp, median=j["median_alpha"], uncond=float(uncond),
                pooled=pooled, **{f"a{k}": out.get(k) for k in f71.RUNGS},
                **{f"eps_mean{k}": erows[k][0] for k in f71.RUNGS},
                **{f"eps_med{k}": erows[k][1] for k in f71.RUNGS},
                med_pooled=med_pool,
                **{f"m{k}": med_a.get(k) for k in f71.RUNGS})


if __name__ == "__main__":
    av = sys.argv[1:]
    if "--tag" in av:                       # e.g. --tag long
        i = av.index("--tag"); TAG = av[i + 1]; del av[i:i + 2]
    cols = av or ["ii"]
    rows = [run(c) for c in cols]
    # Merge, never overwrite: running a single column must not drop the others, which
    # Table 4A reads for every column.
    # Columns may run in parallel: the read-merge-write is held under a file lock.
    import fcntl
    out = RES / MA_NAME(TAG)
    with open(RES / (MA_NAME(TAG) + ".lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        new = pd.DataFrame(rows).set_index("col")
        if out.exists():
            old = pd.read_csv(out).set_index("col")
            new = new.combine_first(old).reindex(
                [c for c in ["ii", "iii", "iv"] if c in set(old.index) | set(new.index)])
        new.reset_index().to_csv(out, index=False)
        fcntl.flock(lk, fcntl.LOCK_UN)
    print(f"\nwrote {RES / 'f85_alpha_by_rung.csv'}")
