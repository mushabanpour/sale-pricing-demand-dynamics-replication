#!/usr/bin/env python3
"""Table 12 Panel B with rivals re-optimizing: one firm removes ITS at observed prices, and its
rivals re-set their regular-week prices while keeping their sale calendar.

For each firm f, two price vectors are solved in every store-week:

    keep    f at its observed prices and sale regimes
    remove  f's sale weeks at the product's reference price, f's sale regimes set to NNN

In both, every rival product that is not on sale that week is priced at the rival's static
best response (the first-order condition of f92_counterfactual.py, solved for those rows only,
all other prices held fixed); every rival product on sale keeps its observed price and
regime.  The difference between the two is f's gain or loss from removing ITS alone once
rivals respond.  Demand, draws, marginal cost and welfare are f92/f98's.

    python3 code/f99_solo_rivals.py               # -> results/f99_solo_rivals_iv_long.csv
    python3 code/f99_solo_rivals.py --markets 200 # smoke test
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f92_counterfactual as f92
from f87_supply import FIRM

FIRMS = ("ABI", "MolsonCoors", "Constellation", "Heineken")


def best_response(p, base, mc, bp, e0, lz, A, wts, own, free, maxit=500, tol=1e-8):
    """Prices of the `free` rows at the static first-order condition, the rest held fixed:
    s_F + (Omega.*D)_FF (p_F - mc_F) + (Omega.*D)_FX (p_X - mc_X) = 0."""
    F, X = free, ~free
    pmax = 5.0 * max(float(p.max()), 1.0)
    for damp, it_max in ((1.0, maxit), (0.5, 2 * maxit)):
        q = p.copy()
        for _ in range(it_max):
            u = (base + bp * q)[:, None] + (-q)[:, None] * e0[None, :] + lz
            mx = np.maximum(0.0, u.max(0))
            ex = np.exp(u - mx)
            P = ex / (np.exp(-mx) + ex.sum(0))
            s = P @ wts
            PA = P * A
            OD = own * (np.diag(PA.sum(1)) - PA @ P.T)
            rhs = s[F] + OD[np.ix_(F, X)] @ (q[X] - mc[X])
            new = q.copy()
            new[F] = np.clip(mc[F] - f92._psolve(OD[np.ix_(F, F)], rhs), 0.05, pmax)
            if np.max(np.abs(new - q)) < tol:
                return new, True
            q = (1 - damp) * q + damp * new
    return q, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:.0f}s] {m}", flush=True)
    prob, delta, sp, sl, bp, d, lam = f92.setup("iv")
    mc = np.load(f92.RES / f"f92_mc_iv_{f92.TAG}.npy")
    assert len(mc) == len(d)
    nodes = np.asarray(prob.agents.nodes)
    w = np.asarray(prob.agents.weights).ravel()
    am = np.asarray(prob.agents.market_ids).ravel()
    ast_ = dict(zip(*np.unique(am, return_index=True)))
    acnt = dict(zip(*np.unique(am, return_counts=True)))
    p0 = d.prices.to_numpy(float)
    sale = d.sale.to_numpy(float) == 1
    preg = np.where(sale, d.p_reg.to_numpy(float), p0)
    lnz = d.log_n_eq.to_numpy(float)
    firm = d.brand_id.map(FIRM).to_numpy()
    fid = pd.factorize(firm)[0]
    pm = d.market_ids.to_numpy()
    order = np.argsort(pm, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(pm[order])) + 1)
    if a.markets:
        groups = groups[:a.markets]
    log(f"setup: {len(d):,} rows, {len(groups):,} markets")

    PS = {(c, f, g): 0.0 for c in ("keep", "remove") for f in FIRMS for g in FIRMS}
    CS = {(c, f): 0.0 for c in ("keep", "remove") for f in FIRMS}
    gap, nfail, nsolve = [], 0, 0
    for gi, idx in enumerate(groups):
        m = pm[idx[0]]
        ai = slice(ast_[m], ast_[m] + acnt[m])
        nu, wts = nodes[ai], w[ai]
        e0 = np.exp(f92.PI + sp * nu[:, 0]); alpha = bp - e0; A = wts * alpha
        lz = np.outer(lnz[idx], sl * nu[:, 1])
        own = (fid[idx][:, None] == fid[idx][None, :]).astype(float)
        fm, sm = firm[idx], sale[idx]
        base_obs = delta[idx] - bp * p0[idx]
        for f in FIRMS:
            free = (fm != f) & ~sm
            for c in ("keep", "remove"):
                mine = (fm == f) & (c == "remove")
                p = np.where(mine, preg[idx], p0[idx])
                base = base_obs - np.where(mine, lam[idx], 0.0)
                if free.any() and (fm == f).any():
                    p, ok = best_response(p, base, mc[idx], bp, e0, lz, A, wts, own, free)
                    nfail += not ok; nsolve += 1
                    if c == "keep":
                        gap.append(np.abs(p[free] - p0[idx][free]))
                cs, _, s, _ = f92.welfare(p, base, bp, e0, lz, alpha, wts, mc[idx])
                CS[(c, f)] += cs
                prof = (p - mc[idx]) * s
                for g in FIRMS:
                    PS[(c, f, g)] += float(prof[fm == g].sum())
        if gi % 2000 == 0:
            log(f"  market {gi:,} of {len(groups):,}; solves {nsolve:,}, not converged {nfail}")
    gap = np.concatenate(gap) if gap else np.zeros(1)
    log(f"rival regular-week prices at best response vs observed: median |diff| "
        f"{np.median(gap):.4f}, 90th pct {np.percentile(gap, 90):.4f} dollars a serving; "
        f"{nfail} of {nsolve:,} solves not converged")
    rows = []
    for f in FIRMS:
        r = dict(firm=f, cs_keep=CS[("keep", f)], cs_remove=CS[("remove", f)])
        for g in FIRMS:
            r[f"ps_{g}_keep"], r[f"ps_{g}_remove"] = PS[("keep", f, g)], PS[("remove", f, g)]
        rows.append(r)
    T = pd.DataFrame(rows).set_index("firm")
    out = f92.RES / f"f99_solo_rivals_iv_{f92.TAG}.csv"
    if not a.markets:
        T.to_csv(out)
    print("\nremoving ITS alone, rivals re-set regular-week prices (percent change from keep):")
    for f in FIRMS:
        own_ = 100 * (T.loc[f, f"ps_{f}_remove"] / T.loc[f, f"ps_{f}_keep"] - 1)
        riv = sum(T.loc[f, f"ps_{g}_remove"] for g in FIRMS if g != f) / \
              sum(T.loc[f, f"ps_{g}_keep"] for g in FIRMS if g != f)
        csx = 100 * (T.loc[f, "cs_remove"] / T.loc[f, "cs_keep"] - 1)
        print(f"  {f:14s} own profit {own_:+6.2f}   rivals' profit {100 * (riv - 1):+6.2f}   "
              f"consumer surplus {csx:+6.2f}")
    log("done" + ("" if a.markets else f" -> {out}"))


if __name__ == "__main__":
    main()
