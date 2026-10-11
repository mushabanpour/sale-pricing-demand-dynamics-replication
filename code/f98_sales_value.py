#!/usr/bin/env python3
"""The value of the sales stores actually ran, column (iv), at observed prices.

Section 6's counterfactuals compare static equilibria, so their MIX benchmark charges the
static Nash price in sale weeks and carries no sale discount.  This evaluates the same
demand system at the prices stores actually charged instead, and removes the sales:

    observed        observed prices, observed sale regimes
    no_sales        sale-week prices at the product's regular (reference) price, every
                    regime set to NNN (the lambda shifts removed)
    no_discount     sale-week prices at the regular price, observed regimes
    no_regimes      observed prices, every regime set to NNN
    solo_<firm>     no_sales for that firm's products only; every rival's prices and
                    sales as observed (the unilateral deviation)

All other prices are held at their observed levels (no re-optimization).  Demand,
contraction, draws, marginal cost and the welfare measure are f92_counterfactual.py's:
CS = E_i[log(1 + sum exp(u)) / |alpha_i|], PS = sum (p - mc) s, mc from no-sale weeks.

    python3 code/f98_sales_value.py              # -> results/f98_sales_value_iv_long.csv
    python3 code/f98_sales_value.py --lam v      # -> results/f98_sales_value_iv_long_lamv.csv

--lam v removes column (v)'s regime shifts (store-product effects, f71_fixtheta_spfe.py)
in place of column (iv)'s.  theta, the mean utilities and marginal cost stay column (iv)'s,
which column (v) shares by construction, so only the part of delta credited to the sale
regimes changes.
"""
import argparse, json
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f92_counterfactual as f92
from f87_supply import FIRM

FIRMS = ("ABI", "MolsonCoors", "Constellation", "Heineken")
# case -> (rows whose sale-week price goes to the regular price, rows whose regime shift
# is removed); None = no rows, "all" = every row, a firm = that firm's rows only
CASES = {"observed": (None, None), "no_sales": ("all", "all"),
         "no_discount": ("all", None), "no_regimes": (None, "all"),
         **{f"solo_{f}": (f, f) for f in FIRMS}}
NQ = 5


def evaluate(d, delta, lam, mc, nodes, w, am, sp, sl, bp, log=print):
    """CS and PS by case, firm and pack-taste quintile; d carries prices, p_reg, sale."""
    p0 = d.prices.to_numpy(float)
    sale = (d.sale.to_numpy(float) == 1)
    firm_all = d.brand_id.map(FIRM).to_numpy()
    def rows(spec):
        if spec is None:
            return np.zeros(len(d), bool)
        return np.ones(len(d), bool) if spec == "all" else (firm_all == spec)
    DROWS = {c: rows(v[0]) for c, v in CASES.items()}
    LROWS = {c: rows(v[1]) for c, v in CASES.items()}
    preg = np.where(sale, d.p_reg.to_numpy(float), p0)
    assert np.all(preg[sale] > p0[sale]), "a sale price at or above its reference"
    lnz = d.log_n_eq.to_numpy(float)
    firm = d.brand_id.map(FIRM).to_numpy()
    firms = sorted(set(firm))
    pm = d.market_ids.to_numpy()
    ast_ = dict(zip(*np.unique(am, return_index=True)))
    acnt = dict(zip(*np.unique(am, return_counts=True)))
    order = np.argsort(pm, kind="stable")
    bounds = np.flatnonzero(np.diff(pm[order])) + 1
    CS = {c: 0.0 for c in CASES}
    PS = {c: {f: 0.0 for f in firms} for c in CASES}
    CSQ = {c: np.zeros(NQ) for c in CASES}
    Q = {c: 0.0 for c in CASES}
    fit = 0.0
    for idx in np.split(order, bounds):
        m = pm[idx[0]]
        ai = slice(ast_[m], ast_[m] + acnt[m])
        nu = nodes[ai]; wts = w[ai]
        e0 = np.exp(f92.PI + sp * nu[:, 0]); alpha = bp - e0
        lz = np.outer(lnz[idx], sl * nu[:, 1])
        q = np.clip((stats.norm.cdf(nu[:, 1]) * NQ).astype(int), 0, NQ - 1)
        base_obs = delta[idx] - bp * p0[idx]
        for c in CASES:
            p = np.where(DROWS[c][idx], preg[idx], p0[idx])
            base = base_obs - np.where(LROWS[c][idx], lam[idx], 0.0)
            cs, _, s, csi = f92.welfare(p, base, bp, e0, lz, alpha, wts, mc[idx])
            CS[c] += cs
            np.add.at(CSQ[c], q, wts * csi)
            prof = (p - mc[idx]) * s
            for f in firms:
                PS[c][f] += float(prof[firm[idx] == f].sum())
            Q[c] += float(s.sum())
            if c == "observed":
                fit = max(fit, float(np.max(np.abs(s - d.shares.to_numpy()[idx]))))
    log(f"  observed shares reproduced to {fit:.2e}")
    rows = []
    for c in CASES:
        r = dict(case=c, cs=CS[c], ps=sum(PS[c].values()), inside_share_sum=Q[c])
        r.update({f"ps_{f}": PS[c][f] for f in firms})
        r.update({f"cs_q{k + 1}": CSQ[c][k] for k in range(NQ)})
        rows.append(r)
    return pd.DataFrame(rows).set_index("case")


def main(lamcol="iv"):
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:.0f}s] {m}", flush=True)
    prob, delta, sp, sl, bp, d, lam = f92.setup("iv")
    log(f"setup: {len(d):,} rows, {d.market_ids.nunique():,} markets")
    sfx = ""
    if lamcol != "iv":
        jv = json.load(open(f92.E / f"f71_col_{lamcol}_{f92.TAG}.json"))
        assert jv["theta_from"] == "iv"
        rc = d[f92.f71.COLS["iv"][0]].to_numpy(); nc = d.n_cans.to_numpy()
        lam = np.zeros(len(d))
        for key, b in jv["beta"].items():
            s_, k = key.rsplit("_", 1)
            lam[(rc == s_) & (nc == int(k))] = b
        sfx = f"_lam{lamcol}"
        log(f"regime shifts from column ({lamcol}): {len(jv['beta'])} cells")
    mc = np.load(f92.RES / f"f92_mc_iv_{f92.TAG}.npy")
    assert len(mc) == len(d), (len(mc), len(d))
    T = evaluate(d, delta, lam, mc, np.asarray(prob.agents.nodes),
                 np.asarray(prob.agents.weights).ravel(),
                 np.asarray(prob.agents.market_ids).ravel(), sp, sl, bp, log)
    T["total"] = T.cs + T.ps
    base = T.loc["observed"]
    pct = (100 * (T - base) / base.abs()).add_suffix("_pct_vs_observed")
    out = pd.concat([T, pct], axis=1)
    out.to_csv(f92.RES / f"f98_sales_value_iv_{f92.TAG}{sfx}.csv")
    # the unilateral deviations against everyone dropping their sales
    print("\nprofit change (%) of each firm: dropping its own sales alone vs all firms dropping")
    for f in FIRMS:
        print(f"  {f:14s} alone {pct.loc[f'solo_{f}', f'ps_{f}_pct_vs_observed']:+7.2f}   "
              f"all {pct.loc['no_sales', f'ps_{f}_pct_vs_observed']:+7.2f}   rivals' profit when it "
              f"drops alone: " + ", ".join(f"{g} {pct.loc[f'solo_{f}', f'ps_{g}_pct_vs_observed']:+.2f}"
                                             for g in FIRMS if g != f))
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(T.round(1).to_string()); print(pct.round(2).to_string())
    log("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", default="iv", help="column whose regime shifts are removed")
    main(ap.parse_args().lam)
