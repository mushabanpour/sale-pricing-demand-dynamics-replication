"""Supply side: markups and marginal costs under two conduct assumptions.

SUPERSEDED, AND STILL IMPORTED.  `f90_supply_mw.py` replaced the estimator here: it
sweeps the conduct parameter over a grid instead of contrasting two fixed ownership
matrices, and Tables 7-9 come from it.  `run(col)` below is kept for reference and is
not a pipeline stage -- no `f87_supply_{col}_{tag}.parquet` exists for the paper's
sample.  What IS live is the `FIRM` brand-to-firm map, imported by `f90_supply_mw.py`,
`f92_counterfactual.py`, `f92_by_firm.py` and `f96_section2_tex.py`.  Do not delete
this file; if you move `FIRM`, move those four imports with it.

For each store-week the demand Jacobian is

    d s_k / d p_j = sum_i w_i alpha_i P_ij (1{j=k} - P_ik),

and the multiproduct first-order condition gives  p - mc = (Omega .* -D)^{-1} s,
where Omega_jk = 1 when j and k are priced by the same agent.  Two assumptions:

  BREWER    Omega_jk = 1 iff brands j and k belong to the same firm (four firms),
            each internalising cannibalisation within its own portfolio only.
  RETAILER  Omega = 1 everywhere, the store pricing its whole shelf jointly.

Marginal cost is read off the NNN weeks alone, where a static pricing model is
defensible; the sale weeks are then held up against it rather than used to fit it.

    python3 code/f87_supply.py iv
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

FIRM = {
    "BUD LIGHT": "ABI", "BUD LIGHT LIME": "ABI", "BUD LIGHT ORANGE": "ABI",
    "BUDWEISER": "ABI", "MICHELOB ULTRA": "ABI",
    "COORS LIGHT": "MolsonCoors", "COORS BANQUET": "MolsonCoors",
    "MILLER LITE": "MolsonCoors", "MILLER HIGH LIFE": "MolsonCoors",
    "CORONA EXTRA": "Constellation", "CORONA LIGHT": "Constellation",
    "MODELO ESPECIAL": "Constellation", "MODELO NEGRA": "Constellation",
    "HEINEKEN": "Heineken",
}


def run(col):
    import pyblp
    pyblp.options.verbose = False
    t0 = time.time()
    j = json.load(open(E / f"f71_col_{col}_{TAG}.json"))
    sp, sl = j["sigma_p"], j["sigma_lsz"]
    bp, pi = f71.price_draw(j)

    d, terms = f71.load(TAG, col)
    assert set(d.brand_id.unique()) <= set(FIRM), set(d.brand_id.unique()) - set(FIRM)
    for k, c in enumerate(f71.ZCOLS):
        d[f"demand_instruments{k}"] = d[c].astype(float)
    prob, delta = f71.demand_at(d, terms, j)          # lognormal, at the estimates

    nodes = np.asarray(prob.agents.nodes)
    w = np.asarray(prob.agents.weights).ravel()
    pm = np.asarray(prob.products.market_ids).ravel()
    am = np.asarray(prob.agents.market_ids).ravel()
    price = d.prices.to_numpy(float)
    lnz = d.log_n_eq.to_numpy(float)
    fid = pd.factorize(d.brand_id.map(FIRM))[0]

    pstart = {m: s for m, s in zip(*np.unique(pm, return_index=True))}
    astart = {m: s for m, s in zip(*np.unique(am, return_index=True))}
    pcount = dict(zip(*np.unique(pm, return_counts=True)))
    acount = dict(zip(*np.unique(am, return_counts=True)))

    mk_b = np.full(len(d), np.nan)      # markup, brewer Nash-Bertrand
    mk_r = np.full(len(d), np.nan)      # markup, multiproduct retailer
    for m in pstart:
        ps, pn = pstart[m], pcount[m]
        as_, an = astart[m], acount[m]
        pj, ai = slice(ps, ps + pn), slice(as_, as_ + an)
        nu = nodes[ai]
        u = (delta[pj][:, None] + np.outer(-price[pj], np.exp(pi + sp * nu[:, 0]))
             + np.outer(lnz[pj], sl * nu[:, 1]))
        mx = np.maximum(0.0, u.max(0))
        ex = np.exp(u - mx)
        P = ex / (np.exp(-mx) + ex.sum(0))
        A = w[ai] * (bp - np.exp(pi + sp * nu[:, 0]))
        PA = P * A
        D = np.diag(PA.sum(1)) - PA @ P.T          # dS_k/dp_j, symmetric
        s = P @ w[ai]
        same = (fid[pj][:, None] == fid[pj][None, :]).astype(float)
        for om, out in ((same, mk_b), (np.ones_like(same), mk_r)):
            try:
                out[pj] = np.linalg.solve(-om * D, s)
            except np.linalg.LinAlgError:
                pass

    out = pd.DataFrame(dict(
        market_ids=d.market_ids.to_numpy(), store=d.store_code_uc.to_numpy(),
        alt=d.alt.to_numpy(), brand=d.brand_id.to_numpy(),
        firm=d.brand_id.map(FIRM).to_numpy(), n_cans=d.n_cans.to_numpy(),
        reg3=d[f71.COLS[col][0]].to_numpy(), price=price,
        mk_brewer=mk_b, mk_retail=mk_r))
    out["mc_brewer"] = out.price - out.mk_brewer
    out["mc_retail"] = out.price - out.mk_retail
    out.to_parquet(RES / f"f87_supply_{col}_{TAG}.parquet", index=False)

    nnn = out[out.reg3 == "NNN"]
    print(f"\ncolumn ({col})  supply side, {len(out):,} rows, "
          f"{len(nnn):,} in the no-sale state\n")
    print(f"{'':28s}{'Brewer-Nash':>14s}{'Retailer':>14s}")
    def line(lab, f):
        print(f"  {lab:26s}{f('brewer'):>14}{f('retail'):>14}")
    line("Median markup ($/144oz)", lambda k: f"{nnn[f'mk_{k}'].median():.3f}")
    line("Median markup (% price)",
         lambda k: f"{(100 * nnn[f'mk_{k}'] / nnn.price).median():.1f}")
    line("Median mc ($/144oz)", lambda k: f"{nnn[f'mc_{k}'].median():.3f}")
    line("Share mc < 0 (%)", lambda k: f"{100 * (nnn[f'mc_{k}'] < 0).mean():.2f}")
    line("Share mc < 0, all weeks (%)",
         lambda k: f"{100 * (out[f'mc_{k}'] < 0).mean():.2f}")
    print(f"\n  median mc ($/144oz) by pack size, no-sale weeks")
    g = nnn.groupby("n_cans")[["mc_brewer", "mc_retail", "price"]].median()
    print(g.round(3).to_string())
    print(f"\n  median markup (% of price) by firm, no-sale weeks")
    nnn = nnn.assign(mb=100 * nnn.mk_brewer / nnn.price,
                     mr=100 * nnn.mk_retail / nnn.price)
    print(nnn.groupby("firm")[["mb", "mr"]].median().round(2).to_string())
    print(f"\n  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    for c in (sys.argv[1:] or ["iv"]):
        run(c)
