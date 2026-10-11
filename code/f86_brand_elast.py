"""Mean elasticities for 12-unit-pack products, after Miller and Weinberg (2017, Table V).

Upper panel: cell (b, b') is the percentage change in the combined quantity of brand b's
12-unit packs from a one percent change in the price of every 12-unit pack of brand b',
averaged over markets, with a final row for the outside good.  Lower panel: the same
column experiment against category quantities, where a category is a pack size or the
domestic/imported split and, as in MW, EXCLUDES the products in the column.

Delta is recovered by inverting observed shares at the estimated sigma, which needs
neither beta nor a weighting matrix, so a single objective evaluation reproduces the
estimated model exactly.

    python3 code/f86_brand_elast.py iv
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
FOCAL = 12                                        # the pack size the matrix is built on
IMPORTED = {"CORONA EXTRA", "CORONA LIGHT", "HEINEKEN",
            "MODELO ESPECIAL", "MODELO NEGRA"}


def run(col):
    import pyblp
    pyblp.options.verbose = False
    t0 = time.time()
    j = json.load(open(E / f"f71_col_{col}_{TAG}.json"))
    sp, sl = j["sigma_p"], j["sigma_lsz"]
    bp, pi = f71.price_draw(j)

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

    BR = sorted(d.brand_id.unique())
    bix = pd.Categorical(d.brand_id, categories=BR).codes.astype(int)
    NB, NR = len(BR), len(f71.RUNGS)
    dom = np.array([b not in IMPORTED for b in BR])
    isdom = dom[bix]

    pstart = {m: s for m, s in zip(*np.unique(pm, return_index=True))}
    astart = {m: s for m, s in zip(*np.unique(am, return_index=True))}
    pcount = dict(zip(*np.unique(pm, return_counts=True)))
    acount = dict(zip(*np.unique(am, return_counts=True)))

    EB = np.zeros((NB + 1, NB)); NBc = np.zeros((NB + 1, NB))      # brands + outside
    EC = np.zeros((NR + 2, NB)); NCc = np.zeros((NR + 2, NB))      # rungs + dom/imp
    for m in pstart:
        ps, pn = pstart[m], pcount[m]
        as_, an = astart[m], acount[m]
        pj, ai = slice(ps, ps + pn), slice(as_, as_ + an)
        nu = nodes[ai]
        alpha = bp - np.exp(pi + sp * nu[:, 0])
        u = (delta[pj][:, None] + np.outer(-price[pj], np.exp(pi + sp * nu[:, 0]))
             + np.outer(lnz[pj], sl * nu[:, 1]))
        mx = np.maximum(0.0, u.max(0))
        ex = np.exp(u - mx)
        P = ex / (np.exp(-mx) + ex.sum(0))
        wi, A = w[ai], w[ai] * (bp - np.exp(pi + sp * nu[:, 0]))
        pr, ru, bi = price[pj], rung[pj], bix[pj]
        T = P.sum(0)
        s0 = 1.0 - float(wi @ T)

        f12 = ru == FOCAL
        Q = np.zeros((NB, an)); R = np.zeros((NB, an)); sb = np.zeros(NB)
        for b in np.unique(bi[f12]):
            msk = f12 & (bi == b)
            Q[b] = P[msk].sum(0)
            R[b] = (P[msk] * pr[msk, None]).sum(0)
            sb[b] = float(wi @ Q[b])
        # category quantities: one per pack size, then domestic and imported
        CAT, sc = np.zeros((NR + 2, an)), np.zeros(NR + 2)
        for q, k in enumerate(f71.RUNGS):
            msk = ru == k
            if msk.any():
                CAT[q] = P[msk].sum(0); sc[q] = float(wi @ CAT[q])
        for q, msk in ((NR, isdom[pj]), (NR + 1, ~isdom[pj])):
            if msk.any():
                CAT[q] = P[msk].sum(0); sc[q] = float(wi @ CAT[q])

        for b2 in range(NB):
            if sb[b2] <= 0:
                continue
            for b1 in range(NB):
                if sb[b1] <= 0:
                    continue
                nm = A @ ((R[b1] if b1 == b2 else 0.0) - Q[b1] * R[b2])
                EB[b1, b2] += nm / sb[b1]; NBc[b1, b2] += 1
            if s0 > 0:
                EB[NB, b2] += float(-(A @ (R[b2] * (1.0 - T)))) / s0
                NBc[NB, b2] += 1
            # categories exclude the column's own products, as in MW
            for q in range(NR + 2):
                own = (q == f71.RUNGS.index(FOCAL)
                       or (q == NR and dom[b2]) or (q == NR + 1 and not dom[b2]))
                den = sc[q] - (sb[b2] if own else 0.0)
                if den <= 0:
                    continue
                Qx = CAT[q] - (Q[b2] if own else 0.0)
                EC[q, b2] += float(-(A @ (Qx * R[b2]))) / den
                NCc[q, b2] += 1

    MB = pd.DataFrame(np.where(NBc > 0, EB / np.maximum(NBc, 1), np.nan),
                      index=BR + ["Outside good"], columns=BR)
    MC = pd.DataFrame(np.where(NCc > 0, EC / np.maximum(NCc, 1), np.nan),
                      index=[f"{k} packs" for k in f71.RUNGS] + ["Domestic", "Imported"],
                      columns=BR)
    MB.to_csv(RES / f"f86_brand_elast_{col}_{TAG}.csv")
    MC.to_csv(RES / f"f86_cat_elast_{col}_{TAG}.csv")
    pd.DataFrame(NBc, index=BR + ["Outside good"], columns=BR).to_csv(
        RES / f"f86_brand_elast_{col}_{TAG}_n.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(f"\ncolumn ({col})  mean elasticities, {FOCAL}-unit packs "
              f"(row = quantity, col = price)")
        print(MB.round(3).to_string())
        print(f"\ncolumn ({col})  cross-elasticities by category")
        print(MC.round(3).to_string())
        print(f"\nmarkets per cell: min {int(NBc[NBc > 0].min()):,}  "
              f"max {int(NBc.max()):,}   [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    for c in (sys.argv[1:] or ["iv"]):
        run(c)
