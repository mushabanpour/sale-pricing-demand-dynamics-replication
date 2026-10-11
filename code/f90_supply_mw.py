"""Supply side on Miller and Weinberg's design.

Two changes from f87-f89, both taken from their replication code.

1.  PACK SIZE IS ABSORBED IN THE COST FIXED EFFECT.  Their cost regressors are
    w = [mcpost dist fess] with fess containing a PRODUCT fixed effect, and a
    product is brand x size, so no pack-size cost gradient is ever identified or
    interpreted.  Cost identification comes from time variation within product.
    Our earlier attempt to read that gradient off the residual required a
    package-level fixed cost of about $2.40 against contract evidence of ~$0.

2.  RETAIL CONDUCT IS A CALIBRATED GRID, NOT A BINARY.  f_omega.m sets
        bMarkup = -(conduct .* (ptr' D'))^{-1} s
        rMarkup = -rc (D')^{-1} s
        cost    = p - bMarkup - rMarkup
    with rc run over {0, .025, .05, .10, .15, .20} and reported as a sensitivity.
    rc scales the retailer from price-taker to full multiproduct monopolist, so
    our old "retailer" model was rc = 1, far outside their range.  Their baseline
    rc = 0 sets ptr = I; we use ptr = I throughout, which is exact at rc = 0 and
    an approximation above it (it omits how the brewer internalises retail
    pass-through, which needs demand second derivatives).

    python3 code/f90_supply_mw.py iv
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import f71_baseline_table as f71                                   # noqa: E402
from f87_supply import FIRM                                        # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
E, RES = ROOT / "estimation", ROOT / "results"
TAG = "long"
RC_GRID = (0.0, 0.025, 0.05, 0.10, 0.15, 0.20)


PI = 0.0       # set by setup(): the shift in alpha_i = bp - exp(PI + sigma_p nu)


def setup(col):
    import pyblp
    pyblp.options.verbose = False
    j = json.load(open(E / f"f71_col_{col}_{TAG}.json"))
    sp, sl = j["sigma_p"], j["sigma_lsz"]
    global PI                     # alpha_i = bp - exp(PI + sigma_p nu); lognormal: bp 0, PI pi_p
    bp, PI = f71.price_draw(j)
    d, terms = f71.load(TAG, col)
    for k, c in enumerate(f71.ZCOLS):
        d[f"demand_instruments{k}"] = d[c].astype(float)
    prob, delta = f71.demand_at(d, terms, j)          # lognormal, at the estimates
    assert {"is_can", "year_month", "store_code_uc"} <= set(d.columns)
    return prob, delta, sp, sl, bp, d


def slices(prob):
    pm = np.asarray(prob.products.market_ids).ravel()
    am = np.asarray(prob.agents.market_ids).ravel()
    ps = dict(zip(*np.unique(pm, return_index=True)))
    as_ = dict(zip(*np.unique(am, return_index=True)))
    pc = dict(zip(*np.unique(pm, return_counts=True)))
    ac = dict(zip(*np.unique(am, return_counts=True)))
    return [(m, slice(ps[m], ps[m] + pc[m]), slice(as_[m], as_[m] + ac[m])) for m in ps]


def _psolve(L, rhs, ridge=1e-10):
    """Solve, falling back to a ridge and then least squares (as in f92): the system
    loses rank when a candidate price drives a share to zero."""
    try:
        return np.linalg.solve(L, rhs)
    except np.linalg.LinAlgError:
        pass
    sc = float(np.trace(L)) / max(len(L), 1)
    try:
        return np.linalg.solve(L + ridge * sc * np.eye(len(L)), rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(L, rhs, rcond=None)[0]


def shares_and_D(p, base, bp, e0, lz, A, w):
    u = (base + bp * p)[:, None] + np.outer(-p, e0) + lz
    mx = np.maximum(0.0, u.max(0))
    ex = np.exp(u - mx)
    P = ex / (np.exp(-mx) + ex.sum(0))
    PA = P * A
    return P @ w, np.diag(PA.sum(1)) - PA @ P.T


def run(col="iv"):
    import pyhdfe
    t0 = time.time()
    prob, delta, sp, sl, bp, d = setup(col)
    SL = slices(prob)
    nodes, w = np.asarray(prob.agents.nodes), np.asarray(prob.agents.weights).ravel()
    p0 = d.prices.to_numpy(float)
    lnz = d.log_n_eq.to_numpy(float)
    fid = pd.factorize(d.brand_id.map(FIRM))[0]
    nnn = (d[f71.COLS[col][0]] == "NNN").to_numpy()
    # MW's prodfe: brand x container x pack size.  Pack size is ABSORBED.
    prodfe = pd.factorize(d.brand_id + "|" + d.is_can.astype(int).astype(str)
                          + "|" + d.n_cans.astype(str) + "|"
                          + d.store_code_uc.astype(str))[0]
    ym = pd.get_dummies(d.year_month, drop_first=True).to_numpy(float)
    print(f"[{time.time() - t0:.0f}s] setup done; {len(d):,} rows, {len(SL):,} markets")

    rows, panel = [], []
    for rc in RC_GRID:
        bmk = np.full(len(d), np.nan)
        rmk = np.full(len(d), np.nan)
        for m, pj, ai in SL:
            nu = nodes[ai]
            e0 = np.exp(PI + sp * nu[:, 0])
            A = w[ai] * (bp - e0)
            s, D = shares_and_D(p0[pj], delta[pj] - bp * p0[pj], bp, e0,
                                np.outer(lnz[pj], sl * nu[:, 1]), A, w[ai])
            own = (fid[pj][:, None] == fid[pj][None, :]).astype(float)
            bmk[pj] = np.linalg.solve(-own * D, s)
            rmk[pj] = rc * np.linalg.solve(-D, s)
        mc = p0 - bmk - rmk

        alg = pyhdfe.create(prodfe[nnn][:, None], drop_singletons=False)
        Z = alg.residualize(np.column_stack([mc[nnn], ym[nnn]]))
        g = np.linalg.lstsq(Z[:, 1:], Z[:, 0], rcond=None)[0]
        r_ = mc - ym @ g
        fe = pd.Series(np.where(nnn, r_, np.nan)).groupby(prodfe).transform("mean")
        mchat = fe.fillna(pd.Series(r_[nnn]).mean()).to_numpy() + ym @ g

        pstar = np.full(len(d), np.nan)
        pmax, nfail = p0.max(), 0
        for m, pj, ai in SL:
            nu = nodes[ai]
            e0 = np.exp(PI + sp * nu[:, 0])
            A = w[ai] * (bp - e0)
            lz = np.outer(lnz[pj], sl * nu[:, 1])
            base = delta[pj] - bp * p0[pj]
            own = (fid[pj][:, None] == fid[pj][None, :]).astype(float)
            # undamped first (converges in almost every market); if it does not,
            # restart damped, with prices kept in range as in f92
            for damp, maxit in ((1.0, 300), (0.5, 1000)):
                p, ok = p0[pj].copy(), False
                for _ in range(maxit):
                    s, D = shares_and_D(p, base, bp, e0, lz, A, w[ai])
                    pn = mchat[pj] + _psolve(-own * D, s) + rc * _psolve(-D, s)
                    pn = np.clip(pn, 0.05, 5.0 * pmax)
                    if np.max(np.abs(pn - p)) < 1e-7:
                        p, ok = pn, True
                        break
                    p = (1 - damp) * p + damp * pn
                if ok:
                    break
            pstar[pj] = p if ok else np.nan
            nfail += not ok

        gap = p0 - pstar
        sale = (d[f71.COLS[col][0]].str[1] == "S").to_numpy()
        panel.append(pd.DataFrame(dict(
            rc=rc, firm=d.brand_id.map(FIRM).to_numpy(), brand=d.brand_id.to_numpy(),
            n_cans=d.n_cans.to_numpy(), reg3=d[f71.COLS[col][0]].to_numpy(),
            nnn=nnn, sale=sale, price=p0, bmk=bmk, rmk=rmk, mc=mc,
            mchat=mchat, pstar=pstar, gap=gap,
            store_code_uc=d.store_code_uc.to_numpy())))
        if rc == 0.0:                       # the by-chain table, at the baseline
            CH = pd.read_parquet(ROOT / "data_work" / "f91_store_chain.parquet")
            ch = dict(zip(CH.store_code_uc, CH.parent))
            b = pd.DataFrame(dict(parent=pd.Series(d.store_code_uc.to_numpy()).map(ch),
                                  store=d.store_code_uc.to_numpy(), price=p0, mc=mc,
                                  mkup=100 * bmk / p0, gap=gap, sale=sale))
            g = b.groupby("parent")
            by = pd.DataFrame(dict(stores=g.store.nunique(), rows=g.size(),
                                   price=g.price.mean(), mc=g.mc.mean(),
                                   mkup=g.mkup.mean(),
                                   gap_sale=b[b.sale].groupby("parent").gap.median()))
            by.sort_values("rows", ascending=False).to_csv(
                RES / f"f90_by_chain_{col}_{TAG}.csv")
            print(f"  wrote f90_by_chain_{col}_{TAG}.csv ({len(by)} chains)")
        rows.append(dict(
            rc=rc, b_mkup=100 * np.median(bmk[nnn] / p0[nnn]),
            r_mkup=100 * np.median(rmk[nnn] / p0[nnn]),
            mc=np.median(mc[nnn]), mc_neg=100 * (mc < 0).mean(),
            gap_nosale=np.nanmedian(gap[~sale]), gap_sale=np.nanmedian(gap[sale]),
            gap_pct=100 * np.nanmedian(gap[sale] / pstar[sale]), nfail=nfail))
        print(f"  rc={rc:<6} brewer mkup {rows[-1]['b_mkup']:5.1f}%  "
              f"retail mkup {rows[-1]['r_mkup']:5.1f}%  mc {rows[-1]['mc']:6.3f}  "
              f"mc<0 {rows[-1]['mc_neg']:4.2f}%  gap: no-sale "
              f"{rows[-1]['gap_nosale']:+.3f}  sale {rows[-1]['gap_sale']:+.3f} "
              f"({rows[-1]['gap_pct']:+.1f}%)  no fixed point {nfail}/{len(SL)} markets  "
              f"[{time.time() - t0:.0f}s]")
    pd.DataFrame(rows).to_csv(RES / f"f90_supply_rcgrid_{col}_{TAG}.csv", index=False)
    pd.concat(panel, ignore_index=True).to_parquet(
        RES / f"f90_supply_panel_{col}_{TAG}.parquet", index=False)
    print(f"\nwrote {RES / f'f90_supply_rcgrid_{col}_{TAG}.csv'}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "iv")
