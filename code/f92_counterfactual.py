"""Section 6: welfare under four pricing regimes.

Second-degree price discrimination here runs on two dimensions, the pack-size ladder
and time.  Switching each off gives four scenarios:

    MIX   both            prices vary by pack size and by week   (observed)
    VD    volume only     prices vary by pack size, not by week
    IT    intertemporal   prices vary by week, not by pack size
    NPD   neither         one price per 144 ounces, all packs, all weeks

A constraint makes a set of products share a price.  With A the group-membership
matrix (p = A p_G), profit is sum_j (p_j - mc_j) s_j and the constrained first-order
condition for the group prices is

    A'[ s + (Omega .* D)(A p_G - mc) ] = 0
    =>  p_G = [A'(Omega .* D)A]^{-1} [ A'(Omega .* D) mc - A' s ],

iterated to a fixed point.  Demand is static, so D is block diagonal across weeks and
the store-level sums accumulate week by week; only the CONSTRAINT couples weeks, and
only for VD and NPD.

Welfare per unit of market size:
    CS = E_i[ log(1 + sum_j exp(delta_j + mu_ij)) / |alpha_i| ]
    PS = sum_j (p_j - mc_j) s_j

Under VD and NPD no week is on sale, so the regime is set to NNN throughout and the
estimated lambda shifts go with it; this assumes the non-price promotional effect
disappears along with the discount.  Under IT and MIX the observed regime is held
fixed rather than recomputed from the counterfactual price path, which would make the
problem dynamic.

    python3 code/f92_counterfactual.py iv --stores 8     # smoke test
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
import f71_baseline_table as f71                                   # noqa: E402
from f87_supply import FIRM                                        # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
E, RES = ROOT / "estimation", ROOT / "results"
TAG = "long"
SCEN = ("MIX", "VD", "IT", "NPD")


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
    # lambda shift carried by each row's OWN regime, so it can be removed for VD/NPD
    lam = np.zeros(len(d))
    for s_ in f71.COLS[col][1]:
        for k in f71.RUNGS:
            key = f"{s_}_{k}"
            if key in j["beta"]:
                lam[(d[f71.COLS[col][0]] == s_).to_numpy()
                    & (d.n_cans == k).to_numpy()] = j["beta"][key]
    return prob, delta, sp, sl, bp, d, lam


def _psolve(L, rhs, ridge=1e-10):
    """Solve, falling back to a ridge and then least squares.  The constrained system
    can lose rank when a candidate price drives a whole group's share to zero."""
    try:
        return np.linalg.solve(L, rhs)
    except np.linalg.LinAlgError:
        pass
    sc = float(np.trace(L)) / max(len(L), 1)
    try:
        return np.linalg.solve(L + ridge * sc * np.eye(len(L)), rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(L, rhs, rcond=None)[0]


def solve(pinit, base, mc, bp, e0, lz, A, wts, own, grp, maxit=400, tol=1e-8):
    """Fixed point in group prices.  grp maps each row to a group index; grp of -1
    would mean unconstrained, which is just one group per product."""
    ng = grp.max() + 1
    M = np.zeros((len(pinit), ng))
    M[np.arange(len(pinit)), grp] = 1.0
    # start ON the constraint: an unconstrained start makes the first step leave the
    # region where shares are bounded away from zero and D becomes singular
    cnt = M.sum(0)
    p = M @ ((M.T @ pinit) / np.where(cnt > 0, cnt, 1.0))
    for it in range(maxit):
        u = (base + bp * p)[:, None] + (-p)[:, None] * e0[None, :] + lz
        mx = np.maximum(0.0, u.max(0))
        ex = np.exp(u - mx)
        P = ex / (np.exp(-mx) + ex.sum(0))
        s = P @ wts
        PA = P * A
        D = np.diag(PA.sum(1)) - PA @ P.T
        OD = own * D
        L = M.T @ (-OD) @ M
        rhs = M.T @ (-OD) @ mc + M.T @ s
        pg = _psolve(L, rhs)
        pnew = np.clip(M @ pg, 0.05, 5.0 * pinit.max())
        if np.max(np.abs(pnew - p)) < tol:
            return pnew, it, True
        p = 0.5 * p + 0.5 * pnew
    return p, maxit, False


def welfare(p, base, bp, e0, lz, alpha, wts, mc):
    u = (base + bp * p)[:, None] + (-p)[:, None] * e0[None, :] + lz
    mx = np.maximum(0.0, u.max(0))
    ex = np.exp(u - mx)
    tot = np.exp(-mx) + ex.sum(0)
    P = ex / tot
    s = P @ wts
    cs = (mx + np.log(tot)) / np.abs(alpha)      # log-sum, undoing the shift
    return float(wts @ cs), float((p - mc) @ s), s, cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("col", nargs="?", default="iv")
    ap.add_argument("--stores", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    prob, delta, sp, sl, bp, d, lam = setup(a.col)
    nodes = np.asarray(prob.agents.nodes)
    w = np.asarray(prob.agents.weights).ravel()
    am = np.asarray(prob.agents.market_ids).ravel()
    pm = np.asarray(prob.products.market_ids).ravel()
    p0 = d.prices.to_numpy(float)
    lnz = d.log_n_eq.to_numpy(float)
    fid = pd.factorize(d.brand_id.map(FIRM))[0]
    mc = np.load(RES / f"f92_mc_{a.col}_{TAG}.npy") if (
        RES / f"f92_mc_{a.col}_{TAG}.npy").exists() else None
    if mc is None:
        pan = pd.read_parquet(RES / f"f90_supply_panel_{a.col}_{TAG}.parquet")
        mc = pan[pan.rc == 0.0].mchat.to_numpy()
        np.save(RES / f"f92_mc_{a.col}_{TAG}.npy", mc)
    ast_ = dict(zip(*np.unique(am, return_index=True)))
    acnt = dict(zip(*np.unique(am, return_counts=True)))
    d = d.assign(_i=np.arange(len(d)))
    stores = sorted(d.store_code_uc.unique())
    if a.stores:
        stores = stores[:a.stores]
    print(f"[{time.time()-t0:.0f}s] setup; {len(stores):,} stores, {len(d):,} rows")

    out = []
    DEC = {sc: np.zeros(10) for sc in SCEN}      # CS by decile of nu1 (price taste)
    NQ = 5
    DECN = {sc: np.zeros(NQ) for sc in SCEN}    # CS by QUINTILE of nu2 (pack taste)
    PQ = {sc: (np.full(len(d), np.nan), np.full(len(d), np.nan)) for sc in SCEN}
    for si, st in enumerate(stores):
        ds = d[d.store_code_uc == st]
        mkts = list(dict.fromkeys(ds.market_ids.tolist()))
        blocks = []
        for m in mkts:
            r_ = ds[ds.market_ids == m]
            idx = r_._i.to_numpy()
            ai = slice(ast_[m], ast_[m] + acnt[m])
            nu = nodes[ai]
            e0 = np.exp(PI + sp * nu[:, 0])
            blocks.append(dict(
                idx=idx, e0=e0, wts=w[ai], alpha=bp - e0,
                A=w[ai] * (bp - e0), lz=np.outer(lnz[idx], sl * nu[:, 1]),
                own=(fid[idx][:, None] == fid[idx][None, :]).astype(float),
                dec=np.clip((stats.norm.cdf(nu[:, 0]) * 10).astype(int), 0, 9),
                decn=np.clip((stats.norm.cdf(nu[:, 1]) * NQ).astype(int), 0, NQ - 1),
                bc=(d.brand_id.to_numpy()[idx] + "|"
                    + d.is_can.to_numpy()[idx].astype(int).astype(str)),
                pack=d.n_cans.to_numpy()[idx]))
        for sc in SCEN:
            drop = sc in ("VD", "NPD")
            res = {}
            if sc in ("MIX", "IT"):
                for bi, b in enumerate(blocks):
                    i = b["idx"]
                    base = delta[i] - bp * p0[i] - (lam[i] if drop else 0.0)
                    g = (pd.factorize(b["bc"])[0] if sc == "IT"
                         else np.arange(len(i)))
                    p, it, ok = solve(p0[i], base, mc[i], bp, b["e0"], b["lz"],
                                      b["A"], b["wts"], b["own"], g)
                    cs, ps, s, csi = welfare(p, base, bp, b["e0"], b["lz"],
                                             b["alpha"], b["wts"], mc[i])
                    np.add.at(DEC[sc], b["dec"], b["wts"] * csi)
                    np.add.at(DECN[sc], b["decn"], b["wts"] * csi)
                    PQ[sc][0][i], PQ[sc][1][i] = p, s
                    res.setdefault("cs", []).append(cs)
                    res.setdefault("ps", []).append(ps)
                    res.setdefault("ok", []).append(ok)
            else:
                # price constrained ACROSS weeks: one system for the whole store
                key = np.concatenate([b["bc"] if sc == "NPD"
                                      else np.char.add(b["bc"], "|"
                                                       + b["pack"].astype(str))
                                      for b in blocks])
                g = pd.factorize(key)[0]
                ng = g.max() + 1
                off, gl = 0, []
                for b in blocks:
                    gl.append(g[off:off + len(b["idx"])])
                    off += len(b["idx"])
                cnt = np.bincount(g, minlength=ng).astype(float)
                pg0 = np.bincount(g, weights=np.concatenate(
                    [p0[b["idx"]] for b in blocks]), minlength=ng) / np.maximum(cnt, 1)
                p = {i: pg0[gl[i]] for i in range(len(blocks))}
                for it in range(400):
                    L = np.zeros((ng, ng)); rhs = np.zeros(ng)
                    for bi, b in enumerate(blocks):
                        i = b["idx"]
                        base = delta[i] - bp * p0[i] - lam[i]
                        u = (base + bp * p[bi])[:, None] \
                            + (-p[bi])[:, None] * b["e0"][None, :] + b["lz"]
                        mx = np.maximum(0.0, u.max(0))
                        ex = np.exp(u - mx)
                        P = ex / (np.exp(-mx) + ex.sum(0))
                        s = P @ b["wts"]
                        PA = P * b["A"]
                        OD = b["own"] * (np.diag(PA.sum(1)) - PA @ P.T)
                        M = np.zeros((len(i), ng)); M[np.arange(len(i)), gl[bi]] = 1
                        L += M.T @ (-OD) @ M
                        rhs += M.T @ (-OD) @ mc[i] + M.T @ s
                    pg = _psolve(L, rhs)
                    pg = np.clip(pg, 0.05, 5.0 * p0.max())
                    gap = 0.0
                    for bi, b in enumerate(blocks):
                        pn = pg[gl[bi]]
                        gap = max(gap, np.max(np.abs(pn - p[bi])))
                        p[bi] = 0.5 * p[bi] + 0.5 * pn
                    if gap < 1e-8:
                        break
                for bi, b in enumerate(blocks):
                    i = b["idx"]
                    base = delta[i] - bp * p0[i] - lam[i]
                    cs, ps, s, csi = welfare(p[bi], base, bp, b["e0"], b["lz"],
                                             b["alpha"], b["wts"], mc[i])
                    np.add.at(DEC[sc], b["dec"], b["wts"] * csi)
                    np.add.at(DECN[sc], b["decn"], b["wts"] * csi)
                    PQ[sc][0][i], PQ[sc][1][i] = p[bi], s
                    res.setdefault("cs", []).append(cs)
                    res.setdefault("ps", []).append(ps)
                    res.setdefault("ok", []).append(True)
            out.append(dict(store=st, scen=sc, cs=float(np.sum(res["cs"])),
                            ps=float(np.sum(res["ps"])), nmkt=len(blocks),
                            conv=float(np.mean(res["ok"]))))
        if (si + 1) % 10 == 0 or si == len(stores) - 1:
            print(f"  {si+1}/{len(stores)} stores  [{time.time()-t0:.0f}s]")
    D = pd.DataFrame(out)
    D.to_csv(RES / f"f92_welfare_{a.col}_{TAG}"
             f"{'_smoke' if a.stores else ''}.csv", index=False)
    g = D.groupby("scen")[["cs", "ps"]].sum()
    g["total"] = g.cs + g.ps
    base = g.loc["MIX"]
    print("\nper unit of market size, summed over stores and weeks")
    print((g.assign(d_cs=g.cs - base.cs, d_ps=g.ps - base.ps,
                    d_tot=g.total - base.total)).round(3).to_string())
    print(f"\nconvergence: {D.conv.min():.3f} min   [{time.time()-t0:.0f}s]")
    keep = np.isfinite(PQ["MIX"][0])
    pq = pd.concat([pd.DataFrame(dict(
        market_ids=d.market_ids.to_numpy()[keep], alt=d.alt.to_numpy()[keep],
        store=d.store_code_uc.to_numpy()[keep], brand=d.brand_id.to_numpy()[keep],
        is_can=d.is_can.to_numpy()[keep], n_cans=d.n_cans.to_numpy()[keep],
        reg3=d[f71.COLS[a.col][0]].to_numpy()[keep],
        p_obs=p0[keep], scen=sc, price=PQ[sc][0][keep], share=PQ[sc][1][keep]))
        for sc in SCEN], ignore_index=True)
    pq.to_parquet(RES / f"f92_pq_{a.col}_{TAG}"
                  f"{'_smoke' if a.stores else ''}.parquet", index=False)
    print(f"\nproduct-level panel: {len(pq):,} rows -> f92_pq_{a.col}_{TAG}.parquet")
    print("\nmedian price and share by pack size")
    t = pq.pivot_table(index="n_cans", columns="scen",
                       values=["price", "share"], aggfunc="median")
    print(t.reindex(columns=list(SCEN), level=1).round(4).to_string())

    dec = pd.DataFrame(DEC, index=[f"d{i+1}" for i in range(10)])
    dec.to_csv(RES / f"f92_cs_decile_{a.col}_{TAG}"
               f"{'_smoke' if a.stores else ''}.csv")
    print("\nCS by decile of the price-sensitivity draw "
          "(d1 = least price sensitive), change vs MIX")
    print((dec[list(SCEN)].sub(dec.MIX, axis=0)).round(3).to_string())
    decn = pd.DataFrame(DECN, index=[f"q{i+1}" for i in range(NQ)])
    decn.to_csv(RES / f"f92_cs_quintile_pack_{a.col}_{TAG}"
                f"{'_smoke' if a.stores else ''}.csv")
    print("\nCS by QUINTILE of the pack-size taste draw "
          "(q1 = weakest preference for large packs), change vs MIX")
    print((decn[list(SCEN)].sub(decn.MIX, axis=0)).round(3).to_string())


if __name__ == "__main__":
    main()
