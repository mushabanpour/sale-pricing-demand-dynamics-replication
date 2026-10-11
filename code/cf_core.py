"""Supply and counterfactual machinery for one demand state, run store by store in parallel.

boot_draw.py calls it once per parameter draw.  Every function reads the demand state from
the module dict S (mean utilities, regime shifts, nonlinear parameters, costs), set by
set_state() before the worker pool is forked, so the point estimates and each bootstrap
draw run through the same code.  The algebra is that of the article's pipeline:

    contract()    mean utilities that reproduce observed shares (the BLP contraction)
    markups()     brewer markups at observed prices, rc = 0            f90_supply_mw.py
    fit_cost()    store-product and year-month cost fit on NNN weeks   f90_supply_mw.py
    pstar()       static Nash prices in every week                     f90_supply_mw.py
    regimes()     MIX, VD, IT, NPD and the third-degree benchmarks     f92_counterfactual.py
    panel_a()     ITS removed at observed prices                       f98_sales_value.py
    panel_b()     one firm removes ITS, rivals re-optimize             f99_solo_rivals.py

Third-degree benchmarks (Hendel and Nevo 2013): consumers are split into K segments by a
taste draw, and each segment is charged its own prices.  Suffix _p segments on the price
draw nu_1, _n on the pack-size draw nu_2.
    NPD3   each segment its own NPD prices: one price per serving per brand and container,
           constant across weeks; no sales (regimes NNN), as in NPD
    MIX3   each segment its own MIX prices: free across pack sizes and weeks
"""
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
from scipy import stats

import f71_baseline_table as f71
import f92_counterfactual as f92
import f99_solo_rivals as f99
from f87_supply import FIRM

FIRMS = ("ABI", "MolsonCoors", "Constellation", "Heineken")
SEGS = {"p": 0, "n": 1}                 # segment suffix -> column of the agent draws
SCEN = ("MIX", "VD", "IT", "NPD") + tuple(f"{x}3_{s}" for s in SEGS for x in ("NPD", "MIX"))
CASES = {"observed": (None, None), "no_sales": ("all", "all"),
         "no_discount": ("all", None), "no_regimes": (None, "all"),
         **{f"solo_{f}": (f, f) for f in FIRMS}}                      # as f98
NQ = 5
NPROC = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
S = {}


# ---------------------------------------------------------------- frame and state
def frame(stores=0):
    """Column (iv)'s estimation frame in pyblp's row order, with the agent draws pyblp
    builds for it (f71.lognormal_problem): Halton, NDRAWS per market, seed 0."""
    import pyblp
    d, terms = f71.load("long", "iv")
    if stores:
        keep = sorted(d.store_code_uc.unique())[:stores]
        d = d[d.store_code_uc.isin(keep)].reset_index(drop=True)
    ids, nodes, w = pyblp.Integration("halton", size=f71.NDRAWS,
                                      specification_options={"seed": 0}) \
        ._build_many(2, np.unique(np.asarray(d.market_ids, dtype=object)))
    return d, terms, np.asarray(ids).ravel(), np.asarray(nodes), np.asarray(w).ravel()


def lam_from(beta, d):
    """Each row's own sale regime x pack-size shift, the part of delta that NNN removes."""
    reg = d[f71.COLS["iv"][0]].to_numpy()
    nc = d.n_cans.to_numpy()
    lam = np.zeros(len(d))
    for s_ in f71.COLS["iv"][1]:
        for k in f71.RUNGS:
            key = f"{s_}_{k}"
            if key in beta:
                lam[(reg == s_) & (nc == k)] = beta[key]
    return lam


def init(d, am, nodes, w, nseg=2):
    """The fixed part of the state: data arrays and the store -> market -> rows/agents map."""
    pm = d.market_ids.to_numpy()
    ast_ = dict(zip(*np.unique(am, return_index=True)))
    acnt = dict(zip(*np.unique(am, return_counts=True)))
    st = d.store_code_uc.to_numpy()
    stores = []
    bounds = np.flatnonzero(pm[1:] != pm[:-1]) + 1      # d is sorted by market
    starts = np.r_[0, bounds]
    ends = np.r_[bounds, len(d)]
    cur = None
    for a, b in zip(starts, ends):
        m = pm[a]
        if cur is None or st[a] != cur["store"]:
            cur = dict(store=st[a], mk=[])
            stores.append(cur)
        cur["mk"].append((m, np.arange(a, b), slice(ast_[m], ast_[m] + acnt[m])))
    stores.sort(key=lambda s: -sum(len(i) for _, i, _ in s["mk"]))   # big stores first
    reg = d[f71.COLS["iv"][0]].to_numpy()
    p0 = d.prices.to_numpy(float)
    sale = d.sale.to_numpy(float) == 1
    firm = d.brand_id.map(FIRM).to_numpy()
    S.update(
        stores=stores, p0=p0, pmax=float(p0.max()), lnz=d.log_n_eq.to_numpy(float),
        lnS=np.log(d.shares.to_numpy(float)), shares=d.shares.to_numpy(float),
        reg=reg, nnn=reg == "NNN", sale3=np.array([r[1] == "S" for r in reg]),
        sale=sale, preg=np.where(sale, d.p_reg.to_numpy(float), p0),
        firm=firm, fid=pd.factorize(firm)[0],
        bc=(d.brand_id.astype(str) + "|" + d.is_can.astype(int).astype(str)).to_numpy(),
        pack=d.n_cans.to_numpy(), nodes=nodes, w=w,
        decn=np.clip((stats.norm.cdf(nodes[:, 1]) * NQ).astype(np.int8), 0, NQ - 1),
        decp=np.clip((stats.norm.cdf(nodes[:, 0]) * NQ).astype(np.int8), 0, NQ - 1),
        seg={s: np.clip((stats.norm.cdf(nodes[:, c]) * nseg).astype(np.int8), 0, nseg - 1)
             for s, c in SEGS.items()},
        nseg=nseg,
        prodfe=pd.factorize(d.brand_id + "|" + d.is_can.astype(int).astype(str) + "|"
                            + d.n_cans.astype(str) + "|"
                            + d.store_code_uc.astype(str))[0],
        ym=pd.get_dummies(d.year_month, drop_first=True).to_numpy(float))
    assert np.all(S["preg"][sale] > p0[sale]), "a sale price at or above its reference"


def set_theta(sp, sl, pi_p):
    """The nonlinear parameters; alpha_i = -exp(pi_p + sigma_p nu_1i) (bp = 0)."""
    S.update(sp=float(sp), sl=float(sl), pi=float(pi_p), bp=0.0)
    f92.PI = float(pi_p)                    # f92.welfare / f99.best_response read it


def pmap(fn):
    """fn over every store, on NPROC forked workers that inherit S."""
    idx = range(len(S["stores"]))
    if NPROC <= 1:
        return [fn(k) for k in idx]
    with mp.get_context("fork").Pool(NPROC) as P:
        return P.map(fn, idx, chunksize=1)


def _mkt(m, idx, ai):
    """Per-market draws: e0 = exp(pi + sigma_p nu1), A = w alpha, the pack-size term."""
    nu = S["nodes"][ai]
    e0 = np.exp(S["pi"] + S["sp"] * nu[:, 0])
    w = S["w"][ai]
    lz = np.outer(S["lnz"][idx], S["sl"] * nu[:, 1])
    own = (S["fid"][idx][:, None] == S["fid"][idx][None, :]).astype(float)
    return nu, e0, w, -e0, w * -e0, lz, own


def _shares(delta, p, e0, lz, w):
    u = delta[:, None] - np.outer(p, e0) + lz
    mx = np.maximum(0.0, u.max(0))
    ex = np.exp(u - mx)
    P = ex / (np.exp(-mx) + ex.sum(0))
    return P, P @ w


# ---------------------------------------------------------------- demand
def _contract(k, tol=1e-12, maxit=20000):
    out, nfail = [], 0
    for m, idx, ai in S["stores"][k]["mk"]:
        nu, e0, w, _, _, lz, _ = _mkt(m, idx, ai)
        mu = -np.outer(S["p0"][idx], e0) + lz
        dl = S["delta_start"][idx].copy()
        lnS = S["lnS"][idx]
        ok = False
        for _ in range(maxit):
            u = dl[:, None] + mu
            mx = np.maximum(0.0, u.max(0))
            ex = np.exp(u - mx)
            s = (ex / (np.exp(-mx) + ex.sum(0))) @ w
            step = lnS - np.log(s)
            dl += step
            if np.max(np.abs(step)) < tol:
                ok = True
                break
        nfail += not ok
        out.append((idx, dl))
    return out, nfail


def contract():
    """Mean utilities that reproduce observed shares at the nonlinear parameters in S."""
    delta, nfail = np.empty(len(S["p0"])), 0
    for out, nf in pmap(_contract):
        nfail += nf
        for idx, dl in out:
            delta[idx] = dl
    return delta, nfail


# ---------------------------------------------------------------- supply (rc = 0)
def _markups(k):
    out = []
    for m, idx, ai in S["stores"][k]["mk"]:
        _, e0, w, _, A, lz, own = _mkt(m, idx, ai)
        P, s = _shares(S["delta"][idx], S["p0"][idx], e0, lz, w)
        PA = P * A
        D = np.diag(PA.sum(1)) - PA @ P.T
        out.append((idx, np.linalg.solve(-own * D, s)))
    return out


def markups():
    bmk = np.empty(len(S["p0"]))
    for out in pmap(_markups):
        for idx, b in out:
            bmk[idx] = b
    return bmk


def fit_cost(mc):
    """f90_supply_mw.py: store-product and year-month effects fitted on NNN weeks."""
    import pyhdfe
    nnn, prodfe, ym = S["nnn"], S["prodfe"], S["ym"]
    alg = pyhdfe.create(prodfe[nnn][:, None], drop_singletons=False)
    Z = alg.residualize(np.column_stack([mc[nnn], ym[nnn]]))
    g = np.linalg.lstsq(Z[:, 1:], Z[:, 0], rcond=None)[0]
    r_ = mc - ym @ g
    fe = pd.Series(np.where(nnn, r_, np.nan)).groupby(prodfe).transform("mean")
    return fe.fillna(pd.Series(r_[nnn]).mean()).to_numpy() + ym @ g


def _pstar(k):
    out, nfail = [], 0
    for m, idx, ai in S["stores"][k]["mk"]:
        _, e0, w, _, A, lz, own = _mkt(m, idx, ai)
        base, mc = S["delta"][idx] - S["bp"] * S["p0"][idx], S["mc"][idx]
        for damp, maxit in ((1.0, 300), (0.5, 1000)):       # as f90
            p, ok = S["p0"][idx].copy(), False
            for _ in range(maxit):
                P, s = _shares(base + S["bp"] * p, p, e0, lz, w)
                PA = P * A
                D = np.diag(PA.sum(1)) - PA @ P.T
                pn = np.clip(mc + f92._psolve(-own * D, s), 0.05, 5.0 * S["pmax"])
                if np.max(np.abs(pn - p)) < 1e-7:
                    p, ok = pn, True
                    break
                p = (1 - damp) * p + damp * pn
            if ok:
                break
        nfail += not ok
        out.append((idx, p if ok else np.full(len(idx), np.nan)))
    return out, nfail


def pstar():
    ps, nfail = np.full(len(S["p0"]), np.nan), 0
    for out, nf in pmap(_pstar):
        nfail += nf
        for idx, p in out:
            ps[idx] = p
    return ps, nfail


# ---------------------------------------------------------------- pricing regimes
def _blocks(k):
    bl = []
    for m, idx, ai in S["stores"][k]["mk"]:
        _, e0, w, alpha, A, lz, own = _mkt(m, idx, ai)
        bl.append(dict(idx=idx, e0=e0, wts=w, alpha=alpha, A=A, lz=lz, own=own,
                       decn=S["decn"][ai], decp=S["decp"][ai], **{f"seg_{s}": S["seg"][s][ai] for s in SEGS},
                       bc=S["bc"][idx], pack=S["pack"][idx]))
    return bl


def _cut(b, a):
    """Block b restricted to the agents in mask a (a consumer segment)."""
    return dict(b, e0=b["e0"][a], wts=b["wts"][a], alpha=b["alpha"][a], A=b["A"][a],
                lz=b["lz"][:, a], decn=b["decn"][a], decp=b["decp"][a])


def _store_fp(blocks, key):
    """f92's store-level fixed point for prices shared across weeks (VD, NPD), regimes NNN."""
    delta, lam, p0, mc, bp = S["delta"], S["lam"], S["p0"], S["mc"], S["bp"]
    kk = np.concatenate([b["bc"] if key == "NPD" else
                         np.char.add(np.char.add(b["bc"].astype(str), "|"), b["pack"].astype(str))
                         for b in blocks])
    g = pd.factorize(kk)[0]
    ng = g.max() + 1
    gl, off = [], 0
    for b in blocks:
        gl.append(g[off:off + len(b["idx"])])
        off += len(b["idx"])
    cnt = np.bincount(g, minlength=ng).astype(float)
    pg0 = np.bincount(g, weights=np.concatenate([p0[b["idx"]] for b in blocks]),
                      minlength=ng) / np.maximum(cnt, 1)
    p = [pg0[gl[i]] for i in range(len(blocks))]
    ok = False
    for _ in range(400):
        L = np.zeros((ng, ng)); rhs = np.zeros(ng)
        for bi, b in enumerate(blocks):
            i = b["idx"]
            base = delta[i] - bp * p0[i] - lam[i]
            P, s = _shares(base + bp * p[bi], p[bi], b["e0"], b["lz"], b["wts"])
            PA = P * b["A"]
            OD = b["own"] * (np.diag(PA.sum(1)) - PA @ P.T)
            M = np.zeros((len(i), ng)); M[np.arange(len(i)), gl[bi]] = 1
            L += M.T @ (-OD) @ M
            rhs += M.T @ (-OD) @ mc[i] + M.T @ s
        pg = np.clip(f92._psolve(L, rhs), 0.05, 5.0 * S["pmax"])
        gap = 0.0
        for bi in range(len(blocks)):
            pn = pg[gl[bi]]
            gap = max(gap, np.max(np.abs(pn - p[bi])))
            p[bi] = 0.5 * p[bi] + 0.5 * pn
        if gap < 1e-8:
            ok = True
            break
    return p, ok


def _acc(R, sc, b, p, base):
    """Add one block's CS, profit by firm and CS by pack-taste quintile to R[sc]."""
    i = b["idx"]
    cs, _, s, csi = f92.welfare(p, base, S["bp"], b["e0"], b["lz"], b["alpha"], b["wts"],
                                S["mc"][i])
    r = R.setdefault(sc, dict(cs=0.0, ps=np.zeros(4), csq=np.zeros(NQ), csqp=np.zeros(NQ), ok=[]))
    r["cs"] += cs
    np.add.at(r["ps"], S["fid"][i], (p - S["mc"][i]) * s)
    np.add.at(r["csq"], b["decn"], b["wts"] * csi)
    np.add.at(r["csqp"], b["decp"], b["wts"] * csi)
    if S.get("rows") and sc in SCEN[:4]:            # Table 11: prices and shares by row
        r.setdefault("rows", []).append((i, p, s))


def _regimes(k):
    delta, lam, p0, mc, bp = S["delta"], S["lam"], S["p0"], S["mc"], S["bp"]
    blocks = _blocks(k)
    R = {}
    for sc in ("MIX", "IT"):                                  # per week
        for b in blocks:
            i = b["idx"]
            base = delta[i] - bp * p0[i]
            g = pd.factorize(b["bc"])[0] if sc == "IT" else np.arange(len(i))
            p, _, ok = f92.solve(p0[i], base, mc[i], bp, b["e0"], b["lz"], b["A"],
                                 b["wts"], b["own"], g)
            _acc(R, sc, b, p, base)
            R[sc]["ok"].append(ok)
    for sc in ("VD", "NPD"):                                  # across weeks, NNN
        p, ok = _store_fp(blocks, sc)
        for bi, b in enumerate(blocks):
            i = b["idx"]
            _acc(R, sc, b, p[bi], delta[i] - bp * p0[i] - lam[i])
        R[sc]["ok"].append(ok)
    for sx in (SEGS if S.get("segs", True) else ()):          # third degree, by segment
        for q in range(S["nseg"]):
            seg = [_cut(b, b[f"seg_{sx}"] == q) for b in blocks]
            seg = [b for b in seg if len(b["wts"])]
            p, ok = _store_fp(seg, "NPD")
            for bi, b in enumerate(seg):
                i = b["idx"]
                _acc(R, f"NPD3_{sx}", b, p[bi], delta[i] - bp * p0[i] - lam[i])
            R[f"NPD3_{sx}"]["ok"].append(ok)
            for b in seg:
                i = b["idx"]
                base = delta[i] - bp * p0[i]
                p, _, ok = f92.solve(p0[i], base, mc[i], bp, b["e0"], b["lz"], b["A"],
                                     b["wts"], b["own"], np.arange(len(i)))
                _acc(R, f"MIX3_{sx}", b, p, base)
                R[f"MIX3_{sx}"]["ok"].append(ok)
    return R


def regimes():
    T = {sc: dict(cs=0.0, ps=np.zeros(4), csq=np.zeros(NQ), csqp=np.zeros(NQ), ok=[])
         for sc in SCEN if S.get("segs", True) or "3_" not in sc}
    for R in pmap(_regimes):
        for sc, r in R.items():
            T[sc]["cs"] += r["cs"]; T[sc]["ps"] += r["ps"]; T[sc]["csq"] += r["csq"]
            T[sc]["csqp"] += r["csqp"]
            T[sc]["ok"] += r["ok"]
            for i, p, s in r.get("rows", ()):
                T[sc].setdefault("P", np.full(len(S["p0"]), np.nan))[i] = p
                T[sc].setdefault("Q", np.full(len(S["p0"]), np.nan))[i] = s
    return T


# ---------------------------------------------------------------- observed prices
def _rows(spec, firm):
    if spec is None:
        return np.zeros(len(firm), bool)
    return np.ones(len(firm), bool) if spec == "all" else firm == spec


def _panel_a(k):
    delta, lam, p0, mc, bp = S["delta"], S["lam"], S["p0"], S["mc"], S["bp"]
    R = {c: dict(cs=0.0, ps=np.zeros(4), csq=np.zeros(NQ), csqp=np.zeros(NQ)) for c in CASES}
    fit = 0.0
    for m, idx, ai in S["stores"][k]["mk"]:
        _, e0, w, alpha, _, lz, _ = _mkt(m, idx, ai)
        fm, fi = S["firm"][idx], S["fid"][idx]
        base_obs = delta[idx] - bp * p0[idx]
        for c, (dr, lr) in CASES.items():
            p = np.where(_rows(dr, fm), S["preg"][idx], p0[idx])
            base = base_obs - np.where(_rows(lr, fm), lam[idx], 0.0)
            cs, _, s, csi = f92.welfare(p, base, bp, e0, lz, alpha, w, mc[idx])
            R[c]["cs"] += cs
            np.add.at(R[c]["ps"], fi, (p - mc[idx]) * s)
            np.add.at(R[c]["csq"], S["decn"][ai], w * csi)
            np.add.at(R[c]["csqp"], S["decp"][ai], w * csi)
            if c == "observed":
                fit = max(fit, float(np.max(np.abs(s - S["shares"][idx]))))
    return R, fit


def panel_a():
    T = {c: dict(cs=0.0, ps=np.zeros(4), csq=np.zeros(NQ), csqp=np.zeros(NQ)) for c in CASES}
    fit = 0.0
    for R, f in pmap(_panel_a):
        fit = max(fit, f)
        for c in CASES:
            for x in ("cs", "ps", "csq", "csqp"):
                T[c][x] = T[c][x] + R[c][x]
    return T, fit


def _panel_b(k):
    delta, lam, p0, mc, bp = S["delta"], S["lam"], S["p0"], S["mc"], S["bp"]
    CS = np.zeros((2, 4)); PS = np.zeros((2, 4, 4)); nfail = 0
    for m, idx, ai in S["stores"][k]["mk"]:
        _, e0, w, alpha, A, lz, own = _mkt(m, idx, ai)
        fm, sm, fi = S["firm"][idx], S["sale"][idx], S["fid"][idx]
        base_obs = delta[idx] - bp * p0[idx]
        for f in FIRMS:
            free = (fm != f) & ~sm
            for c in (0, 1):                                   # keep, remove
                mine = (fm == f) & (c == 1)
                p = np.where(mine, S["preg"][idx], p0[idx])
                base = base_obs - np.where(mine, lam[idx], 0.0)
                if free.any() and (fm == f).any():
                    p, ok = f99.best_response(p, base, mc[idx], bp, e0, lz, A, w, own, free)
                    nfail += not ok
                cs, _, s, _ = f92.welfare(p, base, bp, e0, lz, alpha, w, mc[idx])
                CS[c, FIRMS.index(f)] += cs
                np.add.at(PS[c, FIRMS.index(f)], fi, (p - mc[idx]) * s)
    return CS, PS, nfail


def panel_b():
    CS = np.zeros((2, 4)); PS = np.zeros((2, 4, 4)); nfail = 0
    for c, p, n in pmap(_panel_b):
        CS += c; PS += p; nfail += n
    return CS, PS, nfail


def firm_order():
    """Firm names in the order of S['fid'] (pd.factorize order)."""
    return list(pd.unique(S["firm"]))
