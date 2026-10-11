#!/usr/bin/env python3
"""First-stage instrument strength for the RCNL linear form, per endogenous variable.
Linear IV only: no nonlinear GMM, no random coefficients, no PyBLP.

    log(s_jrt) - log(s_0rt) = alpha*p_jrt + rho*log(s_jrt|g) + FE_j + FE_t + xi_jrt

`linearmodels` is not installed, so the estimators and diagnostics are implemented
here directly. Each formula is stated where it is used.

    python3 paper/src/diagnostics/blp_first_stage_strength.py
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
import _paths
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve()
PAPER = _paths.WORK
sys.path.insert(0, str(PAPER / "src" / "common"))
from run_manifest import RunManifest        # noqa: E402

PANELS = PAPER / "outputs" / "blp_panel"
OUT = PAPER / "outputs" / "blp_first_stage"
DIFF = ["d_same_brand_other_rung", "d_same_rung_other_brand",
        "d_same_container_other", "d_other_container"]
COST = ["z_can", "z_bottle", "z_malt"]


# ------------------------------------------------------------------ absorption
def absorb(X, g1, g2, iters=400, tol=1e-10):
    """Two-way within transformation by alternating projections (alternative FE and
    time FE). X is (n, k); g1, g2 are integer group codes."""
    X = np.array(X, dtype=np.float64, copy=True)
    n1, n2 = g1.max() + 1, g2.max() + 1
    c1 = np.bincount(g1, minlength=n1).astype(float)
    c2 = np.bincount(g2, minlength=n2).astype(float)
    c1[c1 == 0] = 1; c2[c2 == 0] = 1
    for _ in range(iters):
        mx = 0.0
        for g, c, n in ((g1, c1, n1), (g2, c2, n2)):
            m = np.zeros((n, X.shape[1]))
            for k in range(X.shape[1]):
                m[:, k] = np.bincount(g, weights=X[:, k], minlength=n) / c
            X -= m[g]
            mx = max(mx, float(np.abs(m).max()))
        if mx < tol:
            break
    return X


# ------------------------------------------------------------------ estimators
def cluster_vcov(X, u, cl, bread):
    """Cluster-robust sandwich. `bread` is (X'X)^-1 or the 2SLS analogue."""
    k = X.shape[1]
    meat = np.zeros((k, k))
    order = np.argsort(cl, kind="stable")
    cs = cl[order]; Xs = X[order]; us = u[order]
    bounds = np.flatnonzero(np.diff(cs)) + 1
    for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(cs)]):
        s = Xs[a:b].T @ us[a:b]
        meat += np.outer(s, s)
    G = len(bounds) + 1
    n = len(cl)
    dfc = (G / (G - 1)) * ((n - 1) / max(n - k, 1))
    return bread @ meat @ bread * dfc, G


def ols(y, X, cl):
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    u = y - X @ b
    V, G = cluster_vcov(X, u, cl, XtX_inv)
    return b, V, u, G


def tsls(y, X, Z, cl):
    """X endogenous+included, Z instruments (>= as many columns as X)."""
    ZtZ_inv = np.linalg.inv(Z.T @ Z)
    Pz_X = Z @ (ZtZ_inv @ (Z.T @ X))
    A = np.linalg.inv(Pz_X.T @ X)
    b = A @ (Pz_X.T @ y)
    u = y - X @ b
    V, G = cluster_vcov(Pz_X, u, cl, A)
    return b, V, u, G


def wald(b, V, idx):
    R = np.zeros((len(idx), len(b)))
    for i, j in enumerate(idx):
        R[i, j] = 1.0
    Rb = R @ b
    return float(Rb @ np.linalg.solve(R @ V @ R.T, Rb))


def hansen_j(y, X, Z, cl):
    """Two-step GMM J with a cluster-robust weight matrix."""
    b0, _, u0, G = tsls(y, X, Z, cl)
    order = np.argsort(cl, kind="stable")
    Zs, us, cs = Z[order], u0[order], cl[order]
    bounds = np.flatnonzero(np.diff(cs)) + 1
    S = np.zeros((Z.shape[1], Z.shape[1]))
    for a, b_ in zip(np.r_[0, bounds], np.r_[bounds, len(cs)]):
        g = Zs[a:b_].T @ us[a:b_]
        S += np.outer(g, g)
    S /= len(y)
    Si = np.linalg.pinv(S)
    A = np.linalg.inv(X.T @ Z @ Si @ Z.T @ X)
    b = A @ (X.T @ Z @ Si @ Z.T @ y)
    u = y - X @ b
    g = Z.T @ u / len(y)
    J = float(len(y) * g @ Si @ g)
    df = Z.shape[1] - X.shape[1]
    return J, df, float(stats.chi2.sf(J, df)) if df > 0 else np.nan, b


def sw_stats(xj, Xother, Z, cl):
    """Sanderson-Windmeijer conditional first-stage statistic for one endogenous
    variable, following Sanderson and Windmeijer (2016, J. Econometrics 190(2)).

    Purge the other endogenous variables from x_j by 2SLS using Z, then test that
    all L instrument coefficients are zero in the regression of the purged variable
    on Z. Numerator df = L - (K - 1). With K = 1 this reduces to the ordinary
    first-stage F on L instruments, which is asserted in the caller as a check."""
    L = Z.shape[1]
    K = 1 + (Xother.shape[1] if Xother is not None and Xother.size else 0)
    if K == 1:
        xperp = xj
    else:
        d, _, _, _ = tsls(xj, Xother, Z, cl)
        xperp = xj - Xother @ d
    b, V, u, G = ols(xperp, Z, cl)
    W = wald(b, V, list(range(L)))
    df = L - (K - 1)
    if df <= 0:
        # SW needs at least as many instruments as endogenous variables; with one
        # instrument and two endogenous regressors the statistic does not exist.
        return dict(sw_chi2=W, sw_df=df, sw_F=np.nan, sw_p=np.nan)
    return dict(sw_chi2=W, sw_df=df, sw_F=W / df,
                sw_p=float(stats.chi2.sf(W, df)))


def cragg_donald(X, Z):
    """Classical minimum-eigenvalue statistic. Both already FE-absorbed."""
    n, L = Z.shape
    K = X.shape[1]
    # X'P_Z X without forming the n x n projection matrix
    ZtX = Z.T @ X
    XPX = ZtX.T @ np.linalg.solve(Z.T @ Z, ZtX)
    XMX = X.T @ X - XPX
    Sig = XMX / (n - L)
    ev, Q = np.linalg.eigh(Sig)
    Sig_ih = Q @ np.diag(1.0 / np.sqrt(np.clip(ev, 1e-300, None))) @ Q.T
    M = Sig_ih @ XPX @ Sig_ih / L
    return float(np.linalg.eigvalsh(M).min())


def kp_rk_wald(X, Z, cl, robust=True):
    """Kleibergen-Paap (2006) rk Wald statistic for H0: rank(Pi) = K-1, and the
    corresponding F = rk / L. Pi is the (K x L) reduced-form coefficient matrix.

    Built from the SVD of the whitened Pi-hat, with a cluster-robust covariance of
    vec(Pi-hat) when robust=True. With the classical covariance the statistic
    coincides with Cragg-Donald, which the caller checks."""
    n, L = Z.shape
    K = X.shape[1]
    ZtZ_inv = np.linalg.inv(Z.T @ Z)
    Pi = (ZtZ_inv @ (Z.T @ X)).T                       # K x L
    U = X - Z @ Pi.T                                    # reduced-form residuals
    if robust:
        order = np.argsort(cl, kind="stable")
        Zs, Us, cs = Z[order], U[order], cl[order]
        bounds = np.flatnonzero(np.diff(cs)) + 1
        meat = np.zeros((K * L, K * L))
        for a, b_ in zip(np.r_[0, bounds], np.r_[bounds, len(cs)]):
            s = np.kron(np.eye(K), Zs[a:b_].T) @ Us[a:b_].T.reshape(-1)
            meat += np.outer(s, s)
        B = np.kron(np.eye(K), ZtZ_inv)
        V = B @ meat @ B.T                              # cov of vec(Pi') stacked by row
    else:
        Sig = (U.T @ U) / (n - L)
        V = np.kron(Sig, ZtZ_inv)
    # whiten so the SVD is scale free
    Th_r = np.linalg.cholesky(np.cov(U.T) + 1e-18 * np.eye(K)) if K > 1 else \
        np.array([[U.std()]])
    Th_c = np.linalg.cholesky(Z.T @ Z / n)
    Pw = np.linalg.solve(Th_r, Pi) @ Th_c               # K x L whitened
    Uu, s, Vt = np.linalg.svd(Pw, full_matrices=True)
    r = K - 1
    A_perp = Uu[:, r:]                                  # K x (K-r)
    B_perp = Vt[r:, :].T                                # L x (K-r)
    lam = A_perp.T @ Pw @ B_perp                        # (K-r) x (K-r)
    # delta method back to the untransformed vec(Pi)
    Aq = np.linalg.solve(Th_r.T, A_perp)
    Bq = Th_c @ B_perp
    Kron = np.kron(Aq.T, Bq.T)                          # maps vec(Pi') -> vec(lam)
    Om = Kron @ V @ Kron.T
    v = lam.reshape(-1)
    rk = float(v @ np.linalg.solve(Om, v))
    df = (K - r) * (L - r)
    return dict(kp_rk=rk, kp_df=df, kp_F=rk / L, kp_p=float(stats.chi2.sf(rk, df)))


def cond_number(Z):
    s = Z / np.where(Z.std(0) > 0, Z.std(0), 1.0)
    sv = np.linalg.svd(s, compute_uv=False)
    return float(sv.max() / sv.min())


# ------------------------------------------------------------------ driver
def build(mkt: str):
    d = pd.read_parquet(PANELS / f"market_panel_{mkt}.parquet")
    # integer keys throughout: 2.6M string concatenations are the memory bottleneck
    d["alt"] = pd.factorize(pd.MultiIndex.from_arrays(
        [d.brand, d.container, d.n_cans]))[0].astype(np.int32)
    d["per"] = pd.factorize(d.period)[0].astype(np.int32)
    d["dmac"] = pd.factorize(d.dma)[0].astype(np.int32)
    d["mkt"] = (d.dmac.astype(np.int64) * (d.per.max() + 1) + d.per).astype(np.int64)
    for c in ("brand", "container", "period", "dma"):
        d[c] = d[c].astype("category")
    tot = d.groupby("mkt", observed=True)["packs"].transform("sum")
    d["s_jg"] = d.packs / tot                                    # within-nest share
    # Hausman: leave-one-out mean price of the same alternative in other DMAs, same period
    ap = (d.alt.astype(np.int64) * (d.per.max() + 1) + d.per)
    g = d.groupby(ap, observed=True)["price"]
    d["z_haus"] = (g.transform("sum") - d.price) / (g.transform("count") - 1)
    # outside share from the FULL market, before any instrument-availability filter:
    # a market's composition must not depend on whether an instrument exists
    for col in ("share_inside", "share_inside_year"):
        d["s0_" + col] = 1.0 - d.groupby("mkt", observed=True)[col].transform("sum")
    n0 = len(d)
    d = d.loc[np.isfinite(d.z_haus)].copy()
    print(f"  {mkt}: {len(d):,} of {n0:,} rows keep a Hausman instrument "
          f"({100 * len(d) / n0:.2f}%)", flush=True)
    return d


def run_cell(d, mkt, variant, out, rows_fs, rows_sys, rows_p3):
    col = "share_inside" if variant == "full" else "share_inside_year"
    s, s0 = d[col], d["s0_" + col]
    ok = (s > 0) & (s0 > 0) & np.isfinite(d.price) & (d.s_jg > 0)
    e = d.loc[ok].copy()
    y = (np.log(e[col]) - np.log(e["s0_" + col])).to_numpy()
    ga = pd.factorize(e.alt)[0]
    gt = pd.factorize(e.per)[0]
    cl = pd.factorize(e.dmac)[0]
    znames = DIFF + ["z_haus"] + COST
    Xe = e[["price"]].to_numpy(float)
    Xn = np.log(e[["s_jg"]].to_numpy(float))
    Z = e[znames].to_numpy(float)
    M = absorb(np.column_stack([y, Xe, Xn, Z]), ga, gt)
    yd, pd_, sd_, Zd = M[:, 0], M[:, [1]], M[:, [2]], M[:, 3:]
    n, L = Zd.shape
    print(f"  {mkt} / {variant}: n={n:,} instruments={L} clusters={len(np.unique(cl))}",
          flush=True)

    # ---- first stages, one per endogenous variable
    for nm, xj, other in (("price", pd_, sd_), ("log_s_jg", sd_, pd_)):
        b, V, u, G = ols(xj[:, 0], Zd, cl)
        se = np.sqrt(np.diag(V))
        F = wald(b, V, list(range(L))) / L
        r2 = 1 - u.var() / xj[:, 0].var()
        sw = sw_stats(xj[:, 0], other, Zd, cl)
        for i, zn in enumerate(znames):
            rows_fs.append(dict(market=mkt, variant=variant, endog=nm, instrument=zn,
                                coef=b[i], se=se[i], t=b[i] / se[i],
                                p=float(2 * stats.norm.sf(abs(b[i] / se[i])))))
        rows_sys.append(dict(market=mkt, variant=variant, endog=nm, n=n, clusters=G,
                             joint_F=F, partial_R2=r2, **sw))
    cd = cragg_donald(np.column_stack([pd_, sd_]), Zd)
    kp = kp_rk_wald(np.column_stack([pd_, sd_]), Zd, cl, robust=True)
    kpc = kp_rk_wald(np.column_stack([pd_, sd_]), Zd, cl, robust=False)
    rows_sys.append(dict(market=mkt, variant=variant, endog="SYSTEM", n=n,
                         cragg_donald=cd, **kp,
                         kp_classical_check=kpc["kp_F"], cond_number=cond_number(Zd)))

    # ---- Part 3: rho = 0 and rho free
    Xf = np.column_stack([pd_, sd_])
    b_iv, V_iv, u_iv, _ = tsls(yd, Xf, Zd, cl)
    b_ols, V_ols, _, _ = ols(yd, Xf, cl)
    b0_iv, V0_iv, _, _ = tsls(yd, pd_, Zd, cl)
    b0_ols, V0_ols, _, _ = ols(yd, pd_, cl)
    J, Jdf, Jp, _ = hansen_j(yd, pd_, Zd, cl)
    sj = e[col].to_numpy(float); sjg = e.s_jg.to_numpy(float)
    p = e.price.to_numpy(float)
    el_logit_iv = b0_iv[0] * p * (1 - sj)
    el_logit_ols = b0_ols[0] * p * (1 - sj)
    rho = b_iv[1]
    el_nl = b_iv[0] * p * (1 / (1 - rho) - rho / (1 - rho) * sjg - sj)
    def q(x):
        return dict(mean=float(np.mean(x)), p10=float(np.percentile(x, 10)),
                    p50=float(np.median(x)), p90=float(np.percentile(x, 90)))
    rows_p3.append(dict(market=mkt, variant=variant, model="logit rho=0, OLS",
                        alpha=b0_ols[0], alpha_se=float(np.sqrt(V0_ols[0, 0])),
                        rho=np.nan, rho_se=np.nan, **{f"elast_{k}": v for k, v in
                                                      q(el_logit_ols).items()}))
    rows_p3.append(dict(market=mkt, variant=variant, model="logit rho=0, IV",
                        alpha=b0_iv[0], alpha_se=float(np.sqrt(V0_iv[0, 0])),
                        rho=np.nan, rho_se=np.nan, hansen_J=J, hansen_df=Jdf, hansen_p=Jp,
                        **{f"elast_{k}": v for k, v in q(el_logit_iv).items()}))
    rows_p3.append(dict(market=mkt, variant=variant, model="nested logit, IV",
                        alpha=b_iv[0], alpha_se=float(np.sqrt(V_iv[0, 0])),
                        rho=rho, rho_se=float(np.sqrt(V_iv[1, 1])),
                        **{f"elast_{k}": v for k, v in q(el_nl).items()}))
    rows_p3.append(dict(market=mkt, variant=variant, model="nested logit, OLS",
                        alpha=b_ols[0], alpha_se=float(np.sqrt(V_ols[0, 0])),
                        rho=b_ols[1], rho_se=float(np.sqrt(V_ols[1, 1])),
                        **{f"elast_{k}": v for k, v in q(
                            b_ols[0] * p * (1 / (1 - b_ols[1])
                                            - b_ols[1] / (1 - b_ols[1]) * sjg - sj)).items()}))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    with RunManifest(out, args=vars(a),
                     inputs=[PANELS / "market_panel_dma_week.parquet",
                             PANELS / "market_panel_dma_month.parquet"],
                     tag="first-stage instrument strength (round 13)") as rm:
        rm.add_note("Linear IV only. Product FE at brand-container-rung and time FE absorbed by "
                    "alternating projections; SEs clustered on DMA. linearmodels is not installed, "
                    "so SW, Cragg-Donald and Kleibergen-Paap are implemented in this file.")
        fs, sysr, p3 = [], [], []
        for mkt in ("dma_week", "dma_month"):
            d = build(mkt)
            for variant in ("full", "year"):
                run_cell(d, mkt, variant, out, fs, sysr, p3)
        pd.DataFrame(fs).to_csv(out / "first_stage_coefficients.csv", index=False)
        pd.DataFrame(sysr).to_csv(out / "identification_statistics.csv", index=False)
        pd.DataFrame(p3).to_csv(out / "part3_linear_models.csv", index=False)
        pd.set_option("display.width", 260)
        print("\n--- identification statistics ---")
        print(pd.DataFrame(sysr).to_string(index=False))
        print("\n--- first-stage coefficients ---")
        print(pd.DataFrame(fs).to_string(index=False))
        print("\n--- Part 3 ---")
        print(pd.DataFrame(p3).to_string(index=False))
        rm.add_outputs(sorted(out.glob("*.csv")))
    return 0


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    sys.exit(main())
