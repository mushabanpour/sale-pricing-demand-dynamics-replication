#!/usr/bin/env python3
"""The `rho` sweep: does the nest account for the elasticity gap against
Miller-Weinberg (2017)?

With `rho` imposed at `r`, the nested logit's linear form is

    log s_jrt - log s_0rt - r*log s_jrt|g = x_j*beta + alpha*p_jrt + FE + xi_jrt

so `log s_j|g` leaves the right-hand side and **price is the only endogenous
variable**. Three differentiation-class instruments then over-identify by two, so a
Hansen J exists at every `r` and is comparable across the sweep.

Elasticities use the exact nested-logit own-price formula, derived from
`d ln s_j / d delta_j` for u_ij = delta_j + zeta_ig + (1-rho)eps_ij:

    eps_jj = alpha * p_jrt * [ 1/(1-r) - r*s_jrt|g/(1-r) - s_jrt ]      (BRACKET)

and the diversion ratio to the outside good, from `ds_0/dp_j = -alpha*s_0*s_j`:

    D_j0  = s_0rt / [ 1/(1-r) - r*s_jrt|g/(1-r) - s_jrt ] = s_0 / BRACKET

At r = 0 the second collapses to `s_0/(1-s_j)`, round 19's formula. Both are exact,
not the `alpha*p/(1-r)` approximation.

Linear IV only. No nonlinear GMM, no random coefficients, no PyBLP.

    python3 paper/src/diagnostics/blp_rho_sweep.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import _paths
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve()
PAPER = _paths.WORK
sys.path.insert(0, str(PAPER / "src" / "common"))
sys.path.insert(0, str(HERE.parent))
from run_manifest import RunManifest                              # noqa: E402
import blp_first_stage_strength as M                              # noqa: E402
from blp_pre_rcnl_diagnostics import absorbN                      # noqa: E402
from blp_frequency_grid import (build_all, diesel_tables, frame,  # noqa: E402
                                CLS_NAMES, R3, MULT, PANEL, DERIVED)

OUT = PAPER / "outputs" / "blp_rho_sweep"
RHOS = [0.0, 0.3, 0.5, 0.6299, 0.7, 0.8299]
MW_RHO_NL, MW_RHO_RCNL = 0.6299, 0.8299
MW_BAND = (-6.10, -4.33)          # RCNL columns (ii)-(v), D042
MW_COL_I = -3.81                  # column (i)


def bracket(rho, s_jg, sj):
    """The exact nested-logit own-elasticity bracket. Positive under a well-behaved
    nest; if it goes non-positive the own elasticity flips sign and the outside
    diversion goes negative, which is the indefensibility condition."""
    return 1.0 / (1.0 - rho) - rho * s_jg / (1.0 - rho) - sj


def gmm_obj(y, X, Z, Si, n):
    """GMM objective at the optimum for a given weight matrix Si = S^-1.
    Returns (J, beta). Using one common Si across nested models makes the
    difference in J a distance-metric test."""
    A = np.linalg.inv(X.T @ Z @ Si @ Z.T @ X)
    b = A @ (X.T @ Z @ Si @ Z.T @ y)
    g = Z.T @ (y - X @ b) / n
    return float(n * g @ Si @ g), b


def cluster_S(Z, u, cl):
    order = np.argsort(cl, kind="stable")
    Zs, us, cs = Z[order], u[order], cl[order]
    bounds = np.flatnonzero(np.diff(cs)) + 1
    S = np.zeros((Z.shape[1], Z.shape[1]))
    for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(cs)]):
        g = Zs[a:b].T @ us[a:b]
        S += np.outer(g, g)
    return S / len(u)


def summarise(rho, alpha, prc, s_jg, sj, s0):
    """Elasticity, diversion and the two-channel decomposition, all exact."""
    br = bracket(rho, s_jg, sj)
    ap = alpha * prc
    el = ap * br
    dv = s0 / br
    bad = ~(br > 0)
    return dict(
        rho=rho,
        median_alpha_p=float(np.median(ap)),
        median_bracket=float(np.median(br)),
        amplification_1_over_1_minus_rho=1.0 / (1.0 - rho),
        elast_p50=float(np.median(el)),
        effective_amplification=float(np.median(el) / np.median(ap)),
        elast_p10=float(np.percentile(el, 10)),
        elast_p90=float(np.percentile(el, 90)),
        median_diversion_pct=float(100 * np.median(dv)),
        mean_outside_share=float(np.mean(s0)),
        pct_elast_positive=float(100 * np.mean(el > 0)),
        pct_bracket_nonpositive=float(100 * np.mean(bad)),
        pct_diversion_negative=float(100 * np.mean(dv < 0)),
        pct_diversion_gt_1=float(100 * np.mean(dv > 1)),
        indefensible=bool(np.mean(bad) > 0),
    )


def cell_run(name, e, groups, rows, dec, diag, free_rows, dm_rows,
             extra_ms=None):
    n = len(e)
    y0 = (np.log(e.sj) - np.log(e.s0)).to_numpy(float)
    lsjg = np.log(e.s_jg).to_numpy(float)
    prc = e.price.to_numpy(float)
    sjv = e.sj.to_numpy(float); s0v = e.s0.to_numpy(float)
    sjg = e.s_jg.to_numpy(float)
    cl = pd.factorize(e.dmac)[0]
    Aa = absorbN(np.column_stack([y0, lsjg, prc, e[R3].to_numpy(float)]), groups)
    yd, lg, p_, Z = Aa[:, 0], Aa[:, [1]], Aa[:, [2]], Aa[:, 3:]
    print(f"\n=== {name} === n={n:,}  clusters={len(np.unique(cl))}  "
          f"instruments={Z.shape[1]}", flush=True)

    # ---- free rho: two endogenous variables, over-identified by one
    Xf = np.column_stack([p_[:, 0], lg[:, 0]])
    bf, Vf, uf, Gf = M.tsls(yd, Xf, Z, cl)
    Sf = cluster_S(Z, uf, cl); Sfi = np.linalg.pinv(Sf)
    Jf, bf2 = gmm_obj(yd, Xf, Z, Sfi, n)
    swf = M.sw_stats(p_[:, 0], lg, Z, cl)
    free = dict(cell=name, n=n, clusters=Gf,
                alpha=float(bf[0]), se_alpha=float(np.sqrt(Vf[0, 0])),
                rho=float(bf[1]), se_rho=float(np.sqrt(Vf[1, 1])),
                sw_F_price=swf["sw_F"], J=Jf, J_df=Z.shape[1] - 2,
                J_p=float(stats.chi2.sf(Jf, Z.shape[1] - 2)),
                rho_admissible=bool(0.0 <= bf[1] < 1.0))
    free_rows.append(free)
    print(f"  free rho: alpha={bf[0]:+.4f} ({np.sqrt(Vf[0,0]):.4f})  "
          f"rho={bf[1]:+.4f} ({np.sqrt(Vf[1,1]):.4f})  admissible={free['rho_admissible']}  "
          f"J={Jf:.2f} df={Z.shape[1]-2} p={free['J_p']:.4f}", flush=True)

    # ---- the sweep
    for r in RHOS:
        yr = yd - r * lg[:, 0]
        b, V, u, G = M.tsls(yr, p_, Z, cl)
        bo, Vo, _, _ = M.ols(yr, p_, cl)
        sw = M.sw_stats(p_[:, 0], np.empty((n, 0)), Z, cl)
        J, Jdf, Jp, _ = M.hansen_j(yr, p_, Z, cl)
        s = summarise(r, b[0], prc, sjg, sjv, s0v)
        rows.append(dict(cell=name, **{k: s[k] for k in ("rho",)},
                         alpha_iv=float(b[0]), se_iv=float(np.sqrt(V[0, 0])),
                         t_iv=float(b[0] / np.sqrt(V[0, 0])),
                         alpha_ols=float(bo[0]), se_ols=float(np.sqrt(Vo[0, 0])),
                         sw_F_price=sw["sw_F"], hansen_J=J, hansen_df=Jdf,
                         hansen_p=Jp,
                         **{k: v for k, v in s.items() if k != "rho"}))
        # distance metric against the free-rho fit, common weight matrix
        Ji, _ = gmm_obj(yr, p_, Z, Sfi, n)
        D = Ji - Jf
        dm_rows.append(dict(cell=name, rho=r, J_free=Jf, J_imposed_commonS=Ji,
                            distance_metric=D,
                            p_chi2_1=float(stats.chi2.sf(max(D, 0.0), 1)),
                            rejects_imposed_rho=bool(D > stats.chi2.ppf(0.95, 1))))
        dec.append(dict(cell=name, rho=r,
                        median_alpha_p=s["median_alpha_p"],
                        median_bracket=s["median_bracket"],
                        nominal_amplification=s["amplification_1_over_1_minus_rho"],
                        effective_amplification=s["effective_amplification"],
                        elast_p50=s["elast_p50"]))
        diag.append(dict(cell=name, rho=r,
                         pct_elast_positive=s["pct_elast_positive"],
                         pct_bracket_nonpositive=s["pct_bracket_nonpositive"],
                         pct_diversion_negative=s["pct_diversion_negative"],
                         pct_diversion_gt_1=s["pct_diversion_gt_1"],
                         indefensible=s["indefensible"]))
        inb = MW_BAND[0] <= s["elast_p50"] <= MW_BAND[1]
        print(f"  rho={r:<7.4f} aIV={b[0]:+.4f} ({np.sqrt(V[0,0]):.4f}) "
              f"aOLS={bo[0]:+.4f}  ap={s['median_alpha_p']:+8.3f} x "
              f"amp={s['median_bracket']:6.3f} -> el50={s['elast_p50']:+9.3f}"
              f"{'  <-- in MW band' if inb else ''}  div={s['median_diversion_pct']:5.1f}% "
              f"F={sw['sw_F']:7.1f} J={J:6.2f} p={Jp:.3g} DM={D:7.2f}", flush=True)

    # ---- market-size sensitivity of the diversion, at MW's column (i) rho
    if extra_ms is not None:
        for lab, (sj2, s02) in extra_ms.items():
            r = MW_RHO_NL
            yr = yd - r * lg[:, 0]
            b, V, _, _ = M.tsls(yr, p_, Z, cl)
            s = summarise(r, b[0], prc, sjg, sj2, s02)
            diag.append(dict(cell=f"{name} [{lab}]", rho=r,
                             pct_elast_positive=s["pct_elast_positive"],
                             pct_bracket_nonpositive=s["pct_bracket_nonpositive"],
                             pct_diversion_negative=s["pct_diversion_negative"],
                             pct_diversion_gt_1=s["pct_diversion_gt_1"],
                             indefensible=s["indefensible"]))
            rows.append(dict(cell=f"{name} [{lab}]", rho=r,
                             alpha_iv=float(b[0]), se_iv=float(np.sqrt(V[0, 0])),
                             t_iv=float(b[0] / np.sqrt(V[0, 0])),
                             alpha_ols=np.nan, se_ols=np.nan,
                             sw_F_price=np.nan, hansen_J=np.nan, hansen_df=np.nan,
                             hansen_p=np.nan,
                             **{k: v for k, v in s.items() if k != "rho"}))
            print(f"  [{lab}] rho={r} s0_mean={s['mean_outside_share']:.3f} "
                  f"el50={s['elast_p50']:+9.3f} div={s['median_diversion_pct']:5.1f}%",
                  flush=True)
    return yd, lg, p_, Z, cl, prc, sjg, sjv, s0v


def part3(name, yd, lg, p_, Z, cl, prc, sjg, sjv, s0v, e, groups, sub,
          rlist=(0.0, MW_RHO_NL)):
    """Subset table over the three classes at rho = MW's column (i) value, plus
    z_dist_diesel, with one endogenous variable."""
    n = len(yd)
    zd = absorbN(e[["z_dist_diesel"]].to_numpy(float), groups)[:, [0]]
    for r in rlist:
        _part3_one(name, yd, lg, p_, Z, zd, cl, prc, sjg, sjv, s0v, sub, r, n)


def _part3_one(name, yd, lg, p_, Z, zd, cl, prc, sjg, sjv, s0v, sub, r, n):
    print(f"\n  -- rho = {r} --", flush=True)
    yr = yd - r * lg[:, 0]
    sets = [(f"all three classes", list(range(3))),
            ("d_same_brand_other_rung only", [0]),
            ("d_same_rung_other_brand only", [1]),
            ("d_same_container_other only", [2]),
            ("two near classes", [0, 1]),
            ("classes 2+3", [1, 2])]
    for lab, idx in sets:
        Zi = Z[:, idx]
        b, V, _, G = M.tsls(yr, p_, Zi, cl)
        sw = M.sw_stats(p_[:, 0], np.empty((n, 0)), Zi, cl)
        if Zi.shape[1] > 1:
            J, Jdf, Jp, _ = M.hansen_j(yr, p_, Zi, cl)
        else:
            J, Jdf, Jp = np.nan, np.nan, np.nan
        s = summarise(r, b[0], prc, sjg, sjv, s0v)
        sub.append(dict(cell=name, rho=r, instruments=lab, n_iv=Zi.shape[1],
                        alpha=float(b[0]), se=float(np.sqrt(V[0, 0])),
                        elast_p50=s["elast_p50"],
                        median_diversion_pct=s["median_diversion_pct"],
                        sw_F_price=sw["sw_F"], hansen_J=J, hansen_df=Jdf,
                        hansen_p=Jp))
        print(f"  {lab:32s} L={Zi.shape[1]} alpha={b[0]:+.4f} ({np.sqrt(V[0,0]):.4f}) "
              f"el50={s['elast_p50']:+9.3f} F={sw['sw_F']:8.1f} "
              f"J={J if J==J else float('nan'):6.2f} p={Jp if Jp==Jp else float('nan'):.3g}",
              flush=True)

    # z_dist_diesel with one endogenous variable
    for lab, Zi in [("z_dist_diesel alone", zd),
                    ("z_dist_diesel + three classes", np.column_stack([zd, Z]))]:
        b, V, _, _ = M.tsls(yr, p_, Zi, cl)
        sw = M.sw_stats(p_[:, 0], np.empty((n, 0)), Zi, cl)
        if Zi.shape[1] > 1:
            J, Jdf, Jp, _ = M.hansen_j(yr, p_, Zi, cl)
        else:
            J, Jdf, Jp = np.nan, np.nan, np.nan
        s = summarise(r, b[0], prc, sjg, sjv, s0v)
        sub.append(dict(cell=name, rho=r, instruments=lab, n_iv=Zi.shape[1],
                        alpha=float(b[0]), se=float(np.sqrt(V[0, 0])),
                        elast_p50=s["elast_p50"],
                        median_diversion_pct=s["median_diversion_pct"],
                        sw_F_price=sw["sw_F"], hansen_J=J, hansen_df=Jdf,
                        hansen_p=Jp))
        print(f"  {lab:32s} L={Zi.shape[1]} alpha={b[0]:+.4f} ({np.sqrt(V[0,0]):.4f}) "
              f"el50={s['elast_p50']:+9.3f} F={sw['sw_F']:8.1f}", flush=True)


def mw_implied_outside_share():
    """Invert MW's published column (i) triple to their implied median outside share.
    D_j0 = s_0/BRACKET and eps_jj = alpha*p*BRACKET, so with eps_jj = -3.81,
    D_j0 = 0.2980 and rho = 0.6299, s_0 = 0.2980 * BRACKET and BRACKET is pinned by
    their product count. Fixed-point on s_j = s_j|g * (1 - s_0)."""
    rho, div, el = MW_RHO_NL, 0.2980, MW_COL_I
    s_jg = 1.0 / 33.71          # 94,656 obs / 2,808 market-months, round 21 Part 2.5
    s0 = 0.5
    for _ in range(200):
        sj = s_jg * (1.0 - s0)
        br = bracket(rho, s_jg, sj)
        s0 = div * br
    ap = el / br
    return dict(assumed_median_s_jg=s_jg, implied_bracket=br,
                implied_median_outside_share=s0, implied_alpha_p=ap,
                implied_median_price_2010usd=ap / -0.1312)


def rho_matching_diversion(rho0, div0_pct, sjg, sjv, s0v, target_pct=29.80):
    """The exact median outside diversion is median(s_0/BRACKET). Solve for the r
    that puts it at MW's column (i) value. Reported as a mapping, not a preference."""
    from scipy import optimize
    f = lambda r: 100.0 * np.median(s0v / bracket(r, sjg, sjv)) - target_pct
    if f(0.0) < 0:
        return np.nan
    try:
        return float(optimize.brentq(f, 0.0, 0.999))
    except Exception:
        return np.nan


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with RunManifest(OUT, args={"rhos": str(RHOS)},
                     inputs=[PANEL, DERIVED / "distance_matrix.csv"],
                     tag="rho sweep (round 22)") as rm:
        rm.add_note("Imposes rho as an offset so price is the only endogenous variable. "
                    "Exact nested-logit elasticity and diversion formulae. Linear IV only.")
        dz = diesel_tables()
        acc, D, A, labs = build_all()
        rows, dec, diag, free_rows, dm_rows, sub, match = [], [], [], [], [], [], []

        mw = mw_implied_outside_share()
        print("\nMW column (i) inverted to an implied outside share:")
        for k, v in mw.items():
            print(f"  {k:38s} {v:.4f}")

        for g, name, use_bd in (("week", "A dma-week defensible "
                                 "(alt+week+dma_x_year+brand_x_dma)", True),
                                ("month", "B dma-month MW-comparable "
                                 "(alt+month+dma_x_year)", False)):
            e = frame(acc, D, A, labs, g, dz)
            ident = float(np.mean(e[CLS_NAMES].sum(1).to_numpy()
                                  == e.J_rt.to_numpy() - 1))
            print(f"\n{g}: {len(e):,} rows, partition identity "
                  f"{100*ident:.4f}%", flush=True)
            ga = pd.factorize(e.alt)[0]; gt = pd.factorize(e.per)[0]
            gdy = pd.factorize(list(zip(e.dmac, e.year)))[0]
            groups = [ga, gt, gdy]
            if use_bd:
                groups = groups + [pd.factorize(list(zip(e.brand, e.dmac)))[0]]
            # full-window market size, for the diversion comparison only
            q = e.groupby(["dmac", "per"], observed=True).packs.sum().rename("q")
            qq = q.reset_index()
            qq["qmaxfull"] = qq.groupby("dmac")["q"].transform("max")
            e2 = e.merge(qq[["dmac", "per", "q", "qmaxfull"]], on=["dmac", "per"],
                         how="left")
            Mfull = MULT * e2.qmaxfull.to_numpy(float)
            sj_full = e2.packs.to_numpy(float) / Mfull
            ins_full = e2.q.to_numpy(float) / Mfull
            extra = {"full-window max market size": (sj_full, 1.0 - ins_full)}
            out = cell_run(name, e, groups, rows, dec, diag, free_rows, dm_rows,
                           extra_ms=extra)
            sjg_, sjv_, s0v_ = out[6], out[7], out[8]
            for tgt, lab in ((29.80, "MW column (i)"), (12.96, "MW RCNL-1")):
                rstar = rho_matching_diversion(0.0, None, sjg_, sjv_, s0v_, tgt)
                match.append(dict(cell=name, target=lab, target_diversion_pct=tgt,
                                  rho_required=rstar))
                print(f"  rho matching {lab} diversion {tgt}%: "
                      f"{rstar:.4f}" if rstar == rstar else
                      f"  rho matching {lab}: not attainable", flush=True)
            print(f"\n--- Part 3 subsets at rho={MW_RHO_NL}, {name} ---", flush=True)
            part3(name, *out, e, groups, sub)

        tables = (("sweep", rows), ("decomposition", dec),
                   ("diagnostics", diag), ("free_rho", free_rows),
                   ("distance_metric", dm_rows), ("part3_subsets", sub),
                   ("diversion_match", match))
        for nm, obj in tables:
            df = pd.DataFrame(obj)
            df.to_csv(OUT / f"rho_{nm}.csv", index=False)
            print(f"\n--- {nm} ---")
            print(df.to_string(index=False))
        (OUT / "mw_implied_outside_share.json").write_text(json.dumps(mw, indent=1))
        rm.add_outputs([OUT / f"rho_{nm}.csv" for nm, _ in tables])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
