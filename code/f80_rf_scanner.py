#!/usr/bin/env python3
"""Section 3A: reduced-form evidence from the SCANNER panel.

SAMPLE IS THE BASELINE SAMPLE.  Reads f70_baseline_dev.parquet -- the same store-week
panel, the same 14 brands and six rungs, the same median-of-prior-4-observed-weeks
reference and 5% sale cut that columns (i)-(iv) are estimated on.  Nothing is re-derived
here, so every number in section 3 refers to the same rows as the demand estimates.

  RF1  dynamic pricing: sale incidence, depth, persistence, spell length, by rung
  RF2  event study: log servings on sale leads and lags, cell + week effects
       -> the acceleration spike at t=0 and the post-sale DIP that is depletion
  RF3  cross-rung substitution: own-rung sale vs sales on SMALLER and LARGER rungs
  RF4  category expansion: does a sale grow the store-week category or just reallocate?
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, pyhdfe

ROOT = Path(__file__).resolve().parent.parent
TAG = "long"
RUNGS = (6, 12, 18, 24, 30, 36)
OUT = ROOT / "results"


def load():
    d = pd.read_parquet(ROOT / "data_work" / f"f70_baseline_{TAG}.parquet")
    M = pd.read_parquet(ROOT / "data_work" / "f38_market_size.parquet",
                        columns=["market_ids", "mkt_size"])
    d = d.merge(M, on="market_ids", how="left")
    d["serv"] = d.shares * d.mkt_size
    d["cell"] = d.store_code_uc.astype(str) + "_" + d.alt.astype(str)
    d = d.sort_values(["cell", "w"]).reset_index(drop=True)
    # `sale` is NaN on 5.1% of rows where the prior-4-week reference cannot be formed.
    # Unknown is NOT no-sale: those rows are dropped from the demand estimation too (a row
    # with no reference has no regime).  Keep them ONLY as lead/lag look-ups, never as
    # estimation rows, so section 3 runs on exactly baseline1's 206,209 rows.
    full = d[d.sale.notna()].copy()
    full["sale"] = full.sale.astype(bool)
    est = d[d.reg1 != ""].copy()
    est["sale"] = est.sale.astype(bool)
    assert est.sale.notna().all()
    print(f"  look-up frame {len(full):,} rows (sale defined); "
          f"ESTIMATION rows {len(est):,} = baseline1 sample")
    return est, full


STATS = []
LABEL = [""]


def ols_fe(y, X, fe, cl, names):
    # a regressor with no within-sample variation (e.g. "a smaller rung is on sale" at
    # rung 6, where none exists) is dropped rather than allowed to make X'X singular
    keep = [i for i in range(X.shape[1]) if np.ptp(X[:, i]) > 0]
    X = X[:, keep]; names = [names[i] for i in keep]
    A = pyhdfe.create(fe, drop_singletons=False)
    R = A.residualize(np.column_stack([y, X]))
    yr, Xr = R[:, :1], R[:, 1:]
    b = np.linalg.solve(Xr.T @ Xr, Xr.T @ yr)
    e = (yr - Xr @ b).ravel()
    G = int(cl.max() + 1)
    U = np.zeros((G, Xr.shape[1])); np.add.at(U, cl, Xr * e[:, None])
    B = np.linalg.pinv(Xr.T @ Xr)
    V = B @ (U.T @ U) @ B * (G / (G - 1))
    se = np.sqrt(np.diag(V))
    # within R^2: after the fixed effects are absorbed, the share of the remaining
    # variation in y that the reported regressors explain
    ssr = float(e @ e); tss = float(((yr - yr.mean()) ** 2).sum())
    out = pd.DataFrame({"coef": b.ravel(), "se": se, "t": b.ravel() / se}, index=names)
    out.attrs.update(n=int(len(yr)), r2_within=1.0 - ssr / tss, clusters=int(G), V=V)
    STATS.append(dict(fit=LABEL[0], n=int(len(yr)), r2_within=1.0 - ssr / tss,
                      clusters=int(G), k=int(Xr.shape[1])))
    return out


def leadlags(d, K, src_df=None):
    """Sale at w+l for l in -K..K, matched on the CALENDAR week, not by row position:
    a cell with a gap must not have the gap silently treated as an adjacent week.
    src_df supplies the look-ups so a neighboring week just outside the estimation
    sample still counts; genuinely unobserved neighbors stay 0 and are counted."""
    base = d[["cell", "w"]].copy()
    look = d if src_df is None else src_df
    cols, names = [], []
    miss = 0
    # EVENT TIME k: the regressor is SALE at week t-k, so a POSITIVE k is a sale k weeks
    # AGO (post-sale response) and a NEGATIVE k is a sale |k| weeks AHEAD (anticipation).
    # The earlier version indexed by t+l but named the columns as if it indexed by t-l,
    # which reversed every lead and lag.
    for k in range(-K, K + 1):
        src = look[["cell", "w", "sale"]].rename(columns={"sale": "_s"})
        src["w"] = src.w + k                      # SALE at w-k lands on row w
        v = base.merge(src, on=["cell", "w"], how="left")._s
        miss += int(v.isna().sum())
        cols.append(v.fillna(False).astype(float).to_numpy())
        names.append("t0" if k == 0 else (f"lead{-k}" if k < 0 else f"lag{k}"))
    print(f"    lead/lag cells with an unobserved neighbor, coded 0: "
          f"{miss:,} of {len(base)*(2*K+1):,} ({100*miss/(len(base)*(2*K+1)):.1f}%)")
    return np.column_stack(cols), names


def rf0(d):
    """Why the reference is a MEDIAN.  min sets the benchmark at the previous sale price,
    so a continuing sale cannot clear the cut and the both-weeks-on-sale state is
    destroyed by construction; mean is dragged down by sale weeks inside the window."""
    print("\n" + "=" * 78)
    print("RF0  CHOICE OF REFERENCE: median vs mean vs min of the prior four weeks")
    print("=" * 78)
    piv = d.pivot_table(index=["store_code_uc", "alt"], columns="w", values="prices")
    P = piv.to_numpy(float); O = np.isfinite(P)
    S = np.stack([np.where(np.pad(O[:, :-l], ((0, 0), (l, 0)), constant_values=False),
                  np.pad(P[:, :-l], ((0, 0), (l, 0)), constant_values=np.nan), np.nan)
                  for l in range(1, 5)], axis=2)
    with np.errstate(all="ignore"):
        refs = {"median": np.nanmedian(S, axis=2), "mean": np.nanmean(S, axis=2),
                "min": np.nanmin(S, axis=2)}
    rows = {}
    for nm, ref in refs.items():
        ok = O & np.isfinite(ref)
        with np.errstate(all="ignore"):
            sale = np.where(ok, (ref - P) / np.where(ref > 0, ref, np.nan) > 0.05, False)
        lo = np.pad(ok[:, :-1], ((0, 0), (1, 0)), constant_values=False)
        ls = np.pad(sale[:, :-1], ((0, 0), (1, 0)), constant_values=False)
        use = ok & lo
        with np.errstate(all="ignore"):
            rows[nm] = dict(sale_rate=100 * sale[ok].mean(),
                            both_weeks=100 * (sale & ls)[use].mean(),
                            p_sale_given_sale=100 * (sale & ls)[use].sum() / max(ls[use].sum(), 1),
                            mean_depth=100 * np.nanmean(np.where(sale, (ref - P) / ref, np.nan)[ok]))
    ref = refs["median"]; ok = O & np.isfinite(ref)
    with np.errstate(all="ignore"):
        sale = np.where(ok, (ref - P) / np.where(ref > 0, ref, np.nan) > 0.05, False)
    contaminated = np.zeros_like(sale)
    nsale = np.zeros(P.shape, dtype=float)
    for l in range(1, 5):
        lag = np.pad(sale[:, :-l], ((0, 0), (l, 0)), constant_values=False)
        contaminated |= lag
        nsale += lag.astype(float)
    chg = np.abs(ref - np.pad(ref[:, :-1], ((0, 0), (1, 0)), constant_values=np.nan)) / \
          np.pad(ref[:, :-1], ((0, 0), (1, 0)), constant_values=np.nan)
    m = ok & np.isfinite(chg)
    t = pd.DataFrame(rows).T
    t.loc["median", "window_has_a_sale"] = 100 * contaminated[ok].mean()
    print(t.round(2).to_string())
    # the premises the median rests on, tested against each cell's MODAL price over the
    # year -- an UNFITTED benchmark.  Deviations from the fitted median are centered by
    # construction, so they cannot be used to test asymmetry.
    dd = d.copy()
    dd["cell"] = dd.store_code_uc.astype(str) + "_" + dd.alt.astype(str)
    mode = (dd.assign(r=dd.prices.round(2)).groupby(["cell", "r"]).size().rename("n")
              .reset_index().sort_values("n").groupby("cell").tail(1)
              .set_index("cell").r.rename("mode_p"))
    gg = dd.join(mode, on="cell"); gg = gg[gg.mode_p > 0]
    x = (gg.prices - gg.mode_p) / gg.mode_p
    nsal = nsale[ok]
    prem = dict(at_mode=100 * (x.abs() <= .01).mean(),
                below_mode=100 * (x < -.01).mean(), above_mode=100 * (x > .01).mean(),
                tail_below_10=100 * (x < -.10).mean(), tail_above_10=100 * (x > .10).mean(),
                mean_dev=100 * x.mean(), median_dev=100 * x.median(),
                window_gt_half=100 * float(np.mean(nsal >= 3)),
                ref_change_median=100 * float(np.nanmedian(chg[m])),
                ref_change_gt5=100 * float(np.mean(chg[m] > .05)),
                dev_min=100 * x.min(), dev_max=100 * x.max())
    print("\n  premises, against each cell's modal price (unfitted):")
    for k, v in prem.items():
        print(f"    {k:20s} {v:8.2f}")
    pd.Series(prem).to_csv(OUT / "f80_rf0_premises.csv", header=["value"])
    t.to_csv(OUT / "f80_rf0_reference.csv")
    return t


def rf1(d):
    print("\n" + "=" * 78)
    print("RF1  DYNAMIC PRICING: how often, how deep, how persistent -- by rung")
    print("=" * 78)
    rows = []
    for r in RUNGS:
        s = d[d.n_cans == r]
        dsale = s.depth[s.sale]
        spell = s.assign(blk=(s.sale != s.groupby("cell").sale.shift()).cumsum())
        sp = spell[spell.sale].groupby(["cell", "blk"]).size()
        prev = s.groupby("cell").sale.shift()
        rows.append(dict(rung=r, rows=len(s),
                         sale_rate=100 * s.sale.mean(),
                         p_sale_given_sale=100 * s.sale[prev == True].mean(),
                         mean_depth_pct=100 * dsale.mean(),
                         med_depth_pct=100 * dsale.median(),
                         mean_spell_wks=sp.mean() if len(sp) else np.nan,
                         p_spell_1wk=100 * (sp == 1).mean() if len(sp) else np.nan))
    t = pd.DataFrame(rows).set_index("rung")
    print(t.round(2).to_string())
    print("\n  persistence ratio P(sale|sale)/P(sale): " +
          "  ".join(f"{r}:{t.loc[r,'p_sale_given_sale']/t.loc[r,'sale_rate']:.2f}" for r in RUNGS))
    t.to_csv(OUT / "f80_rf1_pricing_by_rung.csv")
    return t


def collapse(res, names):
    """Average the pre-sale and post-sale coefficients, with the standard error of the
    average taken from the full covariance matrix rather than treating them as
    independent -- adjacent event-time coefficients are strongly correlated."""
    V = res.attrs["V"]
    idx = {n: i for i, n in enumerate(names)}
    out = {}
    for lab, keys in (("pre-sale average", [f"lead{j}" for j in (4, 3, 2, 1)]),
                      ("sale week", ["t0"]),
                      ("post-sale average", [f"lag{j}" for j in (1, 2, 3, 4)])):
        w = np.zeros(len(names))
        for k in keys:
            w[idx[k]] = 1.0 / len(keys)
        v = float(w @ res.coef.to_numpy())
        se = float(np.sqrt(w @ V @ w))
        out[lab] = dict(coef=v, se=se, t=v / se)
    return pd.DataFrame(out).T


SUMM = {}


def rf2(d, K=4):
    print("\n" + "=" * 78)
    print(f"RF2  EVENT STUDY: log servings on sale leads/lags -{K}..+{K}")
    print("     cell (store x alternative) + week effects, clustered on DMA")
    print("=" * 78)
    X, names = leadlags(d, K, FULL)
    y = np.log(d.serv.clip(lower=1e-9)).to_numpy()[:, None]
    fe = np.column_stack([pd.factorize(d.cell)[0], pd.factorize(d.w)[0]])
    LABEL[0] = "event_all"
    res = ols_fe(y, X, fe, pd.factorize(d.dma_code)[0], names)
    print(res.round(4).to_string())
    res.to_csv(OUT / "f80_rf2_event_study.csv")
    SUMM["all"] = collapse(res, names)
    # same, split by pack group
    d2 = d.assign(grp=np.where(d.n_cans <= 12, "small(6-12)",
                   np.where(d.n_cans <= 24, "mid(18-24)", "large(30+)")))
    out = {}
    for gname, s in d2.groupby("grp"):
        C, _ = leadlags(s, K, FULL)
        LABEL[0] = "event_" + gname.split("(")[0]
        r = ols_fe(np.log(s.serv.clip(lower=1e-9)).to_numpy()[:, None], C,
                   np.column_stack([pd.factorize(s.cell)[0], pd.factorize(s.w)[0]]),
                   pd.factorize(s.dma_code)[0], names)
        out[(gname, "coef")] = r.coef
        out[(gname, "se")] = r.se
        SUMM[gname] = collapse(r, names)
        print(f"\n  {gname}: " + "  ".join(f"{n}:{v:+.3f}" for n, v in r.coef.items()))
    pd.DataFrame(out).to_csv(OUT / "f80_rf2_event_study_by_group.csv")
    pd.concat(SUMM, axis=1).to_csv(OUT / "f80_rf2_summary.csv")
    print("\n  averaged profile (pre = mean of k=-4..-1, post = mean of k=+1..+4):")
    print("   " + pd.concat(SUMM, axis=1).round(4).to_string().replace("\n", "\n   "))
    return res


def rf3(d):
    print("\n" + "=" * 78)
    print("RF3  CROSS-RUNG SUBSTITUTION: own sale vs a sale on a SMALLER / LARGER rung")
    print("     log servings, cell + week effects, clustered on DMA")
    print("=" * 78)
    # within each store-week and brand, does any SMALLER / LARGER rung go on sale?
    k = ["market_ids", "brand_id"]
    d = d.copy()
    d["_one"] = 1.0
    sm, lg = [], []
    for (mid, br), s in d.groupby(k, sort=False):
        nc = s.n_cans.to_numpy(); sa = s.sale.to_numpy()
        sm.append(pd.Series([(sa & (nc < r)).any() for r in nc], index=s.index))
        lg.append(pd.Series([(sa & (nc > r)).any() for r in nc], index=s.index))
    d["sale_smaller"] = pd.concat(sm).reindex(d.index).astype(float)
    d["sale_larger"] = pd.concat(lg).reindex(d.index).astype(float)
    X = d[["sale", "sale_smaller", "sale_larger"]].astype(float).to_numpy()
    y = np.log(d.serv.clip(lower=1e-9)).to_numpy()[:, None]
    fe = np.column_stack([pd.factorize(d.cell)[0], pd.factorize(d.w)[0]])
    LABEL[0] = "cross_all"
    res = ols_fe(y, X, fe, pd.factorize(d.dma_code)[0],
                 ["own sale", "sale on SMALLER rung", "sale on LARGER rung"])
    print(res.round(4).to_string())
    res.to_csv(OUT / "f80_rf3_pooled.csv")        # the pooled estimates quoted in the text
    print("\n  by rung (own sale | smaller on sale | larger on sale):")
    rows = {}
    for r in RUNGS:
        s = d[d.n_cans == r]
        if s.sale.sum() < 100: continue
        LABEL[0] = f"cross_{r}"
        rr = ols_fe(np.log(s.serv.clip(lower=1e-9)).to_numpy()[:, None],
                    s[["sale", "sale_smaller", "sale_larger"]].astype(float).to_numpy(),
                    np.column_stack([pd.factorize(s.cell)[0], pd.factorize(s.w)[0]]),
                    pd.factorize(s.dma_code)[0], ["own", "smaller", "larger"])
        rows[(r, "coef")] = rr.coef
        rows[(r, "se")] = rr.se
        print(f"    rung {r:<3d} " + "  ".join(f"{n} {v:+.3f} ({rr.t[n]:+.2f})"
                                               for n, v in rr.coef.items()))
    pd.DataFrame(rows).to_csv(OUT / "f80_rf3_cross_rung.csv")
    return res, d


def rf4(d):
    print("\n" + "=" * 78)
    print("RF4  CATEGORY EXPANSION: does a sale grow the store-week category?")
    print("=" * 78)
    m = d.groupby(["market_ids", "store_code_uc", "w", "dma_code"], as_index=False).agg(
        tot=("serv", "sum"), nsale=("sale", "sum"), nalt=("alt", "size"),
        big_sale=("sale", lambda x: 0.0))
    big = d[d.n_cans >= 30].groupby("market_ids").sale.max().rename("big_sale_f")
    sml = d[d.n_cans <= 12].groupby("market_ids").sale.max().rename("small_sale_f")
    m = m.merge(big, on="market_ids", how="left").merge(sml, on="market_ids", how="left")
    m[["big_sale_f", "small_sale_f"]] = m[["big_sale_f", "small_sale_f"]].fillna(False)
    m["share_on_sale"] = m.nsale / m.nalt
    X = m[["share_on_sale", "big_sale_f", "small_sale_f"]].astype(float).to_numpy()
    y = np.log(m.tot.clip(lower=1e-9)).to_numpy()[:, None]
    fe = np.column_stack([pd.factorize(m.store_code_uc)[0], pd.factorize(m.w)[0]])
    LABEL[0] = "category"
    res = ols_fe(y, X, fe, pd.factorize(m.dma_code)[0],
                 ["share of alts on sale", "any LARGE-pack sale", "any SMALL-pack sale"])
    print(f"  store-weeks {len(m):,}")
    print(res.round(4).to_string())
    res.to_csv(OUT / "f80_rf4_category_expansion.csv")
    return res


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    d, FULL = load()
    globals()["FULL"] = FULL
    print(f"BASELINE SAMPLE: {len(d):,} rows  {d.market_ids.nunique():,} store-weeks  "
          f"{d.store_code_uc.nunique()} stores  {d.dma_code.nunique()} DMAs  "
          f"{d.brand_id.nunique()} brands  weeks {d.w.min()}-{d.w.max()}")
    rf0(d); rf1(d); rf2(d); res3, d = rf3(d); rf4(d)
    pd.DataFrame(STATS).set_index("fit").to_csv(OUT / "f80_rf_stats.csv")
    print("\nfit statistics:")
    print(pd.DataFrame(STATS).set_index("fit").round(4).to_string())
    print(f"\nCSV written to {OUT}/f80_rf*.csv")
