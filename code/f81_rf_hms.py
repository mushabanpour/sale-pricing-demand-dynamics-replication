#!/usr/bin/env python3
"""Section 3B: reduced-form evidence from the HMS household panel.

SAMPLE RULES, chosen to match the baseline demand sample rather than HMS convention:
  window     the scanner sample's own calendar span, derived from f70_baseline_{TAG}
  products   in_target_pack & known_container & n_cans in {6,12,18,24,30,36}
  brands     is_focal_brand -- the same 14-brand set the demand system uses
Households are not restricted to the sample's 166 stores: HMS panelists shop anywhere, and
conditioning on the scanner stores would select on the outcome.  Reported either way.

  RF5  do households UPSIZE on a deal?           pack size | deal, household + month FE
  RF6  DEPLETION: does a bigger purchase buy a   days to next beer trip | oz bought
       longer gap before the next one?           household FE
  RF7  REPLENISHMENT: does purchase probability  P(buy this week) | weeks since last,
       rise with time since the last purchase?   and size of that purchase
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, pyhdfe

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
def _scanner_window():
    """The calendar span of the scanner sample, read from the sample itself.

    This used to be hard-coded at 2017-07-01..2018-06-23, which was exactly f3_dev's
    52 weeks.  When the paper moved to `long` the constant stayed behind and the
    household panel silently covered a quarter of the scanner period while Section 3
    claimed the two shared a window.  Deriving it means the two cannot diverge again.
    """
    import pandas as _pd
    from _lib import DW, TAG as _TAG
    w = _pd.read_parquet(DW / f"f70_baseline_{_TAG}.parquet", columns=["w"]).w
    d0 = _pd.Timestamp("2017-04-01")          # scanner week 65, the project's epoch
    lo = d0 + _pd.Timedelta(weeks=int(w.min()) - 65)
    hi = d0 + _pd.Timedelta(weeks=int(w.max()) - 65)
    return lo.strftime("%Y-%m-%d"), hi.strftime("%Y-%m-%d")


W0, W1 = _scanner_window()
RUNGS = (6, 12, 18, 24, 30, 36)
COLS = ["household_code", "purchase_date", "panel_year", "n_cans", "oz_per_pack", "quantity",
        "deal_flag_uc", "is_focal_brand", "in_target_pack", "known_container",
        "total_price_paid", "brand_descr", "store_code_uc"]


STATS = []
LABEL = [""]


def ols_fe(y, X, fe, cl, names):
    keep = [i for i in range(X.shape[1]) if np.ptp(X[:, i]) > 0]
    X = X[:, keep]; names = [names[i] for i in keep]
    A = pyhdfe.create(fe, drop_singletons=False)
    R = A.residualize(np.column_stack([y, X]))
    yr, Xr = R[:, :1], R[:, 1:]
    b = np.linalg.solve(Xr.T @ Xr, Xr.T @ yr); e = (yr - Xr @ b).ravel()
    G = int(cl.max() + 1)
    U = np.zeros((G, Xr.shape[1])); np.add.at(U, cl, Xr * e[:, None])
    B = np.linalg.pinv(Xr.T @ Xr)
    V = B @ (U.T @ U) @ B * (G / (G - 1))
    se = np.sqrt(np.diag(V))
    # within R^2: after the fixed effects are absorbed, the share of the remaining
    # variation in y that the reported regressors explain
    ssr = float(e @ e); tss = float(((yr - yr.mean()) ** 2).sum())
    out = pd.DataFrame({"coef": b.ravel(), "se": se, "t": b.ravel() / se}, index=names)
    out.attrs.update(n=int(len(yr)), r2_within=1.0 - ssr / tss, clusters=int(G))
    STATS.append(dict(fit=LABEL[0], n=int(len(yr)), r2_within=1.0 - ssr / tss,
                      clusters=int(G), k=int(Xr.shape[1])))
    return out


def load():
    """The household sample, already restricted to the scanner window, the pack ladder
    and the fourteen brands (data/consumer panel.parquet)."""
    from _lib import DATA
    d = pd.read_parquet(DATA / "consumer panel.parquet")
    d["purchase_date"] = pd.to_datetime(d.purchase_date)
    d["month"] = d.purchase_date.values.astype("datetime64[M]")
    assert d.purchase_date.between(W0, W1).all() and d.n_cans.isin(RUNGS).all()
    print(f"HMS purchases  {len(d):,}")
    print(f"  households {d.household_code.nunique():,}  "
          f"brands {d.brand_descr.nunique()}  deal rate {100*d.deal.mean():.1f}%")
    print(f"  rung mix: {d.n_cans.value_counts().sort_index().to_dict()}")
    # saved in load()'s household-date order; re-sorting could reorder same-day ties
    k = d[["household_code", "purchase_date"]]
    assert (k.shift().le(k).all(axis=1) | k.household_code.ne(k.household_code.shift())).iloc[1:].all()
    return d.reset_index(drop=True)


def rf5(d):
    print("\n" + "=" * 78)
    print("RF5  DO HOUSEHOLDS UPSIZE ON A DEAL?   household + month effects, cl household")
    print("=" * 78)
    hh = pd.factorize(d.household_code)[0]
    fe = np.column_stack([hh, pd.factorize(d.month)[0]])
    for lab, y in (("log oz per pack", np.log(d.oz_per_pack)),
                   ("1{n_cans >= 24}", (d.n_cans >= 24).astype(float)),
                   ("1{n_cans >= 30}", (d.n_cans >= 30).astype(float))):
        r = ols_fe(y.to_numpy()[:, None], d[["deal"]].to_numpy(float), fe, hh, ["deal"])
        print(f"  {lab:18s} deal {r.coef['deal']:+.4f}  (t {r.t['deal']:+.2f})")
    print("\n  deal rate BY RUNG (raw):")
    t = d.groupby("n_cans").agg(n=("deal", "size"), deal_pct=("deal", lambda x: 100*x.mean()))
    print("   " + t.round(2).to_string().replace("\n", "\n   "))
    # what a flagged purchase is worth per ounce, overall and by pack size.  Section 3
    # quotes these; they used to be typed into the prose and went stale when the
    # window changed.
    x = d[d.oz.gt(0) & d.total_price_paid.gt(0)].copy()
    x["ppo"] = x.total_price_paid / x.oz
    g = x.groupby(["n_cans", "deal"]).ppo.mean().unstack()
    t["deal_ppo_gap_pct"] = 100 * (g[1.0] / g[0.0] - 1)
    a = x.groupby("deal").ppo.mean()
    t.attrs = {}
    t.to_csv(OUT / "f81_rf5_deal_by_rung.csv")
    pd.Series({"deal_rate_pct": 100 * d.deal.mean(),
               "deal_ppo_gap_pct": 100 * (a[1.0] / a[0.0] - 1)},
              name="value").to_csv(OUT / "f81_rf5_deal_overall.csv")
    print(f"\n  deal price gap per ounce: overall "
          f"{100 * (a[1.0] / a[0.0] - 1):+.1f}%  by rung "
          + "  ".join(f"{k}:{v:+.1f}" for k, v in t.deal_ppo_gap_pct.items()))


def rf6(d):
    print("\n" + "=" * 78)
    print("RF6  DEPLETION: does buying MORE today delay the next beer purchase?")
    print("     days to next purchase | log oz bought, household effects, cl household")
    print("=" * 78)
    g = d.groupby("household_code")
    d = d.assign(gap=(g.purchase_date.shift(-1) - d.purchase_date).dt.days,
                 prev_gap=(d.purchase_date - g.purchase_date.shift()).dt.days)
    s = d[d.gap.notna() & (d.gap > 0) & (d.gap <= 180)].copy()
    hh = pd.factorize(s.household_code)[0]
    fe = np.column_stack([hh, pd.factorize(s.month)[0]])
    X = np.column_stack([np.log(s.oz), s.deal])
    LABEL[0] = "gap"
    r = ols_fe(s.gap.to_numpy(float)[:, None], X, fe, hh, ["log oz bought", "bought on deal"])
    print(f"  n {len(s):,} purchase spells, {s.household_code.nunique():,} households")
    print(r.round(4).to_string())
    # RAW vs WITHIN-HOUSEHOLD.  Raw is dominated by composition: heavy buyers buy big
    # AND buy often, so the raw gradient runs the WRONG way for an inventory model.
    # Demeaning by household is what isolates the depletion margin.
    s["gap_w"] = s.gap - s.groupby("household_code").gap.transform("mean") + s.gap.mean()
    print("\n  mean days to next purchase by rung bought  (RAW vs WITHIN-household):")
    t = s.groupby("n_cans").agg(size=("gap", "size"), mean=("gap", "mean"),
                                median=("gap", "median"), within=("gap_w", "mean"))
    print("   " + t.round(2).to_string().replace("\n", "\n   "))
    t.to_csv(OUT / "f81_rf6_gap_by_rung.csv")
    r.to_csv(OUT / "f81_rf6_depletion.csv")
    return s


def rf7(d):
    print("\n" + "=" * 78)
    print("RF7  REPLENISHMENT: purchase hazard by weeks since the last beer purchase")
    print("=" * 78)
    d = d.copy()
    d["wk"] = d.purchase_date.values.astype("datetime64[W]").astype("datetime64[D]")
    hh = d.groupby(["household_code", "wk"], as_index=False).agg(oz=("oz", "sum"))
    allw = np.unique(d.wk.to_numpy())
    idx = pd.MultiIndex.from_product([np.sort(hh.household_code.unique()), allw],
                                     names=["household_code", "wk"])
    P = (hh.set_index(["household_code", "wk"]).reindex(idx).reset_index()
           .sort_values(["household_code", "wk"]))

    # A household is at risk only in the years it is actually in the Homescan panel.
    # The cross-product above assumes every household is present every week, which is
    # true enough over one year and false over four: 54 percent of households here
    # appear in a single panel year.  Left uncorrected, the weeks after a departing
    # household's last purchase enter as at-risk zeros and drive the duration
    # coefficient negative -- the pooled estimate then reverses the sign that every
    # individual year gives.
    mem = pd.MultiIndex.from_frame(
        d[["household_code", "panel_year"]].drop_duplicates().astype(
            {"panel_year": int}))
    keep = pd.MultiIndex.from_arrays(
        [P.household_code, P.wk.values.astype("datetime64[Y]").astype(int) + 1970])
    P = P[keep.isin(mem)].copy()
    P["buy"] = P.oz.notna().astype(float)
    wmap = {w: i for i, w in enumerate(allw)}
    P["wknum"] = P.wk.map(wmap).astype(int)

    # week number, and ounces, of the most recent purchase STRICTLY BEFORE this week
    P["_bw"] = P.wknum.where(P.buy > 0)
    P["_bo"] = P.oz.where(P.buy > 0)
    gb = P.groupby("household_code")
    P["prev_w"] = gb["_bw"].shift().groupby(P.household_code).ffill()
    P["prev_oz"] = gb["_bo"].shift().groupby(P.household_code).ffill()
    P["since"] = P.wknum - P.prev_w

    s = P[P.since.between(1, 12)].copy()
    t = s.groupby("since").buy.agg(n="size", hazard="mean")
    t["hazard_pct"] = 100 * t.hazard
    print(f"  household-weeks {len(P):,}  households {P.household_code.nunique():,}  "
          f"at-risk weeks {len(s):,}")
    print("   " + t[["n", "hazard_pct"]].round(2).to_string().replace("\n", "\n   "))
    t.to_csv(OUT / "f81_rf7_hazard.csv")

    # within-household hazard: the raw curve falls with duration only because frequent
    # buyers crowd the short durations.  Demeaned, replenishment shows its true sign.
    s["buy_w"] = s.buy - s.groupby("household_code").buy.transform("mean") + s.buy.mean()
    tw = s.groupby("since").buy_w.mean().rename("hazard_within_pct") * 100
    t = t.join(tw)
    print("\n  hazard RAW vs WITHIN-household (%):")
    print("   " + t[["hazard_pct", "hazard_within_pct"]].round(2).to_string().replace("\n", "\n   "))
    t.to_csv(OUT / "f81_rf7_hazard.csv")

    med = s.prev_oz.median()
    s["large_last"] = (s.prev_oz >= med).astype(int)
    h = (s.groupby(["since", "large_last"]).buy.mean().unstack() * 100)
    h.columns = [f"small last (<{med:.0f}oz)", f"large last (>={med:.0f}oz)"]
    h["gap (large - small)"] = h.iloc[:, 1] - h.iloc[:, 0]
    print("\n  hazard (%) by weeks since last purchase, split on SIZE of that purchase:")
    print("   " + h.round(2).to_string().replace("\n", "\n   "))
    h.to_csv(OUT / "f81_rf7_hazard_by_size.csv")

    # regression form: does a bigger last purchase lower this week's purchase probability?
    #
    # The effect is household x PANEL YEAR, not household.  A panelist's recorded
    # purchase frequency decays over their tenure, so later weeks carry both longer
    # gaps and a lower purchase probability; a household effect alone leaves that
    # decay in the residual and the duration coefficient picks it up.  Over one year
    # there is little room for it -- estimated separately, 2016, 2017, 2018 and 2019
    # each give phi<0 and psi>0 -- but pooled over four it reverses both signs.
    # Household x year absorbs it and returns the within-year answer at four times
    # the sample.
    hhf = pd.factorize(s.household_code)[0]
    hyf = pd.factorize(s.household_code.astype(str) + "|"
                       + s.wk.values.astype("datetime64[Y]").astype(str))[0]
    X = np.column_stack([np.log(s.prev_oz.clip(lower=1)), s.since.astype(float)])
    LABEL[0] = "hazard"
    r = ols_fe(s.buy.to_numpy(float)[:, None], X,
               np.column_stack([hyf, pd.factorize(s.wk)[0]]), hhf,
               ["log oz of last purchase", "weeks since last purchase"])
    print("\n  P(buy this week), household x year + week effects, clustered on household:")
    print("   " + r.round(4).to_string().replace("\n", "\n   "))
    r.to_csv(OUT / "f81_rf7_hazard_reg.csv")


def rf8(d):
    """SORTING is a finding, not a nuisance.  The cross-household gradient that makes the
    raw depletion comparison misleading is itself the reduced-form counterpart of the
    pack-size random coefficient: households differ systematically in the pack size they
    buy, and those differences line up with how fast they consume."""
    print("\n" + "=" * 78)
    print("RF8  SORTING ON PACK SIZE: who buys big, and how fast do they consume?")
    print("     the reduced-form counterpart of sigma_lsz")
    print("=" * 78)
    g = d.groupby("household_code")
    h = pd.DataFrame({"mean_cans": g.n_cans.mean(), "trips": g.purchase_date.nunique(),
                      "oz_total": g.oz.sum()})
    h = h[h.trips >= 4].copy()
    h["oz_per_week"] = h.oz_total / 52.0
    h["q"] = pd.qcut(h.mean_cans, 5, labels=False, duplicates="drop") + 1
    t = h.groupby("q").agg(households=("trips", "size"), mean_cans=("mean_cans", "mean"),
                           trips_per_yr=("trips", "mean"), oz_per_week=("oz_per_week", "mean"))
    print(f"  households with >=4 beer trips: {len(h):,}")
    print("  quintiles of a household's AVERAGE pack size:")
    print("   " + t.round(2).to_string().replace("\n", "\n   "))
    r = np.corrcoef(h.mean_cans, h.oz_per_week)[0, 1]
    rs = h.mean_cans.corr(h.oz_per_week, method="spearman")
    print(f"\n  corr(household mean pack size, oz consumed per week) = {r:+.3f} "
          f"(Spearman {rs:+.3f})")
    print(f"  dispersion IN household mean pack size: sd {h.mean_cans.std():.2f} cans, "
          f"p10 {h.mean_cans.quantile(.1):.1f}  p50 {h.mean_cans.quantile(.5):.1f}  "
          f"p90 {h.mean_cans.quantile(.9):.1f}")
    t.to_csv(OUT / "f81_rf8_sorting.csv")
    h[["mean_cans", "trips", "oz_per_week"]].to_csv(OUT / "f81_rf8_households.csv")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    d = load(); rf5(d); s = rf6(d); rf7(d); rf8(d)
    pd.DataFrame(STATS).set_index("fit").to_csv(OUT / "f81_rf_stats.csv")
    print("\nfit statistics:")
    print(pd.DataFrame(STATS).set_index("fit").round(4).to_string())
    print(f"\nCSV written to {OUT}/f81_rf*.csv")
