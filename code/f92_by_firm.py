#!/usr/bin/env python3
"""Firm-level profit and the equilibrium price/quantity table, from f92's output.

f92_counterfactual.py writes the product-level panel and the marginal costs but not
the two firm-level summaries the counterfactual section needs.  This stage builds
them, so Tables~10 and~11 come from the same run as the welfare totals:

    results/f92_ps_by_firm.csv            profit by firm and pricing regime
    results/f92_scenario_table_firm.csv   price and quantity by firm and pack size

Profit is $(p-mc)s$ summed over products, exactly the quantity f92 accumulates into
the welfare file, so the by-firm columns add to the producer surplus reported there;
that identity is asserted before anything is written.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from f87_supply import FIRM                                        # noqa: E402
from _lib import TAG, COL                                          # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
SCEN = ("MIX", "VD", "IT", "NPD")          # the order f92 concatenates the panel in
FIRMS = ("ABI", "MolsonCoors", "Constellation", "Heineken")
RUNGS = (6, 12, 18, 24, 30, 36)
SERVINGS = 12.0                            # a price per 144 ounces is 12 servings
PER = 10_000.0                             # quantities are per 10,000 servings


def panel():
    """The product-level counterfactual panel, with cost, profit and the sale flag."""
    d = pd.read_parquet(RES / f"f92_pq_{COL}_{TAG}.parquet")
    mc = np.load(RES / f"f92_mc_{COL}_{TAG}.npy")
    n = len(mc)
    assert len(d) == n * len(SCEN), f"{len(d):,} rows against {n:,} costs"
    assert list(d.scen.iloc[::n]) == list(SCEN), "scenario blocks out of order"
    d["mc"] = np.tile(mc, len(SCEN))
    d["firm"] = d.brand.map(FIRM)
    d["sale"] = d.reg3.str[1].eq("S")
    d["pserv"] = d.price / SERVINGS
    d["qty"] = d.share * PER
    d["prof"] = (d.price - d.mc) * d.share
    return d


def by_firm(d):
    ps = d.pivot_table(index="firm", columns="scen", values="prof", aggfunc="sum")
    return ps.reindex(index=sorted(FIRMS), columns=list(SCEN))


def _p(d, sc, firm, sale=None, pack=None):
    """Median price per serving.  Pack is left open where the regime imposes one
    price across the menu, which is what NPD and IT do."""
    m = d[(d.scen == sc) & (d.firm == firm)]
    if sale is not None:
        m = m[m.sale == sale]
    if pack is not None:
        m = m[m.n_cans == pack]
    return round(float(m.pserv.median()), 2) if len(m) else None


def _q(d, sc, firm, pack, sale=None):
    """Mean over markets of the firm's combined quantity at that pack size."""
    m = d[(d.scen == sc) & (d.firm == firm) & (d.n_cans == pack)]
    if sale is not None:
        m = m[m.sale == sale]
    if not len(m):
        return None
    return round(float(m.groupby("market_ids", observed=True).qty.sum().mean()), 1)


def scenario_table(d, ps):
    rows = []
    for f in FIRMS:
        for i, k in enumerate(RUNGS):
            stocked = len(d[(d.firm == f) & (d.n_cans == k)]) > 0
            r = {"firm": f if i == 0 else "", "pack": k}
            if stocked:
                r |= {"ND_Q": _q(d, "NPD", f, k), "VD_P": _p(d, "VD", f, pack=k),
                      "VD_Q": _q(d, "VD", f, k),
                      "IT_RQ": _q(d, "IT", f, k, False), "IT_SQ": _q(d, "IT", f, k, True),
                      "MX_RP": _p(d, "MIX", f, False, k), "MX_RQ": _q(d, "MIX", f, k, False),
                      "MX_SP": _p(d, "MIX", f, True, k), "MX_SQ": _q(d, "MIX", f, k, True)}
            if i == 0:
                # one price per firm under NPD, and one per firm per week under IT
                r |= {"ND_P": _p(d, "NPD", f),
                      "IT_RP": _p(d, "IT", f, False), "IT_SP": _p(d, "IT", f, True)}
                r |= {f"p{s}": round(100 * ps.loc[f, sc] / ps.loc[f, "MIX"], 1)
                      for s, sc in (("ND", "NPD"), ("VD", "VD"),
                                    ("IT", "IT"), ("MX", "MIX"))}
            rows.append(r)
    cols = ["firm", "pack", "ND_P", "ND_Q", "VD_P", "VD_Q", "IT_RP", "IT_RQ",
            "IT_SP", "IT_SQ", "MX_RP", "MX_RQ", "MX_SP", "MX_SQ",
            "pND", "pVD", "pIT", "pMX"]
    return pd.DataFrame(rows).reindex(columns=cols)


def main():
    d = panel()
    ps = by_firm(d)

    # the by-firm columns must reproduce the producer surplus of the welfare file
    W = pd.read_csv(RES / f"f92_welfare_{COL}_{TAG}.csv").groupby("scen").ps.sum()
    gap = (ps.sum() - W.reindex(ps.columns)).abs().max()
    assert gap < 1e-6, f"by-firm profit off producer surplus by {gap:.3g}"
    print(f"profit by firm reconciles with f92_welfare_{COL}_{TAG}.csv "
          f"(max gap {gap:.2g})")

    ps.to_csv(RES / "f92_ps_by_firm.csv")
    scenario_table(d, ps).to_csv(RES / "f92_scenario_table_firm.csv", index=False)
    print(ps.round(1).to_string())
    print(f"\nwrote {RES / 'f92_ps_by_firm.csv'}")
    print(f"wrote {RES / 'f92_scenario_table_firm.csv'}")


if __name__ == "__main__":
    main()
