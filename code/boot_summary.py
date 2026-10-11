#!/usr/bin/env python3
"""Bootstrap standard errors and 95% intervals for the supply side and counterfactuals of
column (iv), from results/boot/draw_<b>.json (boot_draw.py).

Draw 0 is the point estimate; draws 1, 2, ... are the parametric bootstrap.  For every
number the draws record, the standard error is the standard deviation across draws >= 1
and the interval is their 2.5th and 97.5th percentiles.

    python3 code/boot_summary.py          # -> results/boot_summary_iv_long.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
B = ROOT / "results" / "boot"
OUT = ROOT / "results" / "boot_summary_iv_long.csv"
SKIP = {"seconds", "seconds_total", "csq", "pct_csq", "lam", "draw", "workers", "ndraws",
        "stores", "panel_a_fit", "conv", "peak_rss_gb_largest_process"}
FAIL = (("supply", "contraction_fail"), ("supply", "pstar_fail"), ("panel_b", "not_converged"))


def flat(x, pre=""):
    """Every numeric leaf of a draw's json, keyed by its path."""
    out = {}
    for k, v in x.items():
        if k in SKIP or v is None:
            continue
        key = f"{pre}{k}"
        if isinstance(v, dict):
            out |= flat(v, key + ".")
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def main():
    fs = sorted(f for f in B.glob("draw_[0-9][0-9][0-9].json"))
    D = {int(f.stem[5:]): json.load(open(f)) for f in fs}
    if 0 not in D:
        sys.exit("results/boot/draw_000.json missing: run boot_draw.py --draws 0")
    draws = sorted(b for b in D if b > 0)
    for b in draws:
        bad = [f"{a}.{k}={D[b][a][k]}" for a, k in FAIL if D[b].get(a) and D[b][a].get(k)]
        if bad:
            print(f"  draw {b}: " + ", ".join(bad))
    F = pd.DataFrame({b: flat(D[b]) for b in [0] + draws}).T
    X = F.loc[draws]
    T = pd.DataFrame({"estimate": F.loc[0], "se": X.std(ddof=1),
                      "lo95": X.quantile(0.025), "hi95": X.quantile(0.975),
                      "draws": X.notna().sum()})
    T.index.name = "statistic"
    T.to_csv(OUT)
    print(f"{len(draws)} bootstrap draws; wrote {OUT.relative_to(ROOT)} ({len(T)} statistics)\n")
    show = ([k for k in T.index if k.startswith("hendel_nevo.")]
            + [f"welfare_loss.{k}" for k in ("VD", "IT", "NPD", "ratio_IT_to_VD")]
            + [f"regimes.{s}.pct.{k}" for s in ("VD", "IT", "NPD") for k in ("cs", "ps", "total")]
            + [f"panel_a.{c}.pct.{k}" for c in ("no_sales", "no_discount", "no_regimes")
               for k in ("cs", "ps", "total")]
            + [k for k in T.index if k.startswith("panel_b.") and k.endswith(".own")]
            + [f"supply.{k}" for k in ("brewer_markup_pct", "gap_sale", "gap_pct")])
    print(f"  {'statistic':34s}{'estimate':>11s}{'se':>9s}{'95% interval':>22s}")
    for k in show:
        if k in T.index:
            r = T.loc[k]
            print(f"  {k:34s}{r.estimate:>11.3f}{r.se:>9.3f}   [{r.lo95:8.3f}, {r.hi95:8.3f}]")


if __name__ == "__main__":
    main()
