"""Sale definitions A (own prior median) and B (ZIP3 median).

Three byte-near-identical copies exist; this is the single definition site.
Constants are the project's, unchanged.

CAVEAT recorded from audit round 03 (event study, Part 10.2-10.3):
  * the rebound condition can only hold on the FINAL low week of a spell, so
    95.8158% of coded sale weeks fall on the last week of their spell;
  * 39.9994% of low-price weeks fail the rebound test and are not coded as sales
    at all;
  * definition B's zip_median is a plain group median that INCLUDES the focal
    store, so a store's price is compared to a median it helps set.
None of these is resolved here.
"""
from __future__ import annotations
import numpy as np

SALE_DROP_THRESHOLD = 0.10      # sumstat/...:17 ; reduced_form/...:58
NEXT_REBOUND_THRESHOLD = 0.05   # sumstat/...:18 ; reduced_form/...:59
PRIOR_LAGS = (1, 2, 3, 4)

DUPLICATE_SITES = [
    ("sumstat/make_quantity_discounts_sales_table.py", "56-102", "definition A"),
    ("sumstat/make_market_observation_summary.py", "88-120", "definition A"),
    ("reduced_form/estimate_disciplined_structural_iv_gmm_2016_2019.py", "257-268", "definition A"),
    ("reduced_form/estimate_disciplined_structural_iv_gmm_2016_2019.py", "274-290", "definition B"),
]


def prior_median(price_matrix, obs_matrix, lags=PRIOR_LAGS):
    """Median of the strictly-lagged OBSERVED prices. cells x weeks in, cells x weeks out."""
    pp = np.stack([np.pad(price_matrix[:, :-l], ((0, 0), (l, 0)), constant_values=np.nan)
                   for l in lags], axis=2)
    po = np.stack([np.pad(obs_matrix[:, :-l], ((0, 0), (l, 0)), constant_values=False)
                   for l in lags], axis=2)
    with np.errstate(all="ignore"):
        return np.nanmedian(np.where(po, pp, np.nan), axis=2)


def sale_flag(price_matrix, obs_matrix, benchmark,
              drop=SALE_DROP_THRESHOLD, rebound=NEXT_REBOUND_THRESHOLD):
    """1 iff observed, next week observed, at least `drop` below `benchmark`,
    and the next observed price rebounds by at least `rebound`."""
    nxt_p = np.pad(price_matrix[:, 1:], ((0, 0), (0, 1)), constant_values=np.nan)
    nxt_o = np.pad(obs_matrix[:, 1:], ((0, 0), (0, 1)), constant_values=False)
    return (obs_matrix & nxt_o & np.isfinite(benchmark) & (benchmark > 0)
            & (price_matrix <= (1 - drop) * benchmark)
            & (nxt_p >= (1 + rebound) * price_matrix))


def low_price_flag(price_matrix, obs_matrix, benchmark, drop=SALE_DROP_THRESHOLD):
    """The sale condition WITHOUT the rebound test -- the denominator for the
    39.9994% failure rate reported in audit round 03."""
    return obs_matrix & np.isfinite(benchmark) & (benchmark > 0) \
        & (price_matrix <= (1 - drop) * benchmark)
