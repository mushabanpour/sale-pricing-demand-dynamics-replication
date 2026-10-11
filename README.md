# Sale Pricing and Demand Dynamics: replication package

Muhammad Shabanpour, Northeastern University.

The code that produces every estimate in the article, the logs of the runs that produced
them, and the one public data file. The tables, figures and text are built from these
estimates.

## Data

| Data | Source | In this package |
|---|---|---|
| NielsenIQ Retail Scanner (RMS), 2016–2019 | Kilts Center for Marketing, University of Chicago Booth | no: licensed |
| NielsenIQ Consumer Panel (Homescan), 2016–2019 | Kilts Center for Marketing, University of Chicago Booth | no: licensed |
| American Community Survey 5-year estimates by 3-digit zip, 2016–2019 | U.S. Census Bureau | `data/acs_zip3_demographics_2016_2019.csv` |

The NielsenIQ data, and every file derived from them at the store, product or household
level, can be shared only with researchers who hold a NielsenIQ license through the Kilts
Center. With that
license, every one of them is rebuilt from the raw files by `build/`.

The code expects these files in `data/`:

| File | Built by |
|---|---|
| `data/regression ready.parquet` | `build/` (pos1, the article's sample) |
| `data/positions/pos2/regression ready.parquet` … `pos4/` | `build/` (the appendix samples) |
| `data/consumer panel.parquet` | `build/` (the household sample, Table 4 and Figure 3 only) |
| `data/acs_zip3_demographics_2016_2019.csv` | included |

My analyses are calculated or derived in part from data from Nielsen Consumer LLC and
marketing databases provided through the NielsenIQ Datasets at the Kilts Center for
Marketing Data Center at The University of Chicago Booth School of Business. The
conclusions drawn from the NielsenIQ data are my own and do not reflect the views of
NielsenIQ. NielsenIQ is not responsible for, had no role in, and was not involved in
analyzing or preparing the results reported herein.

## Building the data (`build/`)

The scanner samples, from raw RMS:

    export NIELSEN_RMS=/path/to/nielsen_extracts/RMS
    python3 build/build_foodall.py       # scans every beer movement file, builds the store frame
    python3 build/build_positions.py     # -> build/data/pos1/ ... pos4/regression ready.parquet

Move pos1 to `data/` and pos2–pos4 to `data/positions/`. Intermediate files go to
`build/work/`, and a stage whose output exists is skipped. `build/legacy/` holds the
modules `build_foodall.py` calls.

The household sample, from raw Homescan and the RMS products master file:

    python3 build/build_upc_lookup.py                  # -> build/work/upc_lookup_exact_container.parquet
    stata -b do build/HMScleaningdo.do                 # -> build/work/purchases_with_trips_2016_2019.dta
    python3 build/build_consumer_panel.py build/work/purchases_with_trips_2016_2019.dta

Set the Homescan path at the top of `HMScleaningdo.do`. The do-file merges every Homescan
purchase of a beer UPC with the lookup and with its trip (700,279 purchases, 2016–2019);
`build_consumer_panel.py` keeps the fourteen brands' 6- to 36-packs of 12-ounce cans and
bottles in the scanner sample's window: 250,297 purchases by 24,414 households.

Sample rules: RMS beer 2016–2019, units > 0, $0.50–$100 a pack; the fourteen brands, 6- to
36-packs of 12-ounce cans and bottles; food channel in every year the store reports; median
of at least 2,000 cans or bottles a week; every fourth store in store-code order from the
1st (pos1), 2nd (pos2), 3rd (pos3) or 4th (pos4); at least 3% of product-weeks deep sales;
products observed at least 52 weeks on average; price observed this week and last, with
the reference price formed; a smaller pack of the same brand and container stocked that
week, or none exists; inside share between 1% and 95%; up to five stores per DMA, shuffled
within DMA with seed 0. Market size is all malt-beverage servings in the store-week.

| Sample | Stores | DMAs | Chains |
|---|---|---|---|
| pos1 | 290 | 80 | 33 |
| pos2 | 285 | 79 | 34 |
| pos3 | 297 | 82 (+1 store with no DMA code) | 36 |
| pos4 | 305 | 85 | 36 |

## The model

Random coefficients logit estimated with pyblp, with a lognormal price coefficient,
α_i = −exp(π_p + σ_p ν₁ᵢ), a random coefficient σ_n on log pack size, and sale regimes built
from three-week sequences of product-store-week price states, interacted with pack size.
Supply is static Bertrand-Nash; marginal costs are recovered from weeks without a sale.

## Running

The runs were made on Northeastern's Discovery cluster, one node with 24 cores and 96 GB,
Python 3.12 (anaconda3/2024.06). Python packages: numpy, pandas, scipy, pyarrow, pyblp,
pyhdfe.

| Command | Estimates | Log |
|---|---|---|
| `sbatch run_main.sbatch` | pos1: reduced forms, demand columns (i)–(v), elasticities, supply, the four pricing regimes, sales at observed prices, one firm removing ITS | `logs/main_10915696.log` and one log per step |
| `sbatch run_positions.sbatch` | pos2–pos4: column (iv) | `logs/positions_10915697.log`, `logs/positions/` |
| `python3 code/boot_draw.py --prep` | parameter covariance for the bootstrap | `logs/boot_prep.log` |
| `python3 code/boot_draw.py --draws 0 1 … 150` | bootstrap welfare draws | `logs/boot_draw0.log`, `boot_draws.log`, `boot_draws_147.log` |
| `python3 code/boot_draw.py --pq --draws 0 1 … 150` | bootstrap prices, quantities and incidence | `logs/boot_pq0.log`, `boot_pq.log`, `boot_pq_rev.log` |
| `python3 code/f71_fixtheta_weekfe.py` | sale regime terms with week effects | `logs/weekfe.log` |
| `bash run_jdiag.sh` | decomposition of the Hansen J, pos1–pos4 | `logs/jdiag.log`, `logs/positions/` |

`run_main.sbatch` runs prepare → column (i)–(iv) in parallel → elasticities, column (v) and
supply → counterfactuals → firm profits, sales at observed prices, rivals re-optimizing. It
took about 17 hours. With `RESUME=1` it finishes an interrupted run, skipping completed
steps; without it, it first moves earlier `estimation/`, `results/` and `logs/` aside.
`run_positions.sbatch` needs `data/positions/`; `run_jdiag.sh` needs the column (iv)
estimates of all four samples.

Each bootstrap draw takes about 10 minutes and draws the parameters with seed b, so the two
modes share their draws; a draw already on disk is skipped. Draw 0 is the point estimate.
Outputs go to `estimation/` (demand), `results/` (everything else) and `results/boot/`.

Some logs open with a failed first attempt (a NumPy version clash on the cluster) followed
by the completed run.

## Contents

| Path | |
|---|---|
| `code/` | the estimation code: `prepare_data.py`, the `Est_*.py` / `RF_RMS.py` / `Data_HMS.py` steps and the `f*.py` modules they run, `boot_draw.py` and `cf_core.py` (bootstrap), `f71_fixtheta_*.py` (column (v), week effects), `f71_jdiag.py` |
| `build/` | the sample builds from raw RMS and Homescan |
| `data/` | the ACS file |
| `logs/` | the logs of the article's runs |
| `run_main.sbatch`, `run_positions.sbatch`, `run_jdiag.sh` | the cluster jobs |
