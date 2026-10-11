#!/usr/bin/env python3
"""
Construct Nielsen RMS beer panel for Bayesian demand estimation.

Design choices implemented:
  1. No broad pack aggregation.
  2. Keep exact pack sizes k in {6, 12, 18, 24, 30, 36}.
  3. Keep CAN and BOTTLE as separate products.
  4. Keep zero-sale weeks by building a balanced product-store-week panel.
  5. Use quantity-weighted promo_share instead of max(on_promo).
  6. Keep only focal brands in the final estimation panel.
  7. Compute store-week controls from all beer before dropping non-focal brands:
       - other_beer_share_store = non-focal beer can-units / all beer can-units
       - other_beer_price_wavg  = non-focal beer revenue / non-focal beer can-units

Important:
  This script builds the clean base panel. Your Bayesian script still needs a
  separate regime-construction step to define R_jst and any exact-pack version
  of G_SM, G_ML, G_SL, D_S_rise, and D_M_rise.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

DEFAULT_RMS = Path("/Volumes/LaCie/Beer/nielsen_extracts/RMS")
DEFAULT_OUT = Path("/Volumes/LaCie/Beer/estimation_exact_pack_container")
DEFAULT_YEARS = (2018, 2019)
DEFAULT_MODULES = (5000, 5001, 5005, 5010, 5015, 5020)

NIELSEN_ENCODING = "latin1"

TARGET_PACKS = (6, 12, 18, 24, 30, 36)

FOCAL_BRANDS = {
    "BUDWEISER",
    "BUD LIGHT",
    "BUD LIGHT LIME",
    "BUD LIGHT ORANGE",
    "CORONA EXTRA",
    "CORONA LIGHT",
    "MODELO ESPECIAL",
    "MODELO NEGRA",
    "COORS LIGHT",
    "COORS BANQUET",
    "MILLER LITE",
    "MILLER HIGH LIFE",
    "HEINEKEN",
    "MICHELOB ULTRA",
}


# ============================================================
# Basic utilities
# ============================================================

def log(message: str) -> None:
    print(message, flush=True)


def clean_upc(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def write_table(df: pd.DataFrame, path_without_suffix: Path, write_csv: bool = True) -> None:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)

    parquet_path = path_without_suffix.with_suffix(".parquet")
    df.to_parquet(parquet_path, index=False)
    log(f"  Wrote {parquet_path}")

    if write_csv:
        csv_path = path_without_suffix.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        log(f"  Wrote {csv_path}")


def read_header(path: Path) -> list[str]:
    return pd.read_csv(path, sep="\t", nrows=0, encoding=NIELSEN_ENCODING).columns.tolist()


def standardize_brand(value: object) -> str:
    brand = "" if pd.isna(value) else str(value).strip().upper()

    if brand == "BUDWEISER":
        return "BUDWEISER"
    if re.search(r"BUD LIGHT$", brand):
        return "BUD LIGHT"
    if re.search(r"BUD LT LIME|BUD LIGHT LIME", brand):
        return "BUD LIGHT LIME"
    if re.search(r"BUD LT ORANGE|BUD LIGHT ORANGE", brand):
        return "BUD LIGHT ORANGE"
    if re.search(r"CORONA EXTRA", brand):
        return "CORONA EXTRA"
    if re.search(r"CORONA LIGHT", brand):
        return "CORONA LIGHT"
    if re.search(r"MODELO ESP", brand):
        return "MODELO ESPECIAL"
    if re.search(r"MODELO NEG|NEGRA MODELO", brand):
        return "MODELO NEGRA"
    if re.search(r"COORS LIGHT", brand):
        return "COORS LIGHT"
    if re.search(r"COORS BANQUET", brand) or brand == "COORS":
        return "COORS BANQUET"
    if re.search(r"MILLER LITE", brand):
        return "MILLER LITE"
    if re.search(r"MILLER HIGH LIFE|MHL", brand):
        return "MILLER HIGH LIFE"
    if brand == "HEINEKEN":
        return "HEINEKEN"
    if re.search(r"MICHELOB ULTRA", brand):
        return "MICHELOB ULTRA"

    return brand


def infer_container_from_text(text: object) -> object:
    """
    Infer CAN/BOTTLE from product text.

    This is deliberately conservative. Unknowns are dropped from the focal
    estimation panel, but they can be inspected in the unknown-container output.
    """
    if pd.isna(text):
        return pd.NA

    s = str(text).upper()

    # Bottles first because text may include "CAN" in unrelated strings less often,
    # but either way we drop ambiguous cases below. Nielsen UPC descriptions often
    # use compact package codes: NRB/RB for non-returnable/returnable bottles and
    # LN for longneck bottles.
    bottle = bool(re.search(r"\bBOTTLE(S)?\b|\bBTL(S)?\b|\bGLASS\b|\bNRB\b|\bRB\b|\bLN\b", s))
    can = bool(re.search(r"\bCAN(S)?\b|\bCN(S)?\b|\bCANNED\b", s))

    if bottle and not can:
        return "BOTTLE"
    if can and not bottle:
        return "CAN"
    return pd.NA


def find_container_text_columns(products_path: Path) -> list[str]:
    """
    Pick product columns likely to contain package/container descriptions.

    Nielsen extract column names can differ across vintages. This function reads
    the header and keeps broad description/package fields when available.
    """
    columns = read_header(products_path)

    base = {
        "upc",
        "upc_ver_uc",
        "brand_code_uc",
        "brand_descr",
        "department_code",
        "product_module_code",
        "multi",
        "size1_amount",
        "size1_units",
    }

    patterns = [
        "descr",
        "desc",
        "package",
        "pack",
        "container",
        "form",
        "type",
    ]

    extra = []
    for col in columns:
        low = col.lower()
        if col in base:
            continue
        if any(p in low for p in patterns):
            extra.append(col)

    return sorted(base.intersection(columns)) + sorted(set(extra))


# ============================================================
# Path helpers
# ============================================================

def movement_dir_for_year(rms: Path, year: int, preferred_module_dir: int) -> Path:
    preferred = rms / str(year) / "Movement_Files" / f"{preferred_module_dir}_{year}"
    if preferred.exists():
        return preferred

    movement_root = rms / str(year) / "Movement_Files"
    if movement_root.exists():
        candidates = sorted(movement_root.glob(f"*_{year}"))
        if candidates:
            return candidates[0]

    return preferred


# ============================================================
# Product lookup
# ============================================================

def build_upc_lookup(master_dir: Path, out_dir: Path, write_csv: bool) -> pd.DataFrame:
    log("=" * 70)
    log("Step 1: Building UPC lookup")
    log("=" * 70)

    products_path = master_dir / "products.tsv"
    log(f"Products file: {products_path}")

    usecols = find_container_text_columns(products_path)

    required = [
        "upc",
        "brand_descr",
        "department_code",
        "product_module_code",
        "multi",
        "size1_amount",
        "size1_units",
    ]
    missing = [c for c in required if c not in usecols]
    if missing:
        raise ValueError(f"Missing required product columns: {missing}")

    products = pd.read_csv(
        products_path,
        sep="\t",
        usecols=usecols,
        dtype="string",
        encoding=NIELSEN_ENCODING,
        low_memory=False,
    )

    products["upc"] = clean_upc(products["upc"])

    products["department_code"] = pd.to_numeric(products["department_code"], errors="coerce")
    products["product_module_code"] = pd.to_numeric(products["product_module_code"], errors="coerce")
    products["multi"] = pd.to_numeric(products["multi"], errors="coerce")
    products["size1_amount"] = pd.to_numeric(products["size1_amount"], errors="coerce")

    # Beer modules, 11-13 oz unit size.
    products = products.loc[
        (products["department_code"] == 8)
        & products["product_module_code"].between(5000, 5020, inclusive="both")
        & products["size1_amount"].between(11, 13, inclusive="both")
        & (products["size1_units"].astype("string").str.upper() == "OZ")
        & products["multi"].notna()
        & products["multi"].gt(0)
    ].copy()

    products["n_cans"] = products["multi"].round().astype("Int64")
    products = products.loc[products["n_cans"].notna()].copy()

    products["brand_raw"] = products["brand_descr"].astype("string")
    products["brand_descr"] = products["brand_descr"].map(standardize_brand)
    products["is_focal_brand"] = products["brand_descr"].isin(FOCAL_BRANDS)
    products["in_target_pack"] = products["n_cans"].isin(TARGET_PACKS)

    # Conservative container inference from all likely text fields.
    text_cols = [
        c for c in products.columns
        if c not in {
            "department_code",
            "product_module_code",
            "multi",
            "size1_amount",
            "n_cans",
            "is_focal_brand",
            "in_target_pack",
        }
    ]

    products["container_text"] = (
        products[text_cols]
        .astype("string")
        .fillna("")
        .agg(" ".join, axis=1)
    )

    products["container"] = products["container_text"].map(infer_container_from_text).astype("string")
    products["known_container"] = products["container"].isin(["CAN", "BOTTLE"])

    products["oz_per_unit"] = products["size1_amount"].astype(float)
    products["oz_per_pack"] = products["n_cans"].astype(float) * products["oz_per_unit"]

    lookup_cols = [
        "upc",
        "brand_raw",
        "brand_descr",
        "brand_code_uc",
        "product_module_code",
        "n_cans",
        "oz_per_unit",
        "oz_per_pack",
        "container",
        "container_text",
        "known_container",
        "is_focal_brand",
        "in_target_pack",
    ]
    lookup_cols = [c for c in lookup_cols if c in products.columns]

    lookup = (
        products[lookup_cols]
        .drop_duplicates(subset=["upc"])
        .sort_values("upc")
        .reset_index(drop=True)
    )

    log(f"Beer UPCs after beer/unit filters: {len(lookup):,}")
    log(f"Known CAN/BOTTLE container UPCs: {int(lookup['known_container'].sum()):,}")
    log(f"Focal-brand UPCs: {int(lookup['is_focal_brand'].sum()):,}")
    log(f"Focal-brand target-pack UPCs: {int((lookup['is_focal_brand'] & lookup['in_target_pack']).sum()):,}")

    log("\nFocal UPC counts by brand, pack, container:")
    tab = pd.crosstab(
        [lookup.loc[lookup["is_focal_brand"], "brand_descr"],
         lookup.loc[lookup["is_focal_brand"], "n_cans"]],
        lookup.loc[lookup["is_focal_brand"], "container"],
        dropna=False,
    )
    log(str(tab))

    write_table(lookup, out_dir / "upc_lookup_exact_container", write_csv=write_csv)

    unknown = lookup.loc[
        lookup["is_focal_brand"] & lookup["in_target_pack"] & ~lookup["known_container"]
    ].copy()
    if len(unknown):
        write_table(unknown, out_dir / "focal_target_upcs_unknown_container", write_csv=True)
        log(
            "\nWARNING: Some focal target-pack UPCs have unknown container. "
            "Inspect focal_target_upcs_unknown_container.csv."
        )

    return lookup


def load_valid_upcs(annual_dir: Path, year: int) -> pd.DataFrame:
    path = annual_dir / f"rms_versions_{year}.tsv"
    versions = pd.read_csv(
        path,
        sep="\t",
        dtype={"upc": "string", "upc_ver_uc": "string"},
        usecols=["upc", "upc_ver_uc"],
        encoding=NIELSEN_ENCODING,
        low_memory=False,
    )
    versions["upc"] = clean_upc(versions["upc"])
    return versions.drop_duplicates(subset=["upc"]).sort_values("upc")


def build_beer_upcs_for_year(lookup: pd.DataFrame, annual_dir: Path, year: int) -> pd.DataFrame:
    versions = load_valid_upcs(annual_dir, year)

    beer_upcs = lookup.merge(versions[["upc"]], on="upc", how="inner").copy()

    # Keep all known beer UPCs for other-beer controls.
    beer_upcs = beer_upcs.loc[beer_upcs["known_container"]].copy()

    log(f"Valid known-container beer UPCs in {year}: {len(beer_upcs):,}")
    log(f"Valid focal target-pack UPCs in {year}: {int((beer_upcs['is_focal_brand'] & beer_upcs['in_target_pack']).sum()):,}")

    return beer_upcs


# ============================================================
# Movement loading
# ============================================================

def parse_week_end(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(8)
    return pd.to_datetime(text, format="%Y%m%d", errors="coerce")


def movement_chunks(path: Path, beer_upcs: pd.DataFrame, chunksize: int) -> Iterable[pd.DataFrame]:
    meta_cols = [
        "upc",
        "brand_descr",
        "brand_raw",
        "n_cans",
        "oz_per_unit",
        "oz_per_pack",
        "container",
        "is_focal_brand",
        "in_target_pack",
    ]
    meta_cols = [c for c in meta_cols if c in beer_upcs.columns]
    beer_meta = beer_upcs[meta_cols].drop_duplicates(subset=["upc"]).copy()

    dtype = {
        "store_code_uc": "string",
        "upc": "string",
        "week_end": "string",
        "units": "float64",
        "prmult": "float64",
        "price": "float64",
        "feature": "float64",
        "display": "float64",
    }

    usecols = list(dtype.keys())
    available = read_header(path)
    usecols = [c for c in usecols if c in available]
    missing_required = [c for c in ["store_code_uc", "upc", "week_end", "units", "prmult", "price"] if c not in usecols]
    if missing_required:
        raise ValueError(f"{path} is missing required movement columns: {missing_required}")

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype={c: dtype[c] for c in usecols},
        usecols=usecols,
        chunksize=chunksize,
        encoding=NIELSEN_ENCODING,
        low_memory=False,
    ):
        chunk["upc"] = clean_upc(chunk["upc"])
        chunk = chunk.merge(beer_meta, on="upc", how="inner")
        if chunk.empty:
            continue

        chunk["store_code_uc"] = chunk["store_code_uc"].astype("string").str.strip()
        chunk["week_end"] = parse_week_end(chunk["week_end"])

        chunk["prmult"] = pd.to_numeric(chunk["prmult"], errors="coerce").fillna(1.0)
        chunk.loc[chunk["prmult"].le(0), "prmult"] = 1.0

        chunk["units"] = pd.to_numeric(chunk["units"], errors="coerce")
        chunk["price"] = pd.to_numeric(chunk["price"], errors="coerce")
        chunk["price_per_pack"] = chunk["price"] / chunk["prmult"]

        # Movement files only record positive observed purchases.
        # Zero-sales weeks are created later in balance_panel().
        chunk = chunk.loc[
            chunk["units"].gt(0)
            & chunk["price_per_pack"].between(0.50, 200, inclusive="both")
            & chunk["week_end"].notna()
        ].copy()
        if chunk.empty:
            continue

        if "feature" in chunk.columns:
            feature = chunk["feature"].fillna(0).astype(float).gt(0)
        else:
            feature = pd.Series(False, index=chunk.index)

        if "display" in chunk.columns:
            display = chunk["display"].fillna(0).astype(float).gt(0)
        else:
            display = pd.Series(False, index=chunk.index)

        chunk["on_promo_upc"] = (feature | display).astype("int8")

        chunk["n_cans"] = pd.to_numeric(chunk["n_cans"], errors="coerce")
        chunk["price_per_can"] = chunk["price_per_pack"] / chunk["n_cans"]
        chunk["revenue"] = chunk["price_per_pack"] * chunk["units"]
        chunk["can_units"] = chunk["units"] * chunk["n_cans"]
        chunk["price_per_oz"] = chunk["price_per_can"] / chunk["oz_per_unit"].astype(float)

        chunk = chunk.loc[
            chunk["price_per_can"].notna()
            & chunk["can_units"].gt(0)
            & chunk["container"].isin(["CAN", "BOTTLE"])
        ].copy()
        if chunk.empty:
            continue

        yield chunk[
            [
                "store_code_uc",
                "upc",
                "week_end",
                "units",
                "price_per_pack",
                "price_per_can",
                "price_per_oz",
                "revenue",
                "can_units",
                "on_promo_upc",
                "brand_descr",
                "brand_raw",
                "n_cans",
                "oz_per_unit",
                "oz_per_pack",
                "container",
                "is_focal_brand",
                "in_target_pack",
            ]
        ]


# ============================================================
# Aggregation and controls
# ============================================================

def aggregate_focal_products(upc_level: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate only focal brands at exact pack x container level.

    Uses promo_share = sum(can_units * on_promo) / sum(can_units).
    """
    df = upc_level.loc[
        upc_level["is_focal_brand"]
        & upc_level["in_target_pack"]
        & upc_level["container"].isin(["CAN", "BOTTLE"])
        & upc_level["n_cans"].isin(TARGET_PACKS)
    ].copy()

    if df.empty:
        return pd.DataFrame()

    df["promo_qty"] = df["can_units"] * df["on_promo_upc"]

    group_cols = [
        "store_code_uc",
        "week_end",
        "brand_descr",
        "n_cans",
        "container",
    ]

    out = (
        df.groupby(group_cols, dropna=False, observed=True)
        .agg(
            qty=("can_units", "sum"),
            pack_units=("units", "sum"),
            revenue=("revenue", "sum"),
            promo_qty=("promo_qty", "sum"),
            n_upcs=("upc", "nunique"),
            oz_per_unit=("oz_per_unit", "first"),
            oz_per_pack=("oz_per_pack", "first"),
        )
        .reset_index()
    )

    out["price_per_can"] = out["revenue"] / out["qty"]
    out["price_per_pack_equiv"] = out["price_per_can"] * out["n_cans"].astype(float)
    out["promo_share"] = out["promo_qty"] / out["qty"]
    out["k"] = out["n_cans"].astype(int).astype(str) + "_" + out["container"].astype(str)

    out["observed_sales_week"] = 1

    return out.drop(columns=["promo_qty"])


def build_other_beer_controls(upc_level: pd.DataFrame) -> pd.DataFrame:
    """
    Store-week controls from all beer sales before dropping non-focal brands.
    Other beer = all non-focal beer sales in the same store-week.
    """
    df = upc_level.copy()

    df["other_beer_qty_component"] = np.where(
        df["is_focal_brand"],
        0.0,
        df["can_units"],
    )
    df["other_beer_revenue_component"] = np.where(
        df["is_focal_brand"],
        0.0,
        df["revenue"],
    )

    controls = (
        df.groupby(["store_code_uc", "week_end"], dropna=False, observed=True)
        .agg(
            total_beer_qty=("can_units", "sum"),
            total_beer_revenue=("revenue", "sum"),
            other_beer_qty=("other_beer_qty_component", "sum"),
            other_beer_revenue=("other_beer_revenue_component", "sum"),
            total_beer_upcs=("upc", "nunique"),
        )
        .reset_index()
    )

    controls["other_beer_share_store"] = controls["other_beer_qty"] / controls["total_beer_qty"]

    controls["other_beer_price_wavg"] = np.where(
        controls["other_beer_qty"].gt(0),
        controls["other_beer_revenue"] / controls["other_beer_qty"],
        np.nan,
    )

    controls["total_beer_price_wavg"] = controls["total_beer_revenue"] / controls["total_beer_qty"]

    return controls


def merge_stores(df: pd.DataFrame, annual_dir: Path, year: int) -> pd.DataFrame:
    store_cols = [
        "store_code_uc",
        "retailer_code",
        "parent_code",
        "channel_code",
        "store_zip3",
        "fips_state_code",
        "fips_state_descr",
        "fips_county_code",
        "fips_county_descr",
        "dma_code",
        "dma_descr",
    ]

    stores_path = annual_dir / f"stores_{year}.tsv"
    available = read_header(stores_path)
    usecols = [c for c in store_cols if c in available]

    stores = pd.read_csv(
        stores_path,
        sep="\t",
        dtype={c: "string" for c in usecols},
        usecols=usecols,
        encoding=NIELSEN_ENCODING,
        low_memory=False,
    )

    stores["store_code_uc"] = stores["store_code_uc"].astype("string").str.strip()

    # Make the helper safe to call on frames that already include store metadata.
    # This happens after balancing because focal_observed has already been merged
    # with stores, so the left merge below would otherwise create duplicate
    # *_x/*_y columns and fail when inserting year.
    existing_store_cols = [c for c in store_cols if c != "store_code_uc" and c in df.columns]
    base = df.drop(columns=existing_store_cols).copy()

    merged = base.merge(stores, on="store_code_uc", how="left")

    # Keep food channel, as in the previous construction.
    if "channel_code" in merged.columns:
        merged = merged.loc[merged["channel_code"].eq("F")].copy()

    if "year" in merged.columns:
        merged["year"] = merged["year"].fillna(year).astype(int)
    else:
        merged.insert(0, "year", year)
    return merged


# ============================================================
# Balancing and price variables
# ============================================================

def balance_panel(
    focal_observed: pd.DataFrame,
    other_controls: pd.DataFrame,
    min_observed_weeks_per_product_store: int = 4,
    global_balance: bool = False,
) -> pd.DataFrame:
    """
    Build balanced product-store-week panel.

    Product-store cells are observed focal brand x exact-pack x container cells.
    Missing product-week sales are filled as qty=0.

    If global_balance=False, each product-store is expanded only from its first
    observed sale week to its last observed sale week. This avoids treating
    pre-entry/post-exit weeks as zero sales.

    If global_balance=True, each product-store is expanded to all sample weeks.
    """
    if focal_observed.empty:
        return pd.DataFrame()

    focal_observed = focal_observed.copy()
    focal_observed["week_end"] = pd.to_datetime(focal_observed["week_end"])

    all_weeks = pd.DataFrame(
        {"week_end": sorted(other_controls["week_end"].dropna().unique())}
    )
    all_weeks["week_end"] = pd.to_datetime(all_weeks["week_end"])
    all_weeks["week_key"] = 1

    product_cols = [
        "store_code_uc",
        "brand_descr",
        "n_cans",
        "container",
        "k",
        "oz_per_unit",
        "oz_per_pack",
    ]

    product_stats = (
        focal_observed.groupby(product_cols, dropna=False, observed=True)
        .agg(
            first_week=("week_end", "min"),
            last_week=("week_end", "max"),
            observed_weeks=("week_end", "nunique"),
        )
        .reset_index()
    )

    product_stats = product_stats.loc[
        product_stats["observed_weeks"] >= min_observed_weeks_per_product_store
    ].copy()

    product_stats["week_key"] = 1

    panel = product_stats.merge(all_weeks, on="week_key", how="inner").drop(columns=["week_key"])

    if not global_balance:
        panel = panel.loc[
            panel["week_end"].between(panel["first_week"], panel["last_week"])
        ].copy()

    merge_cols = product_cols + ["week_end"]

    balanced = panel.merge(
        focal_observed,
        on=merge_cols,
        how="left",
        suffixes=("", "_obs"),
    )

    balanced["observed_sales_week"] = balanced["observed_sales_week"].fillna(0).astype("int8")

    fill_zero_cols = ["qty", "pack_units", "revenue", "n_upcs"]
    for col in fill_zero_cols:
        if col in balanced.columns:
            balanced[col] = balanced[col].fillna(0.0)

    balanced["promo_share"] = balanced["promo_share"].fillna(0.0)

    # Keep original observed price indicator before imputation.
    balanced["price_observed"] = balanced["price_per_can"].notna().astype("int8")

    # Impute missing prices only within the same store-brand-pack-container product.
    # This is necessary because zero-sale weeks have no transaction price.
    balanced = balanced.sort_values(["store_code_uc", "brand_descr", "k", "week_end"]).copy()

    group_keys = ["store_code_uc", "brand_descr", "k"]

    balanced["price_per_can_observed"] = balanced["price_per_can"]

    balanced["price_per_can"] = (
        balanced.groupby(group_keys, observed=True)["price_per_can"]
        .transform(lambda x: x.ffill().bfill())
    )

    balanced["price_imputed"] = (
        balanced["price_observed"].eq(0)
        & balanced["price_per_can"].notna()
    ).astype("int8")

    balanced["price_per_pack_equiv"] = balanced["price_per_can"] * balanced["n_cans"].astype(float)

    balanced["ln_Q"] = np.log1p(balanced["qty"])
    balanced["log_qty"] = balanced["ln_Q"]

    # Merge store-week other-beer controls after balancing.
    balanced = balanced.merge(
        other_controls,
        on=["store_code_uc", "week_end"],
        how="left",
    )

    # Drop product-weeks for which the store-week has no observed beer market.
    balanced = balanced.loc[balanced["total_beer_qty"].notna()].copy()

    return balanced


def add_price_gap_variables(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add exact-pack price variables useful for the later regime-construction step.

    The current Bayesian code was originally written for three pack categories
    and variables G_SM, G_ML, G_SL. With six exact pack sizes, it is better to
    use adjacent exact-pack gaps. This function creates both exact gaps and a
    backward-compatible 6/12/24 version of G_SM, G_ML, G_SL.
    """
    if panel.empty:
        return panel

    df = panel.copy()

    index_cols = ["store_code_uc", "week_end", "brand_descr", "container"]

    price_wide = (
        df.pivot_table(
            index=index_cols,
            columns="n_cans",
            values="price_per_can",
            aggfunc="mean",
            observed=True,
        )
        .rename(columns=lambda c: f"price_{int(c)}")
        .reset_index()
    )

    for pack in TARGET_PACKS:
        col = f"price_{pack}"
        if col not in price_wide.columns:
            price_wide[col] = np.nan

    # Adjacent per-can volume-discount gaps.
    # Positive value means the smaller pack is more expensive per can.
    price_wide["gap_6_12"] = price_wide["price_6"] - price_wide["price_12"]
    price_wide["gap_12_18"] = price_wide["price_12"] - price_wide["price_18"]
    price_wide["gap_18_24"] = price_wide["price_18"] - price_wide["price_24"]
    price_wide["gap_24_30"] = price_wide["price_24"] - price_wide["price_30"]
    price_wide["gap_30_36"] = price_wide["price_30"] - price_wide["price_36"]

    # Backward-compatible columns for the current Bayesian code.
    # Interpret S=6, M=12, L=24.
    price_wide["G_SM"] = price_wide["price_6"] - price_wide["price_12"]
    price_wide["G_ML"] = price_wide["price_12"] - price_wide["price_24"]
    price_wide["G_SL"] = price_wide["price_6"] - price_wide["price_24"]

    # Average within brand-store-week-container price across target packs.
    price_cols = [f"price_{p}" for p in TARGET_PACKS]
    price_wide["p_K_bar"] = price_wide[price_cols].mean(axis=1, skipna=True)

    df = df.merge(price_wide, on=index_cols, how="left")

    # Bayesian code expects p_tilde.
    df["p_tilde"] = df["price_per_can"]

    # Price movement indicators for exact product.
    group_keys = ["store_code_uc", "brand_descr", "k"]

    df = df.sort_values(group_keys + ["week_end"]).copy()
    df["price_lag"] = df.groupby(group_keys, observed=True)["price_per_can"].shift(1)
    df["price_lead"] = df.groupby(group_keys, observed=True)["price_per_can"].shift(-1)

    df["price_drop_vs_lag"] = (
        (df["price_per_can"] < df["price_lag"]).fillna(False).astype("int8")
    )
    df["price_rise_vs_lag"] = (
        (df["price_per_can"] > df["price_lag"]).fillna(False).astype("int8")
    )
    df["price_rise_next"] = (
        (df["price_lead"] > df["price_per_can"]).fillna(False).astype("int8")
    )

    # Backward-compatible names used by the old MIX design.
    # These are exact-product analogues, not the old three-bin definitions.
    df["D_S_rise"] = np.where(df["n_cans"].eq(6), df["price_rise_next"], 0).astype("int8")
    df["D_M_rise"] = np.where(df["n_cans"].eq(12), df["price_rise_next"], 0).astype("int8")

    return df


# ============================================================
# Year build
# ============================================================

def build_year(
    rms: Path,
    out_dir: Path,
    lookup: pd.DataFrame,
    year: int,
    modules: tuple[int, ...],
    chunksize: int,
    preferred_movement_dir_module: int,
    save_upc_level: bool,
    write_csv: bool,
    min_observed_weeks_per_product_store: int,
    global_balance: bool,
) -> pd.DataFrame:
    log("")
    log("=" * 70)
    log(f"Building {year}")
    log("=" * 70)

    annual_dir = rms / str(year) / "Annual_Files"
    movement_dir = movement_dir_for_year(rms, year, preferred_movement_dir_module)
    log(f"Movement path: {movement_dir}")

    beer_upcs = build_beer_upcs_for_year(lookup, annual_dir, year)

    yearly_chunks: list[pd.DataFrame] = []

    for module in modules:
        path = movement_dir / f"{module}_{year}.tsv"
        if not path.exists():
            log(f"  Missing module {module}: {path}")
            continue

        log(f"  Loading module {module}...")
        module_chunks = []
        for chunk in movement_chunks(path, beer_upcs, chunksize):
            module_chunks.append(chunk)

        if module_chunks:
            module_df = pd.concat(module_chunks, ignore_index=True)
            yearly_chunks.append(module_df)
            log(f"    Observed beer movement rows: {len(module_df):,}")
        else:
            log("    No matching rows")

    if not yearly_chunks:
        raise RuntimeError(f"No movement rows loaded for {year}")

    upc_level = pd.concat(yearly_chunks, ignore_index=True)

    if save_upc_level:
        write_table(upc_level, out_dir / f"movement_beer_upc_level_exact_{year}", write_csv=False)

    log(f"Total observed beer movement rows in {year}: {len(upc_level):,}")
    log(f"Observed focal target-pack rows in {year}: {int((upc_level['is_focal_brand'] & upc_level['in_target_pack']).sum()):,}")

    focal_observed = aggregate_focal_products(upc_level)
    other_controls = build_other_beer_controls(upc_level)

    focal_observed = merge_stores(focal_observed, annual_dir, year)
    other_controls = merge_stores(other_controls, annual_dir, year)

    # Remove duplicate inserted year in other_controls if merge_stores added it.
    if "year" in other_controls.columns:
        other_controls = other_controls.drop(columns=["year"])

    log(f"Observed focal product rows after food-channel filter: {len(focal_observed):,}")
    log(f"Store-week beer control rows after food-channel filter: {len(other_controls):,}")

    balanced = balance_panel(
        focal_observed=focal_observed,
        other_controls=other_controls,
        min_observed_weeks_per_product_store=min_observed_weeks_per_product_store,
        global_balance=global_balance,
    )

    balanced = merge_stores(balanced, annual_dir, year)

    # If merge_stores inserted duplicate year because balanced already has year, clean it.
    year_cols = [c for c in balanced.columns if c == "year" or c.startswith("year_")]
    if "year" not in balanced.columns:
        balanced.insert(0, "year", year)
    for c in year_cols:
        if c != "year" and c in balanced.columns:
            balanced = balanced.drop(columns=[c])

    balanced = add_price_gap_variables(balanced)

    # Helpful product identifiers.
    balanced["product_id"] = (
        balanced["brand_descr"].astype(str)
        + "_"
        + balanced["n_cans"].astype(int).astype(str)
        + "_"
        + balanced["container"].astype(str)
    )

    balanced["cell_id"] = (
        balanced["store_code_uc"].astype(str)
        + "_"
        + balanced["brand_descr"].astype(str)
    )

    balanced["cell_pack_id"] = (
        balanced["store_code_uc"].astype(str)
        + "_"
        + balanced["brand_descr"].astype(str)
        + "_"
        + balanced["k"].astype(str)
    )

    sort_cols = ["store_code_uc", "brand_descr", "k", "week_end"]
    balanced = balanced.sort_values(sort_cols).reset_index(drop=True)

    log(f"Balanced focal panel rows: {len(balanced):,}")
    log(f"Unique stores: {balanced['store_code_uc'].nunique():,}")
    log(f"Unique product cells store-brand-k: {balanced['cell_pack_id'].nunique():,}")
    log(f"Zero-sale rows: {int((balanced['qty'] == 0).sum()):,}")

    write_table(balanced, out_dir / f"beer_exact_container_balanced_{year}", write_csv=write_csv)

    return balanced


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build exact-pack, container-separated Nielsen beer panel."
    )

    parser.add_argument("--rms", type=Path, default=DEFAULT_RMS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)

    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=list(DEFAULT_YEARS),
        help="Years to build. Default: 2018 2019",
    )

    parser.add_argument(
        "--modules",
        type=int,
        nargs="+",
        default=list(DEFAULT_MODULES),
        help="Product modules to load from movement files.",
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=1_000_000,
        help="Rows per movement-file chunk.",
    )

    parser.add_argument(
        "--movement-dir-module",
        type=int,
        default=5001,
        help="Preferred Movement_Files subdirectory prefix, e.g. 5001 for 5001_2019.",
    )

    parser.add_argument(
        "--min-observed-weeks-per-product-store",
        type=int,
        default=4,
        help=(
            "Minimum positive-sales weeks required for a store-brand-pack-container "
            "cell before balancing. Default: 4."
        ),
    )

    parser.add_argument(
        "--global-balance",
        action="store_true",
        help=(
            "Balance each product-store over all sample weeks. Default is to balance "
            "only from first observed sale week to last observed sale week."
        ),
    )

    parser.add_argument(
        "--save-upc-level",
        action="store_true",
        help="Write large UPC-level observed movement files.",
    )

    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Write parquet only. Default writes parquet and CSV for final panels.",
    )

    args, unknown = parser.parse_known_args()

    # Jupyter/IPython injects "-f <kernel.json>" into sys.argv. Ignore only that
    # known notebook argument so normal command-line typos still fail loudly.
    remaining_unknown = []
    i = 0
    while i < len(unknown):
        if (
            unknown[i] == "-f"
            and i + 1 < len(unknown)
            and unknown[i + 1].endswith(".json")
            and "jupyter" in unknown[i + 1].lower()
        ):
            i += 2
            continue
        remaining_unknown.append(unknown[i])
        i += 1

    if remaining_unknown:
        parser.error(f"unrecognized arguments: {' '.join(remaining_unknown)}")

    return args


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    write_csv = not args.no_csv

    master_dir = args.rms / "Master_Files_2006-2020" / "Latest"

    lookup = build_upc_lookup(master_dir, args.out, write_csv=write_csv)

    final_frames = []
    for year in args.years:
        final_frames.append(
            build_year(
                rms=args.rms,
                out_dir=args.out,
                lookup=lookup,
                year=year,
                modules=tuple(args.modules),
                chunksize=args.chunksize,
                preferred_movement_dir_module=args.movement_dir_module,
                save_upc_level=args.save_upc_level,
                write_csv=write_csv,
                min_observed_weeks_per_product_store=args.min_observed_weeks_per_product_store,
                global_balance=args.global_balance,
            )
        )

    if len(final_frames) > 1:
        combined = pd.concat(final_frames, ignore_index=True)
        years_label = "_".join(str(y) for y in args.years)

        # Re-sort after combining.
        combined = combined.sort_values(
            ["store_code_uc", "brand_descr", "k", "week_end"]
        ).reset_index(drop=True)

        write_table(
            combined,
            args.out / f"beer_exact_container_balanced_{years_label}",
            write_csv=write_csv,
        )

        log("")
        log("=" * 70)
        log("Combined panel summary")
        log("=" * 70)
        log(f"Rows: {len(combined):,}")
        log(f"Stores: {combined['store_code_uc'].nunique():,}")
        log(f"Brands: {combined['brand_descr'].nunique():,}")
        log(f"Products brand-pack-container: {combined['product_id'].nunique():,}")
        log(f"Store-brand-pack-container cells: {combined['cell_pack_id'].nunique():,}")
        log(f"Zero-sale rows: {int((combined['qty'] == 0).sum()):,}")
        log(f"Output directory: {args.out}")


if __name__ == "__main__":
    main()
