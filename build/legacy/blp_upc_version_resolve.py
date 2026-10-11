#!/usr/bin/env python3
"""Round 36 Part 1: resolve the UPC version question and build a corrected lookup.

Round 35 found that `create_upc_lookup` joins products.tsv on `upc` alone and keeps the
first row in file order, while `rms_versions_<year>.tsv` names the version present that
year.  It left five UPCs where the year-valid version's `brand_descr` contradicted its own
`upc_descr`, and named `products_extra_<year>.tsv` as the place to resolve them.

It does resolve them, and not all in the same direction.  `products_extra` is keyed on
(upc, upc_ver_uc, panel_year) and carries `flavor_descr` and `container_descr`, neither of
which is a version-generating "core" attribute, so it describes the physical product
independently of the brand coding that generated the version.

This part also replaces the regex container inference with `container_descr`, which is an
observed field rather than a guess, and reports how often the two disagree.

NO PANEL IS REBUILT HERE.  The corrected lookup is written for Part 2 to use.

    python3 paper/src/diagnostics/blp_upc_version_resolve.py
"""
from __future__ import annotations
import importlib.util, json, sys, time
from pathlib import Path
import _paths
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
PAPER = _paths.WORK
sys.path.insert(0, str(PAPER / "src" / "common"))
from run_manifest import RunManifest        # noqa: E402

RMS = _paths.RMS
DOC = RMS / "Reference_Documentation/2006-2020_Documentation"
BUILD = Path(__file__).resolve().parent / "build_exact_pack_container_beer_panel.py"
OUT = PAPER / "outputs" / "blp_storeweek"
YEARS = [2016, 2017, 2018, 2019]
MODULES = [5000, 5001, 5005, 5010, 5015, 5020]
DISPUTED = ["001820012989", "001820016984", "001820020019", "001820020140",
            "001820028401", "003410000345", "001820000910", "001820000170"]
OZ_PER_ML = 0.033814022701843


def load_build():
    sp = importlib.util.spec_from_file_location("bb", BUILD)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m


def products(bb):
    """Every products.tsv row in the beer modules, with the build's derived fields.

    Kept at (upc, upc_ver_uc) granularity -- the file's true key -- rather than collapsed
    to upc, so the year join can pick the right row.
    """
    p = pd.read_csv(RMS / "Master_Files_2006-2020/Latest/products.tsv", sep="\t",
                    dtype=str, encoding="latin-1", on_bad_lines="skip")
    for c in ("product_module_code", "department_code", "multi", "size1_amount"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    b = p.loc[(p.department_code == 8)
              & p.product_module_code.isin(MODULES)].copy()
    b["upc"] = bb.clean_upc(b.upc)
    b["upc_ver_uc"] = b.upc_ver_uc.astype("string").str.strip()
    u = b.size1_units.astype("string").str.upper().fillna("").to_numpy(dtype=object)
    amt = b.size1_amount.to_numpy(dtype=float)
    b["oz_per_unit"] = np.where(u == "OZ", amt,
                                np.where(u == "ML", amt * OZ_PER_ML, np.nan))
    b["servings"] = b.multi * b.oz_per_unit / 12.0          # 12-oz equivalents per pack
    b["brand_raw"] = b.brand_descr.astype("string")
    b["brand_std"] = b.brand_descr.map(bb.standardize_brand)
    b["n_cans"] = b.multi.round()
    b["is_focal_brand"] = b.brand_std.isin(bb.FOCAL_BRANDS)
    b["in_target_pack"] = b.n_cans.isin(bb.TARGET_PACKS)
    b["in_size_window"] = (u == "OZ") & b.size1_amount.between(11, 13,
                                                              inclusive="both").to_numpy()
    tc = [c for c in b.columns
          if c in {"upc", "upc_ver_uc", "upc_descr", "product_module_descr",
                   "product_group_descr", "department_descr", "brand_descr",
                   "brand_raw", "brand_std", "size1_units", "size1_code_uc",
                   "brand_code_uc", "dataset_found_uc", "size1_change_flag_uc"}]
    b["container_regex"] = (b[tc].astype("string").fillna("").agg(" ".join, axis=1)
                            .map(bb.infer_container_from_text).astype("string"))
    return b


def versions(year):
    v = pd.read_csv(RMS / f"{year}/Annual_Files/rms_versions_{year}.tsv", sep="\t",
                    dtype=str, encoding="latin-1", low_memory=False)
    v["upc"] = v.upc.astype("string").str.strip()
    v["upc_ver_uc"] = v.upc_ver_uc.astype("string").str.strip()
    return v.drop_duplicates(["upc", "upc_ver_uc"])[["upc", "upc_ver_uc"]]


def extra(year):
    e = pd.read_csv(RMS / f"{year}/Annual_Files/products_extra_{year}.tsv", sep="\t",
                    dtype=str, encoding="latin-1", on_bad_lines="skip")
    e["upc"] = e.upc.astype("string").str.strip()
    e["upc_ver_uc"] = e.upc_ver_uc.astype("string").str.strip()
    keep = ["upc", "upc_ver_uc", "flavor_descr", "container_descr", "style_descr",
            "type_descr", "product_descr", "variety_descr", "strength_descr",
            "common_consumer_name_descr", "form_descr", "formula_descr"]
    keep = [c for c in keep if c in e.columns]
    return e[keep].drop_duplicates(["upc", "upc_ver_uc"])


CONTAINER_MAP = {"CAN": "CAN", "NON REFILLABLE BOTTLE": "BOTTLE",
                 "REFILLABLE BOTTLE": "BOTTLE", "BOTTLE": "BOTTLE",
                 "GLASS BOTTLE": "BOTTLE", "PLASTIC BOTTLE": "BOTTLE"}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with RunManifest(OUT, args={}, inputs=[RMS, DOC],
                     tag="UPC version resolution (round 36 Part 1)") as rm:
        rm.add_note("Resolves the five disputed UPCs against products_extra and the Kilts "
                    "manual, and writes a (upc, upc_ver_uc, year) lookup for Part 2.")
        t0 = time.time()
        bb = load_build()
        P = products(bb)
        print(f"beer-module products rows {len(P):,}, upcs {P.upc.nunique():,}, "
              f"upcs with >1 version {int((P.groupby('upc').upc_ver_uc.nunique()>1).sum()):,}",
              flush=True)

        # ---------------- 1.1 the disputed UPCs in products_extra
        print(f"\n=== 1.1  products_extra for the disputed UPCs ===", flush=True)
        rows = []
        for y in YEARS:
            e = extra(y); v = versions(y)
            g = e.loc[e.upc.isin(DISPUTED)].copy()
            g["year"] = y
            g["year_valid"] = pd.Series(list(zip(g.upc, g.upc_ver_uc)),
                                        index=g.index).isin(set(zip(v.upc, v.upc_ver_uc)))
            rows.append(g)
        E = pd.concat(rows, ignore_index=True)
        E = E.merge(P[["upc", "upc_ver_uc", "brand_descr", "brand_std", "upc_descr",
                       "multi", "size1_amount", "product_module_code"]],
                    on=["upc", "upc_ver_uc"], how="left")
        cols = ["year", "upc", "upc_ver_uc", "year_valid", "brand_descr", "upc_descr",
                "multi", "size1_amount", "flavor_descr", "container_descr", "type_descr"]
        print(E[cols].sort_values(["upc", "year", "upc_ver_uc"]).to_string(index=False),
              flush=True)
        E.to_csv(OUT / "p1_disputed_extra.csv", index=False)

        # does the flavour agree across versions of the same upc-year?
        print(f"\n  --- do the versions describe the same physical product? ---")
        for u in DISPUTED:
            g = E.loc[E.upc == u]
            for y in YEARS:
                gy = g.loc[g.year == y]
                if len(gy) < 2:
                    continue
                same = gy.flavor_descr.nunique() == 1 and gy.container_descr.nunique() == 1
                print(f"  {u} {y}: versions {sorted(gy.upc_ver_uc)} flavour "
                      f"{sorted(set(gy.flavor_descr.astype(str)))} container "
                      f"{sorted(set(gy.container_descr.astype(str)))} brands "
                      f"{sorted(set(gy.brand_descr.astype(str)))} -> "
                      f"{'SAME PRODUCT, brand recode only' if same else 'DIFFERENT PRODUCT'}",
                      flush=True)

        # ---------------- 1.2 the Kilts documentation
        print(f"\n=== 1.2  Kilts documentation ===", flush=True)
        doc = {"manual": str(DOC / "Retail Scanner Dataset Manual 2006-2020.pdf"),
               "corrections": str(DOC / "Corrected_UPC_Version_Files_2006-2017.xlsx")}
        cx = pd.ExcelFile(DOC / "Corrected_UPC_Version_Files_2006-2017.xlsx")
        cc = []
        for s in cx.sheet_names:
            d = cx.parse(s); d.columns = [c.strip().lower().replace(" ", "_")
                                          for c in d.columns]
            cc.append(d)
        C = pd.concat(cc, ignore_index=True)
        C["upc_k"] = C.upc.astype(str).str.strip().str.zfill(12)
        hit = C.loc[C.upc_k.isin(DISPUTED)]
        beer_hit = C.loc[C.upc_k.isin(set(P.upc))]
        print(f"  Corrected_UPC_Version_Files: {len(C):,} rows, panel years "
              f"{sorted(C.panel_year.unique())}")
        print(f"  disputed UPCs appearing in it: {len(hit)}")
        print(f"  beer-module UPCs appearing in it: {beer_hit.upc_k.nunique()} "
              f"({len(beer_hit)} rows); in 2016-2017: "
              f"{int(beer_hit.panel_year.isin([2016,2017]).sum())}", flush=True)
        doc["corrections_rows"] = int(len(C))
        doc["corrections_years"] = sorted(int(x) for x in C.panel_year.unique())
        doc["disputed_in_corrections"] = int(len(hit))
        doc["beer_upcs_in_corrections"] = int(beer_hit.upc_k.nunique())
        json.dump(doc, open(OUT / "p1_documentation.json", "w"), indent=1)

        # ---------------- 1.3/1.5 build the corrected lookup, per year
        print(f"\n=== 1.3/1.5  the (upc, upc_ver_uc, year) lookup ===", flush=True)
        # the build's lookup: first row per upc, one mapping for all years
        old = P.drop_duplicates(subset=["upc"]).set_index("upc")
        allrows, acc = [], []
        for y in YEARS:
            v = versions(y); e = extra(y)
            L = P.merge(v, on=["upc", "upc_ver_uc"], how="inner").copy()
            L = L.merge(e, on=["upc", "upc_ver_uc"], how="left")
            L["year"] = y
            L["container_extra"] = (L.container_descr.astype("string").str.upper()
                                    .map(CONTAINER_MAP).astype("string"))
            L["container"] = L.container_extra.fillna(L.container_regex)
            L["container_src"] = np.where(L.container_extra.notna(), "products_extra",
                                          "regex")
            dup = int(L.duplicated(["upc"]).sum())
            # the build's assignment for the same upc, for comparison
            ob = old.reindex(L.upc)
            L["old_ver"] = ob.upc_ver_uc.to_numpy()
            L["old_brand_std"] = ob.brand_std.to_numpy()
            L["old_n_cans"] = ob.n_cans.to_numpy()
            L["old_module"] = ob.product_module_code.to_numpy()
            L["old_in_window"] = ob.in_size_window.to_numpy()
            L["ver_changed"] = L.old_ver.ne(L.upc_ver_uc)
            allrows.append(L)
            nfoc_new = int((L.is_focal_brand & L.in_target_pack & L.in_size_window
                            & L.container.isin(["CAN", "BOTTLE"])).sum())
            print(f"  {y}: {len(L):,} beer UPC-versions valid; upcs with >1 valid version "
                  f"{dup}; version differs from the build's for {int(L.ver_changed.sum()):,}"
                  f"; focal target-pack known-container UPCs {nfoc_new}", flush=True)
            acc.append(dict(year=y, valid_upc_versions=len(L), upcs_multi_valid=dup,
                            ver_changed=int(L.ver_changed.sum()),
                            focal_new=nfoc_new))
        LK = pd.concat(allrows, ignore_index=True)

        # container: how often does products_extra disagree with the regex?
        m = LK.container_extra.notna() & LK.container_regex.notna()
        print(f"\n  container source: products_extra covers "
              f"{100*LK.container_extra.notna().mean():.1f}% of beer UPC-years, regex "
              f"{100*LK.container_regex.notna().mean():.1f}%; where both exist they "
              f"disagree on {100*(LK.loc[m,'container_extra']!=LK.loc[m,'container_regex']).mean():.2f}% "
              f"({int((LK.loc[m,'container_extra']!=LK.loc[m,'container_regex']).sum()):,} of {int(m.sum()):,})",
              flush=True)
        foc = LK.loc[LK.is_focal_brand & LK.in_target_pack & LK.in_size_window]
        mf = foc.container_extra.notna() & foc.container_regex.notna()
        print(f"  among FOCAL target-pack in-window UPC-years: products_extra covers "
              f"{100*foc.container_extra.notna().mean():.1f}%, disagreement "
              f"{int((foc.loc[mf,'container_extra']!=foc.loc[mf,'container_regex']).sum())} "
              f"of {int(mf.sum())}", flush=True)

        # ---------------- 1.5 accuracy verification on the focal set
        print(f"\n=== 1.5  accuracy verification on the corrected focal set ===",
              flush=True)
        F = LK.loc[LK.is_focal_brand & LK.in_target_pack & LK.in_size_window
                   & LK.container.isin(["CAN", "BOTTLE"])].copy()
        F["cell"] = F.brand_std + "|" + F.container.astype(str) + "|" \
            + F.n_cans.astype(int).astype(str)
        fails = []
        # (a) one cell per upc-year
        g = F.groupby(["upc", "year"]).cell.nunique()
        if (g > 1).any():
            fails.append(("multiple cells in one upc-year", list(g[g > 1].index)))
        # (b) the assignment agrees with upc_descr on the container
        d = F.upc_descr.astype("string").str.upper()
        says_can = d.str.contains(r"\bCN\b|\bCAN", regex=True, na=False)
        says_btl = d.str.contains(r"\bNRB\b|\bRB\b|\bLN\b|\bBTL|\bBOTTLE", regex=True,
                                  na=False)
        conflict = ((F.container == "CAN") & says_btl & ~says_can) | \
                   ((F.container == "BOTTLE") & says_can & ~says_btl)
        # (c) multi x size1 reconciles with the assigned pack
        recon = np.isclose(F.n_cans.astype(float), F.multi.astype(float))
        oz_ok = F.size1_amount.between(11, 13, inclusive="both")
        print(f"  focal UPC-years: {len(F):,} over {F.upc.nunique()} UPCs")
        print(f"  (a) exactly one brand-container-pack per UPC-year : "
              f"{'PASS' if not (g>1).any() else 'FAIL ' + str(int((g>1).sum()))}")
        print(f"  (b) container agrees with upc_descr               : "
              f"{'PASS' if not conflict.any() else 'FAIL ' + str(int(conflict.sum()))}")
        print(f"  (c) multi reconciles with the assigned pack       : "
              f"{'PASS' if recon.all() else 'FAIL ' + str(int((~recon).sum()))}")
        print(f"  (d) size1_amount in the 11-13 oz window           : "
              f"{'PASS' if oz_ok.all() else 'FAIL ' + str(int((~oz_ok).sum()))}", flush=True)
        if conflict.any():
            print(F.loc[conflict, ["year", "upc", "upc_ver_uc", "upc_descr", "cell",
                                   "container_src", "container_descr"]].to_string(index=False))
        F["accuracy_fail"] = conflict | ~recon | ~oz_ok
        # exclude failures, per the prompt
        Fok = F.loc[~F.accuracy_fail].copy()
        print(f"  excluded on accuracy: {int(F.accuracy_fail.sum())} UPC-years; "
              f"retained {len(Fok):,} over {Fok.upc.nunique()} UPCs, "
              f"{Fok.cell.nunique()} cells", flush=True)

        # what changed against the build's own focal set
        oldF = old.loc[old.is_focal_brand & old.in_target_pack & old.in_size_window
                       & old.container_regex.isin(["CAN", "BOTTLE"])].copy()
        oldF["cell"] = oldF.brand_std + "|" + oldF.container_regex.astype(str) + "|" \
            + oldF.n_cans.astype(int).astype(str)
        print(f"\n  the build's focal set: {len(oldF)} UPCs, {oldF.cell.nunique()} cells")
        chg = []
        for y in YEARS:
            fy = Fok.loc[Fok.year == y].set_index("upc")
            ly = LK.loc[LK.year == y].set_index("upc")
            # the build's own year restriction is on upc alone, so its focal set for the
            # year is the UPCs it calls focal that appear anywhere in the year's version
            # file. That is round 35's 207/210/234/235.
            base = oldF.index.intersection(ly.index)
            same = base.intersection(fy.index)
            moved = same[(fy.loc[same, "cell"].to_numpy()
                          != oldF.loc[same, "cell"].to_numpy())]
            left = base.difference(fy.index)
            new = fy.index.difference(oldF.index)
            chg.append(dict(year=y, build_focal_this_year=len(base),
                            unchanged=len(same) - len(moved), cell_changed=len(moved),
                            left_focal_set=len(left), entered_focal_set=len(new)))
            print(f"  {y}: the build calls {len(base)} focal this year; "
                  f"{len(same)-len(moved)} unchanged, {len(moved)} change focal cell, "
                  f"{len(left)} LEAVE the focal set, {len(new)} enter it", flush=True)
            for u in moved:
                print(f"      moved  {u}: {oldF.loc[u,'cell']} -> {fy.loc[u,'cell']}")
            for u in left:
                print(f"      leaves {u}: {oldF.loc[u,'cell']} (v{oldF.loc[u,'upc_ver_uc']}"
                      f", {oldF.loc[u,'upc_descr'].strip()}) -> v{ly.loc[u,'upc_ver_uc']} "
                      f"{ly.loc[u,'brand_std']} multi {ly.loc[u,'multi']:g}")
        pd.DataFrame(chg).to_csv(OUT / "p1_focal_changes.csv", index=False)
        pd.DataFrame(acc).to_csv(OUT / "p1_year_counts.csv", index=False)
        # the full beer lookup Part 2 needs: upc, year -> servings per pack, focal cell
        keep = ["upc", "upc_ver_uc", "year", "product_module_code", "brand_std",
                "brand_descr", "upc_descr", "multi", "n_cans", "size1_amount",
                "oz_per_unit", "servings", "container", "container_src",
                "is_focal_brand", "in_target_pack", "in_size_window"]
        LK["focal_cell"] = np.where(
            LK.index.isin(Fok.index),
            LK.brand_std.astype(str) + "|" + LK.container.astype(str) + "|"
            + LK.n_cans.fillna(-1).astype(int).astype(str), "")
        LK[keep + ["focal_cell"]].to_parquet(OUT / "p1_lookup_by_year.parquet", index=False)
        print(f"\n  wrote {OUT/'p1_lookup_by_year.parquet'}: {len(LK):,} beer UPC-years, "
              f"{int((LK.focal_cell != '').sum()):,} focal", flush=True)
        rm.add_outputs([OUT / f for f in ("p1_disputed_extra.csv", "p1_documentation.json",
                                          "p1_focal_changes.csv", "p1_year_counts.csv",
                                          "p1_lookup_by_year.parquet")])
        print(f"\ndone [{time.time()-t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
