#!/usr/bin/env python3
"""Round 46 pass 1: rebuild the all-beer store-week scan, plus the 29 excluded UPCs.

Round 36's scan wrote its store-week totals and its focal row file to a session
scratchpad that no longer exists, and round 39's excluded-UPC scan wrote a second row
file to the same place.  Both are needed here -- the DMA-week panel is the store-week
grid re-aggregated -- so this reproduces both in ONE movement pass instead of two.

Identical to `blp_store_week_scan.py` in every screen and unit convention:
`units` is packs, bottles = units x multi, servings = units x multi x size1_amount / 12,
the per-pack price window is [0.50, 200.0], the horizon is the 209 weeks 2016-01-02 to
2019-12-28, and the lookup is round 36's corrected (upc, upc_ver_uc, year) join.  Two
things are dropped -- the build's `drop_duplicates(subset=["upc"])` comparison lookup and
the per-UPC-year accumulator -- because both are already saved outputs of round 36 and
neither is needed downstream.

The 29 excluded UPCs are reconstructed from D090's flavour rule and D095's raw-brand rule
by value, not by re-deriving them, and the reconstruction is asserted against D095's
published totals before the pass starts: 21 + 8 UPCs, 68 + 18 UPC-years, disjoint, 16
cells, 12,289,916 units = 1.360% of focal units.

    python3 paper/src/diagnostics/blp_dma_week_scan.py
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import _paths
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
PAPER = _paths.WORK
sys.path.insert(0, str(PAPER / "src" / "common"))
from run_manifest import RunManifest        # noqa: E402

RMS = _paths.RMS
SW = PAPER / "outputs" / "blp_storeweek"
OUT = PAPER / "outputs" / "blp_dmaweek"
SCRATCH = _paths.WORK
ROW_DTYPE = np.dtype([("s", "i4"), ("w", "i2"), ("c", "i2"),
                      ("pk", "f4"), ("rv", "f4")])
YEARS = [2016, 2017, 2018, 2019]
MODULES = [5000, 5001, 5005, 5010, 5015, 5020]
PRICE_LO, PRICE_HI = 0.50, 200.0
FLUSH = 20_000_000
# D095 raw-brand rule (21 UPCs) and D090 substantive-flavour rule (8 UPCs)
EXCL_BRANDS = {"MICHELOB ULTRA PURE GOLD LIGHT", "MICHELOB ULTRA AMBER LIGHT",
               "MILLER HIGH LIFE LIGHT", "SOUTHERN STAR BMHL BLD AMC ALE",
               "TERRAPIN BR CO&MHL HPBB IP ALE", "MODELO ESPECIAL & NEGRA MODELO"}
EXCL_FLAVOURS = {"LIME CACTUS", "LIME AND PRICKLY PEAR CACTUS", "POMEGRANATE & AGAVE",
                 "POMEGRANATE RASPBERRY", "TUSCAN ORANGE GRAPEFRUIT"}
# round 36 p2_scan_summary.json and round 39 p2_excluded_scan.json
REF = dict(kept=658_541_178, dropped_nomatch=87_247, dropped_price=88_886,
           off_horizon=0, focal_rows=157_167_005, excluded_rows=4_087_340,
           n_stores=53_490, n_weeks=209, n_cells=116)
# D095's published totals for the exclusion set
REF_EXCL = dict(brand_upcs=21, brand_years=68, flav_upcs=8, flav_years=18,
                upcs=29, years=86, cells=16, units=12_289_916, pct=1.360)


def excluded_upcs(LK):
    """The 29 UPCs D090 and D095 identify, rebuilt by value and asserted on their totals."""
    F = LK.loc[LK.focal_cell != ""].copy()
    ex = []
    for y in YEARS:
        e = pd.read_csv(RMS / f"{y}/Annual_Files/products_extra_{y}.tsv", sep="\t",
                        dtype=str, usecols=["upc", "upc_ver_uc", "flavor_descr"],
                        encoding="latin-1", on_bad_lines="skip")
        e["year"] = y
        ex.append(e)
    E = pd.concat(ex, ignore_index=True)
    E["upc"] = E.upc.str.strip()
    F["upc_ver_uc"] = F.upc_ver_uc.astype(str)
    E["upc_ver_uc"] = E.upc_ver_uc.astype(str)
    m = F.merge(E, on=["upc", "upc_ver_uc", "year"], how="left")
    br = m.brand_descr.isin(EXCL_BRANDS)
    fl = m.flavor_descr.isin(EXCL_FLAVOURS)
    got = dict(brand_upcs=m.loc[br, "upc"].nunique(), brand_years=int(br.sum()),
               flav_upcs=m.loc[fl, "upc"].nunique(), flav_years=int(fl.sum()),
               upcs=m.loc[br | fl, "upc"].nunique(), years=int((br | fl).sum()),
               cells=m.loc[br | fl, "focal_cell"].nunique())
    print("  exclusion set rebuilt:", got, flush=True)
    print(f"  flavour coverage of the focal set {100*m.flavor_descr.notna().mean():.1f}% "
          f"(D090: 33.7%); rules disjoint: {int((br & fl).sum()) == 0}", flush=True)
    for k, v in got.items():
        assert v == REF_EXCL[k], (k, v, REF_EXCL[k])
    U = sorted(set(m.loc[br | fl, "upc"]))
    if (SW / "p2_upc_year_totals.parquet").exists():   # replication build: an optional check
        T = pd.read_parquet(SW / "p2_upc_year_totals.parquet")
        T["upc"] = T.upc.astype(str)
        foc = T.loc[T.cell_new != ""]
        un = float(foc.loc[foc.upc.isin(U), "u"].sum())
        pc = 100 * un / float(foc.u.sum())
        print(f"  excluded units {un:,.0f} = {pc:.3f}% of focal units "
              f"(D095: {REF_EXCL['units']:,} = {REF_EXCL['pct']}%)", flush=True)
        assert abs(un - REF_EXCL["units"]) < 1.0 and abs(pc - REF_EXCL["pct"]) < 0.001
    return set(U)


def store_table():
    st = []
    for y in YEARS:
        s = pd.read_csv(RMS / f"{y}/Annual_Files/stores_{y}.tsv", sep="\t",
                        dtype={"store_code_uc": "string", "parent_code": "string",
                               "retailer_code": "string", "store_zip3": "string",
                               "dma_code": "string", "fips_state_code": "string",
                               "fips_county_code": "string"},
                        encoding="latin-1", low_memory=False)
        s["year"] = y
        st.append(s)
    S = pd.concat(st, ignore_index=True)
    S["store_code_uc"] = S.store_code_uc.str.strip()
    return S


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    rowfile = SCRATCH / "focal_rows.bin"
    excfile = SCRATCH / "excluded_rows.bin"
    for f in (rowfile, excfile):
        if f.exists():
            f.unlink()
    with RunManifest(OUT, args={"modules": str(MODULES)},
                     inputs=[RMS, SW / "p1_lookup_by_year.parquet"],
                     tag="all-beer store-week scan rebuild + excluded UPCs (round 46)") as rm:
        rm.add_note("One movement pass reproducing round 36's scan and round 39's "
                    "excluded-UPC scan, whose scratch files were lost with the session "
                    "that wrote them.")
        t0 = time.time()
        LK = pd.read_parquet(SW / "p1_lookup_by_year.parquet")
        cells = sorted(x for x in LK.focal_cell.unique() if x)
        cix = {c: i for i, c in enumerate(cells)}
        print(f"alternatives {len(cells)}", flush=True)
        EX = excluded_upcs(LK)
        ST = store_table()
        stores = pd.Index(sorted(ST.store_code_uc.dropna().unique()))
        nS = len(stores)
        chan = np.full((nS, len(YEARS)), "", dtype=object)
        dma = np.full((nS, len(YEARS)), "", dtype=object)
        par = np.full((nS, len(YEARS)), "", dtype=object)
        for k, y in enumerate(YEARS):
            g = ST.loc[ST.year == y]
            idx = stores.get_indexer(g.store_code_uc)
            ok = idx >= 0
            chan[idx[ok], k] = g.channel_code.to_numpy()[ok]
            dma[idx[ok], k] = g.dma_code.fillna("").to_numpy()[ok]
            par[idx[ok], k] = g.parent_code.fillna("").to_numpy()[ok]
        print(f"stores in the annual files: {nS:,}", flush=True)
        assert nS == REF["n_stores"], (nS, REF["n_stores"])

        wk = pd.date_range("2016-01-02", "2019-12-28", freq="7D")
        wkidx = pd.Index([d.strftime("%Y%m%d") for d in wk])
        nW = len(wk)
        assert nW == REF["n_weeks"]
        print(f"weeks {nW} from {wk[0].date()} to {wk[-1].date()}", flush=True)

        tot_pk = np.zeros((nS, nW), np.float32)
        tot_bt = np.zeros((nS, nW), np.float32)
        tot_sv = np.zeros((nS, nW), np.float32)
        foc_sv = np.zeros((nS, nW), np.float32)
        exc_sv = np.zeros((nS, nW), np.float32)
        mod_sv = {m: np.zeros(len(YEARS)) for m in MODULES}
        buf, nbuf, nrows_written = [], 0, 0
        ebuf, nexc = [], 0
        dropped_price = dropped_nomatch = kept = off_horizon = 0

        for k, y in enumerate(YEARS):
            ly = LK.loc[LK.year == y]
            serv = pd.Series(ly.servings.to_numpy(), index=ly.upc.to_numpy())
            mult = pd.Series(ly.multi.to_numpy(), index=ly.upc.to_numpy())
            modc = pd.Series(ly.product_module_code.to_numpy(), index=ly.upc.to_numpy())
            cellc = pd.Series([cix.get(c, -1) for c in ly.focal_cell],
                              index=ly.upc.to_numpy())
            isex = pd.Series([u in EX for u in ly.upc.to_numpy()],
                             index=ly.upc.to_numpy())
            for mod in MODULES:
                f = RMS / f"{y}/Movement_Files/5001_{y}/{mod}_{y}.tsv"
                if not f.exists():
                    print(f"    missing {f}", flush=True)
                    continue
                for c in pd.read_csv(f, sep="\t", chunksize=2_000_000,
                                     usecols=["store_code_uc", "upc", "week_end",
                                              "units", "prmult", "price"],
                                     dtype={"store_code_uc": "string", "upc": "string",
                                            "week_end": "string"},
                                     encoding="latin-1", low_memory=False):
                    c["upc"] = c.upc.str.strip()
                    sv = serv.reindex(c.upc).to_numpy()
                    good = np.isfinite(sv)
                    dropped_nomatch += int((~good).sum())
                    if not good.any():
                        continue
                    c = c.loc[good]; sv = sv[good]
                    mu = mult.reindex(c.upc).to_numpy()
                    mc = modc.reindex(c.upc).to_numpy()
                    cc = cellc.reindex(c.upc).to_numpy()
                    xx = isex.reindex(c.upc).to_numpy().astype(bool)
                    si = stores.get_indexer(c.store_code_uc.str.strip()).astype(np.int64)
                    ti = wkidx.get_indexer(
                        c.week_end.str.replace(r"\.0$", "", regex=True).str.zfill(8)
                    ).astype(np.int64)
                    off_horizon += int((ti < 0).sum())
                    u = pd.to_numeric(c.units, errors="coerce").to_numpy(float)
                    pm = pd.to_numeric(c.prmult, errors="coerce").to_numpy(float)
                    pm = np.where(np.isfinite(pm) & (pm > 0), pm, 1.0)
                    pr = pd.to_numeric(c.price, errors="coerce").to_numpy(float) / pm
                    ok = (si >= 0) & (ti >= 0) & (u > 0) & np.isfinite(pr)
                    inprice = ok & (pr >= PRICE_LO) & (pr <= PRICE_HI)
                    dropped_price += int((ok & ~inprice).sum())
                    ok = inprice
                    if not ok.any():
                        continue
                    si, ti, u, pr = si[ok], ti[ok], u[ok], pr[ok]
                    sv, mu, mc, cc, xx = sv[ok], mu[ok], mc[ok], cc[ok], xx[ok]
                    kept += int(ok.sum())
                    flat = si * nW + ti
                    np.add.at(tot_pk.reshape(-1), flat, u.astype(np.float32))
                    np.add.at(tot_bt.reshape(-1), flat, (u * mu).astype(np.float32))
                    np.add.at(tot_sv.reshape(-1), flat, (u * sv).astype(np.float32))
                    for m in MODULES:
                        mm = mc == m
                        if mm.any():
                            mod_sv[m][k] += float((u[mm] * sv[mm]).sum())
                    fm = cc >= 0
                    if fm.any():
                        np.add.at(foc_sv.reshape(-1), flat[fm],
                                  (u[fm] * sv[fm]).astype(np.float32))
                        buf.append(np.rec.fromarrays(
                            [si[fm].astype(np.int32), ti[fm].astype(np.int16),
                             cc[fm].astype(np.int16), u[fm].astype(np.float32),
                             (u[fm] * pr[fm]).astype(np.float32)],
                            dtype=ROW_DTYPE))
                        nbuf += int(fm.sum())
                    xm = fm & xx
                    if xm.any():
                        np.add.at(exc_sv.reshape(-1), flat[xm],
                                  (u[xm] * sv[xm]).astype(np.float32))
                        ebuf.append(np.rec.fromarrays(
                            [si[xm].astype(np.int32), ti[xm].astype(np.int16),
                             cc[xm].astype(np.int16), u[xm].astype(np.float32),
                             (u[xm] * pr[xm]).astype(np.float32)],
                            dtype=ROW_DTYPE))
                        nexc += int(xm.sum())
                    if nbuf >= FLUSH:
                        with open(rowfile, "ab") as fh:
                            np.concatenate(buf).tofile(fh)
                        nrows_written += nbuf; buf, nbuf = [], 0
                        if ebuf:
                            with open(excfile, "ab") as fh:
                                np.concatenate(ebuf).tofile(fh)
                            ebuf = []
                print(f"  {y} mod {mod}: kept {kept:,} focal {nrows_written+nbuf:,} "
                      f"excl {nexc:,} [{time.time()-t0:.0f}s]", flush=True)
        if nbuf:
            with open(rowfile, "ab") as fh:
                np.concatenate(buf).tofile(fh)
            nrows_written += nbuf; buf = []
        if ebuf:
            with open(excfile, "ab") as fh:
                np.concatenate(ebuf).tofile(fh)
            ebuf = []
        print(f"\nkept {kept:,} beer movement rows; dropped {dropped_nomatch:,} with no "
              f"year-valid lookup row, {dropped_price:,} outside "
              f"[{PRICE_LO}, {PRICE_HI}] per pack, {off_horizon:,} outside the "
              f"209-week horizon", flush=True)
        print(f"focal store-cell-week rows {nrows_written:,}; excluded {nexc:,}",
              flush=True)
        got = dict(kept=kept, dropped_nomatch=dropped_nomatch,
                   dropped_price=dropped_price, off_horizon=off_horizon,
                   focal_rows=nrows_written, excluded_rows=nexc)
        print("\n=== assertion against round 36's scan and round 39's excluded scan ===")
        bad = []
        for k_, v in got.items():
            r = REF[k_]
            flag = "OK" if v == r else "**DIFFERS**"
            if v != r:
                bad.append((k_, v, r))
            print(f"  {k_:16s} {v:>13,}  reference {r:>13,}  {flag}")
        np.savez(SCRATCH / "storeweek_totals.npz", tot_pk=tot_pk, tot_bt=tot_bt,
                 tot_sv=tot_sv, foc_sv=foc_sv, exc_sv=exc_sv,
                 stores=np.array(list(stores), dtype=object), chan=chan, dma=dma,
                 par=par,
                 weeks=np.array([d.strftime("%Y-%m-%d") for d in wk], dtype=object),
                 cells=np.array(cells, dtype=object),
                 mod_sv=np.array([mod_sv[m] for m in MODULES]))
        json.dump(dict(got=got, reference=REF, mismatches=bad,
                       n_stores=int(nS), n_weeks=int(nW), n_cells=len(cells),
                       modules=MODULES, years=YEARS,
                       excluded_upcs=sorted(EX),
                       module_servings={str(m): list(mod_sv[m]) for m in MODULES}),
                  open(OUT / "p0_scan_summary.json", "w"), indent=1, default=float)
        rm.add_outputs([OUT / "p0_scan_summary.json"])
        print(f"\ndone [{time.time()-t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
