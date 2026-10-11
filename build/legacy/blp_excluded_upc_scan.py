#!/usr/bin/env python3
"""Round 39 Part 2 pass: the movement rows of the 29 excluded UPCs.

`focal_rows.bin` is keyed on (store, week, ALTERNATIVE), so the UPC identity round 38's
string audit needs is not in it.  This scans the movement files for the 29 excluded UPCs
only -- 21 raw-brand mismatches (D095) and 8 substantive-flavour line extensions (D090) --
and writes their store-cell-week contributions in the same layout, so the corrected panel
is the existing grid minus this one.

The excluded units do not leave the data: they are still RMS beer, so they move from the
inside goods into the all-beer outside good, which is what the corrected specification
should do and needs no change to the denominator.

    python3 paper/src/diagnostics/blp_excluded_upc_scan.py
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
OUT = PAPER / "outputs" / "blp_chainweek"
SCRATCH = _paths.WORK
ROW_DTYPE = np.dtype([("s", "i4"), ("w", "i2"), ("c", "i2"),
                      ("pk", "f4"), ("rv", "f4")])
YEARS = [2016, 2017, 2018, 2019]
MODULES = [5000, 5001, 5005, 5010, 5015, 5020]
PRICE_LO, PRICE_HI = 0.50, 200.0
EXCL = _paths.excluded_upcs()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    outbin = SCRATCH / "excluded_rows.bin"
    if outbin.exists():
        outbin.unlink()
    with RunManifest(OUT, args={"n_excluded_upcs": len(EXCL)}, inputs=[RMS],
                     tag="excluded-UPC movement scan (round 39 Part 2)") as rm:
        rm.add_note("Store-cell-week contributions of the 29 UPCs D090 and D095 identify, "
                    "in the focal_rows layout so the corrected panel is a subtraction.")
        t0 = time.time()
        LK = pd.read_parquet(SW / "p1_lookup_by_year.parquet")
        cells = sorted(x for x in LK.focal_cell.unique() if x)
        cix = {c: i for i, c in enumerate(cells)}
        z = np.load(SCRATCH / "storeweek_totals.npz", allow_pickle=True)
        stores = pd.Index(z["stores"].astype(str))
        wk = pd.Index([str(x) for x in z["weeks"].astype(str)])
        wkidx = pd.Index([w.replace("-", "") for w in wk])
        nW = len(wk)
        print(f"{len(EXCL)} excluded UPCs; {len(cells)} alternatives; {nW} weeks",
              flush=True)
        F = LK.loc[(LK.focal_cell != "") & LK.upc.isin(EXCL)]
        print(f"  they are {len(F)} focal UPC-years across {F.focal_cell.nunique()} cells",
              flush=True)
        keep, nrows = [], 0
        for y in YEARS:
            ly = F.loc[F.year == y]
            u2 = dict(zip(ly.upc, zip([cix[c] for c in ly.focal_cell], ly.n_cans)))
            if not u2:
                print(f"  {y}: none focal", flush=True); continue
            want = set(u2)
            for mod in MODULES:
                f = RMS / f"{y}/Movement_Files/5001_{y}/{mod}_{y}.tsv"
                if not f.exists():
                    continue
                got = 0
                for c in pd.read_csv(f, sep="\t", chunksize=2_000_000,
                                     usecols=["store_code_uc", "upc", "week_end",
                                              "units", "prmult", "price"],
                                     dtype={"store_code_uc": "string", "upc": "string",
                                            "week_end": "string"},
                                     encoding="latin-1", low_memory=False):
                    c["upc"] = c.upc.str.strip()
                    c = c.loc[c.upc.isin(want)]
                    if not len(c):
                        continue
                    si = stores.get_indexer(c.store_code_uc.str.strip())
                    ti = wkidx.get_indexer(
                        c.week_end.str.replace(r"\.0$", "", regex=True).str.zfill(8))
                    u = pd.to_numeric(c.units, errors="coerce").to_numpy(float)
                    pm = pd.to_numeric(c.prmult, errors="coerce").to_numpy(float)
                    pm = np.where(np.isfinite(pm) & (pm > 0), pm, 1.0)
                    pr = pd.to_numeric(c.price, errors="coerce").to_numpy(float) / pm
                    ci = np.array([u2[x][0] for x in c.upc], np.int64)
                    ok = ((si >= 0) & (ti >= 0) & (u > 0) & np.isfinite(pr)
                          & (pr >= PRICE_LO) & (pr <= PRICE_HI))
                    if not ok.any():
                        continue
                    rec = np.rec.fromarrays(
                        [si[ok].astype(np.int32), ti[ok].astype(np.int16),
                         ci[ok].astype(np.int16), u[ok].astype(np.float32),
                         (u[ok] * pr[ok]).astype(np.float32)], dtype=ROW_DTYPE)
                    with open(outbin, "ab") as fh:
                        rec.tofile(fh)
                    nrows += len(rec); got += len(rec)
                print(f"  {y} mod {mod}: {got:,} rows [{time.time()-t0:.0f}s]", flush=True)
        print(f"\nwrote {nrows:,} excluded store-cell-week rows -> {outbin}", flush=True)
        json.dump(dict(rows=int(nrows), upcs=len(EXCL),
                       focal_upc_years=int(len(F))),
                  open(OUT / "p2_excluded_scan.json", "w"), indent=1)
        rm.add_outputs([OUT / "p2_excluded_scan.json"])
        print(f"done [{time.time()-t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
