#!/usr/bin/env python3
"""The food-all sample: the paper's rules, with the store universe food-only from the start,
no one-in-four step, and a $100 ceiling on the price per pack.

    python3 build/build_foodall.py            # every stage not yet done
    python3 build/build_foodall.py --from frame
    python3 build/build_foodall.py --from sample --per-dma 3   # -> data/k3/

  scan    every RMS beer movement row 2016-2019, units > 0 and $0.50-$100 a pack
  frame   food channel in every year it reports, median >= 2,000 cans or bottles a week;
          >= 3% deep-sale product-weeks, >= 52 weeks a product; lagged price, smaller pack
          stocked (or none exists), inside share in [1%, 95%].  Built in batches of stores
          (every rule is store by store), then joined with global store indices.
  sample  five stores per DMA (seed 0, DMAs in code order), every drawn store kept;
          weeks 1-208; regimes and instruments
          (f70_baseline_build.py); market size, zip3 and parent chain
          -> data/regression ready.parquet
"""
from __future__ import annotations
import argparse, gc, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc

HERE = Path(__file__).resolve().parent
WORK, LEGACY, OUT = HERE / "work", HERE / "legacy", HERE / "data" / "regression ready.parquet"
RMS = Path(os.environ.get("NIELSEN_RMS", "nielsen_extracts/RMS"))      # the raw RMS extract
ACS = HERE.parent / "data" / "acs_zip3_demographics_2016_2019.csv"
PRICE_HI, FLOOR, NBATCH = 100.0, 2000, 4
NW, PER_DMA, SEED = 209, 5, 0
DMA_ORDER = "code"
W0, D0, YEARS = 65, pd.Timestamp("2017-04-01"), (2016, 2017, 2018, 2019)
FRAME = WORK / "frame_foodall.parquet"
os.environ.update(BUILD_RMS=str(RMS), BUILD_WORK=str(WORK), BUILD_ACS=str(ACS))
sys.path.insert(0, str(LEGACY))


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def universe(z):
    """Food channel in every year the store reports, median >= 2,000 containers a week."""
    chan, bt = z["chan"], z["tot_bt"]
    inF = np.isin(chan, ["F"])
    univ = inF.any(1) & ~(~inF & (chan != "")).any(1) & (bt > 0).any(1)
    med = np.zeros(len(bt))
    for i in np.where(univ)[0]:
        med[i] = np.median(bt[i][bt[i] > 0])
    return np.where(univ & (med >= FLOOR))[0]


def scan():
    import blp_dma_week_scan as M
    M.PRICE_HI = PRICE_HI
    assert M.main() in (0, None)


def frame():
    z = np.load(WORK / "storeweek_totals.npz", allow_pickle=True)
    U = universe(z)
    log(f"universe: {len(U):,} stores")
    import blp_hn_cells as H
    orig = H.store_set
    import f2_build_frame as F2
    holders = [m for m in list(sys.modules.values()) if getattr(m, "store_set", None) is orig]
    parts = []
    for b, pick in enumerate(np.array_split(U, NBATCH)):
        out = WORK / f"frame_batch{b}.parquet"
        if not out.exists():
            for m in holders + [H]:
                m.store_set = (lambda z, p=pick: (p, orig(z)[1], orig(z)[2]))
            F2.OUT, F2.PARENT_CACHE = out, WORK / f"frame_batch{b}_parent.parquet"
            log(f"batch {b}: {len(pick):,} candidate stores")
            assert F2.main() == 0
            gc.collect()
        parts.append((out, pick))
    writer = None
    for out, pick in parts:
        t = pq.read_table(out, columns=["market_ids", "alt", "prices", "shares", "s0_inside",
                                        "z1", "income"])
        m = t.column("market_ids").to_numpy()
        g = pick[m // NW].astype(np.int64) * NW + m % NW          # global store index
        t = t.set_column(0, "market_ids", pa.array(g))
        writer = writer or pq.ParquetWriter(FRAME, t.schema)
        writer.write_table(t)
    writer.close()
    m = pq.read_table(FRAME, columns=["market_ids"]).column(0).to_numpy()
    log(f"frame: {len(m):,} rows, {len(np.unique(m // NW)):,} stores")


def store_files():
    S = pd.concat([pd.read_csv(RMS / f"{y}/Annual_Files/stores_{y}.tsv", sep="\t", dtype=str,
                               encoding="latin-1").assign(year=y) for y in YEARS])
    S["store_code_uc"] = S.store_code_uc.str.strip()
    return S


def sample():
    z = np.load(WORK / "storeweek_totals.npz", allow_pickle=True)
    codes, cells = z["stores"].astype(str), pd.Series(z["cells"].astype(str))
    S = store_files()
    last = S.sort_values("year").drop_duplicates("store_code_uc", keep="last") \
            .set_index("store_code_uc")
    m = pq.read_table(FRAME, columns=["market_ids"]).column(0).to_numpy()
    u, w = m // NW, m % NW
    stores = np.unique(u)
    dma = pd.Series(last.dma_code.reindex(codes[stores]).fillna("").to_numpy(), index=stores)
    rng = np.random.default_rng(SEED)
    per = {d: g.index.to_numpy() for d, g in dma.groupby(dma)}
    # DMA order, which is also the order the shuffles draw from the generator: "code" sorts
    # DMA codes; "first" takes DMAs in order of their first store in store-code order, as the
    # paper's build does (pd.factorize in f2_build_frame.py)
    order = sorted(per) if DMA_ORDER == "code" else list(pd.unique(dma.sort_index().to_numpy()))
    for d in order:
        rng.shuffle(per[d])
    drawn = np.array([x for d in order for x in per[d][:PER_DMA]])
    kept = drawn                      # no draw-window rule: every drawn store is kept
    log(f"draw: {len(drawn)} stores from {dma.nunique()} DMA codes")
    t = pq.read_table(FRAME)
    uu = pc.divide(t.column("market_ids"), NW)
    t = t.filter(pc.is_in(uu, value_set=pa.array(kept.astype(np.int64))))
    d = t.to_pandas()
    del t; gc.collect()
    d["u"], d["w"] = d.market_ids // NW, d.market_ids % NW
    d = d[d.w.between(1, 208)].copy()
    d["store_code_uc"] = codes[d.u.to_numpy()]
    d["dma_code"] = last.dma_code.reindex(d.store_code_uc).fillna("").to_numpy()
    part = cells.str.split("|")
    a = d.alt.to_numpy()
    d["brand_id"] = part.str[0].to_numpy()[a]
    d["is_can"] = (part.str[1] == "CAN").to_numpy()[a].astype(np.int8)
    d["n_cans"] = part.str[2].astype(int).to_numpy()[a].astype(np.int16)
    d["log_n_eq"] = np.log(d.n_cans / 12.0)
    we = D0 + pd.to_timedelta((d.w.to_numpy() - W0) * 7, unit="D")
    d["year_month"] = we.strftime("%Y%m").astype(np.int32)
    d["w"], d["alt"] = d.w.astype(np.int64), d.alt.astype(np.int32)
    KEEP = ["market_ids", "alt", "store_code_uc", "w", "prices", "shares", "s0_inside",
            "n_cans", "log_n_eq", "is_can", "brand_id", "year_month", "dma_code", "z1",
            "income"]
    (WORK / "data_work").mkdir(exist_ok=True)
    d[KEEP].sort_values(["market_ids", "alt"]).reset_index(drop=True) \
        .to_parquet(WORK / "data_work" / "f3_long.parquet", index=False)
    del d; gc.collect()
    import f70_baseline_build as F70
    F70.ROOT = WORK
    F70.main("long")
    D = pd.read_parquet(WORK / "data_work" / "f70_baseline_long.parquet")
    food = S.groupby("store_code_uc").channel_code.agg(lambda c: (c == "F").all())
    n0 = D.store_code_uc.nunique()
    D = D[D.store_code_uc.astype(str).map(food).fillna(False).astype(bool)].reset_index(drop=True)
    log(f"food channel: {n0} -> {D.store_code_uc.nunique()} stores")
    mu = D.market_ids.to_numpy()
    D["mkt_size"] = z["tot_sv"][mu // NW, mu % NW].astype(np.float64)
    D["zip3"] = last.store_zip3.reindex(D.store_code_uc).str.zfill(3).to_numpy()
    D["parent"] = last.parent_code.reindex(D.store_code_uc).to_numpy()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    D.to_parquet(OUT, index=False)
    log(f"wrote {OUT}: {len(D):,} rows, {D.market_ids.nunique():,} store-weeks, "
        f"{D.store_code_uc.nunique()} stores, {D.dma_code.nunique()} DMA codes, "
        f"{D.parent.nunique()} chains; column (iv) rows {(D.reg3 != '').sum():,}")


STAGES = [("scan", scan, WORK / "storeweek_totals.npz"), ("frame", frame, FRAME),
          ("sample", sample, OUT)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", choices=[s[0] for s in STAGES])
    ap.add_argument("--per-dma", type=int, default=PER_DMA)
    a = ap.parse_args()
    if a.per_dma != PER_DMA:
        PER_DMA, OUT = a.per_dma, HERE / "data" / f"k{a.per_dma}" / "regression ready.parquet"
        STAGES[-1] = ("sample", sample, OUT)
    go = False
    for name, fn, out in STAGES:
        go = go or (a.start == name)
        if out.exists() and not go:
            log(f"=== {name}: done, skipped")
            continue
        t0 = time.time(); log(f"=== {name}")
        fn()
        log(f"=== {name} done in {(time.time() - t0) / 60:.1f} min")
