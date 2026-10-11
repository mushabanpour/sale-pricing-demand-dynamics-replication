#!/usr/bin/env python3
"""The frequency grid: alpha and the own-price elasticity across frequency, fixed-effect
tightness and instrument family. Linear IV only.

Panels are built fresh here at three frequencies rather than reused, for two reasons.
The Hausman instrument is dropped (D028/D032/D033), so the filter that removed the 125
single-DMA alternative-weeks is no longer needed -- which is Part 3.2 -- and `J_rt` and
the differentiation classes are carried through together so the partition identity is
preserved, which is round 18's diagnosis.

    python3 paper/src/diagnostics/blp_frequency_grid.py
"""
from __future__ import annotations
import io, json, sys, warnings
from pathlib import Path
import _paths
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve()
PAPER = _paths.WORK
sys.path.insert(0, str(PAPER / "src" / "common"))
sys.path.insert(0, str(HERE.parent))
from run_manifest import RunManifest                    # noqa: E402
import blp_first_stage_strength as M                    # noqa: E402
from blp_pre_rcnl_diagnostics import absorbN            # noqa: E402

PANEL = Path("/Volumes/LaCie/Beer/estimation_exact_pack_container/"
             "beer_exact_container_balanced_2016_2017_2018_2019.parquet")
DERIVED = PAPER / "data" / "derived"
EXT = PAPER / "data" / "external"
OUT = PAPER / "outputs" / "blp_freq_grid"
MULT, SALE_DROP, REBOUND = 1.5, 0.10, 0.05
ABI = ["BUD LIGHT", "BUDWEISER", "BUD LIGHT LIME", "BUD LIGHT ORANGE", "MICHELOB ULTRA"]
MC = ["COORS LIGHT", "COORS BANQUET", "MILLER LITE", "MILLER HIGH LIFE"]
MEX = ["CORONA EXTRA", "CORONA LIGHT", "MODELO ESPECIAL", "MODELO NEGRA"]
NLB = ["HEINEKEN"]
CLS_NAMES = ["d_same_brand_other_rung", "d_same_rung_other_brand",
             "d_same_container_other", "d_other_container"]
R3 = CLS_NAMES[:3]
FAMILIES = [("z_dist_diesel (national)", ["z_dist_diesel"]),
            ("z_dist_diesel (PADD)", ["z_dd_padd"]),
            ("three differentiation classes", R3)]


def diesel_tables():
    prov = json.loads((EXT / "eia_diesel_provenance.json").read_text())
    out = {}
    for sid, meta in prov.items():
        raw = (EXT / f"{sid}w.xls").read_bytes()
        s = pd.read_excel(io.BytesIO(raw), sheet_name=1, skiprows=2)
        s.columns = ["date", "value"]
        s["date"] = pd.to_datetime(s.date)
        out[meta["name"]] = s.dropna()
    return out


def build_all():
    """One pass, three frequencies."""
    f = pq.ParquetFile(PANEL)
    dm, wks, alts = set(), set(), set()
    for i in range(f.metadata.num_row_groups):
        t = f.read_row_group(i, columns=["dma_code", "week_end", "brand_descr",
                                         "container", "n_cans"])
        dm.update(str(x) for x in t["dma_code"].to_pylist())
        wks.update(pd.Timestamp(w) for w in t["week_end"].to_pylist())
        alts.update(zip(t["brand_descr"].to_pylist(), t["container"].to_pylist(),
                        np.asarray(t["n_cans"], dtype=np.int64).tolist()))
    D = pd.Index(sorted(x for x in dm if str(x).strip() and str(x) != "None"))
    A = pd.MultiIndex.from_tuples(sorted(alts))
    aix = {k: i for i, k in enumerate(A)}
    wk = pd.DatetimeIndex(sorted(wks))
    labs = {"week": pd.Index([str(d.date()) for d in wk]),
            "month": pd.Index(sorted({f"{d.year}-{d.month:02d}" for d in wk})),
            "quarter": pd.Index(sorted({f"{d.year}Q{d.quarter}" for d in wk}))}
    key = {"week": {d.value: labs["week"].get_loc(str(d.date())) for d in wk},
           "month": {d.value: labs["month"].get_loc(f"{d.year}-{d.month:02d}") for d in wk},
           "quarter": {d.value: labs["quarter"].get_loc(f"{d.year}Q{d.quarter}") for d in wk}}
    nD, nA = len(D), len(A)
    acc = {g: {k: np.zeros((nD, len(labs[g]), nA)) for k in ("packs", "w", "wp")}
           for g in labs}
    print(f"DMA {nD} alternatives {nA}; periods "
          f"{ {g: len(v) for g, v in labs.items()} }", flush=True)
    for i in range(f.metadata.num_row_groups):
        t = f.read_row_group(i, columns=["dma_code", "week_end", "brand_descr", "container",
                                         "n_cans", "qty", "pack_units", "price_per_pack_equiv"])
        q = np.asarray(t["qty"], dtype=np.float64); keep = q > 0
        if not keep.any():
            continue
        di = D.get_indexer([str(x) for x in t["dma_code"].to_pylist()])[keep]
        ok = di >= 0
        wv = np.asarray(t["week_end"].cast("int64"), dtype=np.int64)[keep][ok]
        ai = np.array([aix[k] for k in zip(t["brand_descr"].to_pylist(),
                                           t["container"].to_pylist(),
                                           np.asarray(t["n_cans"], dtype=np.int64).tolist())]
                      )[keep][ok]
        pk = np.asarray(t["pack_units"], dtype=np.float64)[keep][ok]
        pr = np.asarray(t["price_per_pack_equiv"], dtype=np.float64)[keep][ok]
        di = di[ok]
        for g in labs:
            ti = np.array([key[g][v] for v in wv])
            np.add.at(acc[g]["packs"], (di, ti, ai), pk)
            np.add.at(acc[g]["w"], (di, ti, ai), pk)
            np.add.at(acc[g]["wp"], (di, ti, ai), pk * pr)
        if i % 100 == 0:
            print(f"  rg {i}", flush=True)
    return acc, D, A, labs


def frame(acc, D, A, labs, g, dz):
    packs = acc[g]["packs"]; present = packs > 0
    with np.errstate(all="ignore"):
        price = np.where(acc[g]["w"] > 0, acc[g]["wp"] / acc[g]["w"], np.nan)
    q_rt = packs.sum(2)
    lab = labs[g]
    yr = np.array([int(str(x)[:4]) for x in lab])
    Mry = np.zeros_like(q_rt)
    for y in np.unique(yr):
        s = yr == y
        Mry[:, s] = (MULT * np.where(q_rt[:, s] > 0, q_rt[:, s], 0).max(1))[:, None]
    with np.errstate(all="ignore"):
        sj = np.where(present, packs / np.where(Mry > 0, Mry, np.nan)[:, :, None], np.nan)
    s0 = 1.0 - np.nansum(np.where(present, sj, 0.0), axis=2)
    # sale flag: the round-3 rule applied at the estimation level -- own prior four
    # PERIODS median of present periods, 10% drop, next period rebounds 5%
    nP = len(lab)
    pri = np.full(price.shape + (4,), np.nan)
    pro = np.zeros(price.shape + (4,), dtype=bool)
    for k in range(1, 5):
        pri[:, k:, :, k - 1] = price[:, :-k, :]
        pro[:, k:, :, k - 1] = present[:, :-k, :]
    with np.errstate(all="ignore"):
        pmed = np.nanmedian(np.where(pro, pri, np.nan), axis=3)
    nxt = np.full_like(price, np.nan); nxt[:, :-1, :] = price[:, 1:, :]
    nxo = np.zeros_like(present); nxo[:, :-1, :] = present[:, 1:, :]
    with np.errstate(all="ignore"):
        sale = (present & nxo & np.isfinite(pmed) & (pmed > 0)
                & (price <= (1 - SALE_DROP) * pmed) & (nxt >= (1 + REBOUND) * price))
    brand = np.array([x[0] for x in A]); cont = np.array([x[1] for x in A])
    ncan = np.array([x[2] for x in A])
    sb = brand[:, None] == brand[None, :]; sc = cont[:, None] == cont[None, :]
    sn = ncan[:, None] == ncan[None, :]; eye = np.eye(len(A), dtype=bool)
    CLS = {"d_same_brand_other_rung": sb & sc & ~sn & ~eye,
           "d_same_rung_other_brand": sc & sn & ~sb & ~eye,
           "d_same_container_other": sc & ~sn & ~sb & ~eye,
           "d_other_container": (~sc) & ~eye}
    flat = present.reshape(-1, len(A)).astype(np.float32)
    dc = {k: (flat @ v.astype(np.float32)).reshape(present.shape) for k, v in CLS.items()}
    Jn = present.sum(2)
    di, ti, ai = np.where(present)
    d = pd.DataFrame(dict(dma=D[di], period=np.array(lab)[ti], dmac=di, per=ti, alt=ai,
                          brand=brand[ai], container=cont[ai], n_cans=ncan[ai],
                          packs=packs[present], price=price[present], sj=sj[present],
                          s0=np.repeat(s0[Jn > 0], Jn[Jn > 0]),
                          J_rt=np.repeat(Jn[Jn > 0], Jn[Jn > 0]).astype(float),
                          is_sale=sale[present],
                          **{k: v[present] for k, v in dc.items()}))
    d["year"] = d.period.map(lambda x: int(str(x)[:4]))
    d["s_jg"] = d.packs / d.groupby(["dmac", "per"], observed=True)["packs"].transform("sum")
    # distance and diesel
    DD = pd.read_csv(DERIVED / "distance_matrix.csv", dtype={"dma": str})
    grp = {**{b: "ABI" for b in ABI}, **{b: "MC" for b in MC},
           **{b: "MEX" for b in MEX}, **{b: "NL" for b in NLB}}
    d["pgroup"] = d.brand.str.upper().str.strip().map(grp)
    dmap = {gg: dict(zip(DD.dma, DD[gg])) for gg in ("ABI", "MC", "MEX", "NL")}
    d["distance"] = [dmap[gg].get(x, np.nan) for gg, x in zip(d.pgroup, d.dma.astype(str))]
    reg = json.loads((PAPER / "outputs" / "blp_validity" / "dma_region_map.json").read_text())
    FIPS = {"01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT",
            "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL",
            "18": "IN", "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME", "24": "MD",
            "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT", "31": "NE",
            "32": "NV", "33": "NH", "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
            "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
            "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA", "54": "WV",
            "55": "WI", "56": "WY"}
    PD = {}
    for p, sts in {"PADD1": "CT DE DC FL GA ME MD MA NH NJ NY NC PA RI SC VT VA WV",
                   "PADD2": "IL IN IA KS KY MI MN MO NE ND OH OK SD TN WI",
                   "PADD3": "AL AR LA MS NM TX", "PADD4": "CO ID MT UT WY",
                   "PADD5": "AK AZ CA HI NV OR WA"}.items():
        for x in sts.split():
            PD[x] = p
    d["padd"] = d.dma.astype(str).map(
        lambda x: PD.get(FIPS.get(str(reg.get(x, {}).get("state", "")).zfill(2), ""), ""))
    def agg_diesel(s):
        if g == "week":
            return {str(w.date()): float(s.loc[(s.date > w - pd.Timedelta(days=7))
                                               & (s.date <= w), "value"].iloc[-1])
                    for w in pd.DatetimeIndex([pd.Timestamp(x) for x in labs["week"]])
                    if len(s.loc[(s.date > w - pd.Timedelta(days=7)) & (s.date <= w)])}
        if g == "month":
            return s.assign(k=s.date.dt.strftime("%Y-%m")).groupby("k")["value"].mean().to_dict()
        return s.assign(k=s.date.dt.year.astype(str) + "Q"
                        + s.date.dt.quarter.astype(str)).groupby("k")["value"].mean().to_dict()
    nat = agg_diesel(dz["national"])
    pad = {k: agg_diesel(v) for k, v in dz.items() if k != "national"}
    d["diesel_nat"] = d.period.map(nat)
    d["diesel_padd"] = [pad.get(p, nat).get(t, np.nan) for p, t in zip(d.padd, d.period)]
    d["z_dist_diesel"] = d.distance * d.diesel_nat
    d["z_dd_padd"] = d.distance * d.diesel_padd
    return d.loc[(d.sj > 0) & (d.s0 > 0) & (d.s_jg > 0) & d.z_dist_diesel.notna()
                 & d.z_dd_padd.notna()].copy()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with RunManifest(OUT, args={}, inputs=[PANEL, DERIVED / "distance_matrix.csv"],
                     tag="frequency grid (round 19)") as rm:
        rm.add_note("Linear IV only. Panels built fresh at three frequencies with no Hausman "
                    "filter; J_rt carried with the classes. Within-year max market size.")
        dz = diesel_tables()
        acc, D, A, labs = build_all()
        grid, dec, div, part3 = [], [], [], []
        cells = {}
        for g in ("week", "month", "quarter"):
            e = frame(acc, D, A, labs, g, dz)
            cells[g] = e
            S = e[CLS_NAMES].sum(1).to_numpy()
            ident = float(np.mean(S == e.J_rt.to_numpy() - 1))
            part3.append(dict(frequency=g, n=len(e), identity_holds_pct=100 * ident))
            print(f"\n{g}: {len(e):,} rows, {e.alt.nunique()} alternatives, "
                  f"partition identity holds on {100 * ident:.4f}% of rows", flush=True)
            y = (np.log(e.sj) - np.log(e.s0)).to_numpy()
            cl = pd.factorize(e.dmac)[0]
            pr_ = e.price.to_numpy(float); sjv = e.sj.to_numpy(float)
            s0v = e.s0.to_numpy(float)
            ga = pd.factorize(e.alt)[0]; gt = pd.factorize(e.per)[0]
            gd = pd.factorize(e.dmac)[0]
            gdy = pd.factorize(list(zip(e.dmac, e.year)))[0]
            gbd = pd.factorize(list(zip(e.brand, e.dmac)))[0]
            ZN = ["z_dist_diesel", "z_dd_padd"] + R3
            FE = [("alt+time", [ga, gt]), ("+dma", [ga, gt, gd]),
                  ("+dma_x_year", [ga, gt, gdy]),
                  ("+dma_x_year+brand_x_dma", [ga, gt, gdy, gbd])]
            for felab, groups in FE:
                Aa = absorbN(np.column_stack([y, pr_, e[ZN].to_numpy(float)]), groups)
                yd, p_, Z = Aa[:, 0], Aa[:, [1]], Aa[:, 2:]
                ngroups = sum(int(x.max()) + 1 for x in groups)
                # Part 2.1: share of residual price variance from sale periods
                sm = e.is_sale.to_numpy(bool)
                tot = float(np.sum(p_[:, 0] ** 2))
                dec.append(dict(frequency=g, fe=felab, sale_period_share_of_rows=float(sm.mean()),
                                sale_share_of_resid_price_var=float(
                                    np.sum(p_[sm, 0] ** 2) / tot),
                                nonsale_share=float(np.sum(p_[~sm, 0] ** 2) / tot)))
                for fam, zs in FAMILIES:
                    idx = [ZN.index(z) for z in zs]
                    Zf = Z[:, idx]
                    b, V, _, G = M.tsls(yd, p_, Zf, cl)
                    sw = M.sw_stats(p_[:, 0], np.empty((len(yd), 0)), Zf, cl)
                    if Zf.shape[1] > 1:
                        J, Jdf, Jp, _ = M.hansen_j(yd, p_, Zf, cl)
                    else:
                        J = Jdf = Jp = np.nan
                    el = b[0] * pr_ * (1 - sjv)
                    grid.append(dict(frequency=g, fe=felab, family=fam, n_iv=Zf.shape[1],
                                     n=len(yd), clusters=G, absorbed_groups=ngroups,
                                     alpha=b[0], se=float(np.sqrt(V[0, 0])),
                                     elast_p50=float(np.median(el)), sw_F=sw["sw_F"],
                                     hansen_J=J, hansen_df=Jdf, hansen_p=Jp))
                    print(f"  {g:8s} {felab:24s} {fam[:30]:30s} alpha={b[0]:+9.4f} "
                          f"el50={grid[-1]['elast_p50']:+9.3f} SWF={sw['sw_F']:8.1f}", flush=True)
            # Part 2.3: outside diversion, logit: D_j0 = s0 / (1 - s_j)
            dv = s0v / (1 - sjv)
            div.append(dict(frequency=g, mean_outside_share=float(np.mean(
                e.groupby(["dmac", "per"], observed=True)["s0"].first())),
                median_outside_diversion_pct=float(100 * np.median(dv)),
                p10=float(100 * np.percentile(dv, 10)),
                p90=float(100 * np.percentile(dv, 90))))
        # Part 2.2: the sale-week split, DMA-week brand-by-DMA cell
        e = cells["week"]
        split = []
        for lab, mask in (("all periods", np.ones(len(e), bool)),
                          ("excluding sale periods", ~e.is_sale.to_numpy(bool)),
                          ("sale periods only", e.is_sale.to_numpy(bool))):
            s = e.loc[mask]
            if len(s) < 1000:
                continue
            y = (np.log(s.sj) - np.log(s.s0)).to_numpy()
            Aa = absorbN(np.column_stack([y, s.price.to_numpy(float),
                                          s[R3].to_numpy(float)]),
                         [pd.factorize(s.alt)[0], pd.factorize(s.per)[0],
                          pd.factorize(list(zip(s.dmac, s.year)))[0],
                          pd.factorize(list(zip(s.brand, s.dmac)))[0]])
            cl = pd.factorize(s.dmac)[0]
            b, V, _, _ = M.tsls(Aa[:, 0], Aa[:, [1]], Aa[:, 2:], cl)
            sw = M.sw_stats(Aa[:, 1], np.empty((len(y), 0)), Aa[:, 2:], cl)
            el = b[0] * s.price.to_numpy(float) * (1 - s.sj.to_numpy(float))
            split.append(dict(sample=lab, n=len(s), alpha=b[0], se=float(np.sqrt(V[0, 0])),
                              elast_p50=float(np.median(el)), sw_F=sw["sw_F"]))
            print(f"  split {lab:24s} n={len(s):>9,} alpha={b[0]:+.4f} "
                  f"el50={split[-1]['elast_p50']:+9.3f} SWF={sw['sw_F']:8.1f}", flush=True)
        for nm, obj in (("grid", grid), ("part2_price_decomposition", dec),
                        ("part2_sale_split", split), ("part2_outside_diversion", div),
                        ("part3_identity", part3)):
            pd.DataFrame(obj).to_csv(OUT / f"{nm}.csv", index=False)
        pd.set_option("display.width", 260)
        print("\n--- grid ---"); print(pd.DataFrame(grid).to_string(index=False))
        print("\n--- price decomposition ---"); print(pd.DataFrame(dec).to_string(index=False))
        print("\n--- sale split ---"); print(pd.DataFrame(split).to_string(index=False))
        print("\n--- outside diversion ---"); print(pd.DataFrame(div).to_string(index=False))
        print("\n--- identity ---"); print(pd.DataFrame(part3).to_string(index=False))
        rm.add_outputs(sorted(OUT.glob("*.csv")))
    return 0


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    sys.exit(main())
