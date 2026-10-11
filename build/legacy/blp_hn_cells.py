#!/usr/bin/env python3
"""Round 47 Part 1: which product-store cells have Hu-Nevo (2013) structure?

The unit is the PRODUCT-STORE CELL, not the category and not the market.  HN identify off the
price PATH -- `p_{t-1} > p_t` crossed with `p_t > p_{t+1}` -- with product-store effects and NO
time effects, so the feasibility question is per cell: is there an identifiable regular price,
are there countable deep temporary reductions, and are all four intertemporal states populated?
A category-wide median sale spell of one week (round 30) does not answer it.

Everything is computed on the corrected panel's own grid: 2,716 stores x 209 weeks x 116
alternatives, prices per 144-ounce equivalent, the 29 D090/D095 UPCs subtracted.  `build_grid`
is the shared builder Parts 2 and 3 import.

DESCRIPTIVE ONLY.  Counts, distributions, a selection grid.  No threshold is chosen.

    python3 paper/src/diagnostics/blp_hn_cells.py
"""
from __future__ import annotations
import gc, json, sys, time
from pathlib import Path
import _paths
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
PAPER = _paths.WORK
sys.path.insert(0, str(PAPER / "src" / "common"))
sys.path.insert(0, str(HERE.parent))
from run_manifest import RunManifest                                  # noqa: E402
from sale_definitions import (SALE_DROP_THRESHOLD, PRIOR_LAGS,        # noqa: E402
                              NEXT_REBOUND_THRESHOLD)
import blp_store_week_demand as SWD                                   # noqa: E402
import blp_chain_by_week as CBW                                       # noqa: E402

RMS = _paths.RMS
OUT = PAPER / "outputs" / "blp_hncells"
SCRATCH = _paths.WORK
SWD.SCRATCH = SCRATCH
CBW.SCRATCH = SCRATCH
YEARS = [2016, 2017, 2018, 2019]
BOTTLE_FLOOR, EVERY = 2000, 4
CHUNK = 30_000
MODE_TOL = 0.02                    # HN Figure 1: "flat regular price" band
STATE_TOL = [0.0, 0.01, 0.02]      # tolerance on "p_{t-1} > p_t", since prices are unit values
SPELL_GRID = [4, 8, 12]
DEPTH_GRID = [0.10, 0.15, 0.20]
MIN_WEEKS = 52                     # the "too few observed weeks" tier
STATE_MIN = 20                     # HN feasibility: >= 20 weeks in each of the four states
REF = dict(n=20_493_943, markets=447_440, stores=2_716, weeks=209, alts=116, brands=14,
           s0_mean=0.5683)
# round 30 section 2.3, store-cell spells, by n_cans: (spells, mean, median, one-week share)
R30 = {6: (162_322, 1.33, 1.0, 0.712), 12: (1_568_502, 1.27, 1.0, 0.767),
       18: (900_769, 1.19, 1.0, 0.825), 24: (544_527, 1.26, 1.0, 0.763),
       30: (210_632, 1.24, 1.0, 0.791), 36: (16_818, 1.35, 1.0, 0.683)}


def store_set(z):
    """The corrected panel's store sample: F/L universe, median-week floor, one-in-four."""
    chan, tot_bt = z["chan"], z["tot_bt"]
    nS0 = tot_bt.shape[0]
    inF = np.zeros(nS0, bool); other = np.zeros(nS0, bool)
    for k in range(len(YEARS)):
        inF |= np.isin(chan[:, k], ["F"])
        other |= (~np.isin(chan[:, k], ["F"])) & (chan[:, k] != "")
    present = tot_bt > 0
    univ = inF & ~other & (present.sum(1) > 0)
    med = np.full(nS0, np.nan)
    for i in np.where(univ)[0]:
        v = tot_bt[i][present[i]]
        if len(v):
            med[i] = np.median(v)
    unb = univ & (np.nan_to_num(med) >= BOTTLE_FLOOR)
    return np.where(unb)[0][::EVERY], unb, present


def build_grid(verbose=True):
    """(store, week, alternative) price grid on the corrected panel, with the panel's own
    row filter applied, and the reference row and market counts asserted.

    Returns a dict with the grid, the masks, the labels and the store metadata.
    """
    z = np.load(SCRATCH / "storeweek_totals.npz", allow_pickle=True)
    stores = pd.Index(z["stores"].astype(str))
    weeks = pd.DatetimeIndex(pd.to_datetime(z["weeks"].astype(str)))
    cells = z["cells"].astype(str)
    nW, nA = len(weeks), len(cells)
    si, unb, present = store_set(z)
    nS = len(si)
    brand = np.array([c.split("|")[0] for c in cells])
    cont = np.array([c.split("|")[1] for c in cells])
    ncan = np.array([int(c.split("|")[2]) for c in cells])
    PK, RV, kept = SWD.load_dense(si, nW, nA)
    XPK, XRV, nex = CBW.load_excluded(si, nW, nA)
    PK = np.maximum(PK - XPK, 0.0); RV = np.maximum(RV - XRV, 0.0)
    del XPK, XRV; gc.collect()
    ncan32 = ncan.astype(np.float32)
    TOT = z["tot_sv"][si]
    ins = np.tensordot(PK, ncan32, axes=([2], [0]))
    with np.errstate(all="ignore"):
        S0 = 1.0 - ins / np.where(TOT > 0, TOT, np.nan)
    good = np.isfinite(S0) & (S0 > 0) & (ins > 0)
    with np.errstate(all="ignore"):
        P = RV / (PK * ncan32[None, None, :] / 12.0)
    del RV; gc.collect()
    valid = (PK > 0) & good[:, :, None] & np.isfinite(P) & (P > 0)
    P = np.where(valid, P, np.nan).astype(np.float32)
    n = int(valid.sum()); nM = int(good.sum())
    if verbose:
        print(f"  grid {nS:,} stores x {nW} weeks x {nA} alternatives; rows {n:,} "
              f"(reference {REF['n']:,}); markets {nM:,} (reference {REF['markets']:,})",
              flush=True)
    assert len(set(brand)) == REF["brands"], sorted(set(brand))
    s0_mean = float(np.nanmean(np.where(good, S0, np.nan)))
    if verbose:
        print(f"  `s_0` mean over markets {s0_mean:.4f} (reference {REF['s0_mean']}); "
              f"stores {nS:,}, weeks {nW}, alternatives {nA}, brands "
              f"{len(set(brand))} -- all asserted", flush=True)
    return dict(P=P, PK=PK, valid=valid, good=good, S0=S0, TOT=TOT, ins=ins,
                si=si, stores=stores, weeks=weeks, cells=cells, brand=brand,
                cont=cont, ncan=ncan, nS=nS, nW=nW, nA=nA, n=n, nM=nM,
                dma=z["dma"], par=z["par"], present=present, s0_mean=s0_mean)


def cell_stats(Pc, Oc):
    """Per-cell price-path statistics.  Pc, Oc are (cells, weeks) price and observed."""
    nC, nW = Pc.shape
    out = {}
    cents = np.where(Oc, np.round(Pc.astype(np.float64), 2), np.nan)
    srt = np.sort(cents, axis=1)                      # NaNs sort to the end
    nobs = Oc.sum(1)
    # distinct values and the modal run, from the sorted array
    diff = np.diff(srt, axis=1)
    isnew = np.zeros_like(srt, dtype=bool)
    isnew[:, 0] = np.isfinite(srt[:, 0])
    isnew[:, 1:] = np.isfinite(srt[:, 1:]) & (diff != 0)
    out["n_distinct"] = isnew.sum(1)
    # run id per sorted position, then the longest run's value = the mode
    runid = np.cumsum(isnew, axis=1)
    mode = np.full(nC, np.nan)
    for i in range(nC):
        k = int(out["n_distinct"][i])
        if k == 0:
            continue
        r = runid[i, :int(nobs[i])]
        cnt = np.bincount(r, minlength=k + 1)[1:]
        # ties -> the higher price, since HN's regular price is the high one
        best = np.max(cnt)
        cand = np.where(cnt == best)[0]
        out_idx = cand[-1]
        mode[i] = srt[i, np.searchsorted(r, out_idx + 1)]
    out["mode"] = mode
    with np.errstate(all="ignore"):
        rel = np.abs(Pc / mode[:, None] - 1.0)
        out["share_within_mode"] = np.where(Oc & (rel <= MODE_TOL), 1, 0).sum(1) / \
            np.maximum(nobs, 1)
        mx = np.nanmax(np.where(Oc, Pc, np.nan), axis=1)
        relmx = np.abs(Pc / mx[:, None] - 1.0)
        out["share_within_max"] = np.where(Oc & (relmx <= MODE_TOL), 1, 0).sum(1) / \
            np.maximum(nobs, 1)
    out["p_max"] = mx
    out["n_obs"] = nobs
    return out


def prior_median_obs(Pc, Oc, lags=PRIOR_LAGS):
    """Median of the strictly-lagged OBSERVED prices -- sale_definitions.prior_median."""
    pp = np.stack([np.pad(Pc[:, :-l], ((0, 0), (l, 0)), constant_values=np.nan)
                   for l in lags], axis=2)
    po = np.stack([np.pad(Oc[:, :-l], ((0, 0), (l, 0)), constant_values=False)
                   for l in lags], axis=2)
    with np.errstate(all="ignore"):
        return np.nanmedian(np.where(po, pp, np.nan), axis=2)


def spells_and_states(Pc, Oc, bench, mode):
    """Sale spells under round 30's rule, and the four HN intertemporal states."""
    nC, nW = Pc.shape
    with np.errstate(all="ignore"):
        low = Oc & np.isfinite(bench) & (bench > 0) & \
            (Pc <= (1 - SALE_DROP_THRESHOLD) * bench)
        depth = np.where(low, 1.0 - Pc / bench, np.nan)
        depth_mode = np.where(low, 1.0 - Pc / mode[:, None], np.nan)
        # the rebound variant, for the round-03 caveat only
        nxt_p = np.pad(Pc[:, 1:], ((0, 0), (0, 1)), constant_values=np.nan)
        nxt_o = np.pad(Oc[:, 1:], ((0, 0), (0, 1)), constant_values=False)
        sale = low & nxt_o & (nxt_p >= (1 + NEXT_REBOUND_THRESHOLD) * Pc)
    starts = low & ~np.pad(low[:, :-1], ((0, 0), (1, 0)), constant_values=False)
    out = dict(n_spells=starts.sum(1), n_low_weeks=low.sum(1), n_sale_weeks=sale.sum(1))
    # per-spell length and depth, by segment id
    sid = np.cumsum(starts, axis=1)
    med_depth = np.full(nC, np.nan); mean_depth = np.full(nC, np.nan)
    med_len = np.full(nC, np.nan); mean_len = np.full(nC, np.nan)
    med_depth_mode = np.full(nC, np.nan)
    one_week = np.zeros(nC)
    for i in range(nC):
        k = int(out["n_spells"][i])
        if k == 0:
            continue
        m = low[i]
        s = sid[i][m]
        ln = np.bincount(s, minlength=k + 1)[1:]
        dp = np.zeros(k); dm = np.zeros(k)
        np.maximum.at(dp, s - 1, depth[i][m])
        np.maximum.at(dm, s - 1, depth_mode[i][m])
        med_len[i] = np.median(ln); mean_len[i] = ln.mean()
        one_week[i] = float((ln == 1).mean())
        med_depth[i] = np.median(dp); mean_depth[i] = dp.mean()
        med_depth_mode[i] = np.median(dm)
    out.update(median_spell_len=med_len, mean_spell_len=mean_len,
               one_week_share=one_week, median_depth=med_depth, mean_depth=mean_depth,
               median_depth_vs_mode=med_depth_mode)
    # the four HN states, on price in LEVELS, requiring t-1, t and t+1 all observed
    prv_p = np.pad(Pc[:, :-1], ((0, 0), (1, 0)), constant_values=np.nan)
    prv_o = np.pad(Oc[:, :-1], ((0, 0), (1, 0)), constant_values=False)
    nxt_p = np.pad(Pc[:, 1:], ((0, 0), (0, 1)), constant_values=np.nan)
    nxt_o = np.pad(Oc[:, 1:], ((0, 0), (0, 1)), constant_values=False)
    trip = Oc & prv_o & nxt_o
    out["n_triples"] = trip.sum(1)
    for tol in STATE_TOL:
        with np.errstate(all="ignore"):
            fell = trip & (prv_p > (1 + tol) * Pc)
            willfall = trip & (Pc > (1 + tol) * nxt_p)
        key = f"{tol:g}"
        out[f"s_fall_fall|{key}"] = (fell & willfall).sum(1)
        out[f"s_fall_notfall|{key}"] = (fell & ~willfall & trip).sum(1)
        out[f"s_notfall_fall|{key}"] = (~fell & willfall & trip).sum(1)
        out[f"s_notfall_notfall|{key}"] = (~fell & ~willfall & trip).sum(1)
        with np.errstate(all="ignore"):
            flat_l = trip & (np.abs(prv_p / Pc - 1.0) <= max(tol, 1e-12))
            flat_r = trip & (np.abs(nxt_p / Pc - 1.0) <= max(tol, 1e-12))
        out[f"flat_lag|{key}"] = flat_l.sum(1)
        out[f"flat_lead|{key}"] = flat_r.sum(1)
    return out


def qtab(x, w=None, qs=(1, 5, 10, 25, 50, 75, 90, 95, 99)):
    x = np.asarray(x, float)
    m = np.isfinite(x)
    d = {f"p{q}": float(np.percentile(x[m], q)) for q in qs} if m.any() else {}
    d.update(mean=float(np.nanmean(x)) if m.any() else np.nan,
             sd=float(np.nanstd(x)) if m.any() else np.nan, n=int(m.sum()))
    if w is not None:
        ww = np.asarray(w, float)[m]
        d["wmean"] = float(np.average(x[m], weights=ww)) if ww.sum() > 0 else np.nan
    return d


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with RunManifest(OUT, args={"chunk": CHUNK, "mode_tol": MODE_TOL},
                     inputs=[SCRATCH / "focal_rows.bin", SCRATCH / "excluded_rows.bin"],
                     tag="HN cell structure and the selection grid (round 47 Part 1)") as rm:
        rm.add_note("Descriptive. The unit is the product-store cell. No threshold chosen.")
        t0 = time.time()
        print("=== 0  the reference assertion ===", flush=True)
        G = build_grid()
        nS, nW, nA = G["nS"], G["nW"], G["nA"]
        brand, cont, ncan = G["brand"], G["cont"], G["ncan"]
        # (cell, week) layout: cell = store * nA + alternative
        P = np.ascontiguousarray(G["P"].transpose(0, 2, 1)).reshape(nS * nA, nW)
        O = np.isfinite(P)
        del G["P"]; gc.collect()
        cell_store = np.repeat(np.arange(nS), nA)
        cell_alt = np.tile(np.arange(nA), nS)
        has = O.any(1)
        idx = np.where(has)[0]
        nC = len(idx)
        print(f"\n=== 1  {nC:,} non-empty product-store cells of {nS*nA:,} possible "
              f"({100*nC/(nS*nA):.1f}%) [{time.time()-t0:.0f}s] ===", flush=True)
        acc = {}
        vol = np.zeros(nS * nA)
        PKt = np.ascontiguousarray(G["PK"].transpose(0, 2, 1)).reshape(nS * nA, nW)
        vol = (PKt * ncan[cell_alt][:, None]).sum(1)          # servings per cell
        del PKt, G["PK"]; gc.collect()
        for a in range(0, nC, CHUNK):
            ii = idx[a:a + CHUNK]
            Pc = P[ii]; Oc = O[ii]
            st = cell_stats(Pc, Oc)
            bench = prior_median_obs(Pc, Oc)
            sp = spells_and_states(Pc, Oc, bench, st["mode"])
            for k, v in list(st.items()) + list(sp.items()):
                acc.setdefault(k, []).append(v)
            del Pc, Oc, bench, st, sp
            if a == 0 or (a // CHUNK) % 3 == 0:
                print(f"    cells {a:,}-{min(a+CHUNK, nC):,} [{time.time()-t0:.0f}s]",
                      flush=True)
            gc.collect()
        C = pd.DataFrame({k: np.concatenate(v) for k, v in acc.items()})
        C["cell"] = idx
        C["store"] = cell_store[idx]; C["alt"] = cell_alt[idx]
        C["brand"] = brand[C.alt.to_numpy()]; C["cont"] = cont[C.alt.to_numpy()]
        C["ncan"] = ncan[C.alt.to_numpy()]
        C["servings"] = vol[idx]
        del acc; gc.collect()
        print(f"  cell table built: {len(C):,} cells, {C.n_obs.sum():,} rows "
              f"(panel {REF['n']:,}) [{time.time()-t0:.0f}s]", flush=True)
        assert int(C.n_obs.sum()) == REF["n"], (int(C.n_obs.sum()), REF["n"])
        res = {"n_cells": int(nC), "n_possible": int(nS * nA), "rows": int(C.n_obs.sum())}

        # ---------------- 1.1 is a regular price identifiable?
        print(f"\n=== 1.1  is a regular price identifiable? ===", flush=True)
        res["weeks_observed"] = qtab(C.n_obs, C.servings)
        print(f"  weeks observed per cell: " + "  ".join(
            f"p{q}={res['weeks_observed'][f'p{q}']:.0f}" for q in (5, 25, 50, 75, 95))
            + f"  mean {res['weeks_observed']['mean']:.1f}")
        for nm, col in (("share of weeks within 2% of the cell's MODE", "share_within_mode"),
                        ("share of weeks within 2% of the cell's MAX", "share_within_max"),
                        ("distinct prices (rounded to the cent)", "n_distinct")):
            d = qtab(C[col], C.servings)
            res[col] = d
            print(f"  {nm:44s} mean {d['mean']:7.3f}  volume-weighted {d['wmean']:7.3f}  "
                  + "  ".join(f"p{q}={d[f'p{q}']:.3f}" for q in (10, 25, 50, 75, 90)))
        for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
            m = C.share_within_mode >= thr
            print(f"    cells with >= {100*thr:2.0f}% of weeks within 2% of the mode: "
                  f"{int(m.sum()):>7,} ({100*m.mean():5.1f}%), carrying "
                  f"{100*C.servings[m].sum()/C.servings.sum():5.1f}% of servings")
            res[f"mode_ge_{thr:g}"] = dict(cells=int(m.sum()), share=float(m.mean()),
                                           volume=float(C.servings[m].sum()
                                                        / C.servings.sum()))

        # ---------------- 1.2 how many temporary reductions?
        print(f"\n=== 1.2  temporary reductions, round 30's rule "
              f"(p <= 0.90 x median of the prior four OBSERVED weeks) ===", flush=True)
        for nm, col in (("sale spells per cell", "n_spells"),
                        ("median spell length (weeks)", "median_spell_len"),
                        ("mean spell length (weeks)", "mean_spell_len"),
                        ("median depth vs the prior median", "median_depth"),
                        ("median depth vs the cell's mode", "median_depth_vs_mode"),
                        ("share of observed weeks on sale", None)):
            if col is None:
                v = C.n_low_weeks / np.maximum(C.n_obs, 1)
            else:
                v = C[col]
            d = qtab(v, C.servings); res[nm] = d
            print(f"  {nm:38s} mean {d['mean']:8.3f}  vw {d['wmean']:8.3f}  "
                  + "  ".join(f"p{q}={d[f'p{q}']:7.3f}" for q in (10, 25, 50, 75, 90)))
        print(f"\n  the pooled cross-check against round 30 section 2.3, by n_cans:")
        r30 = []
        for k in sorted(set(ncan)):
            m = C.ncan == k
            sp = int(C.n_spells[m].sum())
            lens = np.repeat(C.mean_spell_len[m].to_numpy(),
                             C.n_spells[m].to_numpy().astype(int))
            ow = float(np.average(C.one_week_share[m][C.n_spells[m] > 0],
                                  weights=C.n_spells[m][C.n_spells[m] > 0])) \
                if (C.n_spells[m] > 0).any() else np.nan
            ref = R30.get(k)
            r30.append(dict(n_cans=int(k), spells=sp,
                            mean_len=float(np.nanmean(lens)) if len(lens) else np.nan,
                            one_week_share=ow,
                            ref_spells=ref[0] if ref else None,
                            ref_mean=ref[1] if ref else None,
                            ref_one_week=ref[3] if ref else None))
            print(f"    n_cans {k:>2}: spells {sp:>9,} mean len "
                  f"{r30[-1]['mean_len']:.3f} one-week {100*ow:5.1f}%   "
                  f"round 30 (5,208-store balanced panel): {ref[0]:>9,} / {ref[1]:.2f} / "
                  f"{100*ref[3]:.1f}%" if ref else "")
        pd.DataFrame(r30).to_csv(OUT / "p1_spells_by_pack.csv", index=False)
        res["round30_check"] = r30
        # the round 03 caveat, measured here
        lw = float(C.n_low_weeks.sum()); sw = float(C.n_sale_weeks.sum())
        print(f"\n  round 03's caveat, re-measured: low-price weeks {lw:,.0f}; of those "
              f"{100*(1-sw/lw):.4f}% fail the 5% rebound test and are not coded as sales "
              f"(round 03: 39.9994%)", flush=True)
        res["rebound_failure_pct"] = float(100 * (1 - sw / lw))

        # ---------------- 1.3 the selection grid
        print(f"\n=== 1.3  the selection grid -- NO threshold is chosen ===", flush=True)
        tot_serv = C.servings.sum()
        grid = []
        for ns in SPELL_GRID:
            for dp in DEPTH_GRID:
                m = (C.n_spells >= ns) & (C.median_depth >= dp) & (C.n_obs >= MIN_WEEKS)
                g = dict(min_spells=ns, min_depth=dp, cells=int(m.sum()),
                         cell_share=float(m.mean()),
                         volume_share=float(C.servings[m].sum() / tot_serv),
                         rows=int(C.n_obs[m].sum()),
                         row_share=float(C.n_obs[m].sum() / C.n_obs.sum()),
                         stores=int(C.store[m].nunique()),
                         alternatives=int(C.alt[m].nunique()),
                         brands=int(C.brand[m].nunique()))
                grid.append(g)
                print(f"  >= {ns:>2} spells, depth >= {100*dp:2.0f}%: cells "
                      f"{g['cells']:>7,} ({100*g['cell_share']:5.1f}%)  volume "
                      f"{100*g['volume_share']:5.1f}%  rows {g['rows']:>10,} "
                      f"({100*g['row_share']:5.1f}%)  stores {g['stores']:>5,}  "
                      f"alts {g['alternatives']:>4}  brands {g['brands']:>3}"
                      + ("   <-- an alternative or brand DISAPPEARS"
                         if g['alternatives'] < nA or g['brands'] < REF['brands'] else ""),
                      flush=True)
        pd.DataFrame(grid).to_csv(OUT / "p1_selection_grid.csv", index=False)
        res["grid"] = grid
        print(f"\n  NOTE: the >= 10% depth column is MECHANICAL -- round 30's rule already "
              f"requires a 10% cut, so every flagged week has depth >= 10% by "
              f"construction and that column measures only the spell count.", flush=True)

        # ---------------- 1.4 the sub-panel at each grid point
        print(f"\n=== 1.4  the selected sub-panel at each grid point "
              f"[{time.time()-t0:.0f}s] ===", flush=True)
        tot_sv = G["TOT"]; good = G["good"]
        allbeer = float(tot_sv[good].sum())
        sub = []
        for g in grid:
            m = ((C.n_spells >= g["min_spells"]) & (C.median_depth >= g["min_depth"])
                 & (C.n_obs >= MIN_WEEKS)).to_numpy()
            sst = np.unique(C.store.to_numpy()[m])
            keep_store = np.zeros(nS, bool); keep_store[sst] = True
            ab = float(tot_sv[keep_store][good[keep_store]].sum())
            foc = float((C.servings.to_numpy()[m]).sum())
            focall = float(C.servings.to_numpy()[np.isin(C.store.to_numpy(), sst)].sum())
            r = dict(min_spells=g["min_spells"], min_depth=g["min_depth"],
                     cells=g["cells"], stores=len(sst), alternatives=g["alternatives"],
                     brands=g["brands"], rows=g["rows"],
                     brands_share_of_allbeer=100 * focall / ab if ab else np.nan,
                     selected_share_of_allbeer=100 * foc / ab if ab else np.nan)
            sub.append(r)
            print(f"  >= {g['min_spells']:>2} spells, depth >= {100*g['min_depth']:2.0f}%: "
                  f"stores {r['stores']:>5,}  cells {r['cells']:>7,}  rows {r['rows']:>10,}"
                  f"  14 brands' share of all-beer in surviving stores "
                  f"{r['brands_share_of_allbeer']:5.2f}% (panel 45.47%); the SELECTED "
                  f"cells alone {r['selected_share_of_allbeer']:5.2f}%", flush=True)
        pd.DataFrame(sub).to_csv(OUT / "p1_subpanels.csv", index=False)
        res["subpanels"] = sub

        # ---------------- 1.5 why cells fail
        print(f"\n=== 1.5  why cells fail (ordered, each cell counted once) ===",
              flush=True)
        fail = []
        rem = np.ones(len(C), bool)
        tests = [("too few observed weeks (< 52)", (C.n_obs < MIN_WEEKS).to_numpy()),
                 ("no identifiable regular price (< 50% of weeks within 2% of mode)",
                  (C.share_within_mode < 0.5).to_numpy()),
                 ("too few spells (< 4)", (C.n_spells < 4).to_numpy()),
                 ("spells too shallow (median depth < 15%)",
                  (C.median_depth < 0.15).to_numpy() | ~np.isfinite(C.median_depth.to_numpy()))]
        for lab, t in tests:
            m = rem & t
            fail.append(dict(reason=lab, cells=int(m.sum()), share=float(m.mean()),
                             volume=float(C.servings.to_numpy()[m].sum() / tot_serv)))
            print(f"  {lab:64s} {int(m.sum()):>7,} ({100*m.mean():5.1f}%)  volume "
                  f"{100*fail[-1]['volume']:5.1f}%")
            rem = rem & ~t
        fail.append(dict(reason="PASS all four", cells=int(rem.sum()),
                         share=float(rem.mean()),
                         volume=float(C.servings.to_numpy()[rem].sum() / tot_serv)))
        print(f"  {'PASS all four':64s} {int(rem.sum()):>7,} ({100*rem.mean():5.1f}%)  "
              f"volume {100*fail[-1]['volume']:5.1f}%", flush=True)
        pd.DataFrame(fail).to_csv(OUT / "p1_failure_taxonomy.csv", index=False)
        res["failures"] = fail

        # ---------------- 1.6 the four intertemporal states
        print(f"\n=== 1.6  the four HN states, price in LEVELS ===", flush=True)
        st_rows = []
        for tol in STATE_TOL:
            k = f"{tol:g}"
            cols = [f"s_fall_fall|{k}", f"s_fall_notfall|{k}",
                    f"s_notfall_fall|{k}", f"s_notfall_notfall|{k}"]
            tot = C[cols].sum(1)
            allfour = np.ones(len(C), bool)
            r = dict(tolerance=tol, triples=int(C.n_triples.sum()))
            for c in cols:
                r[f"weeks_{c.split('|')[0]}"] = int(C[c].sum())
                r[f"pct_{c.split('|')[0]}"] = float(100 * C[c].sum() / tot.sum())
                allfour &= (C[c] >= STATE_MIN).to_numpy()
            r["cells_all_four_ge20"] = int(allfour.sum())
            r["share_all_four_ge20"] = float(allfour.mean())
            r["volume_all_four_ge20"] = float(C.servings.to_numpy()[allfour].sum()
                                              / tot_serv)
            r["flat_lag_pct"] = float(100 * C[f"flat_lag|{k}"].sum() / C.n_triples.sum())
            st_rows.append(r)
            print(f"  tolerance {100*tol:4.1f}%:  fall/fall {r['pct_s_fall_fall']:5.2f}%  "
                  f"fall/notfall {r['pct_s_fall_notfall']:5.2f}%  notfall/fall "
                  f"{r['pct_s_notfall_fall']:5.2f}%  notfall/notfall "
                  f"{r['pct_s_notfall_notfall']:5.2f}%")
            print(f"      cells with >= {STATE_MIN} weeks in ALL FOUR states: "
                  f"**{r['cells_all_four_ge20']:,}** ({100*r['share_all_four_ge20']:.1f}% "
                  f"of cells, {100*r['volume_all_four_ge20']:.1f}% of volume); "
                  f"transitions flat within tolerance {r['flat_lag_pct']:.1f}%", flush=True)
        pd.DataFrame(st_rows).to_csv(OUT / "p1_hn_states.csv", index=False)
        res["states"] = st_rows
        # and on the selected sub-panels
        print(f"\n  the same test on the selected sub-panels (tolerance 1%):")
        k = "0.01"
        cols = [f"s_fall_fall|{k}", f"s_fall_notfall|{k}", f"s_notfall_fall|{k}",
                f"s_notfall_notfall|{k}"]
        for g in grid:
            m = ((C.n_spells >= g["min_spells"]) & (C.median_depth >= g["min_depth"])
                 & (C.n_obs >= MIN_WEEKS)).to_numpy()
            if not m.any():
                continue
            a4 = np.ones(int(m.sum()), bool)
            for c in cols:
                a4 &= (C[c].to_numpy()[m] >= STATE_MIN)
            print(f"    >= {g['min_spells']:>2} spells, depth >= "
                  f"{100*g['min_depth']:2.0f}%: {int(a4.sum()):>7,} of {int(m.sum()):>7,} "
                  f"selected cells have all four states >= {STATE_MIN} weeks "
                  f"({100*a4.mean():5.1f}%)", flush=True)
        C.to_parquet(OUT / "p1_cell_table.parquet", index=False)
        json.dump(res, open(OUT / "p1_summary.json", "w"), indent=1, default=float)
        rm.add_outputs([OUT / f for f in ("p1_summary.json", "p1_cell_table.parquet",
                                          "p1_selection_grid.csv", "p1_subpanels.csv",
                                          "p1_failure_taxonomy.csv", "p1_hn_states.csv",
                                          "p1_spells_by_pack.csv")])
        print(f"\ndone [{time.time()-t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
