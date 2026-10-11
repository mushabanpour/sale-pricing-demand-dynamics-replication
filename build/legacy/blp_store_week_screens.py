#!/usr/bin/env python3
"""Round 36 Part 2: measure every screen before applying it.

The design fixes markets at the store-week, the outside good at all RMS beer outside the
model's brand set, balance at the store over the whole horizon, a 2,000-bottles-per-week
volume floor, and grocery plus liquor.  Two of those -- balance at the store and the
volume floor -- have never been measured on this data and either could empty the sample.
This reports each screen's bite separately and in composition, and stops if the survivors
cannot support estimation.

Runs off the store-week totals `blp_store_week_scan.py` wrote.  No movement pass.

    python3 paper/src/diagnostics/blp_store_week_screens.py
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
OUT = PAPER / "outputs" / "blp_storeweek"
SCRATCH = _paths.WORK
YEARS = [2016, 2017, 2018, 2019]
MODULES = [5000, 5001, 5005, 5010, 5015, 5020]
CHANNELS = ["F", "L"]                      # grocery and liquor, as specified
BOTTLE_FLOOR = 2000
QS = [1, 5, 25, 50, 75, 95, 99]
REF = dict(n_rows=2_595_065, n_markets=39_863, s0_peak=0.3757, s0_pop=0.0243)


def pct(x, qs=QS):
    return {f"p{q}": float(np.percentile(x, q)) for q in qs}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with RunManifest(OUT, args={"channels": str(CHANNELS), "floor": BOTTLE_FLOOR},
                     inputs=[RMS, SCRATCH / "storeweek_totals.npz"],
                     tag="store-week screens (round 36 Part 2)") as rm:
        rm.add_note("Attrition of every screen, separately and in composition. Nothing "
                    "adopted; the surviving panel is reported and brought back.")
        t0 = time.time()
        z = np.load(SCRATCH / "storeweek_totals.npz", allow_pickle=True)
        stores = pd.Index(z["stores"].astype(str)); chan = z["chan"]
        weeks = pd.DatetimeIndex(pd.to_datetime(z["weeks"].astype(str)))
        cells = z["cells"].astype(str)
        tot_bt, tot_pk, tot_sv = z["tot_bt"], z["tot_pk"], z["tot_sv"]
        foc_sv, old_sv = z["foc_sv"], z["old_sv"]
        nS, nW = tot_bt.shape
        yr_w = weeks.year.to_numpy()
        print(f"stores {nS:,}  weeks {nW}  alternatives {len(cells)}", flush=True)

        # ---------------- 2.1 channel and store universe
        print(f"\n=== 2.1  channel and store universe ===", flush=True)
        rows = []
        for k, y in enumerate(YEARS):
            sel = yr_w == y
            v = tot_sv[:, sel].sum(1); b = tot_bt[:, sel].sum(1)
            for ch in sorted({c for c in chan[:, k] if c}):
                m = (chan[:, k] == ch)
                act = m & (v > 0)
                rows.append(dict(year=y, channel=ch, stores_listed=int(m.sum()),
                                 stores_with_beer=int(act.sum()),
                                 servings=float(v[act].sum()),
                                 bottles=float(b[act].sum())))
        CH = pd.DataFrame(rows)
        CH["pct_of_year_servings"] = 100 * CH.servings / CH.groupby("year").servings.transform("sum")
        piv = CH.pivot_table(index="channel", columns="year",
                             values=["stores_with_beer", "servings"])
        print(piv.round(0).to_string())
        print("\n  year-on-year ratio of stores with beer, by channel:")
        sw = CH.pivot_table(index="channel", columns="year", values="stores_with_beer")
        sv = CH.pivot_table(index="channel", columns="year", values="servings")
        for ch in sw.index:
            r = [sw.loc[ch, y] / sw.loc[ch, y - 1] if y - 1 in sw.columns
                 and sw.loc[ch, y - 1] else np.nan for y in YEARS[1:]]
            rv = [sv.loc[ch, y] / sv.loc[ch, y - 1] if y - 1 in sv.columns
                  and sv.loc[ch, y - 1] else np.nan for y in YEARS[1:]]
            flag = "  <-- DISCONTINUITY" if any(x > 2 or x < 0.5 for x in r
                                                if np.isfinite(x)) else ""
            print(f"    {ch}: stores {['%.2f'%x for x in r]}  servings "
                  f"{['%.2f'%x for x in rv]}{flag}")
        CH.to_csv(OUT / "p2_channel_year.csv", index=False)

        # the working universe: a store is in if its channel is F or L in EVERY year it
        # is listed with beer, so a channel switch does not straddle the panel
        inF = np.zeros(nS, bool); anyother = np.zeros(nS, bool)
        for k in range(len(YEARS)):
            inF |= np.isin(chan[:, k], CHANNELS)
            anyother |= (~np.isin(chan[:, k], CHANNELS)) & (chan[:, k] != "")
        univ = inF & ~anyother
        print(f"\n  stores ever F or L: {int(inF.sum()):,}; of those, "
              f"{int((inF & anyother).sum()):,} also appear in another channel in some "
              f"year and are excluded; working universe {int(univ.sum()):,}", flush=True)

        # ---------------- 2.2 balance at the store
        print(f"\n=== 2.2  balance at the store ===", flush=True)
        present = tot_bt > 0
        wpres = present.sum(1)
        u = univ & (wpres > 0)
        print(f"  stores in the universe with any beer week: {int(u.sum()):,}")
        print("  weeks present per store: " + "  ".join(
            f"p{q}={np.percentile(wpres[u], q):.0f}" for q in QS)
            + f"  mean={wpres[u].mean():.1f}  max={wpres[u].max()}")
        wins = [("2016", 2016, 2016), ("2017", 2017, 2017), ("2018", 2018, 2018),
                ("2019", 2019, 2019), ("2016-2017", 2016, 2017),
                ("2017-2018", 2017, 2018), ("2018-2019", 2018, 2019),
                ("2016-2019 (full)", 2016, 2019)]
        wrows = []
        tot_all = float(tot_sv[u].sum())
        for lab, y0, y1 in wins:
            sel = (yr_w >= y0) & (yr_w <= y1)
            bal = u & present[:, sel].all(1)
            vol = float(tot_sv[np.ix_(bal, sel)].sum())
            volw = float(tot_sv[np.ix_(u, sel)].sum())
            wrows.append(dict(window=lab, weeks=int(sel.sum()),
                              stores_balanced=int(bal.sum()),
                              pct_of_universe=100 * bal.sum() / u.sum(),
                              pct_of_window_servings=100 * vol / volw))
            print(f"  {lab:18s} {int(sel.sum()):>3} weeks: {int(bal.sum()):>6,} stores "
                  f"({100*bal.sum()/u.sum():5.1f}% of the universe) carrying "
                  f"{100*vol/volw:5.1f}% of the window's beer servings", flush=True)
        pd.DataFrame(wrows).to_csv(OUT / "p2_balance_windows.csv", index=False)
        bal_full = u & present.all(1)

        # ---------------- 2.3 the volume screen
        print(f"\n=== 2.3  the volume screen ===", flush=True)
        sw_b = tot_bt[u][present[u]]
        sw_p = tot_pk[u][present[u]]
        sw_s = tot_sv[u][present[u]]
        for nm, arr, unit in (("bottles (individual containers)", sw_b, ""),
                              ("packs (movement `units`)", sw_p, ""),
                              ("144-oz equivalents", sw_s / 12.0, "")):
            d = pct(arr)
            print(f"  store-week {nm:32s} mean {arr.mean():9.1f} sd {arr.std():9.1f} "
                  f"min {arr.min():7.1f} " + " ".join(f"p{q}={d[f'p{q}']:9.1f}" for q in QS)
                  + f" max {arr.max():11.1f}")
        med = np.full(nS, np.nan); mn = np.full(nS, np.nan)
        idx = np.where(u)[0]
        for i in idx:
            v = tot_bt[i][present[i]]
            if len(v):
                med[i] = np.median(v); mn[i] = v.min()
        print(f"\n  stores whose MEDIAN store-week bottles >= {BOTTLE_FLOOR}: "
              f"{int(np.nansum(med[u] >= BOTTLE_FLOOR)):,} of {int(u.sum()):,} "
              f"({100*np.nansum(med[u]>=BOTTLE_FLOOR)/u.sum():.1f}%)")
        print(f"  stores whose MINIMUM store-week bottles >= {BOTTLE_FLOOR}: "
              f"{int(np.nansum(mn[u] >= BOTTLE_FLOOR)):,} of {int(u.sum()):,} "
              f"({100*np.nansum(mn[u]>=BOTTLE_FLOOR)/u.sum():.1f}%)", flush=True)
        for f in (500, 1000, 2000, 3000, 5000):
            print(f"    floor {f:>5}: median rule {int(np.nansum(med[u]>=f)):>6,} stores, "
                  f"minimum rule {int(np.nansum(mn[u]>=f)):>6,} stores")

        # ---------------- composition of the screens
        print(f"\n=== 2.4  the screens in composition ===", flush=True)
        steps = [("all RMS stores", np.ones(nS, bool)),
                 ("+ channel F or L, never another", univ),
                 ("+ any beer week", u),
                 ("+ balanced on all 209 weeks", bal_full),
                 (f"+ median week >= {BOTTLE_FLOOR} bottles",
                  bal_full & (np.nan_to_num(med) >= BOTTLE_FLOOR)),
                 (f"+ minimum week >= {BOTTLE_FLOOR} bottles",
                  bal_full & (np.nan_to_num(mn) >= BOTTLE_FLOOR))]
        srows = []
        for lab, m in steps:
            vol = float(tot_sv[m].sum())
            srows.append(dict(step=lab, stores=int(m.sum()),
                              pct_of_universe_servings=100 * vol / tot_all if tot_all else 0,
                              store_weeks=int(present[m].sum())))
            print(f"  {lab:36s} stores {int(m.sum()):>7,}  store-weeks "
                  f"{int(present[m].sum()):>10,}  beer servings "
                  f"{100*vol/tot_all:6.2f}% of the F/L universe", flush=True)
        pd.DataFrame(srows).to_csv(OUT / "p2_screen_ladder.csv", index=False)
        surv = steps[-1][1]

        # ---------------- 2.4 composition of the survivors
        ST = []
        for y in YEARS:
            s = pd.read_csv(RMS / f"{y}/Annual_Files/stores_{y}.tsv", sep="\t",
                            dtype={"store_code_uc": "string", "parent_code": "string",
                                   "retailer_code": "string", "dma_code": "string",
                                   "fips_state_descr": "string"},
                            encoding="latin-1", low_memory=False)
            s["year"] = y; ST.append(s)
        ST = pd.concat(ST, ignore_index=True)
        ST["store_code_uc"] = ST.store_code_uc.str.strip()
        last = ST.sort_values("year").drop_duplicates("store_code_uc", keep="last") \
                 .set_index("store_code_uc")
        def comp(mask, name):
            sc = stores[mask]
            g = last.reindex(sc)
            top = g.parent_code.value_counts(normalize=True)
            return dict(set=name, stores=int(mask.sum()),
                        parents=int(g.parent_code.nunique()),
                        dmas=int(g.dma_code.nunique()),
                        states=int(g.fips_state_descr.nunique()),
                        pct_channel_F=float(100 * (g.channel_code == "F").mean()),
                        pct_channel_L=float(100 * (g.channel_code == "L").mean()),
                        top1_parent_pct=float(100 * top.iloc[0]) if len(top) else np.nan,
                        top3_parent_pct=float(100 * top.head(3).sum()) if len(top) else np.nan,
                        top5_parent_pct=float(100 * top.head(5).sum()) if len(top) else np.nan)
        crows = [comp(u, "F/L universe"), comp(bal_full, "balanced 209 weeks"),
                 comp(surv, f"balanced + min >= {BOTTLE_FLOOR}")]
        C = pd.DataFrame(crows)
        print(f"\n  composition:"); print(C.round(2).to_string(index=False), flush=True)
        if surv.sum():
            g = last.reindex(stores[surv])
            print(f"\n  top parents among survivors: "
                  f"{g.parent_code.value_counts(normalize=True).head(8).mul(100).round(1).to_dict()}")
            print(f"  top DMAs: {g.dma_code.value_counts().head(6).to_dict()}")
        C.to_csv(OUT / "p2_composition.csv", index=False)

        # ---------------- 2.5 the outside good
        print(f"\n=== 2.5  the outside good ===", flush=True)
        for nm, m in (("F/L universe", u), ("balanced 209", bal_full),
                      (f"balanced + min>={BOTTLE_FLOOR}", surv)):
            if not m.sum():
                continue
            pr = present[m]
            tv = tot_sv[m][pr]; fv = foc_sv[m][pr]; ov = old_sv[m][pr]
            ins = np.clip(fv / np.where(tv > 0, tv, np.nan), 0, None)
            ins = ins[np.isfinite(ins)]
            d = pct(ins)
            print(f"  {nm:26s} store-weeks {len(ins):>10,}  model brands' share of beer "
                  f"servings {100*fv.sum()/tv.sum():5.2f}%  inside share mean "
                  f"{ins.mean():.4f} " + " ".join(f"p{q}={d[f'p{q}']:.4f}" for q in QS)
                  + f"  >0.95 {int((ins>0.95).sum()):,}  <0.001 {int((ins<0.001).sum()):,}",
                  flush=True)
            print(f"      under the BUILD's lookup the same share is "
                  f"{100*ov.sum()/tv.sum():5.2f}%")
        # module composition of the outside good
        mod_sv = z["mod_sv"]
        tot_mod = mod_sv.sum()
        print(f"\n  all-beer servings by module (all stores): "
              + ", ".join(f"{m} {100*mod_sv[i].sum()/tot_mod:.1f}%"
                          for i, m in enumerate(MODULES)), flush=True)

        # ---------------- 2.6 the final counts
        print(f"\n=== 2.6  the final counts ===", flush=True)
        final = dict(stores=int(surv.sum()), weeks=int(nW),
                     store_weeks=int(present[surv].sum()),
                     alternatives=len(cells))
        print(f"  surviving stores {final['stores']:,}; store-weeks "
              f"{final['store_weeks']:,} (markets); the standing frame has "
              f"{REF['n_markets']:,} markets and {REF['n_rows']:,} rows")
        json.dump(dict(final=final, floor=BOTTLE_FLOOR, channels=CHANNELS,
                       universe=int(u.sum()), balanced=int(bal_full.sum()),
                       med_rule=int(np.nansum(med[u] >= BOTTLE_FLOOR)),
                       min_rule=int(np.nansum(mn[u] >= BOTTLE_FLOOR))),
                  open(OUT / "p2_final_counts.json", "w"), indent=1)
        np.savez(SCRATCH / "survivors.npz", surv=surv, bal_full=bal_full, univ=u,
                 med=med, mn=mn, present=present)
        rm.add_outputs([OUT / f for f in ("p2_channel_year.csv", "p2_balance_windows.csv",
                                          "p2_screen_ladder.csv", "p2_composition.csv",
                                          "p2_final_counts.json")])
        print(f"\ndone [{time.time()-t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
