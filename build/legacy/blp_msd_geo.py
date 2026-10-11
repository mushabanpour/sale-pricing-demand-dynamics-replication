#!/usr/bin/env python3
"""Round 55 Part 1: the market-geography frontier.  Build four geographies, choose none.

Requirement 1 (a credible `alpha`) pulls toward store-week; requirements 2 (rho or
sigma_constant, which needs `s_0` variance to survive absorption) and 3 (Pi, which needs
within-market demographic dispersion) pull coarser.  Rounds 22, 45 and 46 measured the
tension at the two ends; **the geography BETWEEN store and DMA has never been built.**

  store-week      market = (store, week)                    1 store
  chain-DMA-week  market = (parent_code, dma_code, week)    5-50
  county-week     market = (state+county FIPS, week)        2-30
  DMA-week        market = (dma_code, week)                 10-200

Everything is aggregated from the SAME store-week grid, so the four geographies differ only
in how stores are grouped.  Revenue is reconstructed as `P * PK * ncan / 12`, exactly the
quantity `build_grid` divided to form `P`.  The full panel is used at every geography,
because a cell selection defined on store-cells does not survive re-aggregation; the
like-for-like store-week anchor is therefore round 50's FULL-panel row, not the adopted
panel's.

    python3 paper/src/diagnostics/blp_msd_geo.py
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
import blp_first_stage_strength as M                                  # noqa: E402
from blp_frequency_grid import R3                                     # noqa: E402
from blp_store_week_decomp import classes                             # noqa: E402
from blp_hn_cells import build_grid                                   # noqa: E402
from blp_dma_week_scan import store_table                             # noqa: E402
from blp_config_e import ACS, SCRATCH                                 # noqa: E402

OUT = PAPER / "outputs" / "blp_msd"
YEARS = [2016, 2017, 2018, 2019]
Z2 = ["d_same_rung_other_brand", "d_same_container_other"]
CRIT = 1.0 / 3.0
REF = dict(sw_rows=20_493_943, sw_markets=447_440, sw_s0=0.5683, sw_s0_surv=0.0888,
           sw_alpha_full=-1.1230, sw_se_full=0.2414, sw_F_full=17.09, sw_Jp_full=0.2288,
           sw_el_full=-13.94, adopted_alpha=-1.1103, adopted_se=0.1643,
           adopted_F=23.35, adopted_Jp=0.0724, adopted_el=-12.60, adopted_epsM=-7.0814)


def resid(X, groups):
    """pyhdfe residualisation, with the largest surviving group mean reported."""
    import pyhdfe
    ids = np.column_stack([np.asarray(g, np.int64) for g in groups])
    alg = pyhdfe.create(ids, drop_singletons=False, compute_degrees=False)
    A = np.asarray(alg.residualize(np.asarray(X, np.float64)), np.float64)
    mx = 0.0
    for g in groups:
        g = np.asarray(g, np.int64)
        c = np.maximum(np.bincount(g), 1.0)
        for k in range(A.shape[1]):
            m = np.bincount(g, weights=A[:, k], minlength=len(c)) / c
            mx = max(mx, float(np.abs(m).max()) / max(float(A[:, k].std()), 1e-30))
    return A, mx


def store_meta(stores):
    """county, zip3, retailer, parent and dma per store, from the LAST year it appears."""
    S = store_table()
    S = S.sort_values("year").drop_duplicates("store_code_uc", keep="last")
    S = S.set_index("store_code_uc").reindex(stores)
    fips = (S.fips_state_code.fillna("").str.zfill(2)
            + S.fips_county_code.fillna("").str.zfill(3))
    return pd.DataFrame({
        "county": fips.to_numpy(), "zip3": S.store_zip3.fillna("").str.zfill(3).to_numpy(),
        "retailer": S.retailer_code.fillna("").to_numpy(),
        "parent": S.parent_code.fillna("").to_numpy(),
        "dma": S.dma_code.fillna("").to_numpy(),
        "state": S.fips_state_code.fillna("").str.zfill(2).to_numpy()}, index=stores)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with RunManifest(OUT, args={}, inputs=[SCRATCH / "focal_rows.bin", ACS],
                     tag="round 55 Part 1: the market-geography frontier") as rm:
        rm.add_note("Four geographies, all four requirements, no choice made. Full panel "
                    "at every geography.")
        t0 = time.time()
        G = build_grid()
        nS, nW, nA = G["nS"], G["nW"], G["nA"]
        ncan = G["ncan"].astype(np.float32)
        P = G["P"]
        PK = G["PK"]
        valid = G["valid"]
        TOT = G["TOT"].astype(np.float64)
        si = G["si"]
        stores = pd.Index(G["stores"][si])
        weeks = G["weeks"]
        yidx = np.array([YEARS.index(d.year) for d in weeks])
        brand = G["brand"]; cont = G["cont"]
        bcode = pd.factorize(brand)[0]
        CLS = classes(nA, brand, cont, G["ncan"])
        assert all(c in R3 for c in Z2)
        SV = np.where(valid, PK * ncan[None, None, :], np.float32(0.0))
        RV = np.where(valid, P * PK * ncan[None, None, :] / np.float32(12.0),
                      np.float32(0.0))
        SP = np.where(valid, P, np.nan).astype(np.float32)      # the store-week price
        del P, PK, G["P"], G["PK"]; gc.collect()
        MD = store_meta(stores)
        A = pd.read_csv(ACS)
        A["store_zip3"] = A.store_zip3.astype(int).astype(str).str.zfill(3)
        inc = (A.groupby("store_zip3").median_hh_income.mean() / 1000.0)
        MD["income"] = inc.reindex(MD.zip3).to_numpy()
        MD["income"] = MD.income.fillna(MD.income.mean())
        pop = A.groupby("store_zip3").population.mean()
        MD["pop"] = pop.reindex(MD.zip3).to_numpy()
        MD["pop"] = MD["pop"].fillna(MD["pop"].mean())
        print(f"  store metadata: {MD.county.nunique():,} counties, "
              f"{MD.zip3.nunique():,} zip3s, {MD.parent.nunique():,} parents, "
              f"{MD.dma.nunique():,} DMAs across {nS:,} stores; income sd across stores "
              f"{MD.income.std():.3f} [{time.time()-t0:.0f}s]", flush=True)

        GEOS = [("store-week", np.arange(nS)),
                ("chain-DMA-week", pd.factorize(MD.parent + "|" + MD.dma)[0]),
                ("county-week", pd.factorize(MD.county)[0]),
                ("DMA-week", pd.factorize(MD.dma)[0])]
        rows, s0rows, incrows, aggrows = [], [], [], []
        for lab, gid in GEOS:
            gid = np.asarray(gid, np.int64)
            nU = int(gid.max()) + 1
            print(f"\n=== {lab}: {nU:,} market units, "
                  f"{np.bincount(gid).mean():.2f} stores per unit (max "
                  f"{np.bincount(gid).max()}) [{time.time()-t0:.0f}s] ===", flush=True)
            # aggregate servings, revenue, all-beer size and presence to the unit
            from scipy import sparse
            Ind = sparse.csr_matrix(
                (np.ones(nS, np.float32), (gid, np.arange(nS))), shape=(nU, nS))
            aSV = np.asarray(Ind @ SV.reshape(nS, -1)).reshape(nU, nW, nA)
            aRV = np.asarray(Ind @ RV.reshape(nS, -1)).reshape(nU, nW, nA)
            PRcnt = np.asarray(Ind @ valid.reshape(nS, -1).astype(np.float32)
                               ).reshape(nU, nW, nA)
            aPR = PRcnt > 0
            aM = np.asarray(Ind @ TOT.astype(np.float32))
            aSV = aSV.astype(np.float64); aRV = aRV.astype(np.float64)
            aM = aM.astype(np.float64)
            eq = aSV / 12.0
            with np.errstate(all="ignore"):
                aP = np.where(eq > 0, aRV / np.where(eq > 0, eq, np.nan), np.nan)
                inside = aSV.sum(2)
                S0 = 1.0 - inside / np.where(aM > 0, aM, np.nan)
            goodm = np.isfinite(S0) & (S0 > 0) & (S0 < 1) & (inside > 0)
            rowok = aPR & goodm[:, :, None] & np.isfinite(aP) & (aP > 0) & (aSV > 0)
            n = int(rowok.sum()); nM = int(goodm.sum())
            ui, wi, ai = np.where(rowok)
            print(f"  rows {n:,}  markets {nM:,}  alternatives per market "
                  f"{n/max(nM,1):.2f}", flush=True)

            # ---- requirement 2: s_0's variance surviving the operative effects
            mu, mw = np.where(goodm)
            s0v = S0[mu, mw]
            lad = [("raw", []), ("week", [mw]), ("+ unit", [mw, mu]),
                   ("+ unit x year", [mw, pd.factorize(mu * 4 + yidx[mw])[0]])]
            sh = {}
            v0 = float(np.var(s0v))
            for nmL, gg in lad:
                if not gg:
                    sh[nmL] = 1.0
                    continue
                RR, mx = resid(s0v.reshape(-1, 1), gg)
                sh[nmL] = float(np.var(RR[:, 0]) / v0)
            s0rows.append(dict(geography=lab, markets=nM, s0_mean=float(s0v.mean()),
                               s0_sd=float(s0v.std()), **{f"surv_{k}": v
                                                          for k, v in sh.items()}))
            print(f"  **`s_0` surviving: week {100*sh['week']:.2f}%, + unit "
                  f"{100*sh['+ unit']:.2f}%, + unit x year "
                  f"**{100*sh['+ unit x year']:.2f}%** against the {100*CRIT:.1f}% "
                  f"criterion and store-week's {100*REF['sw_s0_surv']:.2f}%**",
                  flush=True)

            # ---- requirement 3: within-market income dispersion
            df = pd.DataFrame({"u": gid, "inc": MD.income.to_numpy(),
                              "zip3": MD.zip3.to_numpy()})
            g = df.groupby("u")
            sd = g.inc.std().fillna(0.0)
            nz = g.zip3.nunique()
            incrows.append(dict(geography=lab, units=nU,
                                sd_mean=float(sd.mean()), sd_p50=float(sd.median()),
                                sd_p90=float(sd.quantile(0.9)),
                                sd_max=float(sd.max()),
                                share_units_multi_zip3=float((nz > 1).mean()),
                                zip3_mean=float(nz.mean()), zip3_max=int(nz.max()),
                                sd_across_units=float(g.inc.mean().std())))
            w = incrows[-1]
            print(f"  **within-market income sd: mean {w['sd_mean']:.3f}, median "
                  f"{w['sd_p50']:.3f}, p90 {w['sd_p90']:.3f}, max {w['sd_max']:.3f}; "
                  f"{100*w['share_units_multi_zip3']:.2f}% of units span more than one "
                  f"zip3** (between-unit sd {w['sd_across_units']:.3f})", flush=True)

            # ---- the aggregation cost, round 12's measure
            if lab != "store-week":
                # round 12's measure, volume-weighted: the aggregated price aP IS the
                # volume-weighted mean of the store-week prices inside the cell, so the
                # within component is exactly what aggregation averages away.  Streamed
                # over stores so no (nS, nW, nA) index array is materialised.
                sw = swv = tot = 0.0
                pbar_all = float(np.nansum(np.where(valid, SP, 0.0) * SV)
                                 / np.where(SV.sum() > 0, SV.sum(), 1.0))
                nstore = PRcnt
                for sidx in range(nS):
                    m = valid[sidx]
                    if not m.any():
                        continue
                    d = SP[sidx][m].astype(np.float64) - aP[gid[sidx]][m]
                    wt = SV[sidx][m].astype(np.float64)
                    swv += float(np.nansum(wt * d * d))
                    sw += float(np.nansum(wt))
                    tot += float(np.nansum(
                        wt * (SP[sidx][m].astype(np.float64) - pbar_all) ** 2))
                aggrows.append(dict(geography=lab, within_share=swv / tot,
                                    total_var=tot / sw, within_var=swv / sw,
                                    cells_pooled=int((nstore > 0).sum()),
                                    mean_stores_per_cell=float(
                                        nstore[nstore > 0].mean())))
                w = aggrows[-1]
                print(f"  aggregation cost: **{100*w['within_share']:.2f}% of the "
                      f"store-week price variance is WITHIN market-alternative and is "
                      f"averaged away** ({w['mean_stores_per_cell']:.2f} stores per "
                      f"market-alternative, {w['cells_pooled']:,} cells)", flush=True)
                gc.collect()
            else:
                aggrows.append(dict(geography=lab, within_share=0.0, total_var=float("nan"),
                                    within_var=0.0, cells_pooled=0,
                                    mean_stores_per_cell=1.0))
                print(f"  aggregation cost: 0 by construction", flush=True)

            # ---- requirements 1 and 3': the linear IV at this geography
            sj = aSV[ui, wi, ai] / aM[ui, wi]
            s0r = S0[ui, wi]
            pr = aP[ui, wi, ai]
            PRf = aPR.reshape(-1, nA).astype(np.float32)
            zc = np.zeros((n, len(Z2)))
            for j, k in enumerate(Z2):
                cnt = (PRf @ CLS[k].astype(np.float32)).reshape(nU, nW, nA)
                zc[:, j] = cnt[ui, wi, ai]
                del cnt
            del PRf; gc.collect()
            y = np.log(sj) - np.log(s0r)
            gr = [ai, wi, pd.factorize(ui * 4 + yidx[wi])[0],
                  pd.factorize(ui * 100 + bcode[ai])[0]]
            AA, mx = resid(np.column_stack([y, pr, zc]), gr)
            udma = pd.Series(MD.dma.to_numpy()).groupby(gid).first().reindex(
                range(nU)).fillna("").to_numpy()
            cl = pd.factorize(udma[ui])[0]
            b, V, u_, Gc = M.tsls(AA[:, [0]].ravel(), AA[:, [1]], AA[:, 2:], cl)
            bo, Vo, _, _ = M.ols(AA[:, [0]].ravel(), AA[:, [1]], cl)
            f1 = M.sw_stats(AA[:, 1], np.empty((n, 0)), AA[:, 2:], cl)
            J, Jdf, Jp, _ = M.hansen_j(AA[:, [0]].ravel(), AA[:, [1]], AA[:, 2:], cl)
            fs = []
            for j in range(len(Z2)):
                bb, VV, _, _ = M.ols(AA[:, 1], AA[:, [2 + j]], cl)
                fs.append((float(bb[0]), float(np.sqrt(VV[0, 0]))))
            pbar = float(np.average(pr, weights=sj))
            el = float(np.median(b[0] * pr * (1.0 - sj)))
            epsM = float(b[0] * pbar * np.median(s0r))
            rows.append(dict(geography=lab, units=nU, rows=n, markets=nM,
                             alts_per_market=n / max(nM, 1),
                             clusters=int(Gc), alpha=float(b[0]),
                             se=float(np.sqrt(V[0, 0])), alpha_ols=float(bo[0]),
                             swF=float(f1["sw_F"]), J=float(J), Jdf=int(Jdf), Jp=float(Jp),
                             own_elast=el, pbar=pbar, s0_median=float(np.median(s0r)),
                             eps_M=epsM, resid_max=mx,
                             z1_b=fs[0][0], z1_se=fs[0][1], z1_sign="+" if fs[0][0] > 0 else "-",
                             z2_b=fs[1][0], z2_se=fs[1][1], z2_sign="+" if fs[1][0] > 0 else "-"))
            w = rows[-1]
            print(f"  **alpha {w['alpha']:+.4f} ({w['se']:.4f})** OLS "
                  f"{w['alpha_ols']:+.4f}  SW F **{w['swF']:.2f}**  J {w['J']:.2f}/"
                  f"{w['Jdf']} p **{w['Jp']:.4f}**  clusters {w['clusters']}", flush=True)
            print(f"  first stages: {Z2[0]} {fs[0][0]:+.6f} ({fs[0][1]:.6f}) t "
                  f"{fs[0][0]/fs[0][1]:+.2f} **{w['z1_sign']}**; {Z2[1]} "
                  f"{fs[1][0]:+.6f} ({fs[1][1]:.6f}) t {fs[1][0]/fs[1][1]:+.2f} "
                  f"**{w['z2_sign']}**", flush=True)
            print(f"  own elasticity **{el:+.2f}**; eps_M **{epsM:+.4f}** = alpha "
                  f"{w['alpha']:+.4f} x p-bar {pbar:.4f} x s_0 {w['s0_median']:.4f}",
                  flush=True)
            del aSV, aRV, aPR, PRcnt, aM, aP, S0, goodm, rowok, AA, Ind
            gc.collect()

        for nm, r in (("p1_geography", rows), ("p1_s0_variance", s0rows),
                      ("p1_income_dispersion", incrows), ("p1_aggregation_cost", aggrows)):
            pd.DataFrame(r).to_csv(OUT / f"{nm}.csv", index=False)
        json.dump(dict(geography=rows, s0=s0rows, income=incrows, agg=aggrows,
                       criterion=CRIT, reference=REF),
                  open(OUT / "p1_frontier.json", "w"), indent=1, default=float)
        rm.add_outputs([OUT / f"{nm}.csv" for nm in
                        ("p1_geography", "p1_s0_variance", "p1_income_dispersion",
                         "p1_aggregation_cost")] + [OUT / "p1_frontier.json"])
        print(f"\ndone part 1 [{time.time()-t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
