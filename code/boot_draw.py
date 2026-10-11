#!/usr/bin/env python3
"""Parametric bootstrap of the supply side and counterfactuals, and the Hendel-Nevo
third-degree benchmark, for column (iv).

Draw b takes (sigma_p, sigma_n, pi_p, beta) from the normal distribution centred on the
column (iv) estimates with their full DMA-clustered covariance (draw 0 is the estimates
themselves), re-solves the contraction so mean utilities reproduce observed shares at the
drawn nonlinear parameters, takes the sale regime shifts from the drawn beta, and reruns:
marginal cost and the sale-week gap (Tables 7-9), the four pricing regimes and the
third-degree benchmarks (Tables 10-11), and ITS at observed prices (Table 12, Panels A, B).

    python3 code/boot_draw.py --draws 0          # point estimates: the Hendel-Nevo shares,
                                                 # and a check against results/ (Tables 7-12)
    python3 code/boot_draw.py --prep             # full covariance of column (iv), once
    python3 code/boot_draw.py --draws 1 2 3      # bootstrap draws (need --prep)
    BLP_DRAWS=50 python3 code/boot_draw.py --draws 0 --stores 3     # smoke test

Each draw writes results/boot/draw_<b>.json with its wall time per stage and peak memory.
The store loops run on SLURM_CPUS_PER_TASK forked workers (all cores outside SLURM).
"""
import os
import sys
if "--prep" not in sys.argv:              # one BLAS thread per worker process
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cf_core as C                                                # noqa: E402
import f71_baseline_table as f71                                   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
E, RES = ROOT / "estimation", ROOT / "results"
OUT = RES / "boot"
J = json.load(open(E / "f71_col_iv_long.json"))
VF, VL = E / "f71_vcov_full_iv_long.npy", E / "f71_vcov_full_iv_long_labels.json"
REGS = ("NNN", "NNS", "NSN", "NSS", "SNN", "SNS", "SSN", "SSS")


def prep():
    """The joint covariance of (sigma_p, sigma_n, pi_p, beta) at the column (iv) estimates:
    pyblp's two-step clustered covariance with the nonlinear parameters held at the
    estimates (no optimization), so W is updated from residuals at the estimates."""
    import pyblp
    pyblp.options.verbose = True
    t0 = time.time()
    d, terms = f71.load("long", "iv")
    for k, c in enumerate(f71.ZCOLS):
        d[f"demand_instruments{k}"] = d[c].astype(float)
    prob = f71.lognormal_problem(d, terms)
    r = prob.solve(sigma=np.diag([J["sigma_p"], J["sigma_lsz"]]),
                   pi=np.array([[J["pi_p"]], [0.0]]),
                   optimization=pyblp.Optimization("return"),
                   iteration=pyblp.Iteration("squarem", {"atol": 1e-12}),
                   method="2s", se_type="clustered", W_type="clustered")
    V = np.asarray(r.parameter_covariances) / prob.N
    lab = ["sigma_p", "sigma_lsz", "pi_p"] + list(terms)
    assert V.shape == (len(lab), len(lab)), V.shape
    se = np.sqrt(np.diag(V))
    ref = np.array([J["sigma_p_se"], J["sigma_lsz_se"], J["pi_p_se"]]
                   + [J["se"][t] for t in terms])
    rel = se / ref - 1
    print(f"\nstandard errors against the article's (relative difference):")
    for l, a, b_, x in zip(lab[:3], se[:3], ref[:3], rel[:3]):
        print(f"  {l:10s} {a:.5f}  article {b_:.5f}  {x:+.3f}")
    print(f"  beta: max |relative difference| {np.max(np.abs(rel[3:])):.3f}")
    db = np.asarray(r.beta).ravel() - np.array([J["beta"][t] for t in terms])
    print(f"  beta at this W against the article's: max |difference| {np.max(np.abs(db)):.2e}")
    np.save(VF, V)
    json.dump(lab, open(VL, "w"))
    print(f"wrote {VF.name}  [{time.time() - t0:.0f}s]")


def theta(b):
    """(sigma_p, sigma_n, pi_p, beta) for draw b; draw 0 is the estimates."""
    beta = dict(J["beta"])
    if b == 0:
        return J["sigma_p"], J["sigma_lsz"], J["pi_p"], beta
    V, lab = np.load(VF), json.load(open(VL))
    c = np.array([J["sigma_p"], J["sigma_lsz"], J["pi_p"]] + [beta[t] for t in lab[3:]])
    ev, U = np.linalg.eigh(V)
    x = c + (U * np.sqrt(np.clip(ev, 0, None))) @ np.random.default_rng(b).standard_normal(len(c))
    sp, sl = np.clip(x[:2], 1e-6, 5.0)                 # the estimation's bounds
    pi_p = float(np.clip(x[2], -5.0, 3.0))
    return float(sp), float(sl), pi_p, dict(zip(lab[3:], x[3:].tolist()))


def pct(a, b):
    return 100.0 * (a / b - 1.0)


def pq_cells(R, d):
    """Table 11's cells (f92_by_firm.scenario_table, unrounded): median price per serving
    and mean over markets of the firm's quantity per 10,000 servings, by firm, pack size
    and sale week."""
    S = C.S
    out = {}
    for sc in ("MIX", "VD", "IT", "NPD"):
        x = pd.DataFrame(dict(firm=S["firm"], pack=S["pack"], sale=S["sale3"],
                              mk=d.market_ids.to_numpy(), p=R[sc]["P"] / 12.0,
                              q=R[sc]["Q"] * 1e4))
        assert x.p.notna().all(), sc
        qs = x.groupby(["firm", "pack", "sale", "mk"]).q.sum().groupby(["firm", "pack", "sale"]).mean()
        qa = x.groupby(["firm", "pack", "mk"]).q.sum().groupby(["firm", "pack"]).mean()
        for (f, k), v in qa.items():
            out[f"{f}|{k}|{sc}|Q"] = float(v)
        for (f, k, sl), v in qs.items():
            out[f"{f}|{k}|{sc}|Q{'S' if sl else 'R'}"] = float(v)
        for (f, k), v in x.groupby(["firm", "pack"]).p.median().items():
            out[f"{f}|{k}|{sc}|P"] = float(v)
        for (f, k, sl), v in x.groupby(["firm", "pack", "sale"]).p.median().items():
            out[f"{f}|{k}|{sc}|P{'S' if sl else 'R'}"] = float(v)
        for f, v in x.groupby("firm").p.median().items():
            out[f"{f}|all|{sc}|P"] = float(v)
        for (f, sl), v in x.groupby(["firm", "sale"]).p.median().items():
            out[f"{f}|all|{sc}|P{'S' if sl else 'R'}"] = float(v)
    return out


# Table 11's columns (f92_by_firm.scenario_table) -> pq_cells keys
T11 = {"ND_P": ("all", "NPD", "P"), "ND_Q": (None, "NPD", "Q"), "VD_P": (None, "VD", "P"),
       "VD_Q": (None, "VD", "Q"), "IT_RP": ("all", "IT", "PR"), "IT_RQ": (None, "IT", "QR"),
       "IT_SP": ("all", "IT", "PS"), "IT_SQ": (None, "IT", "QS"), "MX_RP": (None, "MIX", "PR"),
       "MX_RQ": (None, "MIX", "QR"), "MX_SP": (None, "MIX", "PS"), "MX_SQ": (None, "MIX", "QS")}


def report_pq(r):
    """Draw 0's Table 11 cells against results/f92_scenario_table_firm.csv, its CS by
    pack-taste quintile against Table 10, and CS by price-sensitivity quintile."""
    Q = pd.read_csv(RES / "f92_cs_quintile_pack_iv_long.csv", index_col=0)
    g = r["incidence"]["regimes"]
    for sc in ("MIX", "VD", "IT", "NPD"):
        gap = max(abs(x - Q.loc[f"q{k + 1}", sc]) / Q.loc[f"q{k + 1}", sc]
                  for k, x in enumerate(g[sc]["csq"]))
        assert gap < 1e-6, (sc, gap)
        assert abs(sum(g[sc]["csqp"]) - g[sc]["cs"]) < 1e-6 * g[sc]["cs"], sc
    print("\nCS by quintile of the pack-size draw reproduces Table 10; percent change from MIX "
          "by quintile of the price-sensitivity draw nu_1 (1 = least price-sensitive):")
    for sc in ("VD", "IT", "NPD"):
        print(f"  {sc:4s}" + "".join(f"{pct(x, y):8.1f}" for x, y in zip(g[sc]["csqp"], g["MIX"]["csqp"])))
    pa = r["incidence"]["panel_a"]
    for c in ("no_sales", "no_discount", "no_regimes"):
        print(f"  {c:12s}" + "".join(f"{pct(x, y):8.1f}"
                                     for x, y in zip(pa[c]["csqp"], pa["observed"]["csqp"])))
    T = pd.read_csv(RES / "f92_scenario_table_firm.csv")
    T["firm"] = T.firm.replace("", np.nan).ffill()
    worst = 0.0
    for _, row in T.iterrows():
        for c, (k, sc, s) in T11.items():
            if pd.isna(row[c]):
                continue
            key = f"{row.firm}|{'all' if k else int(row.pack)}|{sc}|{s}"
            ours = r["table11"][key]
            tol = 0.005 + 1e-9 if c.endswith("P") else 0.05 + 1e-9
            worst = max(worst, abs(ours - row[c]) / tol)
            assert abs(ours - row[c]) <= tol, (key, ours, row[c])
    print(f"\ndraw 0 reproduces Table 11 (results/f92_scenario_table_firm.csv) to its rounding; "
          f"worst cell at {worst:.2f} of the rounding unit")


def run(b, a, d, t_all):
    t0, T = time.time(), {}
    def tick(k):
        T[k] = round(time.time() - t0 - sum(T.values()), 1)
        print(f"  [{time.time() - t_all:.0f}s] draw {b}: {k} {T[k]:.0f}s", flush=True)
    sp, sl, pi_p, beta = theta(b)
    C.set_theta(sp, sl, pi_p)
    S = C.S
    d0 = OUT / "delta0.npy"
    if d0.exists() and not a.stores and len(np.load(d0)) == len(S["p0"]):
        S["delta_start"] = np.load(d0)
    else:                                              # logit start
        s0 = 1.0 - pd.Series(S["shares"]).groupby(d.market_ids.to_numpy()).transform("sum")
        S["delta_start"] = S["lnS"] - np.log(s0.to_numpy())
    S["delta"], nf_c = C.contract()
    if b == 0 and not a.stores and f71.NDRAWS == 1000:
        OUT.mkdir(exist_ok=True)
        np.save(d0, S["delta"])
    S["lam"] = C.lam_from(beta, d)
    tick("contraction")

    p0 = S["p0"]
    bmk = C.markups()
    mc_raw = p0 - bmk
    S["mc"] = C.fit_cost(mc_raw)
    ps, nf_p = C.pstar()
    gap, sale, nnn, reg = p0 - ps, S["sale3"], S["nnn"], S["reg"]
    sup = dict(
        brewer_markup_pct=100 * float(np.median(bmk[nnn] / p0[nnn])),
        mc_median=float(np.median(mc_raw[nnn])), mc_neg=int((mc_raw < 0).sum()),
        gap_nosale=float(np.nanmedian(gap[~sale])), gap_sale=float(np.nanmedian(gap[sale])),
        gap_pct=100 * float(np.nanmedian(gap[sale] / ps[sale])),
        gap_by_regime={r: float(np.nanmedian(gap[reg == r])) for r in REGS if (reg == r).any()},
        gap_pct_by_pack={int(k): 100 * float(np.nanmedian((gap / ps)[sale & (S["pack"] == k)]))
                         for k in f71.RUNGS if (sale & (S["pack"] == k)).any()},
        pstar_fail=int(nf_p), contraction_fail=int(nf_c))
    tick("supply")

    firms = C.firm_order()
    R = C.regimes()
    if a.pq:          # Table 11's prices and quantities, and CS by quintile of nu_1 and nu_2
        PA, _ = C.panel_a()
        inc = dict(regimes={sc: dict(csq=R[sc]["csq"].tolist(), csqp=R[sc]["csqp"].tolist(),
                                     cs=R[sc]["cs"]) for sc in ("MIX", "VD", "IT", "NPD")},
                   panel_a={c: dict(csq=r["csq"].tolist(), csqp=r["csqp"].tolist(), cs=r["cs"])
                            for c, r in PA.items()})
        res = dict(draw=b, theta=dict(sigma_p=sp, sigma_lsz=sl, pi_p=pi_p),
                   table11=pq_cells(R, d), incidence=inc,
                   conv={sc: float(np.mean(R[sc]["ok"])) for sc in ("MIX", "VD", "IT", "NPD")},
                   seconds_total=round(time.time() - t0, 1))
        (OUT / "pq2").mkdir(parents=True, exist_ok=True)
        f = OUT / "pq2" / f"draw_{b:03d}{'_smoke' if a.stores else ''}.json"
        json.dump(res, open(f, "w"), indent=1)
        print(f"  draw {b}: {res['seconds_total']:.0f}s -> {f.relative_to(ROOT)}", flush=True)
        return res
    reg_out = {}
    for sc, r in R.items():
        reg_out[sc] = dict(cs=r["cs"], ps=float(r["ps"].sum()), total=r["cs"] + float(r["ps"].sum()),
                           ps_firm=dict(zip(firms, r["ps"].tolist())), csq=r["csq"].tolist(),
                           conv=float(np.mean(r["ok"])) if r["ok"] else float("nan"))
    m = reg_out["MIX"]
    for sc, r in reg_out.items():
        r["pct"] = {k: pct(r[k], m[k]) for k in ("cs", "ps", "total")}
        r["pct_firm"] = {f: pct(r["ps_firm"][f], m["ps_firm"][f]) for f in firms}
        r["pct_csq"] = [pct(x, y) for x, y in zip(r["csq"], m["csq"])]
    loss = {sc: m["total"] - reg_out[sc]["total"] for sc in ("VD", "IT", "NPD")}
    psn = reg_out["NPD"]["ps"]
    hn = {den: {sc: ((reg_out[sc]["ps"] - psn) / (reg_out[den]["ps"] - psn)
                     if reg_out[den]["ps"] != psn else float("nan"))
                for sc in ("MIX", "VD", "IT")} for den in reg_out if "3_" in den}
    tick("regimes")

    PA, fit = C.panel_a()
    pa = {}
    for c, r in PA.items():
        pa[c] = dict(cs=r["cs"], ps=float(r["ps"].sum()), total=r["cs"] + float(r["ps"].sum()),
                     ps_firm=dict(zip(firms, r["ps"].tolist())), csq=r["csq"].tolist())
    o = pa["observed"]
    for c, r in pa.items():
        r["pct"] = {k: pct(r[k], o[k]) for k in ("cs", "ps", "total")}
        r["pct_firm"] = {f: pct(r["ps_firm"][f], o["ps_firm"][f]) for f in firms}
    tick("panel A")

    pb = None
    if not a.no_panel_b:
        CS, PS, nf_b = C.panel_b()
        fi = {f: firms.index(f) for f in C.FIRMS if f in firms}
        pb = {}
        for k, f in enumerate(C.FIRMS):
            if f not in fi:
                continue
            riv = [fi[g] for g in fi if g != f]
            pb[f] = dict(own=pct(PS[1, k, fi[f]], PS[0, k, fi[f]]),
                         rivals=pct(PS[1, k, riv].sum(), PS[0, k, riv].sum()),
                         cs=pct(CS[1, k], CS[0, k]))
        pb["not_converged"] = int(nf_b)
        tick("panel B")

    mem = max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    mem_gb = mem / (2**30 if sys.platform == "darwin" else 2**20)
    res = dict(draw=b, theta=dict(sigma_p=sp, sigma_lsz=sl, pi_p=pi_p,
                                  median_alpha=-float(np.exp(pi_p))),
               lam=beta, supply=sup, regimes=reg_out,
               welfare_loss=dict(loss, ratio_IT_to_VD=loss["IT"] / loss["VD"]),
               hendel_nevo=hn, panel_a=pa, panel_a_fit=fit, panel_b=pb,
               seconds=T, seconds_total=round(time.time() - t0, 1),
               peak_rss_gb_largest_process=round(mem_gb, 2), workers=C.NPROC,
               ndraws=f71.NDRAWS, stores=len(S["stores"]))
    OUT.mkdir(exist_ok=True)
    f = OUT / f"draw_{b:03d}{'_smoke' if a.stores else ''}.json"
    json.dump(res, open(f, "w"), indent=1)
    print(f"  draw {b}: {res['seconds_total']:.0f}s, peak {mem_gb:.1f} GB in the largest "
          f"process -> {f.relative_to(ROOT)}", flush=True)
    return res


def report(r, full):
    """The point estimates against the article's results files (draw 0, full sample)."""
    print("\nHendel-Nevo: (profit_X - profit_NPD) / (profit_3rd - profit_NPD), "
          f"{a_.nseg} segments; _p by the price draw nu_1, _n by the pack-size draw nu_2")
    for den, v in r["hendel_nevo"].items():
        print(f"  third degree = {den:7s}  " + "  ".join(f"{k} {x:6.3f}" for k, x in v.items()))
    g = r["regimes"]
    print("  profit: " + "  ".join(f"{sc} {g[sc]['ps']:,.0f}" for sc in g))
    if not full:
        return
    rows = []
    def chk(name, art, ours):
        rows.append((name, art, ours, ours - art))
    sg = pd.read_csv(RES / "f90_supply_rcgrid_iv_long.csv").set_index("rc").loc[0.0]
    chk("brewer markup %", sg.b_mkup, r["supply"]["brewer_markup_pct"])
    chk("gap, no-sale weeks", sg.gap_nosale, r["supply"]["gap_nosale"])
    chk("gap, sale weeks", sg.gap_sale, r["supply"]["gap_sale"])
    chk("gap, sale weeks %", sg.gap_pct, r["supply"]["gap_pct"])
    W = pd.read_csv(RES / "f92_welfare_iv_long.csv").groupby("scen")[["cs", "ps"]].sum()
    for sc in ("MIX", "VD", "IT", "NPD"):
        chk(f"{sc} CS", W.loc[sc, "cs"], g[sc]["cs"])
        chk(f"{sc} PS", W.loc[sc, "ps"], g[sc]["ps"])
    F = pd.read_csv(RES / "f92_ps_by_firm.csv", index_col=0)
    for f in F.index:
        chk(f"MIX profit {f}", F.loc[f, "MIX"], g["MIX"]["ps_firm"][f])
    A = pd.read_csv(RES / "f98_sales_value_iv_long.csv", index_col=0)
    for c in ("no_sales", "no_discount", "no_regimes"):
        for k in ("cs", "ps", "total"):
            chk(f"Panel A {c} {k} %", A.loc[c, f"{k}_pct_vs_observed"], r["panel_a"][c]["pct"][k])
    if r["panel_b"]:
        B = pd.read_csv(RES / "f99_solo_rivals_iv_long.csv", index_col=0)
        for f in C.FIRMS:
            chk(f"Panel B own {f} %", pct(B.loc[f, f"ps_{f}_remove"], B.loc[f, f"ps_{f}_keep"]),
                r["panel_b"][f]["own"])
    print("\ndraw 0 against results/ (the article's numbers):")
    for n, x, y, z in rows:
        print(f"  {n:28s} {x:14.4f} {y:14.4f}  diff {z:+.2e}")
    print(f"  Panel A shares reproduced to {r['panel_a_fit']:.1e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--draws", type=int, nargs="*", default=[0])
    ap.add_argument("--stores", type=int, default=0, help="first k stores (smoke test)")
    ap.add_argument("--nseg", type=int, default=2, help="third-degree segments")
    ap.add_argument("--no-panel-b", action="store_true")
    ap.add_argument("--pq", action="store_true",
                    help="Table 11 only: regimes without the third-degree benchmarks or Panels A-B")
    a_ = ap.parse_args()
    if a_.prep:
        prep()
        sys.exit()
    if any(b > 0 for b in a_.draws):
        assert VF.exists(), "run --prep first"
    t_all = time.time()
    d, _, am, nodes, w = C.frame(a_.stores)
    C.init(d, am, nodes, w, nseg=a_.nseg)
    if a_.pq:
        C.S.update(rows=True, segs=False)
    print(f"[{time.time() - t_all:.0f}s] frame: {len(d):,} rows, {len(C.S['stores'])} stores, "
          f"{f71.NDRAWS} draws per market, {C.NPROC} workers", flush=True)
    for b in a_.draws:
        if b > 0 and not a_.stores and (OUT / ("pq2/" if a_.pq else "") / f"draw_{b:03d}.json").exists():
            print(f"  draw {b}: already done, skipped", flush=True)  # a restart resumes
            continue
        r = run(b, a_, d, t_all)
        if a_.pq:
            if b == 0 and not a_.stores and f71.NDRAWS == 1000:
                report_pq(r)
            continue
        if b == 0:
            report(r, full=not a_.stores and f71.NDRAWS == 1000)
