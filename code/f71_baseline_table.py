#!/usr/bin/env python3
"""The four-column baseline table.  Reads f70_baseline_<tag>.parquet, writes/renders.

    (i)   MW2017 nested logit, rho ESTIMATED by 2SLS, baseline1 regime x rung controls
    (ii)  RCL, baseline1 regime <t-1><t>   x rung
    (iii) RCL, baseline2 regime <t><t+1>   x rung
    (iv)  RCL, baseline3 regime <t-1><t><t+1> x rung   (7 states, 42 terms)

Common to (ii)-(iv), LOGNORMAL PRICE COEFFICIENT (this package's robustness specification):
alpha_i = -exp(pi_p + sigma_p nu_1i), so price is in X2 only (I(-prices), rc type 'log') and
pi_p is its element of Pi; X1 = regime x rung dummies and the demographic interactions,
absorbing brand + year_month; log_n_eq carries a linear rc; instruments z1, z_haus,
sq_dev_lsz, sum_lsz, sum_can.  NO rung main effects anywhere: price per 144oz and log pack
size are two descriptions of the same schedule.  Pack size enters only interacted with the
regime, and as dispersion in X2.  The omitted state (NN, or NNN in (iv)) is the reference at
EVERY rung, which is what makes the columns readable down a rung.

df = MD - K is pinned at 2 for (ii)-(iv) (sigma_p, sigma_lsz, pi_p against five instruments)
and 3 for (i), regardless of the term count: each exogenous regressor brings its own moment
as well as its own parameter.

The median price coefficient is -exp(pi_p), with a delta-method standard error.

    python3 code/f71_baseline_table.py --col ii          # one column
    python3 code/f71_baseline_table.py --render          # the table, from saved JSON
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np, pandas as pd, pyhdfe
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
E = ROOT / "estimation"
# IV SETS.  `cur` is f61's.  Entry 38 predicted that count-like instruments correlate with
# xi; sum_lsz and sum_can are sums over the rest of the choice set and are the two moments
# that individually fail, on BOTH samples.  `a` swaps them for the parent's configuration-E
# income interactions (entry 39, "not counts"); `d` keeps only moments that hold and adds a
# differentiation IV on the can dummy.  DISCLOSE: this selection was informed by the J.
IVSETS = {
    "cur": ["z1", "z_haus", "sq_dev_lsz", "sum_lsz", "sum_can"],
    "a":   ["z1", "z_haus", "sq_dev_lsz", "z1_x_inc", "zh_x_inc"],
    "d":   ["z1", "z_haus", "sq_dev_lsz", "sq_dev_can"],
}
ZCOLS = IVSETS["cur"]
RUNGS = (6, 12, 18, 24, 30, 36)
POOL36 = False        # fold 36-packs into a 30+ cell for the regime interaction
SLOPE = False         # regime level + slope in log pack size, not a free cell each
PACKFE = False        # pack-size dummies in X1 (12-pack omitted); off in the paper
OPTCHECK = "both"     # "gradient" skips the Hessian, which is only a local-minimum check
ST2 = ("NS", "SN", "SS")
ST3 = ("NNS", "NSN", "NSS", "SNN", "SNS", "SSN", "SSS")
COLS = {"i": ("reg1", ST2), "ii": ("reg1", ST2), "iii": ("reg2", ST2), "iv": ("reg3", ST3)}
TAGSFX = ""
RCCONST = False
DEMO = True
DEMO_RAW = False
ABSORB = "C(brand_id) + C(year_month)"
NDRAWS = int(os.environ.get("BLP_DRAWS", "1000"))   # BLP_DRAWS: smoke tests only
def log(m=""): print(m, flush=True)


from _lib import ACS                                        # the one ACS path
DEMOS = {"inc": "median_hh_income", "age": "median_age", "edu": "share_ba_plus",
         "hh": "avg_household_size", "urb": "people_per_sqmi"}


def add_demographics(d):
    """Five zip3 ACS demographics, standardised, interacted with log pack size.

    The ACS is a zip3-by-year aggregate, so every consumer in a market carries the
    same value: the interaction is constant within market and belongs in X1, not in
    a random coefficient.  Identification is cross-store, 166 stores in 83 DMAs.
    """
    _sz = ROOT / "data_work" / "store_zip3_all.parquet"
    sc = pd.read_parquet(_sz if _sz.exists()
                         else ROOT / "data_work" / "f91_store_chain.parquet")
    z = dict(zip(sc.store_code_uc.astype(int),
                 sc.zip3.astype(int).astype(str).str.zfill(3)))
    A = pd.read_csv(ACS)
    yrs = sorted(set(d.year_month.astype(int) // 100) & set(A.year))
    A = A[A.year.isin(yrs)].copy()                   # the ACS years the sample spans
    A["z"] = A.store_zip3.astype(int).astype(str).str.zfill(3)
    A["people_per_sqmi"] = np.log1p(A.people_per_sqmi)
    g = A.groupby("z")[list(DEMOS.values())].mean()
    zz = d.store_code_uc.astype(int).map(z)
    names = []
    for k, v in DEMOS.items():
        x = zz.map(g[v]).astype(float)
        assert x.notna().all(), f"{v}: unmatched stores"
        if not DEMO_RAW:
            x = (x - x.mean()) / x.std()              # one SD across observations
        d[f"dx_{k}"] = x.to_numpy() * d.log_n_eq.to_numpy()
        names.append(f"dx_{k}")
    log(f"  demographics: {', '.join(names)} (zip3 ACS {yrs[0]}-{yrs[-1]}, x log pack size)")
    return names


def load(tag, col, x1extra=()):
    rc, states = COLS[col]
    d = pd.read_parquet(ROOT / "data_work" / f"f70_baseline_{tag}.parquet")
    d = d[d[rc] != ""].copy()
    terms = []
    if SLOPE:
        # One level and one slope in log pack size per sale regime, in place of a free
        # cell at every pack.  log_n_eq is ln(n/12), so the level is the regime effect
        # at the 12-pack and the slope its change per log equivalent unit.  The 36-pack
        # observations inform the slope instead of carrying seven parameters of their
        # own, which is where the cell design runs out of clusters.
        for s in states:
            ind = (d[rc] == s).to_numpy().astype(float)
            d[f"{s}_on"] = ind
            d[f"{s}_lnn"] = ind * d.log_n_eq.to_numpy()
            terms += [f"{s}_on", f"{s}_lnn"]
    else:
        # 36-packs are 0.6 percent of the sample in 24 of 83 DMAs, so their own cells
        # are backed by a quarter of the clusters.  --pool36 folds them into a 30+ cell
        # for the interaction only; log_n_eq in X2 still sees the true pack size.
        rungs = tuple(k for k in RUNGS if k != 36) if POOL36 else RUNGS
        cell = d.n_cans.where(d.n_cans < 36, 30) if POOL36 else d.n_cans
        for s in states:
            for k in rungs:
                d[f"{s}_{k}"] = ((d[rc] == s) & (cell == k)).astype(float)
                terms.append(f"{s}_{k}")
    if PACKFE:
        # Pack-size dummies in the LINEAR part, 12-pack omitted.  The baseline leaves
        # these out because price per 144oz and pack size describe the same schedule:
        # regressed on brand and year-month effects, pack dummies already explain 69
        # percent of price, so putting them in X1 hands most of beta_p's identifying
        # variation to them.  Available so that cost can be checked, not as a default.
        for k in RUNGS:
            if k == 12:
                continue
            d[f"pk_{k}"] = (d.n_cans == k).astype(float)
            terms.insert(0, f"pk_{k}")
    if DEMO and col != "i":            # not the nested logit benchmark
        terms = add_demographics(d) + terms
    terms = list(x1extra) + terms
    d["product_id"] = (d.brand_id.astype(str) + "|" + d.is_can.astype(int).astype(str)
                       + "|" + d.n_cans.astype(int).astype(str))
    d["store_product"] = d.store_code_uc.astype(str) + "|" + d.product_id
    d = d.sort_values(["market_ids", "alt"]).reset_index(drop=True)
    vc = d[rc].value_counts()
    log(f"  {rc}: {len(d):,} rows  {d.market_ids.nunique():,} markets  "
        f"{d.dma_code.nunique()} clusters")
    log("  states: " + "  ".join(f"{k}:{v:,}" for k, v in vc.items()))
    return d, terms


def clustered_2sls(Y, X, Z, cl):
    """2SLS with cluster-robust vcov and a Hansen J.  Never forms an N x N projection."""
    G = int(cl.max() + 1); n = len(Y)
    ZZi = np.linalg.pinv(Z.T @ Z); XZ = X.T @ Z
    b = np.linalg.solve(XZ @ ZZi @ XZ.T, XZ @ ZZi @ Z.T @ Y)
    e = (Y - X @ b)
    m = Z * e
    M = np.zeros((G, Z.shape[1])); np.add.at(M, cl, m)
    B = np.linalg.pinv(XZ @ ZZi @ XZ.T)
    V = B @ (XZ @ ZZi @ M.T @ M @ ZZi @ XZ.T) @ B * (G / (G - 1))
    S = M.T @ M / n; gb = m.mean(0)[:, None]
    j = n * float(np.squeeze(gb.T @ np.linalg.pinv(S) @ gb))
    df = Z.shape[1] - X.shape[1]
    return b.ravel(), np.sqrt(np.diag(V)), V, j, df, (1 - stats.chi2.cdf(j, df) if df > 0 else np.nan), G


def nested_logit(tag, iv="cur", x1extra=()):
    t0 = time.time()
    d, terms = load(tag, "i", x1extra)
    d = d.assign(lsg=np.log(d.sjg), y=np.log(d.shares) - np.log(d.s0_inside))
    fe = np.column_stack([pd.factorize(d.brand_id)[0], pd.factorize(d.year_month)[0]])
    A = pyhdfe.create(fe, drop_singletons=False)
    cols = ["y", "prices", "lsg"] + terms + ZCOLS
    R = A.residualize(d[cols].to_numpy(float))
    nx = 2 + len(terms)
    Y, X = R[:, :1], R[:, 1:1 + nx]
    Z = np.hstack([R[:, 1 + nx:], R[:, 3:1 + nx]])        # excluded IVs + exogenous terms
    cl = pd.factorize(d.dma_code)[0]
    b, se, V, j, df, p, G = clustered_2sls(Y, X, Z, cl)
    lab = ["prices", "rho"] + terms
    log("=" * 72)
    log("(i) MW2017 NESTED LOGIT, rho by 2SLS, one nest = all inside goods")
    log(f"    n {len(d):,}  df {df}  clusters {G}")
    log("=" * 72)
    for l, v, s in list(zip(lab, b, se))[:2]:
        log(f"  {l:10s}{v:>12.5f}{s:>11.5f}  t {v/s:>6.2f}"
            + ("    MW2017 0.6299" if l == "rho" else ""))
    log(f"  HANSEN J {j:.3f}  df {df}  p {p:.4f}  {'PASSES' if p >= .05 else 'REJECTS'}")
    json.dump(dict(col="i", kind="nested_logit", tag=tag, n=int(len(d)), clusters=int(G),
                   J=j, df=int(df), p=p, beta=dict(zip(lab, b.tolist())),
                   se=dict(zip(lab, se.tolist())), seconds=time.time() - t0),
              open(E / f"f71_col_i_{tag}{TAGSFX}.json", "w"), indent=1)
    np.save(E / f"f71_vcov_i_{tag}{TAGSFX}.npy", V)
    json.dump(lab, open(E / f"f71_vcov_i_{tag}{TAGSFX}_labels.json", "w"))
    log(f"\n[{time.time()-t0:.0f}s]")


def lognormal_problem(d, terms):
    """The random coefficients logit with a LOGNORMAL price coefficient,
    alpha_i = -exp(pi_p + sigma_p nu_1i): every consumer's price coefficient is negative and
    1/|alpha_i| has a finite mean.  Price leaves X1; in X2, I(-prices) carries the rc of type
    'log' and pi_p is its element of Pi on a constant demographic.  The draws are pyblp's own
    Halton draws for this integration, built here and passed as agent data (passing an
    Integration together with a demographic makes pyblp's setup scan every agent row once
    per market, about 16 hours at this size)."""
    import pyblp
    X1 = pyblp.Formulation("0 + " + " + ".join(terms), absorb=ABSORB)
    X2 = pyblp.Formulation("0 + I(-prices) + log_n_eq")
    ids, nodes, w = pyblp.Integration("halton", size=NDRAWS, specification_options={"seed": 0}) \
        ._build_many(2, np.unique(np.asarray(d.market_ids, dtype=object)))
    agents = pd.DataFrame({"market_ids": ids, "nodes0": nodes[:, 0], "nodes1": nodes[:, 1],
                           "weights": w.ravel()})
    return pyblp.Problem((X1, X2), d, pyblp.Formulation("1"), agents, rc_types=["log", "linear"])


def price_draw(j):
    """(beta_p, shift) with alpha_i = beta_p - exp(shift + sigma_p nu_1i) for the estimates j:
    the lognormal has beta_p = 0 and shift = pi_p; the original form has shift 0."""
    return (0.0, float(j["pi_p"])) if "pi_p" in j else (j["median_alpha"] + 1.0, 0.0)


def demand_at(d, terms, j):
    """The problem and mean utilities at the estimates j: one contraction, no optimization."""
    import pyblp
    prob = lognormal_problem(d, terms)
    r = prob.solve(sigma=np.diag([j["sigma_p"], j["sigma_lsz"]]),
                   pi=np.array([[j["pi_p"]], [0.0]]),
                   optimization=pyblp.Optimization("return"),
                   iteration=pyblp.Iteration("squarem", {"atol": 1e-12}), method="1s")
    return prob, np.asarray(r.delta).ravel()


def rcl(tag, col, iv="cur", x1extra=()):
    import pyblp
    pyblp.options.verbose = True
    assert not RCCONST, "the lognormal specification has no rc on the constant"
    t0 = time.time()
    d, terms = load(tag, col, x1extra)
    for k, c in enumerate(ZCOLS):
        d[f"demand_instruments{k}"] = d[c].astype(float)
    prob = lognormal_problem(d, terms)
    K = prob.K1 + prob.K2 + 1                       # + pi_p
    log(f"\n({col}, lognormal)  rows {len(d):,}  markets {prob.T:,}  K1 {prob.K1}  K2 {prob.K2}  "
        f"MD {prob.MD}  df {prob.MD - K}")
    assert prob.MD == len(ZCOLS) + len(terms), f"MD {prob.MD} unexpected"
    # start at the original form's median at its estimates: -exp(pi) = -0.708
    r = prob.solve(sigma=np.diag([0.25, 1.00]),
                   sigma_bounds=(np.diag([1e-6, 1e-6]), np.diag([5.0, 5.0])),
                   pi=np.array([[np.log(0.708)], [0.0]]),
                   pi_bounds=(np.array([[-5.0], [0.0]]), np.array([[3.0], [0.0]])),
                   optimization=pyblp.Optimization(
                       "l-bfgs-b", {"gtol": 1e-6, "maxfun": 80, "maxiter": 50}),
                   iteration=pyblp.Iteration("squarem", {"atol": 1e-12}),
                   method="2s", se_type="clustered", W_type="clustered",
                   check_optimality=OPTCHECK)
    lab = list(terms)
    b = np.asarray(r.beta).ravel(); bse = np.asarray(r.beta_se).ravel()
    sp, sl, pp = float(r.sigma[0, 0]), float(r.sigma[1, 1]), float(r.pi[0, 0])
    sse = np.asarray(r.sigma_se); sps, sls = float(sse[0, 0]), float(sse[1, 1])
    pps = float(np.asarray(r.pi_se)[0, 0])
    med, meds = -np.exp(pp), np.exp(pp) * pps                  # delta method
    xi = np.asarray(r.xi).ravel()
    Zd = np.asarray(prob.products.ZD)
    cl = pd.factorize(d.clustering_ids)[0]; G = int(cl.max() + 1)
    m = Zd * xi.reshape(-1, 1)
    M = np.zeros((G, Zd.shape[1])); np.add.at(M, cl, m)
    S = M.T @ M / len(m); gb = m.mean(0)[:, None]
    j = len(m) * float(np.squeeze(gb.T @ np.linalg.pinv(S) @ gb)); df = Zd.shape[1] - K
    p = 1 - stats.chi2.cdf(j, df) if df > 0 else np.nan
    log("\n" + "=" * 72)
    for l, v, s in zip(lab, b, bse):
        log(f"  {l:10s}{v:>12.5f}{s:>11.5f}  t {v/s:>6.2f}" + ("  *" if abs(v/s) >= 1.96 else ""))
    log(f"  {'pi_p':10s}{pp:>12.5f}{pps:>11.5f}  t {pp/pps:>6.2f}")
    log(f"  {'sigma_p':10s}{sp:>12.5f}{sps:>11.5f}  t {sp/sps:>6.2f}")
    log(f"  {'sigma_lsz':10s}{sl:>12.5f}{sls:>11.5f}  t {sl/sls:>6.2f}")
    log(f"  {'med alpha_i':10s}{med:>12.5f}{meds:>11.5f}  t {med/meds:>6.2f}")
    log("=" * 72)
    log(f"  HANSEN J {j:.3f}  df {df}  p {p:.4f}  clusters {G}  "
        f"{'PASSES' if p >= .05 else 'REJECTS'}")
    e_ = np.asarray(r.extract_diagonals(r.compute_elasticities())).ravel()
    by = [(int(k), float(v)) for k, v in d.assign(o=e_).groupby("n_cans").o.mean().items()]
    log(f"  own eps median {np.median(e_):+.4f}   " + "  ".join(f"{k}:{v:+.2f}" for k, v in by))
    nsig = sum(1 for v, s in zip(b, bse) if abs(v / s) >= 1.96)
    log(f"  terms significant: {nsig} of {len(terms)}")
    json.dump(dict(col=col, kind="rcl", form="lognormal", tag=tag, regime=COLS[col][0],
                   J=j, df=int(df), p=p, n=int(prob.N), markets=int(prob.T),
                   MD=int(prob.MD), K=int(K), clusters=G,
                   beta=dict(zip(lab, b.tolist())), se=dict(zip(lab, bse.tolist())),
                   pi_p=pp, pi_p_se=pps,
                   sigma_p=sp, sigma_p_se=sps, sigma_lsz=sl, sigma_lsz_se=sls,
                   median_alpha=med, median_alpha_se=meds,
                   by_rung=by, eps_median=float(np.median(e_)), n_sig=nsig,
                   seconds=time.time() - t0),
              open(E / f"f71_col_{col}_{tag}{TAGSFX}.json", "w"), indent=1)
    try:
        V = np.asarray(r.parameter_covariances) / prob.N
        nth = 3                                     # sigma_p, sigma_lsz, pi_p
        Vb = V[nth:nth + len(lab), nth:nth + len(lab)]
        chk = float(np.max(np.abs(np.sqrt(np.diag(Vb)) - bse)))
        log(f"  saved beta vcov {Vb.shape}  max |sqrt(diag)-beta_se| {chk:.2e}"
            + ("  MISMATCH" if chk > 1e-6 else "  OK"))
        np.save(E / f"f71_vcov_{col}_{tag}{TAGSFX}.npy", Vb)
        json.dump(lab, open(E / f"f71_vcov_{col}_{tag}{TAGSFX}_labels.json", "w"))
    except Exception as ex:
        log(f"  vcov save FAILED: {type(ex).__name__}: {ex}")
    log(f"\n[{time.time()-t0:.0f}s]")


def cells(tag, which=("ii", "iii", "iv")):
    """Per-column regime x rung: coefficient, t, and cell size, both normalisations."""
    d0 = pd.read_parquet(ROOT / "data_work" / f"f70_baseline_{tag}.parquet",
                         columns=["reg1", "reg2", "reg3", "n_cans"])
    for col in which:
        f = E / f"f71_col_{col}_{tag}{TAGSFX}.json"
        if not f.exists():
            print(f"\n{col}: not estimated"); continue
        r = json.load(open(f)); rc, states = COLS[col]
        V = np.load(E / f"f71_vcov_{col}_{tag}{TAGSFX}.npy")
        L = json.load(open(E / f"f71_vcov_{col}_{tag}{TAGSFX}_labels.json"))
        ix = {k: i for i, k in enumerate(L)}
        b = np.array([r["beta"][k] for k in L])
        u = d0[d0[rc] != ""]
        W = pd.crosstab(u[rc], u.n_cans)
        ref = [x for x in W.index if x not in states][0]
        allst = (ref,) + tuple(states)

        def vec(st, k):
            w = np.zeros(len(L))
            if st != ref:
                w[ix[f"{st}_{k}"]] = 1.0
            return w
        print(f"\n{'='*86}\n({col})  regime {rc} = <{'><'.join(rc[-1])}>  "
              f"reference {ref}   sigma_lsz {r['sigma_lsz']:.4f}   "
              f"median alpha_i {r['median_alpha']:+.4f}   J {r['J']:.3f} p {r['p']:.4f}\n{'='*86}")
        for norm in ("raw", "centered"):
            print(f"\n  {ref} = 0 reference" if norm == "raw" else "\n  CENTRED within rung")
            print(f"  {'':6s}" + "".join(f"{'rung '+str(k):>12s}" for k in RUNGS))
            for st in allst:
                cf, ts, nn = [], [], []
                for k in RUNGS:
                    w_ = W[k] / W[k].sum()
                    mb = sum(w_[q] * vec(q, k) for q in allst) if norm == "centered" else 0
                    x = vec(st, k) - mb
                    v = float(np.squeeze(x @ b)); sd = float(np.sqrt(x @ V @ x))
                    if sd == 0:
                        cf.append("0 (ref)"); ts.append(""); nn.append(f"[{W[k][st]:,}]")
                    else:
                        t = v / sd
                        cf.append(f"{v:+.3f}" + ("***" if abs(t) >= 2.58 else
                                  "**" if abs(t) >= 1.96 else "*" if abs(t) >= 1.64 else ""))
                        ts.append(f"({t:+.2f})"); nn.append(f"[{W[k][st]:,}]")
                print(f"  {st:6s}" + "".join(f"{x:>12s}" for x in cf))
                print(f"  {'':6s}" + "".join(f"{x:>12s}" for x in ts))
                print(f"  {'':6s}" + "".join(f"{x:>12s}" for x in nn))
        print("\n  *** |t|>=2.58  ** >=1.96  * >=1.64    [n] = rows in cell")


def render(tag):
    import textwrap
    rows = []
    for c in ("i", "ii", "iii", "iv"):
        f = E / f"f71_col_{c}_{tag}{TAGSFX}.json"
        rows.append(json.load(open(f)) if f.exists() else None)
    W = 20
    hdr = ["(i) nested logit", "(ii) baseline1", "(iii) baseline2", "(iv) baseline3"]
    sub = ["reg1 controls", "<t-1><t>", "<t><t+1>", "<t-1><t><t+1>"]
    print("=" * (14 + W * 4))
    print(f"{'BASELINE TABLE':^{14+W*4}}")
    print(f"{'store-week | median reference k=4 | 5% cut | sample f3_'+tag:^{14+W*4}}")
    print("=" * (14 + W * 4))
    print(f"{'':14s}" + "".join(f"{h:>{W}s}" for h in hdr))
    print(f"{'':14s}" + "".join(f"{s:>{W}s}" for s in sub))
    print("-" * (14 + W * 4))

    def line(lab, fn):
        print(f"{lab:14s}" + "".join(
            f"{(fn(r) if r else '--'):>{W}s}" for r in rows))
    line("alpha / med a_i", lambda r: (f"{r['beta']['prices']:+.4f} ({abs(r['beta']['prices']/r['se']['prices']):.2f})"
                                       if r['kind'] == 'nested_logit'
                                       else f"{r['median_alpha']:+.4f} ({abs(r['median_alpha']/r['median_alpha_se']):.2f})"))
    line("rho", lambda r: (f"{r['beta']['rho']:+.4f} ({r['beta']['rho']/r['se']['rho']:.2f})"
                           if r['kind'] == 'nested_logit' else "--"))
    line("sigma_p", lambda r: "--" if r['kind'] == 'nested_logit'
                               else f"{r['sigma_p']:.4f} ({r['sigma_p']/r['sigma_p_se']:.2f})")
    line("sigma_lsz", lambda r: "--" if r['kind'] == 'nested_logit'
                                 else f"{r['sigma_lsz']:.4f} ({r['sigma_lsz']/r['sigma_lsz_se']:.2f})")
    print("-" * (14 + W * 4))
    line("median own eps", lambda r: "--" if r['kind'] == 'nested_logit' else f"{r['eps_median']:.4f}")
    line("  eps rung 6", lambda r: "--" if r['kind'] == 'nested_logit' else f"{dict(r['by_rung'])[6]:.2f}")
    line("  eps rung 36", lambda r: "--" if r['kind'] == 'nested_logit' else f"{dict(r['by_rung'])[36]:.2f}")
    print("-" * (14 + W * 4))
    line("Hansen J", lambda r: f"{r['J']:.3f}")
    line("  df", lambda r: f"{r['df']}")
    line("  p", lambda r: f"{r['p']:.4f}")
    line("  verdict", lambda r: "PASSES" if r['p'] >= .05 else "REJECTS")
    line("N", lambda r: f"{r['n']:,}")
    line("markets", lambda r: "--" if r['kind'] == 'nested_logit' else f"{r['markets']:,}")
    line("clusters", lambda r: f"{r['clusters']}")
    line("regime terms", lambda r: "18" if r['col'] in ("i", "ii", "iii") else "42")
    print("=" * (14 + W * 4))
    print(textwrap.dedent("""
    NOTES
      Regimes are <state> letters over the weeks shown; the omitted state (NN, NNN in
      (iv)) is the reference AT EVERY RUNG and there are no rung main effects, so a
      coefficient is that regime-rung's mean-utility shift against no-sale at the same
      pack size, same price, same brand, same month.
      baseline1 NS = sale starts this week, SN = sale just ended, SS = continuing.
      baseline2 NS = a sale arrives NEXT week, SN = on sale now and ending, SS = continuing.
      beta_p is a ceiling: median alpha_i = beta_p - 1, same standard error.
      Per-column regime x rung coefficients, t-statistics and cell counts: --cells.
    """).rstrip())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="est")
    ap.add_argument("--col", choices=["i", "ii", "iii", "iv"])
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--cells", action="store_true")
    ap.add_argument("--iv", default="cur", choices=list(IVSETS))
    ap.add_argument("--lnz", action="store_true",
                    help="put log_n_eq in X1 (its MEAN; the rc stays in X2)")
    ap.add_argument("--no-demo", action="store_true", dest="no_demo",
                    help="drop the five zip3 ACS demographic interactions, which are "
                         "in X1 of columns (ii) to (iv) by default")
    ap.add_argument("--demo-raw", action="store_true", dest="demo_raw",
                    help="as --demo but with the demographics in raw units, "
                         "not centered and scaled")
    ap.add_argument("--fe", default="brand", choices=["brand", "product", "storeproduct"],
                    help="absorbed set: brand+year_month (default) or "
                         "product+year_month, or storeproduct+year_month; "
                         "product = brand x container x pack")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--rcconst", action="store_true",
                    help="add a random coefficient on the constant")
    ap.add_argument("--no-hessian", action="store_true", dest="no_hessian",
                    help="skip the Hessian optimality check, which needs a second pass "
                         "over every market; estimates and standard errors are unchanged")
    ap.add_argument("--packfe", action="store_true",
                    help="add pack-size dummies to X1, 12-pack omitted; they absorb "
                         "most of the price variation that identifies beta_p")
    ap.add_argument("--pool36", action="store_true",
                    help="fold 36-packs into a 30+ cell in the regime interaction; "
                         "the random coefficient still sees the true pack size")
    ap.add_argument("--slope", action="store_true",
                    help="one level and one log-pack-size slope per sale regime "
                         "instead of a free coefficient at every pack size")
    ap.add_argument("--can", action="store_true",
                    help="add a can dummy (is_can) to X1, the mean utility of the container")
    ap.add_argument("--draws", type=int, default=NDRAWS,
                    help="Halton draws; simulation error inflates the J")
    a = ap.parse_args()
    sfx0 = a.suffix or (a.iv + ("_lnz" if a.lnz else ""))
    globals()["TAGSFX"] = "" if sfx0 in ("cur", "") else "_" + sfx0
    if a.cells:
        cells(a.tag, (a.col,) if a.col else ("ii", "iii", "iv"))
    elif a.render:
        render(a.tag)
    else:
        globals()["ZCOLS"] = IVSETS[a.iv]
        globals()["RCCONST"] = a.rcconst
        globals()["NDRAWS"] = a.draws
        xt = (("log_n_eq",) if a.lnz else ()) + (("is_can",) if a.can else ())
        globals()["OPTCHECK"] = "gradient" if a.no_hessian else "both"
        globals()["PACKFE"] = a.packfe
        globals()["POOL36"] = a.pool36
        globals()["SLOPE"] = a.slope
        globals()["DEMO"] = (not a.no_demo) or a.demo_raw
        globals()["DEMO_RAW"] = a.demo_raw
        if a.fe == "product":
            globals()["ABSORB"] = "C(product_id) + C(year_month)"
        elif a.fe == "storeproduct":
            globals()["ABSORB"] = "C(store_product) + C(year_month)"
        sfx = a.suffix or (a.iv + ("_lnz" if a.lnz else "") + ("_rcc" if a.rcconst else "")
                           + ("_prodfe" if a.fe == "product" else
                              "_spfe" if a.fe == "storeproduct" else "")
                           + ("_slope" if a.slope else "_pool36" if a.pool36 else "")
                           + ("_packfe" if a.packfe else "") + ("_can" if a.can else "")
                           + ("_demoraw" if a.demo_raw else "_nodemo" if a.no_demo else ""))
        globals()["TAGSFX"] = "" if sfx in ("cur", "") else "_" + sfx
        log(f"IV set {a.iv}: {ZCOLS}   X1 extra: {list(xt)}   output suffix '{TAGSFX}'")
        if a.col == "i":
            nested_logit(a.tag, a.iv, xt)
        elif a.col:
            rcl(a.tag, a.col, a.iv, xt)
        else:
            ap.error("give --col or --render")
