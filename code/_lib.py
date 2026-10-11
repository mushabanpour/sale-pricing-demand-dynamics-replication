"""Shared paths and helpers for the pipeline scripts."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
CODE, DW, RES, EST = ROOT / "code", ROOT / "data_work", ROOT / "results", ROOT / "estimation"
TEX, FIG = ROOT / "tex", ROOT / "figures"
DEST = ROOT / "manuscript"
DOC = "Sale Pricing and Demand Dynamics"
TAG, COL = "long", "iv"
sys.path.insert(0, str(CODE))
for d in (TEX, FIG, RES, EST, DW):
    d.mkdir(exist_ok=True)


def uniform(tex):
    """One layout for every table: columns flush to both margins with the space spread
    between them, and signed numbers set with a true minus/plus in math mode."""
    import re
    def spec(m):
        cols = re.sub(r"@\{[^{}]*(\{[^{}]*\}[^{}]*)*\}", "", m.group(2))
        return m.group(1) + "{@{}" + cols[0] + r"@{\extracolsep\fill}" + cols[1:] + "@{}}"
    tex = re.sub(r"(\\begin\{tabular\*\}\{[^}]*\})\{((?:[^{}]|\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})*)\}", spec, tex)
    num = re.compile(r"(?<=[&\s])([+-])(\d[\d,]*(?:\.\d+)?)(?=\s*(?:&|\\\\))")
    def arg(t, i):                       # the balanced {...} starting at t[i]
        d = 0
        for j in range(i, len(t)):
            d += (t[j] == "{") - (t[j] == "}")
            if d == 0:
                return t[i:j + 1], j + 1
    def bold(c):
        c0 = c.strip()
        if not c0:
            return c
        if c0.startswith(r"\multicolumn"):
            a1, k = arg(c0, len(r"\multicolumn"))
            a2, k = arg(c0, k)
            a3, _ = arg(c0, k)
            return r"\multicolumn" + a1 + a2 + "{" + bold(a3[1:-1]) + "}"
        return r"{\boldmath\textbf{" + c0 + "}}"
    lines, head = tex.split("\n"), False
    for i, line in enumerate(lines):     # header rows: every cell bold, math included
        if r"\toprule" in line:
            head = True
            continue
        if r"\midrule" in line:
            head = False
        if head and "&" in line and "Panel" not in line:
            k = line.index("\\\\") if "\\\\" in line else len(line)
            lines[i] = " & ".join(bold(c) for c in line[:k].split("&")) + line[k:]
    tex = "\n".join(lines)
    out = []
    for line in tex.split("\n"):
        if "&" in line and not line.lstrip().startswith(("\\caption", "\\item")):
            line = num.sub(lambda m: "$" + m.group(1) + m.group(2).replace(",", "{,}") + "$", line)
        out.append(line)
    return "\n".join(out)


def write_table(n, body):
    """Write one float to manuscript/TableN.tex and report it."""
    DEST.mkdir(exist_ok=True)
    p = DEST / f"Table{n}.tex"
    from texutil import firm_names
    p.write_text(firm_names(uniform(body.rstrip("\n"))) + "\n")
    print(f"wrote {p.relative_to(ROOT)}  ({len(body.splitlines())} lines)")
    return p


DATA = ROOT / "data"
ACS = DATA / "acs_zip3_demographics_2016_2019.csv"


def acs_years(tag=TAG):
    """The ACS years the estimation sample spans -- what add_demographics averages."""
    import pandas as pd
    ym = pd.read_parquet(DW / f"f70_baseline_{tag}.parquet",
                         columns=["year_month"]).year_month.astype(int)
    yrs = sorted(set(ym // 100) & set(pd.read_csv(ACS, usecols=["year"]).year))
    return yrs[0], yrs[-1]
