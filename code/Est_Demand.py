#!/usr/bin/env python3
"""Demand estimation: the nested logit benchmark and three random coefficients
specifications differing only in the weeks that define the sale regime.

    -> estimation/f71_col_{i,ii,iii,iv}_long.json and the parameter covariances
"""
import subprocess, sys
from _lib import CODE, ROOT
for col in ("i", "ii", "iii", "iv"):
    print(f"--- column ({col}) ---", flush=True)
    r = subprocess.run([sys.executable, str(CODE / "f71_baseline_table.py"),
                        "--col", col, "--tag", "long"], cwd=ROOT)
    if r.returncode:
        sys.exit(r.returncode)

