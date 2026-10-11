#!/usr/bin/env python3
"""Elasticities and the price coefficient by pack size, from the estimated demand
system.  One objective evaluation at the estimated sigma recovers delta exactly.

    -> results/f85_alpha_by_rung.csv, results/f86_brand_elast_iv_dev.csv
"""
import subprocess, sys
from _lib import CODE, ROOT
for script, args in (("f85_alpha_by_rung.py", ["ii", "iii", "iv"]),
                     ("f86_brand_elast.py", ["iv"])):
    r = subprocess.run([sys.executable, str(CODE / script)] + args, cwd=ROOT)
    if r.returncode:
        sys.exit(r.returncode)
