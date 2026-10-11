#!/usr/bin/env python3
"""Supply side: markups and marginal costs over the retail conduct grid, with pack
size absorbed in the cost fixed effect and cost fitted on no-sale weeks only.

    -> results/f90_supply_rcgrid_iv_dev.csv, results/f90_supply_panel_iv_dev.parquet
"""
import subprocess, sys
from _lib import CODE, ROOT
r = subprocess.run([sys.executable, str(CODE / "f90_supply_mw.py"), "iv"], cwd=ROOT)
sys.exit(r.returncode)
