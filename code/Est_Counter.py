#!/usr/bin/env python3
"""Counterfactual pricing regimes NPD, VD, IT and MIX: constrained Nash prices,
consumer and producer surplus, and the equilibrium price and quantity panel.

    -> results/f92_welfare_iv_dev.csv, f92_cs_quintile_pack_iv_dev.csv, f92_pq_iv_dev.parquet
"""
import subprocess, sys
from _lib import CODE, ROOT
r = subprocess.run([sys.executable, str(CODE / "f92_counterfactual.py"), "iv"], cwd=ROOT)
sys.exit(r.returncode)
