#!/usr/bin/env python3
"""Firm-level profit and the equilibrium price/quantity table for Tables 10 and 11.

    -> results/f92_ps_by_firm.csv, f92_scenario_table_firm.csv
"""
import subprocess, sys
from _lib import CODE, ROOT
r = subprocess.run([sys.executable, str(CODE / "f92_by_firm.py")], cwd=ROOT)
sys.exit(r.returncode)
