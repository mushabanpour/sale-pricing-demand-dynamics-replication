#!/usr/bin/env python3
"""The value of the sales stores ran, at observed prices (Table 12).

    -> results/f98_sales_value_iv_long.csv
"""
import subprocess, sys
from _lib import CODE, ROOT
r = subprocess.run([sys.executable, str(CODE / "f98_sales_value.py")], cwd=ROOT)
sys.exit(r.returncode)
