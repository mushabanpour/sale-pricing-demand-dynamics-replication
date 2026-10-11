#!/usr/bin/env python3
"""Table 12 Panel B: one firm removes ITS while its rivals re-set their regular-week prices.

    -> results/f99_solo_rivals_iv_long.csv
"""
import subprocess, sys
from _lib import CODE, ROOT
r = subprocess.run([sys.executable, str(CODE / "f99_solo_rivals.py")], cwd=ROOT)
sys.exit(r.returncode)
