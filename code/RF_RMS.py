#!/usr/bin/env python3
"""Scanner reduced forms: the sale definition's premises, the event study, cross-pack
substitution and category expansion, on the baseline estimation sample.

    -> results/f80_rf*.csv
"""
import runpy
from _lib import CODE
runpy.run_path(str(CODE / "f80_rf_scanner.py"), run_name="__main__")
