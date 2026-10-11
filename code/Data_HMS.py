#!/usr/bin/env python3
"""Household panel data, regression ready.

Restricts Nielsen Homescan to the demand sample's window, brands and pack sizes, and
derives the purchase-timing variables: days to the next beer purchase, weeks since the
last one, the size of that purchase, and the deal flag recorded with the transaction.

    -> results/f81_rf*.csv  (the household regressions and the series the figures use)
"""
import runpy
from _lib import CODE
runpy.run_path(str(CODE / "f81_rf_hms.py"), run_name="__main__")
