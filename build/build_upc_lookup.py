#!/usr/bin/env python3
"""The UPC lookup build/HMScleaningdo.do merges Homescan purchases with: brand, container and
pack size of every beer UPC, from the RMS products master file (step 1 of
build/legacy/build_exact_pack_container_beer_panel.py).

    NIELSEN_RMS=/path/to/nielsen_extracts/RMS python3 build/build_upc_lookup.py
        -> build/work/upc_lookup_exact_container.parquet
"""
import os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "legacy"))
from build_exact_pack_container_beer_panel import build_upc_lookup

RMS = Path(os.environ.get("NIELSEN_RMS", "nielsen_extracts/RMS"))
(HERE / "work").mkdir(exist_ok=True)
build_upc_lookup(RMS / "Master_Files_2006-2020" / "Latest", HERE / "work", write_csv=False)
