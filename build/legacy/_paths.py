"""Where the legacy modules read and write.  build_regression_ready.py sets these
environment variables before it imports any of them."""
import json, os
from pathlib import Path

RMS = Path(os.environ.get("BUILD_RMS", "RMS"))
WORK = Path(os.environ.get("BUILD_WORK", "work"))
ACS = Path(os.environ.get("BUILD_ACS", "acs_zip3_demographics_2016_2019.csv"))
EXCL = Path(__file__).resolve().parent.parent / "excluded_upcs.json"


def excluded_upcs():
    return json.load(open(EXCL))
