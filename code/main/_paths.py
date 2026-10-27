"""Resolve the isolated stage directory created by reproduce.py."""
from pathlib import Path
import os

def stage_root():
    value = os.environ.get("WLS_STAGE_ROOT")
    if not value:
        raise RuntimeError("Use the package entry point: python reproduce.py --help. "
                           "The scripts expect the work folder prepared by reproduce.py.")
    root = Path(value).resolve()
    if not (root / ".wls_stage.json").is_file():
        raise RuntimeError("Unprepared WLS_STAGE_ROOT: use reproduce.py prepare first")
    return root
