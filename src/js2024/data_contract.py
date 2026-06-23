"""Raw-data contract checks for the Jane Street 2024 competition download.

These helpers verify that a ``data/raw`` directory looks like a real Kaggle
download *before* any training is attempted: required files are present and the
``train`` parquet exposes the schema the rest of the pipeline relies on. Nothing
here materializes the full dataset — only lazy schema / min-max probes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from .data import (
    get_required_train_columns,
    resolve_parquet_scan_path,
    scan_train_data,
)

# Files / directories a genuine competition download must contain.
RAW_REQUIRED_FILES: list[str] = [
    "train.parquet",
    "lags.parquet",
    "features.csv",
    "responders.csv",
]

# How many rows we materialize from train as a cheap "is it readable" probe.
_SAMPLE_ROWS = 5


def _path_size_bytes(p: Path) -> int:
    """Total size in bytes of a file, or recursive sum of a directory's files."""
    if p.is_file():
        return p.stat().st_size
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return 0


def _describe_required_file(raw_dir: Path, name: str) -> dict[str, Any]:
    p = raw_dir / name
    exists = p.exists()
    if not exists:
        return {"name": name, "exists": False, "type": None, "size_bytes": 0}
    file_type = "dir" if p.is_dir() else "file"
    return {
        "name": name,
        "exists": True,
        "type": file_type,
        "size_bytes": _path_size_bytes(p),
    }


def check_raw_data_contract(raw_dir: str | Path) -> dict[str, Any]:
    """Inspect ``raw_dir`` and return a structured contract report.

    The returned dict always contains the same keys so callers can render a
    summary regardless of how far the checks got:

    ``raw_dir``, ``exists``, ``required_files`` (list of per-file dicts),
    ``train_scan_path``, ``train_columns_count``, ``missing_train_columns``,
    ``date_min``, ``date_max``, ``sample_rows_checked``.

    A non-empty ``missing_train_columns`` (or any missing required file) means
    the contract is *not* satisfied; callers decide how to surface that.
    """
    raw_dir = Path(raw_dir)
    report: dict[str, Any] = {
        "raw_dir": str(raw_dir),
        "exists": raw_dir.exists(),
        "required_files": [],
        "train_scan_path": None,
        "train_columns_count": None,
        "missing_train_columns": None,
        "date_min": None,
        "date_max": None,
        "sample_rows_checked": 0,
    }

    report["required_files"] = [
        _describe_required_file(raw_dir, name) for name in RAW_REQUIRED_FILES
    ]

    if not raw_dir.exists():
        return report

    train_path = raw_dir / "train.parquet"
    if not train_path.exists():
        # Can't probe the schema; required-file check above already flags it.
        return report

    try:
        scan_path = resolve_parquet_scan_path(train_path)
    except (FileNotFoundError, ValueError):
        return report
    report["train_scan_path"] = str(scan_path)

    lf = scan_train_data(train_path)
    schema_names = lf.collect_schema().names()
    report["train_columns_count"] = len(schema_names)

    required_cols = get_required_train_columns()
    available = set(schema_names)
    report["missing_train_columns"] = [c for c in required_cols if c not in available]

    if "date_id" in available:
        bounds = lf.select(
            pl.col("date_id").min().alias("date_min"),
            pl.col("date_id").max().alias("date_max"),
        ).collect()
        dmin = bounds.item(0, "date_min")
        dmax = bounds.item(0, "date_max")
        report["date_min"] = None if dmin is None else int(dmin)
        report["date_max"] = None if dmax is None else int(dmax)

    # Cheap readability probe: pull a handful of rows.
    sample = lf.head(_SAMPLE_ROWS).collect()
    report["sample_rows_checked"] = sample.height

    return report


def contract_ok(report: dict[str, Any]) -> bool:
    """True iff the raw dir exists, all required files exist and no train cols missing."""
    if not report.get("exists"):
        return False
    if any(not f["exists"] for f in report.get("required_files", [])):
        return False
    missing = report.get("missing_train_columns")
    if missing is None or len(missing) > 0:
        return False
    return True
