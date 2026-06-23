"""Data loading utilities for the Jane Street 2024 train.parquet.

These helpers wrap polars ``scan_parquet`` so we only read the columns and
date-range we need. They never assume the data is present: missing files and
missing columns raise clear, actionable errors.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

FEATURE_COLUMNS: list[str] = [f"feature_{i:02d}" for i in range(79)]
TARGET_COLUMN: str = "responder_6"
WEIGHT_COLUMN: str = "weight"
ID_COLUMNS: list[str] = ["date_id", "time_id", "symbol_id"]

_DATA_HINT = (
    "Download the Jane Street 2024 competition data from Kaggle and place "
    "train.parquet at data/raw/train.parquet (this repo does not ship the data)."
)


def get_default_columns(include_target: bool = True, include_weight: bool = True) -> list[str]:
    """Return the default column projection: ids + features (+ target/weight)."""
    cols = list(ID_COLUMNS) + list(FEATURE_COLUMNS)
    if include_weight:
        cols.append(WEIGHT_COLUMN)
    if include_target:
        cols.append(TARGET_COLUMN)
    return cols


def validate_data_path(path: str | Path) -> Path:
    """Return ``path`` as a Path if it exists, else raise FileNotFoundError with a hint."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Training data not found at '{p}'.\n{_DATA_HINT}"
        )
    return p


def load_train_data(
    path: str | Path,
    columns: list[str] | None = None,
    start_date_id: int | None = None,
    end_date_id: int | None = None,
    collect: bool = True,
) -> pl.DataFrame | pl.LazyFrame:
    """Lazily read selected columns / date-range from train.parquet.

    Parameters
    ----------
    path
        Path to the parquet file. Validated for existence first.
    columns
        Column projection; defaults to :func:`get_default_columns`.
    start_date_id, end_date_id
        Optional inclusive ``date_id`` bounds.
    collect
        If True (default) return a materialized ``pl.DataFrame``; otherwise return
        the lazy ``pl.LazyFrame`` so the caller can chain more operations.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If any requested column is absent from the parquet schema.
    """
    p = validate_data_path(path)
    if columns is None:
        columns = get_default_columns()

    lf = pl.scan_parquet(p)

    available = set(lf.collect_schema().names())
    missing = [c for c in columns if c not in available]
    if missing:
        raise ValueError(
            f"Requested columns missing from {p}: {missing}. "
            f"File has {len(available)} columns."
        )

    lf = lf.select(columns)

    if start_date_id is not None:
        lf = lf.filter(pl.col("date_id") >= start_date_id)
    if end_date_id is not None:
        lf = lf.filter(pl.col("date_id") <= end_date_id)

    return lf.collect() if collect else lf


def get_date_id_range(df: pl.DataFrame) -> tuple[int, int]:
    """Return ``(min_date_id, max_date_id)`` from a materialized DataFrame."""
    if "date_id" not in df.columns:
        raise ValueError(f"'date_id' not found in columns {df.columns}")
    col = df.get_column("date_id")
    if col.len() == 0:
        raise ValueError("DataFrame is empty; cannot compute date_id range.")
    return int(col.min()), int(col.max())
