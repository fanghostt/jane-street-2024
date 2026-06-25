"""Shared test fixtures: build tiny fake raw/train data without Kaggle access."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from js2024.data.data import FEATURE_COLUMNS, TARGET_COLUMN, WEIGHT_COLUMN


def make_train_frame(n: int = 12, drop_columns: list[str] | None = None) -> pl.DataFrame:
    """A small DataFrame with the full required train schema (optionally dropping cols)."""
    drop_columns = drop_columns or []
    data: dict[str, list] = {
        "date_id": [i // 3 for i in range(n)],
        "time_id": [i % 3 for i in range(n)],
        "symbol_id": [i % 2 for i in range(n)],
    }
    for c in FEATURE_COLUMNS:
        # Introduce some nulls into feature_00 so missing-ratio is non-trivial.
        if c == "feature_00":
            data[c] = [None if i % 4 == 0 else float(i) for i in range(n)]
        else:
            data[c] = [float(i) for i in range(n)]
    data[WEIGHT_COLUMN] = [1.0 + 0.1 * i for i in range(n)]
    data[TARGET_COLUMN] = [0.1 * i for i in range(n)]
    for col in drop_columns:
        data.pop(col, None)
    return pl.DataFrame(data)


def make_raw_dir(
    root: Path,
    partitioned: bool = False,
    drop_train_columns: list[str] | None = None,
    missing_files: list[str] | None = None,
) -> Path:
    """Create a minimal fake Kaggle raw dir under ``root`` and return it.

    Produces ``train.parquet`` (single file or partitioned dir) plus tiny fake
    ``lags.parquet``, ``features.csv`` and ``responders.csv``. ``missing_files``
    lists required files to deliberately *not* create.
    """
    missing_files = missing_files or []
    root.mkdir(parents=True, exist_ok=True)
    df = make_train_frame(drop_columns=drop_train_columns)

    if "train.parquet" not in missing_files:
        train_path = root / "train.parquet"
        if partitioned:
            for pid in (0, 1):
                part_dir = train_path / f"partition_id={pid}"
                part_dir.mkdir(parents=True)
                df.write_parquet(part_dir / "part-0.parquet")
        else:
            df.write_parquet(train_path)

    if "lags.parquet" not in missing_files:
        pl.DataFrame({"date_id": [0, 1], "responder_6_lag_1": [0.0, 0.1]}).write_parquet(
            root / "lags.parquet"
        )
    if "features.csv" not in missing_files:
        (root / "features.csv").write_text("feature,tag\nfeature_00,x\n", encoding="utf-8")
    if "responders.csv" not in missing_files:
        (root / "responders.csv").write_text("responder,tag\nresponder_6,y\n", encoding="utf-8")

    return root


@pytest.fixture
def raw_dir_factory(tmp_path):
    """Factory fixture returning :func:`make_raw_dir` rooted under a tmp path."""

    def _factory(name: str = "raw", **kwargs) -> Path:
        return make_raw_dir(tmp_path / name, **kwargs)

    return _factory


@pytest.fixture
def write_train(tmp_path):
    """Factory fixture: write a small train parquet and return its path."""

    def _factory(name: str = "train.parquet", **kwargs) -> Path:
        df = make_train_frame(**kwargs)
        path = tmp_path / name
        df.write_parquet(path)
        return path

    return _factory
