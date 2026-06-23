"""Time-aware validation utilities (date_id based holdout splits).

Financial time-series must never be split randomly: rows from the future would
leak into training. We split strictly by ``date_id`` and optionally insert a gap
between train and valid to reduce leakage from autocorrelated targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl


@dataclass
class DateRangeSplit:
    """Inclusive ``date_id`` boundaries for a train/valid holdout split."""

    train_start: int | None
    train_end: int
    valid_start: int
    valid_end: int


def build_holdout_split(
    min_date_id: int,
    max_date_id: int,
    valid_days: int,
    gap_days: int = 0,
) -> DateRangeSplit:
    """Build a holdout split where the most recent ``valid_days`` are validation.

    Layout (inclusive boundaries)::

        [train_start ... train_end] (gap_days) [valid_start ... valid_end]

    Parameters
    ----------
    min_date_id, max_date_id
        Observed inclusive range of ``date_id`` in the data.
    valid_days
        Number of trailing distinct ``date_id`` units reserved for validation.
    gap_days
        Number of ``date_id`` units dropped between train and valid.

    Raises
    ------
    ValueError
        If inputs are invalid or there is not enough history to form a train set.
    """
    if valid_days <= 0:
        raise ValueError(f"valid_days must be > 0, got {valid_days}")
    if gap_days < 0:
        raise ValueError(f"gap_days must be >= 0, got {gap_days}")
    if max_date_id < min_date_id:
        raise ValueError(
            f"max_date_id ({max_date_id}) < min_date_id ({min_date_id})"
        )

    valid_end = max_date_id
    valid_start = max_date_id - valid_days + 1
    train_end = valid_start - gap_days - 1
    train_start = min_date_id

    if valid_start <= min_date_id:
        raise ValueError(
            f"Not enough history: valid window starts at {valid_start} which is "
            f"<= min_date_id ({min_date_id}); reduce valid_days/gap_days."
        )
    if train_end < train_start:
        raise ValueError(
            f"Not enough history for training: train_end ({train_end}) < "
            f"train_start ({train_start}); reduce valid_days/gap_days."
        )

    return DateRangeSplit(
        train_start=train_start,
        train_end=train_end,
        valid_start=valid_start,
        valid_end=valid_end,
    )


def filter_by_date_range(
    df: pl.DataFrame,
    date_col: str,
    start: int | None,
    end: int | None,
) -> pl.DataFrame:
    """Filter a polars DataFrame to ``start <= date_col <= end`` (both inclusive).

    ``start`` or ``end`` may be ``None`` to leave that side unbounded.
    """
    if date_col not in df.columns:
        raise ValueError(f"date_col '{date_col}' not found in columns {df.columns}")

    out = df
    if start is not None:
        out = out.filter(pl.col(date_col) >= start)
    if end is not None:
        out = out.filter(pl.col(date_col) <= end)
    return out


def summarize_date_split(
    df: pl.DataFrame,
    split: DateRangeSplit,
    date_col: str = "date_id",
) -> dict[str, Any]:
    """Summarize row/day counts for each side of a split."""
    train_df = filter_by_date_range(df, date_col, split.train_start, split.train_end)
    valid_df = filter_by_date_range(df, date_col, split.valid_start, split.valid_end)

    return {
        "train_start": split.train_start,
        "train_end": split.train_end,
        "valid_start": split.valid_start,
        "valid_end": split.valid_end,
        "train_rows": train_df.height,
        "valid_rows": valid_df.height,
        "train_days": train_df.get_column(date_col).n_unique(),
        "valid_days": valid_df.get_column(date_col).n_unique(),
    }
