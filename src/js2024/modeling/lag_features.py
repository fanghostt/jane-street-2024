"""Train-reconstructed responder lag features.

The competition serves each day the *previous* day's responders
(``responder_0_lag_1 … responder_8_lag_1``) via ``lags.parquet``, but that file
is synthetic and does not match this repo's ``train.parquet``. For offline
training we reconstruct the lags locally from ``train.parquet`` responders, with
strict leakage control: only ``date_id`` D-1 (and earlier) may inform date D.

Reconstruction is a self-join: copy the responder panel, shift ``date_id`` by
``+1``, rename ``responder_i -> responder_i_lag_1``, and left-join back on
``(date_id, time_id, symbol_id)``. The first ``date_id`` has no predecessor, so
its lag columns are null; current-date responders are never used as their own
lag. ``lags.parquet`` is never read here.
"""

from __future__ import annotations

import polars as pl

from ..data.data import ID_COLUMNS

RESPONDER_COLUMNS: list[str] = [f"responder_{i}" for i in range(9)]
LAG_FEATURE_COLUMNS: list[str] = [f"responder_{i}_lag_1" for i in range(9)]


def get_lag_feature_columns() -> list[str]:
    """Return the 9 day-lagged responder feature columns."""
    return list(LAG_FEATURE_COLUMNS)


def add_responder_lags_from_train(df: pl.DataFrame) -> pl.DataFrame:
    """Reconstruct D-1 responders as ``responder_i_lag_1`` features on each D row.

    The join key is ``(date_id, time_id, symbol_id)``; the first ``date_id`` has
    null lags; current-date responders are never used as their own lag.

    Raises
    ------
    ValueError
        If any id or responder column is missing.
    """
    missing = [c for c in ID_COLUMNS + RESPONDER_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"add_responder_lags_from_train: missing columns {missing}")

    lags = (
        df.select(ID_COLUMNS + RESPONDER_COLUMNS)
        .with_columns((pl.col("date_id") + 1).alias("date_id"))
        .rename({f"responder_{i}": f"responder_{i}_lag_1" for i in range(9)})
    )
    return df.join(lags, on=ID_COLUMNS, how="left")
