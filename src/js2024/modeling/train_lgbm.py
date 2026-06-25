"""CLI entry point for the LightGBM V0 baseline.

Usage
-----
From the project root::

    PYTHONPATH=src python -m js2024.modeling.train_lgbm --config configs/lgbm_v0.yaml

The script reads ``train.parquet`` (path from config), builds a date-based
holdout split, trains a single ``LGBMRegressor`` with sample weights and early
stopping, scores the validation fold with the competition metric, and writes the
model, out-of-fold validation predictions, and a markdown report.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from .config import LGBMConfig, load_lgbm_config, resolve_project_path
from ..data.data import (
    TARGET_COLUMN,
    WEIGHT_COLUMN,
    get_default_columns,
    get_date_id_range,
    load_train_data,
    validate_data_path,
)
from .features import get_v0_feature_columns, prepare_lgbm_frame
from .metrics import weighted_zero_mean_r2
from .reporting import write_lgbm_report
from .validation import build_holdout_split, filter_by_date_range, summarize_date_split


def _series_summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


@dataclass
class LGBMRunResult:
    """Structured result of a single :func:`run` invocation."""

    run_name: str
    score: float
    model_path: Path
    oof_path: Path
    report_path: Path
    split_summary: dict
    feature_count: int
    best_iteration: int
    prediction_summary: dict
    target_summary: dict
    feature_importance_top20: list[tuple[str, int]] = field(default_factory=list)


def run(
    config: LGBMConfig,
    run_name: str = "lgbm_v0",
    df: pl.DataFrame | None = None,
) -> LGBMRunResult:
    """Run the full baseline pipeline and return a structured result.

    Parameters
    ----------
    config
        The (already validated) LGBM configuration.
    run_name
        Artifact basename — model/OOF/report files are named after it. Defaults
        to ``"lgbm_v0"`` so the standard CLI keeps writing ``lgbm_v0.*``.
    df
        Optional pre-loaded train frame. When provided the data load is skipped
        and the split is taken from ``df`` directly — this lets a caller load
        the (large) parquet once and reuse it across many splits. When ``None``
        the frame is loaded from ``config.train_path``.
    """
    import lightgbm as lgb

    feature_cols = get_v0_feature_columns(include_symbol=True, include_time=True)
    # Columns we must read: ids + features + target + weight.
    columns = get_default_columns(include_target=True, include_weight=True)

    if df is None:
        train_path = resolve_project_path(config.train_path)
        # Fail early & clearly if the data has not been placed yet.
        validate_data_path(train_path)
        print(f"[js2024] Loading data from {train_path} ...")
        df = load_train_data(
            train_path,
            columns=columns,
            start_date_id=config.start_date_id,
            end_date_id=config.end_date_id,
            collect=True,
        )
    min_date, max_date = get_date_id_range(df)
    print(f"[js2024] Loaded {df.height:,} rows; date_id range [{min_date}, {max_date}]")

    split = build_holdout_split(
        min_date_id=min_date,
        max_date_id=max_date,
        valid_days=config.valid_days,
        gap_days=config.gap_days,
    )
    split_summary = summarize_date_split(df, split, date_col="date_id")
    print(f"[js2024] Split: {split_summary}")

    train_df = filter_by_date_range(df, "date_id", split.train_start, split.train_end)
    valid_df = filter_by_date_range(df, "date_id", split.valid_start, split.valid_end)

    X_train, y_train, w_train = prepare_lgbm_frame(
        train_df, feature_cols, TARGET_COLUMN, WEIGHT_COLUMN
    )
    X_valid, y_valid, w_valid = prepare_lgbm_frame(
        valid_df, feature_cols, TARGET_COLUMN, WEIGHT_COLUMN
    )

    model = lgb.LGBMRegressor(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        device_type=config.device_type,
        max_bin=config.max_bin,
        gpu_use_dp=config.gpu_use_dp,
        random_state=config.random_state,
        n_jobs=-1,
    )

    print("[js2024] Training LGBMRegressor ...")
    model.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        eval_set=[(X_valid, y_valid)],
        eval_sample_weight=[w_valid],
        callbacks=[
            lgb.early_stopping(config.early_stopping_rounds),
            lgb.log_evaluation(period=100),
        ],
    )

    preds = model.predict(X_valid)
    score = weighted_zero_mean_r2(y_valid.to_numpy(), preds, w_valid.to_numpy())
    print(f"[js2024] Validation weighted zero-mean R²: {score:.6f}")

    # --- Persist artifacts (namespaced by run_name) ---
    model_dir = resolve_project_path(config.model_dir)
    output_dir = resolve_project_path(config.output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "oof").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{run_name}.txt"
    model.booster_.save_model(str(model_path))
    print(f"[js2024] Saved model -> {model_path}")

    oof_path = output_dir / "oof" / f"{run_name}_valid_predictions.parquet"
    oof = valid_df.select(["date_id", "time_id", "symbol_id", TARGET_COLUMN, WEIGHT_COLUMN])
    oof = oof.with_columns(pl.Series("prediction", preds))
    oof.write_parquet(oof_path)
    print(f"[js2024] Saved OOF predictions -> {oof_path}")

    importances = sorted(
        zip(feature_cols, model.feature_importances_.tolist()),
        key=lambda kv: kv[1],
        reverse=True,
    )
    report_path = output_dir / "reports" / f"{run_name}_report.md"
    prediction_summary = _series_summary(preds)
    target_summary = _series_summary(y_valid.to_numpy())
    write_lgbm_report(
        path=report_path,
        config=config,
        split_summary=split_summary,
        score=score,
        feature_cols=feature_cols,
        prediction_summary=prediction_summary,
        target_summary=target_summary,
        feature_importance=importances,
    )
    print(f"[js2024] Saved report -> {report_path}")

    best_iteration = int(getattr(model, "best_iteration_", 0) or 0)
    return LGBMRunResult(
        run_name=run_name,
        score=float(score),
        model_path=model_path,
        oof_path=oof_path,
        report_path=report_path,
        split_summary=dict(split_summary),
        feature_count=len(feature_cols),
        best_iteration=best_iteration,
        prediction_summary=prediction_summary,
        target_summary=target_summary,
        feature_importance_top20=importances[:20],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the LightGBM V0 baseline.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config, e.g. configs/lgbm_v0.yaml",
    )
    args = parser.parse_args(argv)

    try:
        config = load_lgbm_config(args.config)
        run(config)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        # Clean, actionable message instead of a traceback for the common
        # config / data errors (missing file, bad hyperparameter, missing
        # config key, missing/invalid columns, too-small date range, ...).
        print(f"[js2024] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
