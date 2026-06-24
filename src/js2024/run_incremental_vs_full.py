r"""CLI: compare *full* vs *incremental* (daily-refit) LightGBM on a fixed test block.

Both modes share one initial model fit on the training region and are scored on the
**same** trailing ``test_days`` block (default the last 200 ``date_id``s), so their
weighted zero-mean R² is directly comparable. This is the LightGBM, leakage-clean
analog of the "with vs without online learning" rows in the
``evgeniavolkova/kagglejanestreet`` writeup.

Layout (inclusive ``date_id`` boundaries)::

    [ start ......... es_holdout ][ TEST = last test_days ]
      \________ train region ____/ \____ scored block ____/
                (es_holdout = last `valid_days` of train, for early stopping)

- ``full``        : fit on the train region, predict the whole test block once.
- ``incremental`` : same fit, then walk the test block day-by-day, refitting leaf
  values on each freshly-revealed day before predicting the next (cadence-controlled).

The big train frame is loaded **once** and reused for both modes. Heavy artifacts go
under ``--out-dir`` (gitignored); the only committed output is the markdown doc.

This stage introduces NO feature engineering, GRU, auxiliary targets, ensembling, or
prediction clipping.

Usage
-----
    uv run js2024-run-incremental-vs-full \\
        --config configs/lgbm_v0_incremental.yaml \\
        --out-dir outputs/incremental_vs_full/lgbm_v0 \\
        --docs-out docs/experiments/lgbm_v0_incremental_vs_full.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import polars as pl

from .config import LGBMConfig, load_lgbm_config, resolve_project_path
from .data import (
    TARGET_COLUMN,
    WEIGHT_COLUMN,
    get_default_columns,
    get_date_id_range,
    load_train_data,
    validate_data_path,
)
from .estimators import LGBMEstimator
from .features import get_v0_feature_columns
from .validation import build_holdout_split, filter_by_date_range, summarize_date_split
from .walk_forward import WalkForwardResult, walk_forward_evaluate

MODES = ("full", "incremental")


def _lgbm_params(config: LGBMConfig) -> dict[str, Any]:
    return {
        "n_estimators": config.n_estimators,
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "random_state": config.random_state,
    }


def run_modes(
    config: LGBMConfig,
    df: pl.DataFrame,
    *,
    feature_cols: list[str],
) -> dict[str, Any]:
    """Fit + walk-forward evaluate both modes on a shared frame; return results."""
    min_date, max_date = get_date_id_range(df)

    # Fixed trailing test block of `test_days`; train region is everything before it.
    test_split = build_holdout_split(
        min_date_id=min_date,
        max_date_id=max_date,
        valid_days=config.test_days,
        gap_days=0,
    )
    test_start, test_end = test_split.valid_start, test_split.valid_end
    train_lo, train_hi = min_date, test_start - 1

    # Early-stopping holdout = last `valid_days` of the TRAIN region (never the test).
    es_split = build_holdout_split(
        min_date_id=train_lo,
        max_date_id=train_hi,
        valid_days=config.valid_days,
        gap_days=config.gap_days,
    )
    train_df = filter_by_date_range(df, "date_id", es_split.train_start, es_split.train_end)
    es_valid_df = filter_by_date_range(df, "date_id", es_split.valid_start, es_split.valid_end)

    print(
        f"[js2024] train region [{train_lo}, {train_hi}] "
        f"(fit [{es_split.train_start}, {es_split.train_end}], "
        f"es-holdout [{es_split.valid_start}, {es_split.valid_end}]); "
        f"TEST [{test_start}, {test_end}] ({config.test_days} days)"
    )

    results: dict[str, WalkForwardResult] = {}
    for mode in MODES:
        print(f"\n[js2024] === mode={mode} ===")
        est = LGBMEstimator(
            feature_cols=feature_cols,
            target_col=TARGET_COLUMN,
            weight_col=WEIGHT_COLUMN,
            params=_lgbm_params(config),
            early_stopping_rounds=config.early_stopping_rounds,
            refit_decay=config.refit_decay,
        )
        print(f"[js2024] fitting initial model ({mode}) ...")
        est.fit(train_df, es_valid_df)
        res = walk_forward_evaluate(
            est,
            df,
            test_start,
            test_end,
            mode=mode,
            update_cadence=config.update_cadence,
            target_col=TARGET_COLUMN,
            weight_col=WEIGHT_COLUMN,
        )
        print(
            f"[js2024] {mode}: R²={res.score:.6f} "
            f"(updates={res.n_updates}, test_rows={res.n_test_rows:,})"
        )
        results[mode] = res

    return {
        "results": results,
        "train_lo": train_lo,
        "train_hi": train_hi,
        "es_split": summarize_date_split(df, es_split),
        "test_start": test_start,
        "test_end": test_end,
    }


def _summary_rows(bundle: dict[str, Any], config: LGBMConfig) -> list[dict[str, Any]]:
    rows = []
    for mode in MODES:
        res = bundle["results"][mode]
        rows.append(
            {
                "mode": res.mode,
                "test_start": res.test_start,
                "test_end": res.test_end,
                "test_days": res.n_test_days,
                "test_rows": res.n_test_rows,
                "update_method": config.update_method,
                "update_cadence": config.update_cadence,
                "n_updates": res.n_updates,
                "score": res.score,
                "prediction_mean": res.prediction_summary.get("mean"),
                "prediction_std": res.prediction_summary.get("std"),
                "prediction_min": res.prediction_summary.get("min"),
                "prediction_max": res.prediction_summary.get("max"),
            }
        )
    return rows


def render_docs(bundle: dict[str, Any], config: LGBMConfig, status: str) -> str:
    full = bundle["results"]["full"]
    inc = bundle["results"]["incremental"]
    delta = inc.score - full.score
    rel = (delta / full.score * 100) if full.score not in (0, None) else float("nan")

    L: list[str] = []
    L.append("# LGBM V0 — incremental (daily refit) vs full training")
    L.append("")
    L.append("## Purpose")
    L.append("")
    L.append(
        "Compare a statically-trained LightGBM (**full**) against the same model "
        "updated **incrementally** (LightGBM leaf-value refit, "
        f"`update_method={config.update_method}`, cadence "
        f"`{config.update_cadence}` day) as it walks a fixed final test block. This "
        "is the leakage-clean LightGBM analog of the *with vs without online "
        "learning* comparison in the evgeniavolkova writeup."
    )
    L.append("")
    L.append("## Protocol")
    L.append("")
    L.append(f"- data start: `date_id >= {config.start_date_id}` (her cutoff).")
    L.append(
        f"- **fixed test block:** last `{config.test_days}` date_ids = "
        f"[{bundle['test_start']}, {bundle['test_end']}] (shared by both modes)."
    )
    es = bundle["es_split"]
    L.append(
        f"- train region: [{bundle['train_lo']}, {bundle['train_hi']}]; "
        f"early-stopping holdout (from train tail, `valid_days={config.valid_days}`): "
        f"[{es['valid_start']}, {es['valid_end']}] — the test block is **never** used "
        "for early stopping."
    )
    L.append(
        "- raw features only (`feature_00..feature_78` + `symbol_id` + `time_id`); "
        "no feature engineering, GRU, auxiliary targets, ensemble, or clipping."
    )
    L.append("")
    L.append("## Results")
    L.append("")
    L.append(f"- **status:** {status}")
    L.append("")
    L.append("| mode | test_range | test_rows | n_updates | R² |")
    L.append("| --- | --- | ---: | ---: | ---: |")
    for mode in MODES:
        r = bundle["results"][mode]
        L.append(
            f"| {r.mode} | {r.test_start}–{r.test_end} | {r.n_test_rows:,} | "
            f"{r.n_updates} | {r.score:.6f} |"
        )
    L.append("")
    L.append(
        f"- **delta (incremental − full):** {delta:+.6f} "
        f"({rel:+.1f}% relative)."
    )
    L.append("")
    L.append("## Interpretation")
    L.append("")
    if delta > 0:
        L.append(
            "- Daily leaf-value refitting **improves** the score: adapting to the "
            "most recent days helps on this test block, consistent with the "
            "writeup's finding that online updating is the single biggest lever."
        )
    else:
        L.append(
            "- Daily leaf-value refitting does **not** improve (or hurts) the score "
            "here. LightGBM `refit` only re-weights existing leaves; the structural "
            "online gains reported for the GRU may need continued boosting / retrain "
            "or richer features. Recorded as a negative result."
        )
    L.append(
        "- The **full** number here is leakage-clean (early stopping uses a train-tail "
        "holdout, not the test block), so it may differ slightly from the recent700 "
        "baseline R²=0.010469, which used the last-200 block as its eval_set."
    )
    L.append("")
    L.append("## Next steps")
    L.append("")
    L.append("1. Try `continue` (init_model) and `retrain` (expanding) update methods.")
    L.append("2. Vary `update_cadence` (1 / 20 / 50) for the cost-vs-benefit curve.")
    L.append("3. Repo-style 2-fold CV + 200-day gap protocol.")
    L.append("4. Feature engineering parity, then the GRU estimator behind the same API.")
    L.append("")
    return "\n".join(L)


def _write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare full vs incremental (daily-refit) LightGBM on a fixed "
        "trailing test block."
    )
    parser.add_argument("--config", default="configs/lgbm_v0_incremental.yaml")
    parser.add_argument("--test-days", type=int, default=None)
    parser.add_argument("--update-method", default=None)
    parser.add_argument("--update-cadence", type=int, default=None)
    parser.add_argument("--out-dir", default="outputs/incremental_vs_full/lgbm_v0")
    parser.add_argument(
        "--docs-out", default="docs/experiments/lgbm_v0_incremental_vs_full.md"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--early-stopping-rounds", type=int, default=None)
    args = parser.parse_args(argv)

    try:
        config = load_lgbm_config(args.config)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"[js2024] ERROR: {exc}", file=sys.stderr)
        return 1

    # CLI overrides (re-validate via dataclasses.replace).
    import dataclasses

    overrides: dict[str, Any] = {}
    if args.test_days is not None:
        overrides["test_days"] = args.test_days
    if args.update_method is not None:
        overrides["update_method"] = args.update_method
    if args.update_cadence is not None:
        overrides["update_cadence"] = args.update_cadence
    if args.n_estimators is not None:
        overrides["n_estimators"] = args.n_estimators
    if args.early_stopping_rounds is not None:
        overrides["early_stopping_rounds"] = args.early_stopping_rounds
    if overrides:
        from .config import validate_lgbm_config

        config = validate_lgbm_config(dataclasses.replace(config, **overrides))

    feature_cols = get_v0_feature_columns(include_symbol=True, include_time=True)

    print(
        f"[js2024] incremental-vs-full | start={config.start_date_id} "
        f"test_days={config.test_days} method={config.update_method} "
        f"cadence={config.update_cadence}"
    )
    if args.dry_run:
        print("[js2024] Dry run: modes =", ", ".join(MODES), "(no training).")
        return 0

    out_dir = resolve_project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        train_path = resolve_project_path(config.train_path)
        validate_data_path(train_path)
        columns = get_default_columns(include_target=True, include_weight=True)
        print(f"[js2024] Loading shared train frame from {train_path} ...")
        df = load_train_data(
            train_path,
            columns=columns,
            start_date_id=config.start_date_id,
            end_date_id=config.end_date_id,
            collect=True,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[js2024] ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"[js2024] Shared frame: {df.height:,} rows.")

    try:
        bundle = run_modes(config, df, feature_cols=feature_cols)
    except ValueError as exc:
        print(f"[js2024] ERROR: {exc}", file=sys.stderr)
        return 1

    rows = _summary_rows(bundle, config)
    _write_summary_csv(rows, out_dir / "summary.csv")
    print(f"\n[js2024] Wrote {out_dir / 'summary.csv'}")

    docs_path = resolve_project_path(args.docs_out)
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(render_docs(bundle, config, "completed"), encoding="utf-8")
    print(f"[js2024] Wrote experiment doc -> {docs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
