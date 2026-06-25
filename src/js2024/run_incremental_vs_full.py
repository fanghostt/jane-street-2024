r"""CLI: compare *full* vs *incremental* LightGBM on a fixed trailing test block.

All variants share one training region and are scored on the **same** trailing
``test_days`` block (default the last 200 ``date_id``s), so their weighted zero-mean
R² is directly comparable. This is the LightGBM, leakage-clean analog of the "with
vs without online learning" comparison in ``evgeniavolkova/kagglejanestreet``.

Layout (inclusive ``date_id`` boundaries)::

    [ start ......... es_holdout ][ TEST = last test_days ]
      \________ train region ____/ \____ scored block ____/
                (es_holdout = last `valid_days` of train, for early stopping)

Variants:

- ``full``     : fit once, predict the whole test block; never update.
- ``refit``    : daily ``Booster.refit`` of leaf values on each revealed day.
- ``continue`` : daily continued boosting (add ``continue_rounds`` trees).
- ``retrain``  : expanding retrain from scratch on all data so far (coarse cadence).

The big train frame is loaded **once** and reused by every variant. Heavy artifacts
go under ``--out-dir`` (gitignored); the only committed output is the markdown doc.

This stage introduces NO feature engineering, GRU, auxiliary targets, ensembling, or
prediction clipping.

Usage
-----
    uv run js2024-run-incremental-vs-full \\
        --config configs/lgbm_v0_incremental.yaml \\
        --methods refit,continue,retrain \\
        --out-dir experiments/incremental_vs_full/lgbm_v0 \\
        --docs-out experiments/incremental_vs_full/lgbm_v0/report.md
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Any

import polars as pl

from .config import LGBMConfig, load_lgbm_config, resolve_project_path, validate_lgbm_config
from .data import (
    TARGET_COLUMN,
    WEIGHT_COLUMN,
    get_default_columns,
    get_date_id_range,
    load_train_data,
    validate_data_path,
)
from .estimators import UPDATE_METHODS, LGBMEstimator
from .features import get_v0_feature_columns
from .validation import build_holdout_split, filter_by_date_range, summarize_date_split
from .walk_forward import WalkForwardResult, walk_forward_evaluate

# Per-method default cadence: refit/continue are cheap (daily); retrain is a full
# fit per step, so it defaults to a coarse cadence.
DEFAULT_METHOD_CADENCE = {"refit": 1, "continue": 1, "retrain": 50}


def _lgbm_params(config: LGBMConfig) -> dict[str, Any]:
    return {
        "n_estimators": config.n_estimators,
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "random_state": config.random_state,
    }


def _make_estimator(config: LGBMConfig, feature_cols: list[str], method: str) -> LGBMEstimator:
    return LGBMEstimator(
        feature_cols=feature_cols,
        target_col=TARGET_COLUMN,
        weight_col=WEIGHT_COLUMN,
        params=_lgbm_params(config),
        early_stopping_rounds=config.early_stopping_rounds,
        update_method=method,
        refit_decay=config.refit_decay,
        continue_rounds=config.continue_rounds,
    )


def run_suite(
    config: LGBMConfig,
    df: pl.DataFrame,
    *,
    feature_cols: list[str],
    specs: list[tuple[str, str, int]],
) -> dict[str, Any]:
    """Fit + walk-forward evaluate ``full`` plus each spec on a shared frame.

    ``specs`` is a list of ``(label, method, cadence)`` — letting the caller run
    distinct labels (e.g. a retrain cadence sweep ``retrain_c25``/``retrain_c50``).
    """
    min_date, max_date = get_date_id_range(df)

    # Fixed trailing test block of `test_days`; train region is everything before it.
    test_split = build_holdout_split(
        min_date_id=min_date, max_date_id=max_date, valid_days=config.test_days, gap_days=0
    )
    test_start, test_end = test_split.valid_start, test_split.valid_end
    train_lo, train_hi = min_date, test_start - 1

    # Early-stopping holdout = last `valid_days` of the TRAIN region (never the test).
    es_split = build_holdout_split(
        min_date_id=train_lo, max_date_id=train_hi, valid_days=config.valid_days, gap_days=config.gap_days
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
    cadence_used: dict[str, int] = {"full": 0}

    # full: estimator method is irrelevant (no updates happen).
    print("\n[js2024] === full ===")
    est_full = _make_estimator(config, feature_cols, "refit")
    est_full.fit(train_df, es_valid_df)
    results["full"] = walk_forward_evaluate(
        est_full, df, test_start, test_end, mode="full",
        target_col=TARGET_COLUMN, weight_col=WEIGHT_COLUMN,
    )
    print(f"[js2024] full: R²={results['full'].score:.6f}")

    method_of: dict[str, str] = {"full": "full"}
    for label, method, cadence in specs:
        print(f"\n[js2024] === {label} ({method}, cadence={cadence}) ===")
        est = _make_estimator(config, feature_cols, method)
        est.fit(train_df, es_valid_df)
        res = walk_forward_evaluate(
            est, df, test_start, test_end, mode="incremental", update_cadence=cadence,
            target_col=TARGET_COLUMN, weight_col=WEIGHT_COLUMN,
        )
        print(f"[js2024] {label}: R²={res.score:.6f} (updates={res.n_updates})")
        results[label] = res
        cadence_used[label] = cadence
        method_of[label] = method

    return {
        "results": results,
        "cadence_used": cadence_used,
        "method_of": method_of,
        "train_lo": train_lo,
        "train_hi": train_hi,
        "es_split": summarize_date_split(df, es_split),
        "test_start": test_start,
        "test_end": test_end,
        "labels": ["full", *[s[0] for s in specs]],
    }


def _summary_rows(bundle: dict[str, Any], config: LGBMConfig) -> list[dict[str, Any]]:
    rows = []
    for label in bundle["labels"]:
        res = bundle["results"][label]
        rows.append(
            {
                "variant": label,
                "test_start": res.test_start,
                "test_end": res.test_end,
                "test_days": res.n_test_days,
                "test_rows": res.n_test_rows,
                "cadence": bundle["cadence_used"][label],
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
    labels = bundle["labels"]

    L: list[str] = []
    L.append("# LGBM V0 — incremental vs full training")
    L.append("")
    L.append("## Purpose")
    L.append("")
    L.append(
        "Compare a statically-trained LightGBM (**full**) against three incremental "
        "update strategies as the model walks a fixed final test block: **refit** "
        "(leaf-value refit), **continue** (continued boosting), and **retrain** "
        "(expanding retrain from scratch). Leakage-clean LightGBM analog of the *with "
        "vs without online learning* comparison in the evgeniavolkova writeup."
    )
    L.append("")
    L.append("## Protocol")
    L.append("")
    L.append(f"- data start: `date_id >= {config.start_date_id}` (her cutoff).")
    L.append(
        f"- **fixed test block:** last `{config.test_days}` date_ids = "
        f"[{bundle['test_start']}, {bundle['test_end']}] (shared by every variant)."
    )
    es = bundle["es_split"]
    L.append(
        f"- train region: [{bundle['train_lo']}, {bundle['train_hi']}]; early-stopping "
        f"holdout (train tail, `valid_days={config.valid_days}`): "
        f"[{es['valid_start']}, {es['valid_end']}] — the test block is **never** used "
        "for early stopping."
    )
    L.append(
        f"- update params: `refit_decay={config.refit_decay}`, "
        f"`continue_rounds={config.continue_rounds}`."
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
    L.append("| variant | cadence | n_updates | R² | Δ vs full | pred[min,max] |")
    L.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for label in labels:
        r = bundle["results"][label]
        cad = bundle["cadence_used"][label]
        delta = "" if label == "full" else f"{r.score - full.score:+.6f}"
        pmin = r.prediction_summary.get("min")
        pmax = r.prediction_summary.get("max")
        L.append(
            f"| {label} | {cad if cad else '–'} | {r.n_updates} | {r.score:.6f} | "
            f"{delta} | [{pmin:.2f}, {pmax:.2f}] |"
        )
    L.append("")
    best = max(labels, key=lambda lb: bundle["results"][lb].score)
    L.append(f"- **best variant:** `{best}` (R²={bundle['results'][best].score:.6f}).")
    L.append("")
    method_of = bundle["method_of"]
    labels_for = lambda m: [lb for lb in labels if method_of.get(lb) == m]

    L.append("## Interpretation")
    L.append("")
    L.append(
        "- **full** is leakage-clean (early stopping uses a train-tail holdout, not the "
        "test block), so it may sit slightly below the recent700 baseline R²=0.010469, "
        "which used the last-200 block as its eval_set."
    )
    for lb in labels_for("refit"):
        if bundle["results"][lb].score < full.score:
            L.append(
                f"- **{lb}** degrades (R²={bundle['results'][lb].score:.6f}): each daily "
                "leaf-refit re-weights *all* leaves toward one noisy day; cumulative "
                "refits drag the fit off. `Booster.refit` is the wrong online analog for "
                "a tree model."
            )
    for lb in labels_for("continue"):
        cs = bundle["results"][lb].score
        if cs < 0:
            L.append(
                f"- **{lb}** blows up (R²={cs:.6f}): adding trees per day on a single "
                "day's rows compounds into a heavily over-fit ensemble (prediction range "
                "explodes past the target's [-5, 5])."
            )
    retrain_labels = labels_for("retrain")
    if retrain_labels:
        best_rt = max(retrain_labels, key=lambda lb: bundle["results"][lb].score)
        verb = "is the strongest" if best in retrain_labels else "did not win"
        L.append(
            f"- **retrain** {verb}: expanding retrain reincorporates the early-stopping "
            "holdout and every revealed day, so it trains on more data than the "
            "static/online variants — the closest analog to her expanding CV."
        )
        if len(retrain_labels) > 1:
            curve = ", ".join(
                f"{lb}={bundle['results'][lb].score:.6f}({bundle['cadence_used'][lb]}d/"
                f"{bundle['results'][lb].n_updates}u)"
                for lb in retrain_labels
            )
            L.append(
                f"- retrain cadence sweep (R²/cadence/updates): {curve}. Best: "
                f"`{best_rt}`. Finer cadence ⇒ more retrains (cost) for the marginal R²."
            )
    L.append("")
    L.append("## Next steps")
    L.append("")
    L.append("1. Repo-style 2-fold CV + 200-day gap protocol.")
    L.append("2. Feature engineering parity, then the GRU estimator behind the same API.")
    L.append("")
    return "\n".join(L)


def _write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(path)


def _parse_methods(text: str) -> list[str]:
    methods = [m.strip() for m in str(text).split(",") if m.strip()]
    bad = [m for m in methods if m not in UPDATE_METHODS]
    if bad:
        raise ValueError(f"unknown update method(s) {bad}; choose from {list(UPDATE_METHODS)}")
    return methods


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare full vs incremental (refit/continue/retrain) LightGBM on "
        "a fixed trailing test block."
    )
    parser.add_argument("--config", default="configs/lgbm_v0_incremental.yaml")
    parser.add_argument(
        "--methods", default="refit,continue,retrain",
        help="Comma list from {refit,continue,retrain}.",
    )
    parser.add_argument("--test-days", type=int, default=None)
    parser.add_argument(
        "--cadence", type=int, default=None,
        help="Override cadence for ALL selected methods (else per-method defaults).",
    )
    parser.add_argument("--retrain-cadence", type=int, default=None)
    parser.add_argument(
        "--retrain-cadences", default=None,
        help="Sweep mode: comma list of retrain cadences, e.g. '100,50,25'. When set, "
        "runs full + retrain at each cadence (ignores --methods).",
    )
    parser.add_argument("--out-dir", default="experiments/incremental_vs_full/lgbm_v0")
    parser.add_argument("--docs-out", default="experiments/incremental_vs_full/lgbm_v0/report.md")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--early-stopping-rounds", type=int, default=None)
    args = parser.parse_args(argv)

    try:
        config = load_lgbm_config(args.config)
        if args.retrain_cadences:
            # Sweep mode: full + retrain at each listed cadence.
            cads = [int(c) for c in args.retrain_cadences.split(",") if c.strip()]
            if not cads:
                raise ValueError("--retrain-cadences was empty")
            specs = [(f"retrain_c{c}", "retrain", c) for c in cads]
        else:
            methods = _parse_methods(args.methods)
            cadences = {m: DEFAULT_METHOD_CADENCE[m] for m in methods}
            if args.cadence is not None:
                cadences = {m: args.cadence for m in methods}
            if args.retrain_cadence is not None and "retrain" in cadences:
                cadences["retrain"] = args.retrain_cadence
            specs = [(m, m, cadences[m]) for m in methods]
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"[js2024] ERROR: {exc}", file=sys.stderr)
        return 1

    overrides: dict[str, Any] = {}
    if args.test_days is not None:
        overrides["test_days"] = args.test_days
    if args.n_estimators is not None:
        overrides["n_estimators"] = args.n_estimators
    if args.early_stopping_rounds is not None:
        overrides["early_stopping_rounds"] = args.early_stopping_rounds
    if overrides:
        config = validate_lgbm_config(dataclasses.replace(config, **overrides))

    feature_cols = get_v0_feature_columns(include_symbol=True, include_time=True)

    print(
        f"[js2024] incremental-vs-full | start={config.start_date_id} "
        f"test_days={config.test_days} | variants: full, "
        + ", ".join(f"{lb}(cad={cad})" for lb, _, cad in specs)
    )
    if args.dry_run:
        print("[js2024] Dry run: no training performed.")
        return 0

    out_dir = resolve_project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        train_path = resolve_project_path(config.train_path)
        validate_data_path(train_path)
        columns = get_default_columns(include_target=True, include_weight=True)
        print(f"[js2024] Loading shared train frame from {train_path} ...")
        df = load_train_data(
            train_path, columns=columns, start_date_id=config.start_date_id,
            end_date_id=config.end_date_id, collect=True,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[js2024] ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"[js2024] Shared frame: {df.height:,} rows.")

    try:
        bundle = run_suite(config, df, feature_cols=feature_cols, specs=specs)
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
