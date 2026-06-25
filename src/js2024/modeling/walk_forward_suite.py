"""Model-agnostic walk-forward suite shared by every config-driven experiment.

Given a :class:`~js2024.modeling.registry.ModelSpec`, a typed config and a loaded
frame, this carves the *same* fixed-test-block layout used by the LightGBM baseline
and runs each requested variant through :func:`walk_forward_evaluate`::

    [ start ......... es_holdout ][ TEST = last test_days ]
      \\________ train region ____/ \\____ scored block ____/

Variants map to walk-forward modes:

- ``full``        -> ``mode="full"`` (fit once, never update).
- ``incremental`` -> ``mode="incremental"`` at ``config.update_cadence``
  (NN fine-tuning or LightGBM online update on each revealed chunk).

The estimator is always fit on the train region only; the engine never trains it and
never feeds a test day's labels to ``update`` before that day is predicted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import polars as pl

from ..data.data import TARGET_COLUMN, WEIGHT_COLUMN, get_date_id_range
from .validation import build_holdout_split, filter_by_date_range, summarize_date_split
from .walk_forward import WalkForwardResult, walk_forward_evaluate

if TYPE_CHECKING:
    from .registry import ModelSpec

# variant name -> walk-forward mode.
VARIANT_MODES: dict[str, str] = {
    "full": "full",
    "incremental": "incremental",
}


@dataclass
class SuiteBundle:
    """Results + split metadata for one config-driven walk-forward run."""

    model: str
    results: dict[str, WalkForwardResult]
    cadence_used: dict[str, int]
    labels: list[str]
    train_lo: int
    train_hi: int
    es_split: dict[str, Any]
    test_start: int
    test_end: int


def validate_variants(variants: list[str]) -> list[str]:
    """Return ``variants`` if all are known, else raise ``ValueError``."""
    if not variants:
        raise ValueError(f"choose at least one variant from {sorted(VARIANT_MODES)}")
    bad = [v for v in variants if v not in VARIANT_MODES]
    if bad:
        raise ValueError(
            f"unknown variant(s) {bad}; choose from {sorted(VARIANT_MODES)}"
        )
    return variants


def run_walk_forward_suite(
    spec: "ModelSpec",
    config: Any,
    df: pl.DataFrame,
    *,
    variants: list[str],
    feature_cols: list[str],
) -> SuiteBundle:
    """Fit + walk-forward evaluate ``spec``'s estimator for each variant."""
    validate_variants(variants)

    min_date, max_date = get_date_id_range(df)
    test_split = build_holdout_split(
        min_date_id=min_date, max_date_id=max_date, valid_days=config.test_days, gap_days=0
    )
    test_start, test_end = test_split.valid_start, test_split.valid_end
    train_lo, train_hi = min_date, test_start - 1

    es_split = build_holdout_split(
        min_date_id=train_lo, max_date_id=train_hi,
        valid_days=config.valid_days, gap_days=config.gap_days,
    )
    train_df = filter_by_date_range(df, "date_id", es_split.train_start, es_split.train_end)
    valid_df = filter_by_date_range(df, "date_id", es_split.valid_start, es_split.valid_end)

    print(
        f"[js2024] train region [{train_lo}, {train_hi}] "
        f"(fit [{es_split.train_start}, {es_split.train_end}], "
        f"es-holdout [{es_split.valid_start}, {es_split.valid_end}]); "
        f"TEST [{test_start}, {test_end}] ({config.test_days} days)"
    )

    results: dict[str, WalkForwardResult] = {}
    cadence_used: dict[str, int] = {}
    labels: list[str] = []

    for variant in variants:
        mode = VARIANT_MODES[variant]
        label = f"{spec.name}_{variant}"
        cadence = 0 if mode == "full" else config.update_cadence
        print(f"\n[js2024] === {label} (cadence={cadence}) ===")
        est = spec.make_estimator(config, feature_cols)
        est.fit(train_df, valid_df)
        results[label] = walk_forward_evaluate(
            est, df, test_start, test_end,
            mode=mode, update_cadence=max(cadence, 1),
            target_col=TARGET_COLUMN, weight_col=WEIGHT_COLUMN,
        )
        cadence_used[label] = cadence
        labels.append(label)
        print(
            f"[js2024] {label}: R²={results[label].score:.6f} "
            f"(updates={results[label].n_updates})"
        )

    return SuiteBundle(
        model=spec.name,
        results=results,
        cadence_used=cadence_used,
        labels=labels,
        train_lo=train_lo,
        train_hi=train_hi,
        es_split=summarize_date_split(df, es_split),
        test_start=test_start,
        test_end=test_end,
    )


def summary_rows(bundle: SuiteBundle) -> list[dict[str, Any]]:
    """Flatten ``bundle`` into rows for ``summary.csv``."""
    rows = []
    for label in bundle.labels:
        res = bundle.results[label]
        rows.append(
            {
                "model": bundle.model,
                "variant": label,
                "test_start": res.test_start,
                "test_end": res.test_end,
                "test_days": res.n_test_days,
                "test_rows": res.n_test_rows,
                "cadence": bundle.cadence_used[label],
                "n_updates": res.n_updates,
                "score": res.score,
                "prediction_mean": res.prediction_summary.get("mean"),
                "prediction_std": res.prediction_summary.get("std"),
                "prediction_min": res.prediction_summary.get("min"),
                "prediction_max": res.prediction_summary.get("max"),
            }
        )
    return rows


def render_report(bundle: SuiteBundle, spec: "ModelSpec", config: Any, status: str) -> str:
    """Render a generic markdown report (protocol + per-variant R² table)."""
    L: list[str] = []
    L.append(f"# {spec.title} — walk-forward")
    L.append("")
    L.append("## Protocol")
    L.append("")
    L.append(f"- data start: `date_id >= {config.start_date_id}`.")
    L.append(
        f"- **fixed test block:** last `{config.test_days}` date_ids = "
        f"[{bundle.test_start}, {bundle.test_end}] (shared by every variant)."
    )
    es = bundle.es_split
    L.append(
        f"- train region: [{bundle.train_lo}, {bundle.train_hi}]; early-stopping "
        f"holdout (train tail, `valid_days={config.valid_days}`): "
        f"[{es['valid_start']}, {es['valid_end']}] — the test block is **never** used "
        "for early stopping."
    )
    for line in spec.describe(config):
        L.append(f"- {line}")
    L.append("")
    L.append("## Results")
    L.append("")
    L.append(f"- **status:** {status}")
    L.append("")
    L.append("| variant | cadence | n_updates | R² | pred[min,max] |")
    L.append("| --- | ---: | ---: | ---: | --- |")
    for label in bundle.labels:
        r = bundle.results[label]
        cad = bundle.cadence_used[label]
        pmin = r.prediction_summary.get("min")
        pmax = r.prediction_summary.get("max")
        L.append(
            f"| {label} | {cad if cad else '–'} | {r.n_updates} | {r.score:.6f} | "
            f"[{pmin:.2f}, {pmax:.2f}] |"
        )
    L.append("")
    best = max(bundle.labels, key=lambda lb: bundle.results[lb].score)
    L.append(f"- **best variant:** `{best}` (R²={bundle.results[best].score:.6f}).")
    L.append("")
    return "\n".join(L)
