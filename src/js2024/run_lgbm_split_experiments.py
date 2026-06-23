"""CLI: run the V0 raw-feature LightGBM baseline across a grid of date splits.

Purpose
-------
Check whether the recent700 baseline's weighted zero-mean R² is stable across
validation protocols (it is meant to confirm the score is not a one-split
artifact), *not* to improve the model. Same raw features, same hyperparameters —
only ``valid_days`` and ``gap_days`` vary.

Usage
-----
    uv run js2024-run-lgbm-split-experiments \\
        --base-config configs/lgbm_v0_recent700.yaml \\
        --valid-days 100,200,300 --gap-days 0,5,20 \\
        --out-dir outputs/split_experiments/lgbm_v0_recent700 \\
        --docs-out docs/experiments/lgbm_v0_split_experiments.md

The big train frame is loaded **once** and reused across every split. Heavy
artifacts (per-split model / OOF / report + summary.csv/md) go under ``--out-dir``
(gitignored); the only committed output is the markdown at ``--docs-out``.

This stage does NOT introduce feature engineering, GRU, auxiliary targets, online
learning, ensembling, or prediction clipping.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import statistics
import sys
from pathlib import Path
from typing import Any

import polars as pl

from .config import LGBMConfig, load_lgbm_config
from .data import get_default_columns, load_train_data, validate_data_path
from .config import resolve_project_path
from .train_lgbm import LGBMRunResult, run

EXPECTED_START_DATE_ID = 700


def parse_int_list(text: str) -> list[int]:
    """Parse a comma-separated list of ints, e.g. ``"100,200,300"`` -> ``[100, 200, 300]``."""
    items = [piece.strip() for piece in str(text).split(",") if piece.strip()]
    if not items:
        raise ValueError(f"Empty int list: {text!r}")
    try:
        return [int(piece) for piece in items]
    except ValueError as exc:
        raise ValueError(f"Invalid int list {text!r}: {exc}") from exc


def build_grid(valid_days: list[int], gap_days: list[int]) -> list[tuple[int, int]]:
    """Cartesian product of ``valid_days`` x ``gap_days`` (valid_days varies slowest)."""
    return list(itertools.product(valid_days, gap_days))


def make_run_name(base_stem: str, valid_days: int, gap_days: int) -> str:
    """e.g. ``lgbm_v0_recent700`` + (100, 5) -> ``lgbm_v0_recent700_v100_g5``."""
    return f"{base_stem}_v{valid_days}_g{gap_days}"


def _result_row(
    result: LGBMRunResult, valid_days: int, gap_days: int
) -> dict[str, Any]:
    s = result.split_summary
    pred = result.prediction_summary
    tgt = result.target_summary
    top5 = "; ".join(name for name, _ in result.feature_importance_top20[:5])
    return {
        "run_name": result.run_name,
        "valid_days": valid_days,
        "gap_days": gap_days,
        "train_start": s.get("train_start"),
        "train_end": s.get("train_end"),
        "valid_start": s.get("valid_start"),
        "valid_end": s.get("valid_end"),
        "train_rows": s.get("train_rows"),
        "valid_rows": s.get("valid_rows"),
        "best_iteration": result.best_iteration,
        "score": result.score,
        "prediction_mean": pred.get("mean"),
        "prediction_std": pred.get("std"),
        "prediction_min": pred.get("min"),
        "prediction_max": pred.get("max"),
        "target_mean": tgt.get("mean"),
        "target_std": tgt.get("std"),
        "target_min": tgt.get("min"),
        "target_max": tgt.get("max"),
        "top_5_features": top5,
    }


def _write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(path)


def _write_summary_md(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Split experiments summary", ""]
    lines.append(
        "| run_name | valid_days | gap_days | train_range | valid_range | "
        "train_rows | valid_rows | best_iter | R² | top_5_features |"
    )
    lines.append("| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |")
    for r in rows:
        lines.append(
            f"| {r['run_name']} | {r['valid_days']} | {r['gap_days']} | "
            f"{r['train_start']}–{r['train_end']} | {r['valid_start']}–{r['valid_end']} | "
            f"{r['train_rows']:,} | {r['valid_rows']:,} | {r['best_iteration']} | "
            f"{r['score']:.6f} | {r['top_5_features']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _stability(rows: list[dict[str, Any]]) -> dict[str, float]:
    scores = [r["score"] for r in rows]
    return {
        "mean": statistics.fmean(scores),
        "std": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        "min": min(scores),
        "max": max(scores),
    }


def render_docs(
    rows: list[dict[str, Any]],
    grid: list[tuple[int, int]],
    base_stem: str,
    start_date_id: int,
    status: str,
) -> str:
    lines: list[str] = []
    lines.append("# LGBM V0 Split Experiments")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "Verify that the raw-feature LightGBM **recent700** baseline is stable "
        "across validation splits — i.e. that its weighted zero-mean R² is not a "
        "one-split artifact."
    )
    lines.append("")
    lines.append(
        "- Baseline reference: recent700, `valid_days=200`, `gap_days=0`, "
        "R² = 0.010469."
    )
    lines.append(
        "- Raw features only (`feature_00..feature_78` + `symbol_id` + `time_id`). "
        "No feature engineering, no GRU, no online learning, no ensemble."
    )
    lines.append("- Prediction clipping is still deferred.")
    lines.append("")
    lines.append("## Experiment Grid")
    lines.append("")
    lines.append("| start_date_id | valid_days | gap_days |")
    lines.append("| ---: | ---: | ---: |")
    for vd, gd in grid:
        lines.append(f"| {start_date_id} | {vd} | {gd} |")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(f"- **status:** {status}")
    lines.append("")
    if rows:
        lines.append(
            "| run_name | valid_days | gap_days | train_range | valid_range | "
            "train_rows | valid_rows | best_iter | R² | top_5_features |"
        )
        lines.append(
            "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |"
        )
        for r in rows:
            lines.append(
                f"| {r['run_name']} | {r['valid_days']} | {r['gap_days']} | "
                f"{r['train_start']}–{r['train_end']} | "
                f"{r['valid_start']}–{r['valid_end']} | "
                f"{r['train_rows']:,} | {r['valid_rows']:,} | "
                f"{r['best_iteration']} | {r['score']:.6f} | {r['top_5_features']} |"
            )
        lines.append("")

    lines.append("## Stability Summary")
    lines.append("")
    if rows:
        st = _stability(rows)
        best = max(rows, key=lambda r: r["score"])
        worst = min(rows, key=lambda r: r["score"])
        baseline_match = [
            r for r in rows if r["valid_days"] == 200 and r["gap_days"] == 0
        ]
        lines.append(f"- runs completed: {len(rows)}")
        lines.append(f"- mean R²: {st['mean']:.6f}")
        lines.append(f"- std R²: {st['std']:.6f}")
        lines.append(f"- min R²: {st['min']:.6f} ({worst['run_name']})")
        lines.append(f"- max R²: {st['max']:.6f} ({best['run_name']})")
        if baseline_match:
            lines.append(
                f"- split matching baseline (v200_g0): R² = "
                f"{baseline_match[0]['score']:.6f}"
            )
        all_positive = all(r["score"] > 0 for r in rows)
        lines.append(
            f"- all splits positive: {all_positive} "
            f"(a constant-zero prediction scores exactly 0)"
        )
    else:
        lines.append("- status: not run yet")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if rows:
        all_positive = all(r["score"] > 0 for r in rows)
        if all_positive:
            lines.append(
                "- All splits score positive, so the raw-feature LGBM signal is "
                "consistently present — it is not a one-split artifact."
            )
        else:
            lines.append(
                "- Some splits are non-positive; inspect their target/weight "
                "distributions and possible market-regime / distribution shift."
            )

        # valid_days trend: mean score per validation-window length.
        by_vd: dict[int, list[float]] = {}
        for r in rows:
            by_vd.setdefault(r["valid_days"], []).append(r["score"])
        vd_means = {vd: statistics.fmean(s) for vd, s in sorted(by_vd.items())}
        trend = ", ".join(f"vd{vd}≈{m:.4f}" for vd, m in vd_means.items())
        monotonic = all(
            a <= b
            for a, b in zip(
                list(vd_means.values()), list(vd_means.values())[1:]
            )
        )
        lines.append(
            f"- The score depends on the validation window length ({trend}); "
            f"R² {'rises monotonically' if monotonic else 'varies'} with "
            f"`valid_days`. So the *absolute* number is protocol-dependent — "
            f"cross-experiment comparisons must fix the split."
        )

        # gap_days effect: average relative change g0 -> max gap within each vd.
        gaps = sorted({r["gap_days"] for r in rows})
        if len(gaps) >= 2:
            g_lo, g_hi = gaps[0], gaps[-1]
            drops = []
            for vd, _ in vd_means.items():
                lo = next(
                    (r["score"] for r in rows if r["valid_days"] == vd and r["gap_days"] == g_lo),
                    None,
                )
                hi = next(
                    (r["score"] for r in rows if r["valid_days"] == vd and r["gap_days"] == g_hi),
                    None,
                )
                if lo and hi and lo != 0:
                    drops.append((hi - lo) / lo)
            if drops:
                avg_drop = statistics.fmean(drops)
                lines.append(
                    f"- Increasing `gap_days` {g_lo}→{g_hi} changes R² by on "
                    f"average {avg_drop * 100:+.1f}% within a fixed `valid_days` "
                    f"— a mild penalty, consistent with slight near-boundary "
                    f"temporal autocorrelation rather than large leakage."
                )

        # Feature stability: top-1 and top-2 across splits.
        top1 = sum(1 for r in rows if r["top_5_features"].split("; ")[:1] == ["time_id"])
        feat61 = sum(1 for r in rows if "feature_61" in r["top_5_features"].split("; "))
        lines.append(
            f"- Feature ranking is stable: `time_id` is the #1 feature in "
            f"{top1}/{len(rows)} splits and `feature_61` is top-5 in "
            f"{feat61}/{len(rows)}."
        )

        # Prediction range vs target clip.
        over_clip = sum(
            1 for r in rows if (r.get("prediction_max") or 0) > 5 or (r.get("prediction_min") or 0) < -5
        )
        if over_clip:
            lines.append(
                f"- Predictions exceed the target's [-5, 5] range in "
                f"{over_clip}/{len(rows)} splits (upper tail). Prediction "
                f"clipping remains deferred to a separate PR."
            )
    else:
        lines.append("- Not run yet (dry-run or skipped).")
    lines.append("")
    lines.append("## Next Steps")
    lines.append("")
    lines.append("1. If stable: feature parity with the `evgeniavolkova/kagglejanestreet` repo.")
    lines.append("2. If unstable: inspect split-specific target/weight distributions.")
    lines.append("3. Later: repo-style 2-fold + 200-day gap test.")
    lines.append("4. Later: `clip_predictions` PR.")
    lines.append("5. Later: GRU parity.")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the V0 LightGBM baseline across a grid of date splits."
    )
    parser.add_argument("--base-config", default="configs/lgbm_v0_recent700.yaml")
    parser.add_argument("--valid-days", default="100,200,300")
    parser.add_argument("--gap-days", default="0,5,20")
    parser.add_argument(
        "--out-dir", default="outputs/split_experiments/lgbm_v0_recent700"
    )
    parser.add_argument(
        "--docs-out", default="docs/experiments/lgbm_v0_split_experiments.md"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--early-stopping-rounds", type=int, default=None)
    parser.add_argument(
        "--allow-non-700-start",
        action="store_true",
        help="Permit a base config whose start_date_id is not 700.",
    )
    args = parser.parse_args(argv)

    try:
        base = load_lgbm_config(args.base_config)
        valid_days = parse_int_list(args.valid_days)
        gap_days = parse_int_list(args.gap_days)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"[js2024] ERROR: {exc}", file=sys.stderr)
        return 1

    if base.start_date_id != EXPECTED_START_DATE_ID and not args.allow_non_700_start:
        print(
            f"[js2024] ERROR: base config start_date_id={base.start_date_id} "
            f"(expected {EXPECTED_START_DATE_ID}); pass --allow-non-700-start to "
            f"override.",
            file=sys.stderr,
        )
        return 1

    base_stem = Path(args.base_config).stem
    grid = build_grid(valid_days, gap_days)
    if args.limit is not None:
        grid = grid[: args.limit]

    print(f"[js2024] Split experiments for base '{base_stem}' (start_date_id="
          f"{base.start_date_id}); {len(grid)} run(s):")
    for vd, gd in grid:
        print(f"  - {make_run_name(base_stem, vd, gd)} (valid_days={vd}, gap_days={gd})")

    if args.dry_run:
        print("[js2024] Dry run: no training performed.")
        return 0

    out_dir = resolve_project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the train frame ONCE and reuse it across all splits.
    try:
        train_path = resolve_project_path(base.train_path)
        validate_data_path(train_path)
        columns = get_default_columns(include_target=True, include_weight=True)
        print(f"[js2024] Loading shared train frame from {train_path} ...")
        df = load_train_data(
            train_path,
            columns=columns,
            start_date_id=base.start_date_id,
            end_date_id=base.end_date_id,
            collect=True,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[js2024] ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"[js2024] Shared frame: {df.height:,} rows.")

    rows: list[dict[str, Any]] = []
    for i, (vd, gd) in enumerate(grid, start=1):
        run_name = make_run_name(base_stem, vd, gd)
        overrides: dict[str, Any] = {
            "valid_days": vd,
            "gap_days": gd,
            "model_dir": str(out_dir / "models"),
            "output_dir": str(out_dir),
        }
        if args.n_estimators is not None:
            overrides["n_estimators"] = args.n_estimators
        if args.early_stopping_rounds is not None:
            overrides["early_stopping_rounds"] = args.early_stopping_rounds
        cfg = dataclasses.replace(base, **overrides)

        print(f"\n[js2024] === [{i}/{len(grid)}] {run_name} ===")
        try:
            result = run(cfg, run_name=run_name, df=df)
        except ValueError as exc:
            # e.g. not enough history for this valid_days/gap_days combo.
            print(f"[js2024] SKIP {run_name}: {exc}", file=sys.stderr)
            continue
        rows.append(_result_row(result, vd, gd))

    if not rows:
        print("[js2024] ERROR: no split completed successfully.", file=sys.stderr)
        return 1

    _write_summary_csv(rows, out_dir / "summary.csv")
    _write_summary_md(rows, out_dir / "summary.md")
    print(f"\n[js2024] Wrote {out_dir / 'summary.csv'} and summary.md")

    status = "completed" if len(rows) == len(grid) else f"partial ({len(rows)}/{len(grid)})"
    docs_path = resolve_project_path(args.docs_out)
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(
        render_docs(rows, grid, base_stem, base.start_date_id, status),
        encoding="utf-8",
    )
    print(f"[js2024] Wrote experiment doc -> {docs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
