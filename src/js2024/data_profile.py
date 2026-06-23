"""CLI: write a markdown profile of the train data (lazy, optional date range).

Usage
-----
    uv run js2024-data-profile \\
        --train-path data/raw/train.parquet \\
        --out outputs/reports/data_profile.md \\
        --start-date-id 1200 --end-date-id 1698

Everything is computed with a single (plus one for missing-ratios) lazy
aggregation so the full dataset never lands in memory. If the full scan is too
slow, restrict it with ``--start-date-id`` / ``--end-date-id``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import polars as pl

from .data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    WEIGHT_COLUMN,
    scan_train_data,
)

_TOP_MISSING = 30


def _summary_exprs(col: str, prefix: str) -> list[pl.Expr]:
    return [
        pl.col(col).mean().alias(f"{prefix}_mean"),
        pl.col(col).std().alias(f"{prefix}_std"),
        pl.col(col).min().alias(f"{prefix}_min"),
        pl.col(col).max().alias(f"{prefix}_max"),
    ]


def profile_train_data(
    train_path: str | Path,
    start_date_id: int | None = None,
    end_date_id: int | None = None,
) -> dict[str, Any]:
    """Compute profile statistics for the train data; return a dict of results."""
    lf = scan_train_data(
        train_path, start_date_id=start_date_id, end_date_id=end_date_id
    )
    available = lf.collect_schema().names()
    available_set = set(available)

    agg: list[pl.Expr] = [pl.len().alias("row_count")]
    if "date_id" in available_set:
        agg += [
            pl.col("date_id").min().alias("date_min"),
            pl.col("date_id").max().alias("date_max"),
        ]
    if "symbol_id" in available_set:
        agg.append(pl.col("symbol_id").n_unique().alias("symbol_n_unique"))
    if "time_id" in available_set:
        agg.append(pl.col("time_id").n_unique().alias("time_n_unique"))
    if TARGET_COLUMN in available_set:
        agg += _summary_exprs(TARGET_COLUMN, "target")
    if WEIGHT_COLUMN in available_set:
        agg += _summary_exprs(WEIGHT_COLUMN, "weight")

    stats_df = lf.select(agg).collect()
    stats = stats_df.row(0, named=True)

    feature_cols = [c for c in FEATURE_COLUMNS if c in available_set]
    missing_ratio: list[tuple[str, float]] = []
    if feature_cols:
        miss_df = lf.select(
            [pl.col(c).is_null().mean().alias(c) for c in feature_cols]
        ).collect()
        miss_row = miss_df.row(0, named=True)
        missing_ratio = sorted(
            ((c, float(miss_row[c] if miss_row[c] is not None else 0.0)) for c in feature_cols),
            key=lambda kv: kv[1],
            reverse=True,
        )[:_TOP_MISSING]

    return {
        "train_path": str(train_path),
        "stats": stats,
        "missing_ratio": missing_ratio,
        "n_feature_columns": len(feature_cols),
    }


def render_profile(
    profile: dict[str, Any],
    train_path: str | Path,
    scan_path: str,
    start_date_id: int | None,
    end_date_id: int | None,
) -> str:
    s = profile["stats"]

    def g(key: str) -> Any:
        return s.get(key)

    lines: list[str] = []
    lines.append("# Train data profile")
    lines.append("")
    lines.append(f"- **train path:** `{train_path}`")
    lines.append(f"- **scan path:** `{scan_path}`")
    lines.append(f"- **requested date range:** {start_date_id} … {end_date_id}")
    lines.append(f"- **date_id min/max:** {g('date_min')} … {g('date_max')}")
    lines.append(f"- **row count:** {g('row_count')}")
    lines.append(f"- **symbol_id n_unique:** {g('symbol_n_unique')}")
    lines.append(f"- **time_id n_unique:** {g('time_n_unique')}")
    lines.append(f"- **feature columns profiled:** {profile['n_feature_columns']}")
    lines.append("")
    lines.append("## Target (`responder_6`)")
    lines.append("")
    lines.append(
        f"- mean={g('target_mean')} | std={g('target_std')} | "
        f"min={g('target_min')} | max={g('target_max')}"
    )
    lines.append("")
    lines.append("## Weight (`weight`)")
    lines.append("")
    lines.append(
        f"- mean={g('weight_mean')} | std={g('weight_std')} | "
        f"min={g('weight_min')} | max={g('weight_max')}"
    )
    lines.append("")
    lines.append(f"## Top {_TOP_MISSING} features by missing ratio")
    lines.append("")
    lines.append("| feature | missing_ratio |")
    lines.append("| --- | --- |")
    for name, ratio in profile["missing_ratio"]:
        lines.append(f"| `{name}` | {ratio:.6f} |")
    lines.append("")
    return "\n".join(lines)


def write_profile(
    train_path: str | Path,
    out_path: str | Path,
    start_date_id: int | None = None,
    end_date_id: int | None = None,
) -> Path:
    """Compute and write the markdown profile; return the output path."""
    from .data import resolve_parquet_scan_path

    scan_path = str(resolve_parquet_scan_path(train_path))
    profile = profile_train_data(
        train_path, start_date_id=start_date_id, end_date_id=end_date_id
    )
    md = render_profile(
        profile, train_path, scan_path, start_date_id, end_date_id
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write a markdown profile of the train data."
    )
    parser.add_argument("--train-path", default="data/raw/train.parquet")
    parser.add_argument("--out", default="outputs/reports/data_profile.md")
    parser.add_argument("--start-date-id", type=int, default=None)
    parser.add_argument("--end-date-id", type=int, default=None)
    args = parser.parse_args(argv)

    try:
        out_path = write_profile(
            train_path=args.train_path,
            out_path=args.out,
            start_date_id=args.start_date_id,
            end_date_id=args.end_date_id,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[js2024] ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"[js2024] Wrote data profile -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
