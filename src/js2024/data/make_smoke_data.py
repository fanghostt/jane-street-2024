"""CLI: carve a tiny smoke parquet out of the real train data for fast tests.

Usage
-----
    uv run js2024-make-smoke-data \\
        --train-path data/raw/train.parquet \\
        --out-path data/interim/train_smoke.parquet \\
        --start-date-id 1200 --end-date-id 1210

Reads only the requested ``date_id`` range and default column projection, then
writes a single small parquet. The output lives under ``data/interim`` (gitignored)
and is never committed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .data import get_date_id_range, get_default_columns, load_train_data


def make_smoke_data(
    train_path: str | Path,
    out_path: str | Path,
    start_date_id: int | None = None,
    end_date_id: int | None = None,
    columns: list[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Write a smoke parquet for ``[start_date_id, end_date_id]`` and return stats.

    Raises
    ------
    FileExistsError
        If ``out_path`` exists and ``force`` is False.
    FileNotFoundError
        If ``train_path`` does not exist.
    """
    out_path = Path(out_path)
    if out_path.exists() and not force:
        raise FileExistsError(
            f"Output '{out_path}' already exists; pass --force to overwrite."
        )

    if columns is None:
        columns = get_default_columns()

    df = load_train_data(
        train_path,
        columns=columns,
        start_date_id=start_date_id,
        end_date_id=end_date_id,
        collect=True,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)

    stats: dict[str, Any] = {
        "out_path": str(out_path),
        "rows": df.height,
        "columns": df.width,
        "date_min": None,
        "date_max": None,
    }
    if df.height > 0:
        stats["date_min"], stats["date_max"] = get_date_id_range(df)
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a small smoke parquet from real train data."
    )
    parser.add_argument("--train-path", default="data/raw/train.parquet")
    parser.add_argument("--out-path", default="data/interim/train_smoke.parquet")
    parser.add_argument("--start-date-id", type=int, default=None)
    parser.add_argument("--end-date-id", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    try:
        stats = make_smoke_data(
            train_path=args.train_path,
            out_path=args.out_path,
            start_date_id=args.start_date_id,
            end_date_id=args.end_date_id,
            force=args.force,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"[js2024] ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"[js2024] Wrote {stats['rows']:,} rows x {stats['columns']} cols "
        f"(date_id {stats['date_min']}..{stats['date_max']}) -> {stats['out_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
