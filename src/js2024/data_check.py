"""CLI: verify a local ``data/raw`` directory satisfies the raw-data contract.

Usage
-----
    uv run js2024-data-check --raw-dir data/raw

Prints a markdown-ish summary and exits non-zero when the contract is not
satisfied (missing required files or missing train columns), so it composes
cleanly inside other scripts (e.g. after a download).
"""

from __future__ import annotations

import argparse
from typing import Any

from .data_contract import check_raw_data_contract, contract_ok


def _fmt_size(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(n)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{n} B"


def render_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Raw data contract check")
    lines.append("")
    lines.append(f"- **raw_dir:** `{report['raw_dir']}`")
    lines.append(f"- **exists:** {report['exists']}")
    lines.append("")
    lines.append("## Required files")
    lines.append("")
    lines.append("| file | exists | type | size |")
    lines.append("| --- | --- | --- | --- |")
    for f in report["required_files"]:
        size = _fmt_size(f["size_bytes"]) if f["exists"] else "-"
        lines.append(
            f"| `{f['name']}` | {f['exists']} | {f['type'] or '-'} | {size} |"
        )
    lines.append("")
    lines.append("## Train schema")
    lines.append("")
    lines.append(f"- **scan path:** `{report['train_scan_path']}`")
    lines.append(f"- **column count:** {report['train_columns_count']}")
    lines.append(f"- **date_id range:** {report['date_min']} … {report['date_max']}")
    lines.append(f"- **sample rows checked:** {report['sample_rows_checked']}")
    missing = report["missing_train_columns"]
    if missing is None:
        # train.parquet absent / unscannable -> schema was never checked.
        lines.append("- **missing columns:** not checked")
    elif missing:
        lines.append(f"- **MISSING columns ({len(missing)}):** {missing}")
    else:
        lines.append("- **missing columns:** none")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check that a local raw data directory satisfies the contract."
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Directory holding the raw Kaggle download (default: data/raw).",
    )
    args = parser.parse_args(argv)

    report = check_raw_data_contract(args.raw_dir)
    print(render_report(report))

    ok = contract_ok(report)
    if ok:
        print("\n[js2024] Contract OK.")
        return 0

    print("\n[js2024] ERROR: raw data contract NOT satisfied (see above).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
