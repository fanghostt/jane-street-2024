"""Markdown report writer for the LightGBM V0 baseline run."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


def _fmt_summary(summary: Mapping[str, float]) -> str:
    keys = ["mean", "std", "min", "max"]
    return " | ".join(f"{k}={summary.get(k, float('nan')):.6g}" for k in keys)


def write_lgbm_report(
    path: str | Path,
    config: Any,
    split_summary: Mapping[str, Any],
    score: float,
    feature_cols: Sequence[str],
    prediction_summary: Mapping[str, float],
    target_summary: Mapping[str, float],
    feature_importance: Sequence[tuple[str, float]],
) -> Path:
    """Write a markdown report describing a baseline run; return the path written."""
    cfg = asdict(config) if is_dataclass(config) and not isinstance(config, type) else dict(config)

    lines: list[str] = []
    lines.append("# LightGBM V0 Baseline Report")
    lines.append("")
    lines.append("## Run summary")
    lines.append("")
    lines.append(f"- **Model name:** lgbm_v0")
    lines.append(f"- **Config train_path:** `{cfg.get('train_path')}`")
    lines.append(
        f"- **Train date range:** {split_summary.get('train_start')} "
        f"– {split_summary.get('train_end')}"
    )
    lines.append(
        f"- **Valid date range:** {split_summary.get('valid_start')} "
        f"– {split_summary.get('valid_end')}"
    )
    lines.append(f"- **Train rows:** {split_summary.get('train_rows'):,}")
    lines.append(f"- **Valid rows:** {split_summary.get('valid_rows'):,}")
    lines.append(f"- **Train days:** {split_summary.get('train_days')}")
    lines.append(f"- **Valid days:** {split_summary.get('valid_days')}")
    lines.append(f"- **Feature count:** {len(feature_cols)}")
    lines.append("")
    lines.append("## Score")
    lines.append("")
    lines.append(f"- **Weighted zero-mean R²:** `{score:.6f}`")
    lines.append("")
    lines.append("## Prediction vs. target distribution")
    lines.append("")
    lines.append(f"- **Predictions:** {_fmt_summary(prediction_summary)}")
    lines.append(f"- **Target:** {_fmt_summary(target_summary)}")
    lines.append("")
    lines.append("## Config")
    lines.append("")
    lines.append("```yaml")
    for k, v in cfg.items():
        lines.append(f"{k}: {v}")
    lines.append("```")
    lines.append("")
    lines.append("## Top 30 feature importance")
    lines.append("")
    lines.append("| rank | feature | importance |")
    lines.append("| ---: | --- | ---: |")
    for rank, (name, imp) in enumerate(feature_importance[:30], start=1):
        lines.append(f"| {rank} | {name} | {imp:.6g} |")
    lines.append("")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
