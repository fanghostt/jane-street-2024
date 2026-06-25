"""Small helpers for timestamped experiment output directories."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def timestamp_slug(now: datetime | None = None) -> str:
    """Return a filesystem-friendly local timestamp."""
    dt = now or datetime.now().astimezone()
    return dt.strftime("%Y%m%d_%H%M%S")


def timestamped_run_dir(base_dir: str | Path, *, label: str | None = None) -> Path:
    """Return ``base_dir/<timestamp>[_label]`` without creating it."""
    slug = timestamp_slug()
    if label:
        slug = f"{slug}_{label}"
    return Path(base_dir) / slug


def dated_run_dir(base_dir: str | Path, project: str, *, now: datetime | None = None) -> Path:
    """Return ``base_dir/<YYYYMMDD>_<project>/<HHMMSS>`` without creating it.

    The dated project folder makes ``ls experiments/`` sortable by date so stale runs
    are easy to spot and prune; same-day runs of one project group under one folder.
    """
    dt = now or datetime.now().astimezone()
    return Path(base_dir) / f"{dt.strftime('%Y%m%d')}_{project}" / dt.strftime("%H%M%S")


def write_manifest(path: str | Path, **payload: Any) -> Path:
    """Write a compact JSON manifest for an experiment run."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {"created_at": datetime.now().astimezone().isoformat(timespec="seconds")}
    data.update(payload)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out
