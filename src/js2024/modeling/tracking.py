"""Optional Weights & Biases experiment tracking.

A thin, dependency-soft wrapper: every helper is a no-op unless tracking is
enabled *and* ``wandb`` is importable. Enable per-run with the config flag
``use_wandb: true`` (see :class:`js2024.modeling.config.GRUConfig`).

Auth is never allowed to stall a run. With no ``WANDB_API_KEY`` (and no
``wandb login``) the run falls back to ``offline`` mode: metrics are written
under ``wandb/`` and can be pushed later with ``wandb sync``. Set ``WANDB_MODE``
explicitly to override (e.g. ``online`` / ``disabled``).

The training loop logs through the module-level :func:`log`, which targets
whichever run is currently active (``wandb.run``). That keeps the estimator
decoupled from run setup, which lives in the walk-forward suite.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator


def _wandb() -> Any | None:
    try:
        import wandb
    except ImportError:
        return None
    return wandb


def _resolve_mode() -> str:
    """Pick a wandb mode that never blocks on an interactive login."""
    explicit = os.environ.get("WANDB_MODE")
    if explicit:
        return explicit
    if os.environ.get("WANDB_API_KEY"):
        return "online"
    netrc = os.path.expanduser("~/.netrc")
    try:
        with open(netrc, encoding="utf-8") as fh:
            if "api.wandb.ai" in fh.read():
                return "online"
    except OSError:
        pass
    return "offline"


@contextmanager
def run(
    enabled: bool,
    *,
    project: str,
    name: str,
    config: dict[str, Any],
    group: str | None = None,
    dir: str | None = None,
) -> Iterator[Any | None]:
    """Context manager yielding an active wandb run, or ``None`` when disabled.

    A failure to initialise (offline path included) is downgraded to a warning
    and a ``None`` run so tracking can never break training.
    """
    wb = _wandb() if enabled else None
    if wb is None:
        yield None
        return
    try:
        active = wb.init(
            project=project,
            name=name,
            group=group,
            config=config,
            mode=_resolve_mode(),
            reinit="finish_previous",
            dir=dir,
        )
    except Exception as exc:  # noqa: BLE001 - tracking must never break training
        print(f"[js2024] WARNING: wandb init failed ({exc}); continuing untracked.")
        yield None
        return
    try:
        yield active
    finally:
        wb.finish()


def log(metrics: dict[str, Any], *, step: int | None = None) -> None:
    """Log ``metrics`` to the active run; no-op if none is active."""
    wb = _wandb()
    if wb is None or wb.run is None:
        return
    wb.log(metrics, step=step)
