r"""Config-driven walk-forward experiment runner.

A single entrypoint for every walk-forward model. The YAML config declares which
model to run and which variants to evaluate; the model registry
(:mod:`js2024.modeling.registry`) maps the ``model:`` name to its config loader,
estimator factory, feature columns and frame loader, and the shared suite
(:mod:`js2024.modeling.walk_forward_suite`) carves the fixed test block, fits and
evaluates each variant, and writes ``summary.csv`` / ``manifest.json`` / a markdown
report. Adding a model means adding a :class:`~js2024.modeling.registry.ModelSpec`
and a config file — not a new runner script.

Config keys consumed here (alongside the model's own hyperparameters):

    model: gru                 # selects the registry spec (required)
    variants: [full, incremental] # walk-forward variants to run (optional)

Usage
-----
    uv run js2024-run-experiment --config configs/gru_v0.yaml
    uv run js2024-run-experiment --config configs/gru_v0.yaml --variants full
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from typing import Any

import polars as pl

from ..modeling.config import load_yaml_config, resolve_project_path
from ..modeling.experiments import dated_run_dir, write_manifest
from ..modeling.registry import get_model_spec
from ..modeling.walk_forward_suite import (
    render_report,
    run_walk_forward_suite,
    summary_rows,
    validate_variants,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Config-driven walk-forward experiment runner."
    )
    parser.add_argument("--config", required=True, help="Path to the experiment YAML.")
    parser.add_argument(
        "--variants",
        default=None,
        help="Comma list overriding the config's `variants` (e.g. full,incremental).",
    )
    parser.add_argument("--test-days", type=int, default=None, help="Override test_days.")
    parser.add_argument(
        "--update-cadence", type=int, default=None, help="Override update_cadence."
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--docs-out", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        raw = load_yaml_config(args.config)
        model = raw.get("model")
        if not model:
            raise ValueError("config must set `model:` (e.g. model: gru)")
        spec = get_model_spec(model)
        config = spec.load_config(args.config)

        overrides: dict[str, Any] = {}
        if args.test_days is not None:
            overrides["test_days"] = args.test_days
        if args.update_cadence is not None:
            overrides["update_cadence"] = args.update_cadence
        if overrides:
            config = dataclasses.replace(config, **overrides)

        if args.variants is not None:
            variants = [v.strip() for v in args.variants.split(",") if v.strip()]
        else:
            variants = list(raw.get("variants", ["full", "incremental"]))
        validate_variants(variants)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"[js2024] ERROR: {exc}", file=sys.stderr)
        return 1

    feature_cols = spec.feature_columns(config)
    print(
        f"[js2024] run-experiment | model={spec.name} start={config.start_date_id} "
        f"test_days={config.test_days} cadence={config.update_cadence} | "
        f"variants: " + ", ".join(f"{spec.name}_{v}" for v in variants)
    )
    if args.dry_run:
        print("[js2024] Dry run: no training performed.")
        return 0

    out_dir = resolve_project_path(
        args.out_dir or dated_run_dir("experiments", spec.name)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(f"[js2024] Loading {spec.name} frame ...")
        load_t0 = time.perf_counter()
        df = spec.load_frame(config, feature_cols)
        load_secs = time.perf_counter() - load_t0
        print(f"[js2024] Shared frame: {df.height:,} rows (load {load_secs:.1f}s).")
        bundle = run_walk_forward_suite(
            spec, config, df, variants=variants, feature_cols=feature_cols
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[js2024] ERROR: {exc}", file=sys.stderr)
        return 1

    pl.DataFrame(summary_rows(bundle)).write_csv(out_dir / "summary.csv")
    print(f"\n[js2024] Wrote {out_dir / 'summary.csv'}")

    write_manifest(
        out_dir / "manifest.json",
        runner="js2024-run-experiment",
        model=spec.name,
        config=args.config,
        variants=variants,
        out_dir=str(out_dir),
    )
    print(f"[js2024] Wrote {out_dir / 'manifest.json'}")

    docs_path = resolve_project_path(args.docs_out or (out_dir / "report.md"))
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(render_report(bundle, spec, config, "completed"), encoding="utf-8")
    print(f"[js2024] Wrote experiment doc -> {docs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
