r"""Paired marketroll window × subset × seed sweep (decisive metric: gru_incremental).

Validates whether the GRU online-incremental gain from the engineered market-avg /
per-symbol rolling features holds up across rolling windows, feature subsets and random
seeds — not just at the single (window=1000, subset=top12, seed=42) point measured so
far. The decisive number is the **paired** delta::

    delta(window, subset, seed) = gru_incremental(marketroll) - gru_incremental(baseline)

where ``baseline`` is the same GRU with the engineered features off. The baseline is
invariant to ``window``/``subset``, so it is run **once per seed** and reused across
every cell — making the grid ``len(seeds)`` baseline runs + ``windows × subsets × seeds``
marketroll runs (not the full Cartesian product).

Efficiency: the base frame (features + aux targets, no engineered columns) is loaded
**once**; each ``(window, subset)`` cell computes its engineered columns once and reuses
that frame across all seeds. Per-run rows are appended to ``runs.csv`` as they finish, so
a crash mid-sweep keeps every completed run (this is a multi-hour job).

Usage
-----
    uv run js2024-run-marketroll-sweep \
        --config configs/gru_marketroll_v1.yaml \
        --windows 250,500,1000,2000 --subsets top12,top24,all \
        --seeds 42,43,44,45,46
"""

from __future__ import annotations

import argparse
import dataclasses
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import polars as pl

from ..modeling.config import GRUConfig, load_yaml_config, resolve_project_path
from ..modeling.experiments import dated_run_dir, write_manifest
from ..modeling.market_features import (
    MARKET_ROLL_SUBSETS,
    add_engineered_features,
    resolve_market_roll_features,
)
from ..modeling.registry import get_model_spec
from ..modeling.walk_forward_suite import run_walk_forward_suite

# The sweep only studies the online-incremental gain — the variant where the GRU
# actually generalises (see configs/gru_marketroll_v1.yaml).
VARIANT = "incremental"
SCORE_LABEL = "gru_incremental"


# --- pure aggregation helpers (unit-tested) -------------------------------


def paired_rows(
    runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join each marketroll run to its same-seed baseline and compute the delta.

    ``runs`` rows have keys ``kind`` ('baseline'|'marketroll'), ``window``, ``subset``,
    ``seed``, ``score``. Baseline rows have ``window``/``subset`` = None. Returns one
    paired row per marketroll run that has a matching-seed baseline.
    """
    baseline_by_seed = {r["seed"]: r["score"] for r in runs if r["kind"] == "baseline"}
    out: list[dict[str, Any]] = []
    for r in runs:
        if r["kind"] != "marketroll":
            continue
        base = baseline_by_seed.get(r["seed"])
        if base is None:
            continue
        out.append(
            {
                "window": r["window"],
                "subset": r["subset"],
                "seed": r["seed"],
                "marketroll_score": r["score"],
                "baseline_score": base,
                "delta": r["score"] - base,
            }
        )
    return out


def summarize_cells(paired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate paired deltas per ``(window, subset)`` cell.

    Reports mean/std of the paired delta, the seed count, how many seeds had a positive
    delta (sign stability), and the cell means — sorted best mean delta first.
    """
    cells: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for p in paired:
        cells.setdefault((p["window"], p["subset"]), []).append(p)
    rows: list[dict[str, Any]] = []
    for (window, subset), ps in cells.items():
        deltas = [p["delta"] for p in ps]
        rows.append(
            {
                "window": window,
                "subset": subset,
                "n_seeds": len(ps),
                "mean_delta": statistics.fmean(deltas),
                "std_delta": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                "n_positive": sum(1 for d in deltas if d > 0),
                "mean_marketroll": statistics.fmean([p["marketroll_score"] for p in ps]),
                "mean_baseline": statistics.fmean([p["baseline_score"] for p in ps]),
            }
        )
    rows.sort(key=lambda r: r["mean_delta"], reverse=True)
    return rows


def render_report(
    summary: list[dict[str, Any]], paired: list[dict[str, Any]], n_seeds: int
) -> str:
    L: list[str] = []
    L.append("# Marketroll window × subset × seed sweep — gru_incremental")
    L.append("")
    L.append(
        f"Paired delta = `gru_incremental(marketroll) − gru_incremental(baseline)` at the "
        f"same seed ({n_seeds} seeds/cell). Baseline = same GRU, engineered features off."
    )
    L.append("")
    L.append("## Per-cell summary (best mean Δ first)")
    L.append("")
    L.append("| window | subset | n | mean Δ | std Δ | +seeds | mean mkt | mean base |")
    L.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in summary:
        L.append(
            f"| {r['window']} | {r['subset']} | {r['n_seeds']} | {r['mean_delta']:+.6f} | "
            f"{r['std_delta']:.6f} | {r['n_positive']}/{r['n_seeds']} | "
            f"{r['mean_marketroll']:.6f} | {r['mean_baseline']:.6f} |"
        )
    L.append("")
    if summary:
        best = summary[0]
        L.append(
            f"- **best cell:** window={best['window']}, subset={best['subset']} "
            f"(mean Δ={best['mean_delta']:+.6f}, {best['n_positive']}/{best['n_seeds']} "
            "seeds positive)."
        )
    L.append("")
    return "\n".join(L)


# --- sweep driver ---------------------------------------------------------


def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_str_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _run_score(
    spec,
    config: GRUConfig,
    df: pl.DataFrame,
    feature_cols: list[str],
    *,
    wandb_run_name: str | None = None,
) -> dict[str, Any]:
    """Run the incremental suite on a prepared frame and return score + timing."""
    t0 = time.perf_counter()
    bundle = run_walk_forward_suite(
        spec, config, df, variants=[VARIANT], feature_cols=feature_cols,
        wandb_run_name=wandb_run_name,
    )
    res = bundle.results[SCORE_LABEL]
    return {"score": res.score, "n_updates": res.n_updates, "secs": time.perf_counter() - t0}


def _append_run(runs_csv: Path, row: dict[str, Any]) -> None:
    """Append one run row to runs.csv (write header on first row)."""
    header = not runs_csv.exists()
    line_df = pl.DataFrame([row])
    with runs_csv.open("a") as fh:
        line_df.write_csv(fh, include_header=header)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gru_marketroll_v1.yaml")
    parser.add_argument("--windows", default="250,500,1000,2000")
    parser.add_argument("--subsets", default="top12,top24,all")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip the paired baseline runs (only valid if a previous runs.csv supplies them).",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Enable wandb logging per run (default off for sweeps)."
    )
    parser.add_argument(
        "--wandb-project",
        default="js2024-marketroll-sweep",
        help="wandb project for the sweep (a fresh project = a fresh URL). Used with --wandb.",
    )
    parser.add_argument("--test-days", type=int, default=None, help="Override test_days (smoke).")
    parser.add_argument("--valid-days", type=int, default=None, help="Override valid_days (smoke).")
    parser.add_argument(
        "--start-date-id", type=int, default=None, help="Override start_date_id (smoke)."
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        raw = load_yaml_config(args.config)
        if raw.get("model") != "gru":
            raise ValueError("marketroll sweep requires a GRU config (model: gru)")
        spec = get_model_spec("gru")
        base: GRUConfig = spec.load_config(args.config)
        windows = _parse_int_list(args.windows)
        subsets = _parse_str_list(args.subsets)
        seeds = _parse_int_list(args.seeds)
        bad = [s for s in subsets if s not in MARKET_ROLL_SUBSETS]
        if bad:
            raise ValueError(f"unknown subset(s) {bad}; choose from {list(MARKET_ROLL_SUBSETS)}")
        if not (windows and subsets and seeds):
            raise ValueError("windows, subsets and seeds must each be non-empty")
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"[js2024] ERROR: {exc}", file=sys.stderr)
        return 1

    # Apply smoke overrides + force wandb policy consistently across every cell.
    overrides: dict[str, Any] = {"use_wandb": args.wandb}
    if args.wandb:
        overrides["wandb_project"] = args.wandb_project
    if args.test_days is not None:
        overrides["test_days"] = args.test_days
    if args.valid_days is not None:
        overrides["valid_days"] = args.valid_days
    if args.start_date_id is not None:
        overrides["start_date_id"] = args.start_date_id
    base = dataclasses.replace(base, **overrides)

    n_cells = len(windows) * len(subsets)
    n_runs = (0 if args.no_baseline else len(seeds)) + n_cells * len(seeds)
    print(
        f"[js2024] marketroll sweep | windows={windows} subsets={subsets} seeds={seeds}\n"
        f"[js2024]   {n_cells} cells × {len(seeds)} seeds + "
        f"{0 if args.no_baseline else len(seeds)} baseline = {n_runs} runs."
    )
    if args.dry_run:
        print("[js2024] Dry run: no training performed.")
        return 0

    out_dir = resolve_project_path(args.out_dir or dated_run_dir("experiments", "marketroll_sweep"))
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_csv = out_dir / "runs.csv"
    print(f"[js2024] Writing per-run rows to {runs_csv} as they complete.")

    runs: list[dict[str, Any]] = []
    done = 0
    sweep_t0 = time.perf_counter()

    def progress() -> str:
        elapsed = time.perf_counter() - sweep_t0
        if done == 0:
            return ""
        eta = elapsed / done * (n_runs - done)
        return f" [{done}/{n_runs}, elapsed {elapsed/60:.0f}m, eta {eta/60:.0f}m]"

    # --- baseline: engineered features off; window/subset invariant -------
    base_off = dataclasses.replace(base, use_market_avg=False, use_symbol_rolling=False)
    print("[js2024] Loading base frame (no engineered features) ...")
    base_df = spec.load_frame(base_off, spec.feature_columns(base_off))
    print(f"[js2024] Base frame: {base_df.height:,} rows.")

    if not args.no_baseline:
        base_feats = spec.feature_columns(base_off)
        for seed in seeds:
            cfg = dataclasses.replace(base_off, random_state=seed)
            print(f"\n[js2024] === baseline seed={seed} ==={progress()}")
            r = _run_score(spec, cfg, base_df, base_feats, wandb_run_name=f"base_s{seed}")
            row = {
                "kind": "baseline", "window": None, "subset": None, "seed": seed,
                "score": r["score"], "n_updates": r["n_updates"], "secs": r["secs"],
            }
            runs.append(row)
            _append_run(runs_csv, row)
            done += 1

    # --- marketroll cells: compute engineered frame once, reuse across seeds
    for window in windows:
        for subset in subsets:
            cell = dataclasses.replace(
                base, use_market_avg=True, use_symbol_rolling=True,
                rolling_window=window, market_roll_subset=subset,
            )
            feats_engineered = resolve_market_roll_features(subset)
            print(
                f"\n[js2024] Building engineered frame: window={window} subset={subset} "
                f"({len(feats_engineered)} features) ..."
            )
            eng_df = add_engineered_features(
                base_df, use_market_avg=True, use_symbol_rolling=True,
                window=window, features=feats_engineered,
            )
            cell_feats = spec.feature_columns(cell)
            for seed in seeds:
                cfg = dataclasses.replace(cell, random_state=seed)
                print(
                    f"\n[js2024] === marketroll window={window} subset={subset} "
                    f"seed={seed} ==={progress()}"
                )
                r = _run_score(
                    spec, cfg, eng_df, cell_feats,
                    wandb_run_name=f"mkt_w{window}_{subset}_s{seed}",
                )
                row = {
                    "kind": "marketroll", "window": window, "subset": subset, "seed": seed,
                    "score": r["score"], "n_updates": r["n_updates"], "secs": r["secs"],
                }
                runs.append(row)
                _append_run(runs_csv, row)
                done += 1

    # --- aggregate + write outputs ----------------------------------------
    paired = paired_rows(runs)
    summary = summarize_cells(paired)
    if paired:
        pl.DataFrame(paired).write_csv(out_dir / "paired.csv")
    if summary:
        pl.DataFrame(summary).write_csv(out_dir / "summary.csv")
    report = render_report(summary, paired, n_seeds=len(seeds))
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    write_manifest(
        out_dir / "manifest.json",
        runner="js2024-run-marketroll-sweep",
        config=args.config,
        windows=windows, subsets=subsets, seeds=seeds,
        n_runs=len(runs), out_dir=str(out_dir),
    )
    print(f"\n[js2024] Sweep complete: {len(runs)} runs in {(time.perf_counter()-sweep_t0)/60:.0f}m.")
    print(f"[js2024] Wrote {out_dir/'runs.csv'}, paired.csv, summary.csv, report.md, manifest.json")
    print("\n" + report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
