r"""Paired GRU auxiliary-target-set × seed sweep (decisive metric: gru_incremental).

Measures whether enlarging the GRU's auxiliary-head set beyond the public-solution
``base4`` (responder_10/9/8/7) helps the online-incremental model. The decisive
number is the **paired** delta against ``base4`` at the same seed::

    delta(set, seed) = gru_incremental(set) - gru_incremental(base4)

Efficiency: the auxiliary targets are training targets, **not** model inputs, so the
feature columns are identical across every set. The runner therefore loads **one**
maximal frame (``aux_target_set=all11``, which loads responder_0..8 and generates the
synthetic responder_9/10) and reuses it for every ``(set, seed)`` — only the
estimator's ``aux_cols`` changes per cell, making the pairing exact. Per-run rows are
appended to ``runs.csv`` as they finish so a crash mid-sweep keeps completed runs.

Usage
-----
    uv run js2024-run-aux-sweep \
        --config configs/gru_aux_base4.yaml \
        --aux-sets base4,target_family,all9,all11 --seeds 42,43,44
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
from ..modeling.gru import GRU_AUX_TARGET_SETS
from ..modeling.registry import get_model_spec
from ..modeling.walk_forward_suite import run_walk_forward_suite

# The sweep only studies the online-incremental gain — the variant where the GRU
# actually generalises (see configs/gru_aux_base4.yaml).
VARIANT = "incremental"
SCORE_LABEL = "gru_incremental"
BASELINE_SET = "base4"
# Frame loaded once with the maximal aux set so every cell shares identical inputs.
MAXIMAL_SET = "all11"


# --- pure aggregation helpers (unit-tested) -------------------------------


def paired_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join each run to its same-seed ``base4`` run and compute the delta.

    ``runs`` rows have keys ``aux_set``, ``seed``, ``score``. Returns one paired row
    per non-``base4`` run that has a matching-seed ``base4`` run.
    """
    base_by_seed = {r["seed"]: r["score"] for r in runs if r["aux_set"] == BASELINE_SET}
    out: list[dict[str, Any]] = []
    for r in runs:
        if r["aux_set"] == BASELINE_SET:
            continue
        base = base_by_seed.get(r["seed"])
        if base is None:
            continue
        out.append(
            {
                "aux_set": r["aux_set"],
                "seed": r["seed"],
                "set_score": r["score"],
                "base4_score": base,
                "delta": r["score"] - base,
            }
        )
    return out


def summarize_cells(paired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate paired deltas per ``aux_set``.

    Reports mean/std of the paired delta, seed count, how many seeds had a positive
    delta (sign stability), and the cell means — sorted best mean delta first.
    """
    cells: dict[str, list[dict[str, Any]]] = {}
    for p in paired:
        cells.setdefault(p["aux_set"], []).append(p)
    rows: list[dict[str, Any]] = []
    for aux_set, ps in cells.items():
        deltas = [p["delta"] for p in ps]
        rows.append(
            {
                "aux_set": aux_set,
                "n_seeds": len(ps),
                "mean_delta": statistics.fmean(deltas),
                "std_delta": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                "n_positive": sum(1 for d in deltas if d > 0),
                "mean_set": statistics.fmean([p["set_score"] for p in ps]),
                "mean_base4": statistics.fmean([p["base4_score"] for p in ps]),
            }
        )
    rows.sort(key=lambda r: r["mean_delta"], reverse=True)
    return rows


def render_report(
    summary: list[dict[str, Any]], paired: list[dict[str, Any]], n_seeds: int
) -> str:
    L: list[str] = []
    L.append("# GRU auxiliary-target-set × seed sweep — gru_incremental")
    L.append("")
    L.append(
        f"Paired delta = `gru_incremental(set) − gru_incremental(base4)` at the same "
        f"seed ({n_seeds} seeds/set). base4 = public-solution responder_10/9/8/7."
    )
    L.append("")
    L.append("## Per-set summary (best mean Δ first)")
    L.append("")
    L.append("| aux_set | n | mean Δ | std Δ | +seeds | mean set | mean base4 |")
    L.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in summary:
        L.append(
            f"| {r['aux_set']} | {r['n_seeds']} | {r['mean_delta']:+.6f} | "
            f"{r['std_delta']:.6f} | {r['n_positive']}/{r['n_seeds']} | "
            f"{r['mean_set']:.6f} | {r['mean_base4']:.6f} |"
        )
    L.append("")
    if summary:
        best = summary[0]
        L.append(
            f"- **best set:** {best['aux_set']} (mean Δ={best['mean_delta']:+.6f}, "
            f"{best['n_positive']}/{best['n_seeds']} seeds positive)."
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
    parser.add_argument("--config", default="configs/gru_aux_base4.yaml")
    parser.add_argument("--aux-sets", default="base4,target_family,all9,all11")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--wandb", action="store_true", help="Enable wandb logging per run (default off for sweeps)."
    )
    parser.add_argument(
        "--wandb-project",
        default="js2024-gru-aux-sweep",
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
            raise ValueError("aux sweep requires a GRU config (model: gru)")
        spec = get_model_spec("gru")
        base: GRUConfig = spec.load_config(args.config)
        aux_sets = _parse_str_list(args.aux_sets)
        seeds = _parse_int_list(args.seeds)
        bad = [s for s in aux_sets if s not in GRU_AUX_TARGET_SETS]
        if bad:
            raise ValueError(
                f"unknown aux set(s) {bad}; choose from {list(GRU_AUX_TARGET_SETS)}"
            )
        if BASELINE_SET not in aux_sets:
            raise ValueError(
                f"aux-sets must include the paired baseline {BASELINE_SET!r}; got {aux_sets}"
            )
        if not (aux_sets and seeds):
            raise ValueError("aux-sets and seeds must each be non-empty")
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

    n_runs = len(aux_sets) * len(seeds)
    print(
        f"[js2024] aux sweep | aux_sets={aux_sets} seeds={seeds}\n"
        f"[js2024]   {len(aux_sets)} sets × {len(seeds)} seeds = {n_runs} runs."
    )
    if args.dry_run:
        print("[js2024] Dry run: no training performed.")
        return 0

    out_dir = resolve_project_path(args.out_dir or dated_run_dir("experiments", "gru_aux_sweep"))
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_csv = out_dir / "runs.csv"
    print(f"[js2024] Writing per-run rows to {runs_csv} as they complete.")

    # Load the maximal frame once: identical features across sets, all responders
    # present so each set just selects its own aux_cols from the same frame.
    frame_cfg = dataclasses.replace(base, aux_target_set=MAXIMAL_SET)
    feature_cols = spec.feature_columns(frame_cfg)
    print(f"[js2024] Loading shared frame (aux_target_set={MAXIMAL_SET}) ...")
    df = spec.load_frame(frame_cfg, feature_cols)
    print(f"[js2024] Frame: {df.height:,} rows.")

    runs: list[dict[str, Any]] = []
    done = 0
    sweep_t0 = time.perf_counter()

    def progress() -> str:
        elapsed = time.perf_counter() - sweep_t0
        if done == 0:
            return ""
        eta = elapsed / done * (n_runs - done)
        return f" [{done}/{n_runs}, elapsed {elapsed/60:.0f}m, eta {eta/60:.0f}m]"

    for aux_set in aux_sets:
        for seed in seeds:
            cfg = dataclasses.replace(base, aux_target_set=aux_set, random_state=seed)
            print(f"\n[js2024] === aux_set={aux_set} seed={seed} ==={progress()}")
            r = _run_score(
                spec, cfg, df, feature_cols, wandb_run_name=f"aux_{aux_set}_s{seed}"
            )
            row = {
                "aux_set": aux_set, "seed": seed,
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
        runner="js2024-run-aux-sweep",
        config=args.config,
        aux_sets=aux_sets, seeds=seeds,
        n_runs=len(runs), out_dir=str(out_dir),
    )
    print(f"\n[js2024] Sweep complete: {len(runs)} runs in {(time.perf_counter()-sweep_t0)/60:.0f}m.")
    print(f"[js2024] Wrote {out_dir/'runs.csv'}, paired.csv, summary.csv, report.md, manifest.json")
    print("\n" + report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
