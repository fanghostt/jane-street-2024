"""CLI: audit the semantics & schema of the Jane Street 2024 raw data files.

This is a *read-only* audit. It never trains a model, never engineers features
and never mutates the raw data. It answers four questions:

1. What is ``lags.parquet`` and how does it map to the train/test/evaluation API?
2. Does ``test.parquet`` carry a label (can it be used as local validation)?
3. Is the Kaggle competition still open for officially-scored submissions?
4. What do ``features.csv`` / ``responders.csv`` contain (schema, tags) and how
   do they relate to later feature engineering?

Usage
-----
    uv run js2024-data-semantics-audit \\
        --raw-dir data/raw \\
        --out-dir outputs/data_semantics_audit \\
        --docs-out docs/data/data_semantics_audit.md

Parquet files are inspected *lazily*: we read the schema, a tiny ``head`` sample
and a single lazy aggregation per file, so the full (47M-row) train set never
lands in memory. The small CSVs (``features.csv`` / ``responders.csv``) are read
in full.

Outputs written under ``--out-dir`` (git-ignored, never committed):
``train_schema.csv``, ``test_schema.csv``, ``lags_schema.csv``,
``features_metadata.csv``, ``responders_metadata.csv`` and ``file_summary.json``.

The committed artifact is the markdown doc at ``--docs-out``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import polars as pl

from .data import resolve_parquet_scan_path

FEATURE_COLUMNS: list[str] = [f"feature_{i:02d}" for i in range(79)]
RESPONDER_COLUMNS: list[str] = [f"responder_{i}" for i in range(9)]
LAG_COLUMNS: list[str] = [f"responder_{i}_lag_1" for i in range(9)]
TARGET_COLUMN: str = "responder_6"
WEIGHT_COLUMN: str = "weight"
ID_COLUMNS: list[str] = ["date_id", "time_id", "symbol_id"]

# Verified against the Kaggle CLI on 2026-06-24 (see docs section 7).
KAGGLE_DEADLINE = "2025-07-12"
AUDIT_TODAY = "2026-06-24"


# --------------------------------------------------------------------------- #
# Parquet auditing (lazy)
# --------------------------------------------------------------------------- #
def _summary_exprs(col: str) -> list[pl.Expr]:
    return [
        pl.col(col).mean().alias(f"{col}_mean"),
        pl.col(col).std().alias(f"{col}_std"),
        pl.col(col).min().alias(f"{col}_min"),
        pl.col(col).max().alias(f"{col}_max"),
    ]


def audit_parquet(path: str | Path, *, head_n: int = 3) -> dict[str, Any]:
    """Audit a parquet file/dir lazily: schema, counts, ranges, key summaries.

    Returns a dict with ``exists``; when present it also carries ``schema``
    (list of ``{column, dtype}``), ``n_columns``, ``row_count``, id ranges,
    column-presence flags, per-responder min/max, weight/target summaries and a
    small ``head`` sample. Nothing is collected eagerly except the head and a
    single aggregation row.
    """
    p = Path(path)
    if not p.exists():
        return {"exists": False, "path": str(p)}

    scan_path = resolve_parquet_scan_path(p)
    lf = pl.scan_parquet(scan_path)
    schema = lf.collect_schema()
    cols = schema.names()
    col_set = set(cols)

    agg: list[pl.Expr] = [pl.len().alias("row_count")]
    for c in ID_COLUMNS:
        if c in col_set:
            agg += [
                pl.col(c).min().alias(f"{c}_min"),
                pl.col(c).max().alias(f"{c}_max"),
                pl.col(c).n_unique().alias(f"{c}_n_unique"),
            ]
    for c in RESPONDER_COLUMNS:
        # TARGET_COLUMN's min/max come from its richer summary block below;
        # adding them here too would create duplicate aggregation aliases.
        if c in col_set and c != TARGET_COLUMN:
            agg += [
                pl.col(c).min().alias(f"{c}_min"),
                pl.col(c).max().alias(f"{c}_max"),
            ]
    if TARGET_COLUMN in col_set:
        agg += _summary_exprs(TARGET_COLUMN)
    if WEIGHT_COLUMN in col_set:
        agg += _summary_exprs(WEIGHT_COLUMN)

    stats = lf.select(agg).collect().row(0, named=True)
    head_df = lf.head(head_n).collect()

    present_responders = [c for c in RESPONDER_COLUMNS if c in col_set]
    responder_minmax = {
        c: (stats.get(f"{c}_min"), stats.get(f"{c}_max"))
        for c in present_responders
    }
    all_clipped = bool(present_responders) and all(
        (lo is not None and hi is not None and lo >= -5.0 and hi <= 5.0)
        for lo, hi in responder_minmax.values()
    )

    return {
        "exists": True,
        "path": str(p),
        "scan_path": str(scan_path),
        "schema": [{"column": c, "dtype": str(schema[c])} for c in cols],
        "n_columns": len(cols),
        "row_count": stats.get("row_count"),
        "stats": stats,
        "presence": {
            "all_id_columns": all(c in col_set for c in ID_COLUMNS),
            "weight": WEIGHT_COLUMN in col_set,
            "is_scored": "is_scored" in col_set,
            "row_id": "row_id" in col_set,
            "n_feature_columns": sum(c in col_set for c in FEATURE_COLUMNS),
            "has_all_features": all(c in col_set for c in FEATURE_COLUMNS),
            "n_responder_columns": len(present_responders),
            "has_responder_6": TARGET_COLUMN in col_set,
            "n_lag_columns": sum(c in col_set for c in LAG_COLUMNS),
            "has_all_lags": all(c in col_set for c in LAG_COLUMNS),
        },
        "responder_minmax": responder_minmax,
        "responders_clipped_-5_5": all_clipped,
        "head": head_df.to_dicts(),
        "columns": cols,
    }


# --------------------------------------------------------------------------- #
# Metadata CSV auditing (read in full — these files are tiny)
# --------------------------------------------------------------------------- #
def _audit_tag_csv(path: str | Path, key_col: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"exists": False, "path": str(p)}

    df = pl.read_csv(p)
    cols = df.columns
    tag_cols = [c for c in cols if c.startswith("tag_")]
    names = df.get_column(key_col).to_list() if key_col in cols else []

    tag_summary: dict[str, dict[str, int]] = {}
    groups: dict[str, list[str]] = {}
    for tag in tag_cols:
        col = df.get_column(tag)
        true_count = int(col.sum())
        tag_summary[tag] = {
            "true": true_count,
            "false": int(df.height - true_count),
        }
        if key_col in cols:
            groups[tag] = (
                df.filter(pl.col(tag)).get_column(key_col).to_list()
            )

    # tags-per-row, for a normalized metadata table.
    per_row: list[dict[str, Any]] = []
    for record in df.iter_rows(named=True):
        active = [t for t in tag_cols if record.get(t)]
        per_row.append(
            {
                key_col: record.get(key_col),
                "n_tags": len(active),
                "tags": "|".join(active),
            }
        )

    return {
        "exists": True,
        "path": str(p),
        "row_count": df.height,
        "columns": cols,
        "key_column": key_col,
        "n_tag_columns": len(tag_cols),
        "tag_columns": tag_cols,
        "names": names,
        "tag_summary": tag_summary,
        "groups": groups,
        "per_row": per_row,
        "dataframe": df,
    }


def audit_features_csv(path: str | Path) -> dict[str, Any]:
    """Audit ``features.csv`` (anonymized feature metadata + boolean tags)."""
    return _audit_tag_csv(path, key_col="feature")


def audit_responders_csv(path: str | Path) -> dict[str, Any]:
    """Audit ``responders.csv`` (anonymized responder metadata + boolean tags)."""
    return _audit_tag_csv(path, key_col="responder")


def audit_sample_submission(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"exists": False, "path": str(p)}
    df = pl.read_csv(p)
    return {
        "exists": True,
        "path": str(p),
        "columns": df.columns,
        "row_count": df.height,
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_audit(raw_dir: str | Path) -> dict[str, Any]:
    """Audit every known raw file under ``raw_dir`` and return a results dict."""
    raw = Path(raw_dir)
    return {
        "raw_dir": str(raw),
        "train": audit_parquet(raw / "train.parquet"),
        "test": audit_parquet(raw / "test.parquet"),
        "lags": audit_parquet(raw / "lags.parquet"),
        "features": audit_features_csv(raw / "features.csv"),
        "responders": audit_responders_csv(raw / "responders.csv"),
        "sample_submission": audit_sample_submission(
            raw / "sample_submission.csv"
        ),
    }


def _write_schema_csv(audit: dict[str, Any], out_path: Path) -> None:
    if not audit.get("exists"):
        return
    pl.DataFrame(audit["schema"]).write_csv(out_path)


def write_artifacts(results: dict[str, Any], out_dir: str | Path) -> list[Path]:
    """Write git-ignored audit artifacts under ``out_dir``; return their paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    _write_schema_csv(results["train"], out / "train_schema.csv")
    _write_schema_csv(results["test"], out / "test_schema.csv")
    _write_schema_csv(results["lags"], out / "lags_schema.csv")
    for name in ("train_schema.csv", "test_schema.csv", "lags_schema.csv"):
        if (out / name).exists():
            written.append(out / name)

    feats = results["features"]
    if feats.get("exists"):
        pl.DataFrame(feats["per_row"]).write_csv(out / "features_metadata.csv")
        written.append(out / "features_metadata.csv")

    resp = results["responders"]
    if resp.get("exists"):
        pl.DataFrame(resp["per_row"]).write_csv(out / "responders_metadata.csv")
        written.append(out / "responders_metadata.csv")

    # JSON summary excludes bulky/non-serializable bits (head dicts, dataframes).
    summary = _json_summary(results)
    (out / "file_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    written.append(out / "file_summary.json")
    return written


def _json_summary(results: dict[str, Any]) -> dict[str, Any]:
    def parquet_brief(a: dict[str, Any]) -> dict[str, Any]:
        if not a.get("exists"):
            return {"exists": False}
        return {
            "exists": True,
            "n_columns": a["n_columns"],
            "row_count": a["row_count"],
            "presence": a["presence"],
            "responders_clipped_-5_5": a["responders_clipped_-5_5"],
            "stats": a["stats"],
        }

    def csv_brief(a: dict[str, Any]) -> dict[str, Any]:
        if not a.get("exists"):
            return {"exists": False}
        return {
            "exists": True,
            "row_count": a["row_count"],
            "columns": a["columns"],
            "n_tag_columns": a["n_tag_columns"],
            "tag_summary": a["tag_summary"],
        }

    return {
        "raw_dir": results["raw_dir"],
        "train": parquet_brief(results["train"]),
        "test": parquet_brief(results["test"]),
        "lags": parquet_brief(results["lags"]),
        "features": csv_brief(results["features"]),
        "responders": csv_brief(results["responders"]),
        "sample_submission": results["sample_submission"],
    }


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #
def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def _yn(flag: bool) -> str:
    return "yes" if flag else "no"


def render_docs(results: dict[str, Any]) -> str:
    train = results["train"]
    test = results["test"]
    lags = results["lags"]
    feats = results["features"]
    resp = results["responders"]
    sub = results["sample_submission"]

    L: list[str] = []
    L.append("# Jane Street 2024 — data semantics audit")
    L.append("")
    L.append(
        "Read-only audit of `data/raw/*`. No model is trained, no feature is "
        "engineered and no raw data is mutated. Regenerate with "
        "`uv run js2024-data-semantics-audit` (see README)."
    )
    L.append("")

    # 1. File inventory ----------------------------------------------------- #
    L.append("## 1. File inventory")
    L.append("")
    L.append("| file | exists | role | local scoring? | notes |")
    L.append("| --- | --- | --- | --- | --- |")
    L.append(
        f"| train.parquet | {_yn(train.get('exists', False))} | historical "
        "training data, contains responders | yes (cut by time) | source of "
        "all local validation splits |"
    )
    L.append(
        f"| test.parquet | {_yn(test.get('exists', False))} | mock test "
        "structure / evaluation API input | no | no `responder_6` label; "
        "API/inference compatibility only |"
    )
    L.append(
        f"| lags.parquet | {_yn(lags.get('exists', False))} | responder_0..8 "
        "lagged by one date_id | no | served at first time_id of the "
        "succeeding date; for lag features / online / GRU |"
    )
    L.append(
        f"| features.csv | {_yn(feats.get('exists', False))} | metadata for "
        "anonymized features (boolean tags) | no | feature grouping / parity "
        "analysis |"
    )
    L.append(
        f"| responders.csv | {_yn(resp.get('exists', False))} | metadata for "
        "anonymized responders | no | not the label values themselves |"
    )
    L.append(
        f"| sample_submission.csv | {_yn(sub.get('exists', False))} | "
        "submission format (`row_id`,`responder_6`) | no | shape reference for "
        "the inference gateway |"
    )
    L.append("")
    L.append("**Key conclusions:**")
    L.append("")
    L.append(
        "- `train.parquet`: historical training data; contains responders; "
        "local validation **must** be cut from it by time."
    )
    L.append(
        "- `test.parquet`: mock test structure / evaluation API input; does "
        "**not** contain `responder_6`; cannot be used for local R² validation."
    )
    L.append(
        "- `lags.parquet`: `responder_0..8` lagged by one `date_id`, served by "
        "the evaluation API at the first `time_id` of the succeeding date; "
        "useful for lag features / online learning / GRU, **not** used in V0 "
        "raw LGBM."
    )
    L.append(
        "- `features.csv` / `responders.csv`: anonymized metadata (boolean "
        "tags), **not** the label values."
    )
    L.append("")

    # 2. train.parquet ------------------------------------------------------ #
    L.append("## 2. train.parquet audit")
    L.append("")
    if train.get("exists"):
        s = train["stats"]
        pr = train["presence"]
        L.append(f"- schema columns: **{train['n_columns']}**")
        L.append(f"- row count: **{train['row_count']:,}**")
        L.append(
            f"- has date_id/time_id/symbol_id: {_yn(pr['all_id_columns'])}; "
            f"has weight: {_yn(pr['weight'])}"
        )
        L.append(
            f"- feature_00..feature_78 present: {_yn(pr['has_all_features'])} "
            f"({pr['n_feature_columns']}/79)"
        )
        L.append(
            f"- responder_0..responder_8 present: {_yn(pr['n_responder_columns'] == 9)} "
            f"({pr['n_responder_columns']}/9)"
        )
        L.append(f"- **responder_6 exists: {_yn(pr['has_responder_6'])}**")
        L.append(
            f"- date_id min/max: {_fmt(s.get('date_id_min'))} … "
            f"{_fmt(s.get('date_id_max'))} "
            f"({_fmt(s.get('date_id_n_unique'))} unique)"
        )
        L.append(
            f"- time_id n_unique: {_fmt(s.get('time_id_n_unique'))} "
            f"(min {_fmt(s.get('time_id_min'))}, max {_fmt(s.get('time_id_max'))})"
        )
        L.append(f"- symbol_id n_unique: {_fmt(s.get('symbol_id_n_unique'))}")
        L.append(
            f"- responder_6 summary: mean={_fmt(s.get('responder_6_mean'))} | "
            f"std={_fmt(s.get('responder_6_std'))} | "
            f"min={_fmt(s.get('responder_6_min'))} | "
            f"max={_fmt(s.get('responder_6_max'))}"
        )
        L.append(
            f"- weight summary: mean={_fmt(s.get('weight_mean'))} | "
            f"std={_fmt(s.get('weight_std'))} | "
            f"min={_fmt(s.get('weight_min'))} | max={_fmt(s.get('weight_max'))}"
        )
        L.append(
            f"- **all responders clipped to [-5, 5]: "
            f"{_yn(train['responders_clipped_-5_5'])}** "
            f"(observed per-responder min/max: "
            + ", ".join(
                f"{c}=[{_fmt(lo)},{_fmt(hi)}]"
                for c, (lo, hi) in train["responder_minmax"].items()
            )
            + ")"
        )
    else:
        L.append("_train.parquet not found in raw dir._")
    L.append("")
    L.append(
        "**Observed panel structure (real Kaggle data).** `train.parquet` is a "
        "`(date_id × time_id × symbol_id)` panel, not a flat table:"
    )
    L.append("")
    L.append(
        "- **`weight` is constant within each `(date_id, symbol_id)`** — every "
        "intraday `time_id` of a given symbol-day carries the same weight "
        "(verified: `n_unique(weight) == 1` per symbol on date 0). Weight is a "
        "per-symbol-per-day scalar broadcast across the day."
    )
    L.append(
        "- **The symbol universe grows over time**: ~8 symbols on `date_id=0`, "
        "13 by day 100, 28 by day 500, 35 by day 1000, 39 by day 1698. "
        "Instruments are phased in, so early dates cover far fewer symbols."
    )
    L.append(
        "- **`time_id` is not a fixed grid**: the number of intraday buckets "
        "varies by day (≈849 on date 0 vs 968 on date 1698)."
    )
    L.append(
        "- **Features have a warm-up period**: on `date_id=0` only 44/79 "
        "features are populated; several (`feature_00..04`, `21`, `26`, `27`, "
        "`31`) are 100% null early and only appear once enough history exists."
    )
    L.append("")

    # 3. test.parquet ------------------------------------------------------- #
    L.append("## 3. test.parquet audit")
    L.append("")
    if test.get("exists"):
        s = test["stats"]
        pr = test["presence"]
        L.append(f"- schema columns: **{test['n_columns']}**")
        L.append(f"- row count: **{test['row_count']:,}**")
        L.append(
            f"- has date_id/time_id/symbol_id/weight: "
            f"{_yn(pr['all_id_columns'] and pr['weight'])}; "
            f"has is_scored: {_yn(pr['is_scored'])}; "
            f"has row_id: {_yn(pr['row_id'])}"
        )
        L.append(
            f"- feature_00..feature_78 present: {_yn(pr['has_all_features'])} "
            f"({pr['n_feature_columns']}/79)"
        )
        L.append(f"- **contains responder_6: {_yn(pr['has_responder_6'])}**")
        L.append(
            f"- date_id min/max: {_fmt(s.get('date_id_min'))} … "
            f"{_fmt(s.get('date_id_max'))}; "
            f"time_id min/max: {_fmt(s.get('time_id_min'))} … "
            f"{_fmt(s.get('time_id_max'))}; "
            f"symbol_id n_unique: {_fmt(s.get('symbol_id_n_unique'))}"
        )
    else:
        L.append("_test.parquet not found in raw dir._")
    L.append("")
    L.append(
        "**Conclusion:** `test.parquet` is a *mock* of the evaluation API input "
        "(one served batch: `row_id`, ids, `weight`, `is_scored`, features, but "
        "no `responder_6`). It exists for API/inference compatibility, **not** "
        "local model evaluation. Because there is no label, no R² / weighted-R² "
        "can be computed against it locally."
    )
    L.append("")
    L.append(
        "**The mock is hollow (real Kaggle data).** Beyond the missing label, "
        "the packaged `test.parquet` carries **no real feature values**: all 79 "
        "feature columns are `0.0` / `-0.0` (0/79 have any non-zero value; "
        "64/79 are non-null but literally zero). Only `weight` is populated and "
        "`is_scored` is `false` for all 39 rows (one row per symbol at "
        "`date_id=0`, `time_id=0`). So it is purely a schema/format example for "
        "the inference gateway and carries no modelling signal whatsoever."
    )
    L.append("")

    # 4. lags.parquet ------------------------------------------------------- #
    L.append("## 4. lags.parquet audit")
    L.append("")
    if lags.get("exists"):
        s = lags["stats"]
        pr = lags["presence"]
        L.append(f"- schema columns: **{lags['n_columns']}**")
        L.append(
            "- columns: `date_id`, `time_id`, `symbol_id`, "
            "`responder_0_lag_1` … `responder_8_lag_1`"
        )
        L.append(
            f"- lag columns present: {_yn(pr['has_all_lags'])} "
            f"({pr['n_lag_columns']}/9)"
        )
        L.append(f"- has date_id/time_id/symbol_id: {_yn(pr['all_id_columns'])}")
        L.append(f"- row count: **{lags['row_count']:,}**")
        L.append(
            f"- date_id range: {_fmt(s.get('date_id_min'))} … "
            f"{_fmt(s.get('date_id_max'))}; "
            f"time_id values: {_fmt(s.get('time_id_min'))} … "
            f"{_fmt(s.get('time_id_max'))}; "
            f"symbol_id n_unique: {_fmt(s.get('symbol_id_n_unique'))}"
        )
        if lags.get("head"):
            sample = lags["head"][0]
            keys = ["date_id", "time_id", "symbol_id"] + LAG_COLUMNS[:3]
            L.append(
                "- sample row: "
                + ", ".join(f"{k}={_fmt(sample.get(k))}" for k in keys)
                + ", …"
            )
    else:
        L.append("_lags.parquet not found in raw dir._")
    L.append("")
    L.append(
        "**Semantics:** at a new `date_id` D, the evaluation API delivers the "
        "responders from `date_id` D-1 (all `time_id`s of that prior date) as "
        "`responder_*_lag_1`, handed over at the **first `time_id` of D**. They "
        "are the only responder information available at inference time — the "
        "live API never reveals current-date responders."
    )
    L.append("")
    L.append(
        "**Local reconstruction:** for train-time experiments these lags can be "
        "rebuilt from `train.parquet` responders by shifting one `date_id` "
        "forward (responders of D-1 become features for D). This must avoid "
        "using current- or future-date responders, or it leaks the target. "
        "(V0 raw LGBM does not use lags at all.)"
    )
    L.append("")
    L.append(
        "**The packaged lags are illustrative (real Kaggle data).** Unlike the "
        "hollow `test.parquet`, the lag *values* here are real numbers, but they "
        "do **not** correspond to this `train.parquet`: the mock covers all 39 "
        "symbols at `date_id=0` whereas train's `date_id=0` only has ~8 symbols, "
        "and the lag values do not match train's date-0 responders. Treat the "
        "shipped `lags.parquet` as a synthetic format example; reconstruct real "
        "lags from train as described above."
    )
    L.append("")

    # 5. features.csv ------------------------------------------------------- #
    L.append("## 5. features.csv audit")
    L.append("")
    if feats.get("exists"):
        L.append(f"- row count (features): **{feats['row_count']}**")
        L.append(f"- columns: `{'`, `'.join(feats['columns'])}`")
        L.append(f"- tag columns: **{feats['n_tag_columns']}**")
        L.append("")
        L.append("| tag | true | false |")
        L.append("| --- | --- | --- |")
        for tag, counts in feats["tag_summary"].items():
            L.append(f"| {tag} | {counts['true']} | {counts['false']} |")
        L.append("")
        L.append("**Features grouped by tag (membership, tags overlap):**")
        L.append("")
        for tag, members in feats["groups"].items():
            shown = ", ".join(members) if members else "_(none)_"
            L.append(f"- `{tag}`: {shown}")
        L.append("")
        per_row = {r["feature"]: r["tags"] for r in feats["per_row"]}
        L.append("**Spot checks:**")
        for f in ["feature_09", "feature_10", "feature_11"]:
            if f in per_row:
                L.append(f"- `{f}` tags: {per_row[f] or '_(none)_'}")
        f20_31 = [
            f"feature_{i:02d}" for i in range(20, 32) if f"feature_{i:02d}" in per_row
        ]
        L.append(
            "- `feature_20`…`feature_31`: "
            + "; ".join(f"{f}→{per_row[f] or '∅'}" for f in f20_31)
        )
        if "feature_61" in per_row:
            tags61 = per_row["feature_61"]
            L.append(
                f"- `feature_61` tags: {tags61 or '_(none)_'} "
                f"({'has tags' if tags61 else 'has NO tags'})"
            )
    else:
        L.append("_features.csv not found in raw dir._")
    L.append("")
    L.append(
        "**Parity note (evgeniavolkova repo):** the tag columns group features "
        "that share anonymized structure, so they are a natural unit for "
        "feature-engineering parity — e.g. building per-tag aggregates or "
        "ensuring the same features feed the same derived columns. Do **not** "
        "infer real financial meaning from tags; they are anonymized metadata "
        "only."
    )
    L.append("")

    # 6. responders.csv ----------------------------------------------------- #
    L.append("## 6. responders.csv audit")
    L.append("")
    if resp.get("exists"):
        L.append(f"- row count (responders): **{resp['row_count']}**")
        L.append(f"- columns: `{'`, `'.join(resp['columns'])}`")
        L.append(f"- responder names: {', '.join(resp['names'])}")
        L.append(f"- tag columns: **{resp['n_tag_columns']}**")
        L.append("")
        L.append("| tag | true | false |")
        L.append("| --- | --- | --- |")
        for tag, counts in resp["tag_summary"].items():
            L.append(f"| {tag} | {counts['true']} | {counts['false']} |")
        L.append("")
        has_r6 = "responder_6" in resp["names"]
        L.append(
            f"- `responder_6` is exactly one metadata row: {_yn(has_r6)}."
        )
    else:
        L.append("_responders.csv not found in raw dir._")
    L.append("")
    L.append(
        "**Conclusion:** `responders.csv` is *metadata* describing the nine "
        "anonymized responders and their boolean tags. The actual responder "
        "*values* (including the `responder_6` target) live in "
        "`train.parquet`; this file is metadata only."
    )
    L.append("")

    # 7. Competition submission status ------------------------------------- #
    L.append("## 7. Competition submission status")
    L.append("")
    L.append(
        f"Verified with the Kaggle CLI on {AUDIT_TODAY} "
        "(`kaggle competitions list -s jane-street-real-time` and "
        "`kaggle competitions submissions -c "
        "jane-street-real-time-market-data-forecasting`):"
    )
    L.append("")
    L.append(
        f"- Kaggle-reported competition **deadline: {KAGGLE_DEADLINE}** "
        "(end of the forecasting/evaluation phase)."
    )
    L.append(
        "- The competition's *final submission deadline* for the initial phase "
        "was **Jan 13, 2025**; the forecasting phase then scored submitted "
        f"models against live market data until ~mid-2025 (deadline "
        f"{KAGGLE_DEADLINE}). The leaderboard's last scored submissions are "
        "dated 2025-06-16."
    )
    L.append(f"- Audit date: **{AUDIT_TODAY}** — well past the deadline.")
    L.append(
        "- The configured account *has entered* the competition "
        "(`userHasEntered=True`) but has **0 submissions** "
        "(`No submissions found`)."
    )
    L.append(
        "- **Empirical probe:** a CLI submit "
        "(`kaggle competitions submit -c "
        "jane-street-real-time-market-data-forecasting -f "
        "sample_submission.csv`) was attempted and the server **rejected it "
        "with `400 Bad Request` on `CreateSubmission`**; `submissions` still "
        "shows `No submissions found` afterwards. This is also a *code* "
        "competition (notebook-only submission), so a CSV file submit could "
        "not create a scored entry regardless. Submission is empirically closed."
    )
    L.append("")
    L.append("**Answers:**")
    L.append("")
    L.append("- Can we still train locally? **Yes.**")
    L.append("- Can we still run the evaluation API smoke locally? **Yes.**")
    L.append(
        "- Can we still get a new *official* leaderboard score? **No** — the "
        f"deadline ({KAGGLE_DEADLINE}) has passed, so official scoring is "
        "closed. Even if Kaggle accepts a late submission, it must not be "
        "assumed to affect the official leaderboard or any private rescore."
    )
    L.append("")

    # 8. Validation protocol implications ---------------------------------- #
    L.append("## 8. Implications for validation protocol")
    L.append("")
    L.append(
        "- Local validation **must** come from `train.parquet` (the only file "
        "with labels)."
    )
    L.append(
        "- A random split is **invalid**: `date_id`/`time_id` are chronological, "
        "so random folds leak future information into the past."
    )
    L.append("- `test.parquet` is **not** local validation (no label).")
    L.append(
        "- Lag features require careful leakage control: only D-1 (and earlier) "
        "responders may inform date D."
    )
    L.append(
        "- The panel structure (section 2) shapes the split design: the symbol "
        "universe grows over time (≈8→39), early dates have a feature warm-up "
        "with many nulls, and `weight` is a per-`(date_id, symbol_id)` scalar — "
        "so recent-window validation is cleaner than full history, and weighted "
        "metrics should respect the per-symbol-day weighting."
    )
    L.append("- The next stage should define a **split-protocol registry**:")
    L.append("  - A. `recent700_v200_g0` — recent 700 days, 200-day valid, gap 0")
    L.append("  - B. repo-style 2-fold CV")
    L.append("  - C. 200-day gap test")
    L.append("  - D. test-API smoke")
    L.append("")
    return "\n".join(L)


def write_docs(results: dict[str, Any], docs_out: str | Path) -> Path:
    out = Path(docs_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_docs(results), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit the semantics & schema of the Jane Street 2024 raw "
        "data files (read-only; no training)."
    )
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="outputs/data_semantics_audit")
    parser.add_argument(
        "--docs-out", default="docs/data/data_semantics_audit.md"
    )
    args = parser.parse_args(argv)

    try:
        results = run_audit(args.raw_dir)
        written = write_artifacts(results, args.out_dir)
        docs_path = write_docs(results, args.docs_out)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[js2024] ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"[js2024] Wrote audit docs   -> {docs_path}")
    for p in written:
        print(f"[js2024] Wrote audit artifact -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
