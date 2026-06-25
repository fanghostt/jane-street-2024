# Jane Street 2024 — Rebuild

A clean, reproducible financial ML pipeline rebuilt from scratch for the Kaggle
[Jane Street Real-Time Market Data Forecasting (2024)](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting)
competition.

## Project goals

- Rebuild a maintainable, reproducible financial ML pipeline **from zero**.
- **V0 (current):** a single LightGBM baseline — the competition metric, a
  time-aware date split, lean data loading, and an end-to-end training CLI.
- **Later milestones** (not implemented yet) target parity with
  [`evgeniavolkova/kagglejanestreet`](https://github.com/evgeniavolkova/kagglejanestreet):
  GRU models, auxiliary targets, online learning, and ensembling.

This V0 is deliberately *not* a copy of that repo — it is a foundation we can
extend toward it.

## Data

The competition data is **not** shipped with this repo and is never committed.
Nothing under `data/` (raw downloads, the Kaggle zip, smoke parquets) and no
`kaggle.json` is ever added to git — `data/`, `models/`, and `outputs/` artifacts
are gitignored (directory structure is preserved via `.gitkeep`).

The expected schema used by V0:

- IDs: `date_id`, `time_id`, `symbol_id`
- Features: `feature_00` … `feature_78` (79 columns)
- Weight: `weight`
- Target: `responder_6`

Kaggle ships `train.parquet` as a *partitioned directory*
(`data/raw/train.parquet/partition_id=*/part-0.parquet`). The loader handles both
that directory layout and a single `.parquet` file transparently.

## Data setup

### 1. Configure Kaggle credentials (local only)

The download tool defers entirely to the official `kaggle` CLI and never reads,
prints, or stores your credentials itself. Provide them in **one** of two ways:

- Save an API token to `~/.kaggle/kaggle.json` (Kaggle → *Settings* →
  *Create New Token*), or
- Export `KAGGLE_USERNAME` and `KAGGLE_KEY` in your environment.

You must also **accept the competition rules** on the Kaggle website for
`jane-street-real-time-market-data-forecasting`, or downloads will be rejected.

### 2. Download the competition data

```bash
uv run js2024-download-data \
  --competition jane-street-real-time-market-data-forecasting \
  --out-dir data/raw
```

Downloads + extracts into `data/raw` (existing files are kept unless you pass
`--force`) and then runs the contract check automatically.

### 3. Check the raw data contract

```bash
uv run js2024-data-check --raw-dir data/raw
```

Verifies the required files (`train.parquet`, `lags.parquet`, `features.csv`,
`responders.csv`) exist and that `train` exposes the expected schema. Exits
non-zero if anything is missing.

### 4. Make a small smoke dataset

```bash
uv run js2024-make-smoke-data \
  --train-path data/raw/train.parquet \
  --out-path data/interim/train_smoke.parquet \
  --start-date-id 1200 --end-date-id 1210
```

Carves a tiny date range out of the real train data for fast local iteration.
The smoke parquet lives under `data/interim` and is **not** committed. (Wiring a
smoke-data baseline config is left for a later PR; this PR only provides the
command.)

### 5. Profile the data

```bash
uv run js2024-data-profile \
  --train-path data/raw/train.parquet \
  --out outputs/reports/data_profile.md \
  --start-date-id 1200 --end-date-id 1698
```

Writes a markdown profile (row count, date range, symbol/time cardinality,
target & weight distributions, top-30 features by missing ratio). All stats are
computed lazily; restrict the date range if a full scan is too slow.

### 6. Data semantics audit

```bash
uv run js2024-data-semantics-audit \
  --raw-dir data/raw \
  --out-dir outputs/data_semantics_audit \
  --docs-out docs/data/data_semantics_audit.md
```

Read-only audit that clarifies the semantics of `train` / `test` / `lags` /
`features` / `responders`. Key points it documents:

- `test.parquet` is a mock of the evaluation API input and has **no**
  `responder_6` label — it is **not** a local validation set.
- Local validation must come from time splits of `train.parquet` (the only file
  with labels).
- `lags.parquet` holds `responder_0..8` lagged by one `date_id`, served at the
  first `time_id` of the succeeding date.
- `features.csv` / `responders.csv` are anonymized metadata (boolean tags), not
  label values.

The committed artifact is the markdown doc; the per-file CSV/JSON dumps under
`outputs/data_semantics_audit/` are git-ignored.

## Install

This project is managed with [uv](https://docs.astral.sh/uv/). From the project
root:

```bash
uv sync
```

This creates `.venv/`, installs all dependencies (pinned via `uv.lock`), and
installs the `js2024` package itself in editable mode — so `import js2024` works
without setting `PYTHONPATH`. The `dev` dependency group (which includes
`pytest`) is installed by default.

`pyproject.toml` + `uv.lock` are the single source of truth for dependencies.

## Smoke run (validate the loop first)

Before training on the full 11.5 GB dataset, sanity-check the end-to-end loop on
the tiny smoke parquet. First generate it (see [Data setup](#4-make-a-small-smoke-dataset)):

```bash
uv run js2024-make-smoke-data \
  --train-path data/raw/train.parquet \
  --out-path data/interim/train_smoke.parquet \
  --start-date-id 1200 --end-date-id 1210
```

Then run the baseline against the smoke config:

```bash
uv run js2024-train-lgbm --config configs/lgbm_v0_smoke.yaml
```

This trains a small (100-tree) model on a handful of days purely to confirm the
data → split → train → score → artifact loop works. **The smoke score is not a
real result** — with only a few train days it is essentially noise (near zero or
slightly negative). The generated `model` / `oof` / `report` are gitignored and
must not be committed.

## Run the baseline

From the project root, once `data/raw/train.parquet` is in place. The **formal
baseline starts from `recent700`** (trains on `date_id >= 700`, the start point
we use for later parity with `evgeniavolkova/kagglejanestreet`):

```bash
uv run js2024-train-lgbm --config configs/lgbm_v0_recent700.yaml
```

`configs/lgbm_v0.yaml` is the full-data config (no `start_date_id` bound — heavier,
prefer `recent700` first):

```bash
uv run js2024-train-lgbm --config configs/lgbm_v0.yaml
# or, equivalently:
uv run python -m js2024.modeling.train_lgbm --config configs/lgbm_v0.yaml
```

Outputs (all **gitignored — do not commit**):

- `models/lgbm_v0.txt` — trained LightGBM booster
- `outputs/oof/lgbm_v0_valid_predictions.parquet` — validation predictions
- `outputs/reports/lgbm_v0_report.md` — run report (metric, distributions,
  top-30 feature importance)

All V0 experiment results (baseline, split stability, incremental-vs-full, retrain
cadence sweep) are recorded in the single committed log
[`docs/experiments/lgbm_v0.md`](docs/experiments/lgbm_v0.md). Single-run artifacts
go to `outputs/`; multi-run experiment artifacts go to a separate top-level
`experiments/` directory (both gitignored).

If the data is missing, the CLI prints a clear error pointing to
`data/raw/train.parquet` and exits non-zero — it does not crash or fabricate
results.

### Split-stability experiments

To check that the recent700 baseline R² is not a single-split artifact, run the
same raw-feature model across a grid of `valid_days` × `gap_days` (the big train
frame is loaded once and reused). All artifacts go under `experiments/` (gitignored).

```bash
uv run js2024-run-lgbm-split-experiments \
  --base-config configs/lgbm_v0_recent700.yaml \
  --valid-days 100,200,300 --gap-days 0,5,20 \
  --out-dir experiments/split_experiments/lgbm_v0_recent700
```

Useful flags: `--dry-run` (print the grid only), `--limit N` (first N combos),
`--n-estimators` / `--early-stopping-rounds` (quick-debug overrides). The summary is
folded into [`docs/experiments/lgbm_v0.md`](docs/experiments/lgbm_v0.md) §2.

### Incremental vs full (walk-forward)

Compare a statically-trained LightGBM (**full**) against three **incremental**
update strategies as the model walks a *fixed* trailing test block (the last
`test_days` date_ids, default 200):

- **refit** — daily `Booster.refit` of leaf values on each revealed day;
- **continue** — daily continued boosting (add `continue_rounds` trees);
- **retrain** — expanding retrain from scratch on all data so far (coarse cadence).

This is the leakage-clean LightGBM analog of the "with vs without online learning"
comparison in
[`evgeniavolkova/kagglejanestreet`](https://github.com/evgeniavolkova/kagglejanestreet).
All variants share one training region and are scored on the *same* test block, so
the numbers are directly comparable.

```bash
uv run js2024-run-incremental-vs-full \
  --config configs/lgbm_v0_incremental.yaml \
  --methods refit,continue,retrain \
  --out-dir experiments/incremental_vs_full/lgbm_v0

# or sweep retrain cadence:
uv run js2024-run-incremental-vs-full \
  --config configs/lgbm_v0_incremental.yaml --retrain-cadences 100,50,25
```

The early-stopping holdout is carved from the **train tail**, never the test block,
so the "full" number is leakage-clean (and may differ slightly from the recent700
baseline, which uses the last-200 block as its `eval_set`). The engine is
model-agnostic (`Estimator` protocol: `fit` / `update` / `predict`) so a future GRU
can slot in behind the same API. Useful flags: `--methods`, `--retrain-cadences`,
`--dry-run`, `--test-days`, `--cadence`, `--n-estimators`. Results (incl. the finding
that only **retrain** beats static training) are recorded in
[`docs/experiments/lgbm_v0.md`](docs/experiments/lgbm_v0.md) §3–4.

## Tests

```bash
uv run pytest
```

Tests construct tiny in-memory/temp parquet fixtures and never train a large
model or require the real competition data.

## The metric

The competition uses a **sample-weighted, zero-mean R²**:

```
R2 = 1 - sum(w_i * (y_i - yhat_i)^2) / sum(w_i * y_i^2)
```

The denominator uses `y_i^2`, not variance around a weighted mean, so this is
**not** `sklearn.metrics.r2_score`. A constant-zero prediction scores exactly 0;
a perfect prediction scores 1; bad predictions go negative. See
`src/js2024/modeling/metrics.py` and `notebooks/01_metric_and_split.md`.

## Repository layout

```
pyproject.toml  # project metadata, deps, console script, pytest config
uv.lock         # pinned dependency lockfile (commit this)
configs/        # YAML run configs
data/           # raw / interim / features (gitignored)
models/         # trained models (gitignored)
outputs/        # single-run artifacts: oof, reports, submissions (gitignored)
experiments/    # multi-run experiment artifacts (gitignored)
notebooks/      # markdown notes (metric & split derivation)
src/js2024/     # package, grouped into subpackages:
  data/         #   loading, contract checks, profiling, download, semantics audit
  modeling/     #   config, features, metrics, validation, estimators, training
  runners/      #   experiment CLIs (split experiments, incremental-vs-full)
tests/          # pytest suite
```

Console scripts (defined in `pyproject.toml`) are unchanged by the layout — e.g.
`uv run js2024-train-lgbm`, `uv run js2024-run-incremental-vs-full`.

## V0 acceptance criteria

- [x] Tests pass (`uv run pytest`)
- [x] Competition metric implemented correctly
- [x] Time-aware date split implemented correctly
- [x] CLI gives a clear error when data is missing (no crash, no fake results)
- [ ] With data present: produces `model` / `oof` / `report` (run once data is placed)

## Explicitly not implemented yet

- GRU / neural models
- Auxiliary targets
- Online learning
- Ensembling
- Kaggle inference gateway / submission packaging
- Full `evgeniavolkova/kagglejanestreet` parity
