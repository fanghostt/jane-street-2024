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

From the project root, once `data/raw/train.parquet` is in place:

```bash
uv run js2024-train-lgbm --config configs/lgbm_v0.yaml
# or, equivalently:
uv run python -m js2024.train_lgbm --config configs/lgbm_v0.yaml
```

Outputs:

- `models/lgbm_v0.txt` — trained LightGBM booster
- `outputs/oof/lgbm_v0_valid_predictions.parquet` — validation predictions
- `outputs/reports/lgbm_v0_report.md` — run report (metric, distributions,
  top-30 feature importance)

If the data is missing, the CLI prints a clear error pointing to
`data/raw/train.parquet` and exits non-zero — it does not crash or fabricate
results.

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
`src/js2024/metrics.py` and `notebooks/01_metric_and_split.md`.

## Repository layout

```
pyproject.toml  # project metadata, deps, console script, pytest config
uv.lock         # pinned dependency lockfile (commit this)
configs/        # YAML run configs
data/           # raw / interim / features (gitignored)
models/         # trained models (gitignored)
outputs/        # oof predictions, reports, submissions (gitignored)
notebooks/      # markdown notes (EDA, metric & split)
src/js2024/     # package: metrics, validation, data, features, config, training
tests/          # pytest suite
```

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
