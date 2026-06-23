# 00 — Exploratory Data Analysis (template)

> Notes template. Once `data/raw/train.parquet` is in place, work through each
> section with polars and record findings here. No real numbers yet.

## 0. Setup

```python
import polars as pl
from js2024.data import load_train_data, get_default_columns

df = load_train_data("data/raw/train.parquet")  # ids + features + weight + target
df.shape
```

## 1. Field / schema check

- Confirm presence of `date_id`, `time_id`, `symbol_id`.
- Confirm `feature_00` … `feature_78` (79 features).
- Confirm `weight` and `responder_6`.
- Dtypes per column; any unexpected strings/categoricals?

## 2. `responder_6` (target) distribution

- mean / std / min / max / quantiles.
- Histogram; is it roughly symmetric and zero-centered?
- Tail behaviour / clipping.

## 3. `weight` distribution

- mean / std / min / max.
- Fraction of zero-weight rows (they don't contribute to the metric).
- Correlation of weight with target magnitude.

## 4. Rows per `date_id`

- Count of distinct `date_id`.
- Rows-per-date over time: stable? trending? regime breaks?
- This informs how many trailing days to hold out for validation.

## 5. Rows per `symbol_id`

- Number of distinct symbols.
- Symbols entering/leaving over time (panel is unbalanced).
- Rows per symbol.

## 6. Missing-value ratio

- Per-feature NaN fraction.
- Do missing patterns cluster by date or symbol?
- (V0 leaves NaNs for LightGBM to handle natively.)

## 7. Takeaways for modeling

- Chosen `valid_days` / `gap_days`.
- Features that look degenerate / constant.
- Anything that motivates feature engineering in a later milestone.
