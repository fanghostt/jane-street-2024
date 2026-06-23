# 01 — Metric & Time Split

## 1. Zero-mean weighted R²

The competition scores predictions of `responder_6` with:

```
R2 = 1 - sum(w_i * (y_i - yhat_i)^2) / sum(w_i * y_i^2)
```

Key points:

- **Weighted.** Each row contributes proportionally to its `weight`. Zero-weight
  rows do not affect the score.
- **Zero-mean.** The denominator is `sum(w * y^2)`, i.e. error measured against a
  *constant-zero* prediction — **not** the variance around a weighted mean. This
  is why we cannot use `sklearn.metrics.r2_score`.
- **Reference points:**
  - Perfect prediction → `R2 = 1`.
  - Constant zero prediction → `R2 = 0`.
  - Worse than zero (e.g. wrong sign) → `R2 < 0`.

Because the baseline is "predict 0", any signal must beat zero on a
weighted-squared-error basis to score positive. Implemented in
`src/js2024/metrics.py::weighted_zero_mean_r2`.

## 2. Why we cannot split randomly

This is a time series of market data:

- Rows are ordered by `date_id` (and `time_id` within a day).
- Targets are autocorrelated across nearby times and dates.
- A random split would place future rows in the training set, leaking
  information and producing optimistic, non-reproducible validation scores.

We therefore split **strictly by `date_id`**: train on earlier dates, validate on
later dates.

## 3. Holdout split (V0)

The last `valid_days` distinct `date_id`s become the validation fold; everything
before is training.

```
[ train_start ............ train_end ]   [ valid_start ... valid_end ]
```

Implemented in `build_holdout_split`. Example (`valid_days=200`, `gap_days=0`,
dates `0..999`): valid = `800..999`, train = `0..799`.

## 4. Gap split

Because targets near the train/valid boundary are correlated, we can drop
`gap_days` dates between the two folds so validation more honestly reflects
out-of-sample performance:

```
[ train ... train_end ]  (gap_days dropped)  [ valid_start ... valid_end ]
```

With `gap_days=g`, `train_end = valid_start - g - 1`.

## 5. Toward repo parity (later)

Future milestones will likely move beyond a single holdout to:

- **Rolling / expanding-window** cross-validation by date.
- **Purged & embargoed** splits around fold boundaries.
- Walk-forward evaluation that mirrors the competition's online setting.

These belong to the parity milestone; V0 ships only the single date-based
holdout (with optional gap).
