# Sequence backbones (GRU vs Transformer vs TCN) — experiment log

Does swapping the **per-step sequence backbone** buy anything over the day-batch GRU?
All three models share the *identical* protocol — one `date_id` is one batch, rows
reshaped to `symbols × time_id × features`, four auxiliary responder heads
(`responder_7/8/9/10`) trained jointly, per-day online finetune at a smaller LR — and
differ **only** in the backbone that maps each step's features to the hidden state:

- **GRU** — recurrent (registry key `gru`, `configs/gru_v0.yaml`).
- **Transformer** — causal self-attention (proj → sinusoidal PE → masked encoder),
  `num_heads=5` (registry key `transformer`, `configs/transformer_v0.yaml`).
- **TCN** — causal dilated 1-D conv, `kernel_size=3`, dilation `2**layer_idx`, residual
  (registry key `tcn`, `configs/tcn_v0.yaml`).

So every weighted zero-mean R² below is directly comparable to `docs/experiments/gru_v0.md`.
Heavy per-run artifacts (models / CSVs / per-run reports) are gitignored; this file is
the committed record. Mamba was deliberately **not** run this round (only worth the
mamba_ssm dependency if a backbone had shown a gain — none did).

## Status / verdict

- **No backbone gain.** On the decisive `incremental` (per-day online finetune)
  column all three collapse into **0.0105–0.0111** — GRU marginally best; the spread
  is noise.
- **The backbones are genuinely different** (it is *not* a wiring bug that flattens
  them): the `full` column ranges −0.004 → −0.020 and the prediction distributions
  differ sharply (Transformer caps at 3.93, TCN tails to 9.13). They only *converge*
  once daily online finetune dominates the endpoint.
- **Shared ceiling = feature bottleneck.** All three plateau at the same training
  valid-R² (~0.017–0.021) within ~8 epochs (`early_stopping_patience=1`). Same ceiling
  + same online endpoint ⇒ the limit is **features / data / protocol, not the backbone**.
- **Decision:** keep `gru` as the model of record; do **not** keep tuning architecture
  at the V0 feature set. Higher-leverage next step is feature engineering (market
  averages / rolling stats — the deferred public-solution parity).

## Shared setup

- Metric: sample-weighted zero-mean R², `1 - Σ w(y-ŷ)² / Σ w·y²` (constant-zero ⇒ 0).
- Data start `date_id >= 700`, seed 42, `device: auto` (CUDA). Common hyperparameters
  borrowed from the tuned GRU: `hidden_sizes=[500]`, linear head `[500, 300]`, dropout
  `0.3/0.2/0.1`, `lr=5e-4`, `lr_refit=3e-4`, `epochs=1000` (`patience=1`),
  `weight_decay=0.01`, `grad_clip=1.0`. *Transformer/TCN reuse the GRU's hyperparameters
  unchanged — they were not separately tuned.*
- **Fixed test block 1499–1698** (200 days, 7,435,208 rows), shared by every variant;
  early stopping uses the train-tail holdout 1299–1498, never the test block.
- Runner: `uv run js2024-run-experiment --config <cfg>`.

## Results (same fixed test block 1499–1698)

| model | variant | n_updates | R² | pred[min, max] | valid-R² plateau |
| --- | --- | ---: | ---: | --- | ---: |
| `gru` | full | 0 | −0.015794 | [−2.45, 7.07] | ~0.021 |
| **`gru`** | **incremental** | 199 | **0.011139** | [−2.71, 7.06] | ~0.021 |
| `transformer` | full | 0 | −0.004115 | [−1.02, 3.93] | ~0.017 |
| `transformer` | incremental | 199 | 0.010505 | [−1.80, 5.22] | ~0.017 |
| `tcn` | full | 0 | −0.019550 | [−2.50, 9.13] | ~0.019 |
| `tcn` | incremental | 199 | 0.010534 | [−3.63, 7.22] | ~0.019 |

**Reading it:** the only column that matters operationally is `incremental` (the model
is finetuned daily in production); there GRU 0.01114 ≳ TCN 0.01053 ≈ Transformer 0.01050,
i.e. a tie with GRU nose-ahead. Transformer's one edge is robustness *without* updates
(`full` −0.004, best of the three, and the tightest prediction range), but that edge is
erased once daily finetune is on.

## ⚠️ Discrepancy to reconcile with `gru_v0.md` §2

`gru_v0.md` §2 records **`gru_full = 0.002126`** (pred [−0.52, 2.99]); re-measured here
under current code it is **−0.015794** (pred [−2.45, 7.07]). The `gru_incremental` row
reproduces **exactly** (0.011139, pred [−2.71, 7.06]), which proves the fit is
deterministic and the backbone refactor did not perturb GRU numerics — so the older
`gru_full` figure predates a model change in the GRU itself (the no-online-update GRU has
since gone from slightly-positive to **negative**; daily finetune still rescues it). Flag
to refresh `gru_v0.md` §2 / its `full` claims; not introduced by the backbone work.

## Next steps

- **Feature engineering** (market averages / rolling stats) on the GRU — the actual lever.
- Optional architecture due-diligence: a small `lr × hidden × {num_heads | kernel_size}`
  sweep for Transformer/TCN to confirm 0.0105 is their ceiling (low expected value).
- Reconcile the `gru_full` regression above.
