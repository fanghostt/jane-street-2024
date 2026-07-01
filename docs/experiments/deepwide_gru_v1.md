# Deep & Wide GRU — experiment log

## Motivation

The day-batch GRU's per-head architecture (`gru_mlp`) is a deep RNN stack → FC. The
deep recurrent path is good at carrying *temporal* structure across `time_id`s, but the
final prediction at time `t` still has to route the **current-step** signal through the
whole recurrent bottleneck. The "deep & wide" idea (Cheng et al., 2016) is to add a
shallow **wide** branch that sees the raw standardized inputs `x_t` directly, so the
model can memorize simple current-step feature→target relationships without competing
for recurrent capacity. The wide branch is **not new data** — the deep branch already
sees `x_t` — so any gain is from optimization path / inductive bias, and the expected
signature is a *small, stable* improvement with no variance blow-up, not a step change.

Two variants (see `src/js2024/modeling/gru.py` `_build_model`, `architecture` knob):

| architecture | per-head structure |
| --- | --- |
| `gru_mlp` (default) | RNN stack → FC. Unchanged from the public solution. |
| `deep_wide_gru` | deep repr **⊕** per-timestep wide MLP over `x_t` → fusion MLP → pred |
| `deep_wide_residual` | `deep_pred + wide_residual_scale · wide_pred` (few extra params) |

`deep_wide_residual` is the conservative baseline (one scaled scalar added to the deep
prediction); `deep_wide_gru` is the higher-capacity concat-fusion version. The training
loop, auxiliary-head protocol, and online inference update are **untouched** — this
isolates the architecture change. `gru_mlp` allocates no wide/fusion modules, so existing
models and configs are bit-for-bit unaffected.

## Method

Decisive metric is the **paired** online-incremental test R² delta vs the SOTA baseline
(`gru_marketroll_v1`) at the same seed:

    delta(arch, seed) = gru_incremental(arch) − gru_incremental(gru_marketroll_v1)

All arms inherit `gru_marketroll_v1`'s settings verbatim — market-avg + per-symbol
rolling features **on**, `aux_target_set=all9`, recent700 window, shared last-200 test
block `[1499, 1698]`, fp32 (`use_amp=false`) for comparability with the recorded
baseline. The **only** change is the `architecture` block. Seeds 42/43/44 (config copies
with `random_state` overridden, since the runner has no `--random-state` flag).

Wide/fusion shapes: `deepwide_gru_w256` uses `wide_hidden_sizes=[256,256]`; the fusion
MLP defaults to `hidden_sizes_linear/dropout_rates_linear` (`[500,300]`). `residual`
uses `wide_hidden_sizes=[128]`, `wide_residual_scale=0.1`.

Configs: `configs/deepwide_gru_{w128,w256,residual}.yaml`. A seed-42 sanity pass also
covered `w128` (dropped after one seed — see below). Heavy per-run artifacts are
gitignored; this file is the committed record.

## Status / verdict

**`deep_wide_gru` (w256) is a small, sign-stable positive; `deep_wide_residual` is a
loser.** w256 beats the baseline on the paired delta with **3/3 seeds positive**, mean
**+0.00043** (~+3.7% relative on a ~0.0117 baseline) — same order of magnitude as the
aux-set and market-feature wins. Crucially it shows the predicted clean signature: the
prediction std does **not** inflate (0.103/0.101/0.103 vs baseline 0.103/0.105/0.105 —
if anything tighter), and no seed diverges. `deep_wide_residual` is 0/3 (consistently
−0.0002): the single scaled scalar adds nothing and slightly hurts.

### Single-seed sanity (seed 42, all four)

| config | test R² | Δ vs baseline | best_epoch | pred_std |
| --- | ---: | ---: | ---: | ---: |
| gru_marketroll_v1 (baseline) | 0.011917 | — | 6 | 0.1033 |
| **deepwide_w256** | 0.012127 | +0.000210 | 6 | 0.1032 |
| deepwide_w128 | 0.011511 | −0.000406 | 8 | 0.1071 |
| deepwide_residual | 0.011533 | −0.000383 | 7 | 0.1070 |

`w128` scored *worst* of the three at seed 42 despite having fewer params than `w256`
(an overfit story would predict the opposite) — read as seed noise, and dropped to keep
the paired sweep cheap. `w256` and `residual` carried to 3 seeds.

### Paired 3-seed (42/43/44, decisive)

| config | n | mean Δ | std Δ | +seeds | mean R² | mean baseline R² | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **deepwide_w256** | 3 | **+0.000434** | 0.000381 | **3/3** | 0.012214 | 0.011780 | **keep** |
| deepwide_residual | 3 | −0.000196 | 0.000136 | 0/3 | 0.011584 | 0.011780 | drop |

Per-seed deltas — w256: +0.000210 / +0.000970 / +0.000122; residual: −0.000383 /
−0.000139 / −0.000065.

- **verdict:** `deep_wide_gru` (w256) is **worth pursuing**, not yet confirmed SOTA. The
  mean gain is real (3/3, clean variance) but modest and seed-43-weighted (the other two
  seeds are only +0.0001–0.0002, inside the seed std 0.00038). Treat as a candidate
  pending the wide-size ablation below. **Drop `deep_wide_residual`** (0/3).
- `GRUConfig`'s default stays `gru_mlp` (bit-equivalent to prior runs); only the deepwide
  configs opt in. `gru_marketroll_v1` remains the documented SOTA until w256 is confirmed.

## Risks watched (and what we saw)

- **More params → better valid but worse test?** No. We judged on incremental *test* R²
  only; w256's gain holds on the test block, and `best_epoch` matches baseline (6 at seed
  42) — no later-stopping / overfit-to-valid pattern.
- **Wide branch makes online update fragile?** No. `pred_std` is stable/tighter across all
  3 seeds; no tail blow-up (pred max ~7, same as baseline), no seed-divergent scores.

## Next ablations (not yet run)

To decide whether w256 is the right operating point and whether fusion (concat) beats the
residual shortcut on *capacity* rather than noise:

| ablation | question |
| --- | --- |
| `wide_hidden_sizes=[64,64]` | is a small wide branch enough? |
| `[128,128]` / `[256,256]` / `[512,512]` | where does it overfit? |
| `deep_wide_residual` scale ∈ {0.05, 0.1, 0.2} | is the shortcut just mis-scaled? |

Candidate to run the sweep under `use_amp=true` (paired across *all* arms) for ~1.5×
speedup, since it would be a self-contained A/B not tied to the fp32 baseline.

## Reproduce

```bash
# single configs (one seed each):
uv run js2024-run-experiment --config configs/deepwide_gru_w128.yaml     --variants incremental
uv run js2024-run-experiment --config configs/deepwide_gru_w256.yaml     --variants incremental
uv run js2024-run-experiment --config configs/deepwide_gru_residual.yaml --variants incremental

# paired baseline (same seed) for the delta:
uv run js2024-run-experiment --config configs/gru_marketroll_v1.yaml     --variants incremental

# other seeds: copy the config and override `random_state` (no --random-state flag yet).
```
