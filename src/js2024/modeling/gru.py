"""GRU model closer to evgeniavolkova/kagglejanestreet.

This implementation keeps the public solution's core neural-net protocol:
one date is one batch, rows are reshaped into ``symbols x time_id x features``,
four auxiliary responder heads are trained jointly with the final target, and
online fine-tuning uses a smaller learning rate.
"""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Any

import numpy as np
import polars as pl

from ..data.data import FEATURE_COLUMNS, TARGET_COLUMN, WEIGHT_COLUMN
from . import tracking
from .features import fit_feature_standardizer
from .metrics import weighted_zero_mean_r2

GRU_AUX_COLUMNS = ["responder_10", "responder_9", "responder_8", "responder_7"]

# Named auxiliary-target sets selectable via the `aux_target_set` config key. Each
# value is the ordered list of responder columns trained as auxiliary heads (a
# final linear head combines them into the target). `base4` reuses GRU_AUX_COLUMNS
# verbatim so the default reproduces the public-solution behaviour bit-for-bit.
# responder_9/10 are synthetic (see add_gru_aux_targets); the real responders are
# responder_0..8.
GRU_AUX_TARGET_SETS: dict[str, list[str]] = {
    "base4": list(GRU_AUX_COLUMNS),
    "target_family": ["responder_6", "responder_7", "responder_8", "responder_9", "responder_10"],
    "all9": [f"responder_{i}" for i in range(9)],
    "all11": [f"responder_{i}" for i in range(11)],
}


def resolve_gru_aux_targets(name: str) -> list[str]:
    """Map an ``aux_target_set`` name to its ordered auxiliary-responder list."""
    try:
        return list(GRU_AUX_TARGET_SETS[name])
    except KeyError:
        known = ", ".join(GRU_AUX_TARGET_SETS)
        raise ValueError(
            f"unknown aux_target_set {name!r}; choose from {known}"
        ) from None


def get_gru_feature_columns(include_time: bool = True) -> list[str]:
    """Public-solution-style raw inputs: all features except 09-11, plus time_id."""
    cols = [c for c in FEATURE_COLUMNS if c not in {"feature_09", "feature_10", "feature_11"}]
    if include_time:
        cols.append("time_id")
    return cols


def add_gru_aux_targets(df: pl.DataFrame) -> pl.DataFrame:
    """Add responder_9/10 auxiliary targets described in the public solution."""
    needed = {"responder_6", "responder_7", "responder_8", "symbol_id"}
    missing = sorted(needed.difference(df.columns))
    if missing:
        raise ValueError(f"Missing columns for GRU auxiliaries: {missing}")
    return df.with_columns(
        (
            pl.col("responder_8")
            + pl.col("responder_8").shift(-4).over("symbol_id")
        )
        .fill_null(0.0)
        .alias("responder_9"),
        (
            pl.col("responder_6")
            + pl.col("responder_6").shift(-20).over("symbol_id")
            + pl.col("responder_6").shift(-40).over("symbol_id")
        )
        .fill_null(0.0)
        .alias("responder_10"),
    )


def _weighted_r2_loss(pred, target, weight):
    denom = (weight * target.pow(2)).sum().clamp_min(1e-6)
    return (weight * (pred - target).pow(2)).sum() / denom


def _day_tensor(
    df_day: pl.DataFrame,
    feature_cols: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    *,
    target_col: str,
    weight_col: str,
    aux_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``X, aux, y, w`` as ``symbols x time x ...`` arrays for one date."""
    df_day = df_day.sort(["time_id", "symbol_id"])
    n_times = int(df_day.get_column("time_id").max()) + 1
    n_rows = df_day.height
    if n_rows % n_times != 0:
        raise ValueError(
            f"Cannot reshape date {df_day.get_column('date_id')[0]}: "
            f"{n_rows} rows is not divisible by n_times={n_times}"
        )
    n_symbols = n_rows // n_times

    x = df_day.select(feature_cols).to_numpy().astype(np.float32)
    x = np.nan_to_num((x - mean) / std, nan=0.0)
    aux = df_day.select(aux_cols).to_numpy().astype(np.float32)
    y = df_day.get_column(target_col).to_numpy().astype(np.float32)
    w = df_day.get_column(weight_col).to_numpy().astype(np.float32)

    x = x.reshape(n_times, n_symbols, len(feature_cols)).swapaxes(0, 1)
    aux = aux.reshape(n_times, n_symbols, len(aux_cols)).swapaxes(0, 1)
    y = y.reshape(n_times, n_symbols).swapaxes(0, 1)
    w = w.reshape(n_times, n_symbols).swapaxes(0, 1)
    return x, aux, y, w


SEQ_MODEL_TYPES = ("gru", "lstm", "transformer", "tcn")


def _make_backbone_layer(model_type: str, input_dim: int, hidden: int, layer_idx: int, params):
    """Build one sequence-backbone layer mapping ``(B, T, input_dim) -> (B, T, hidden)``.

    Every layer exposes the same ``forward(x, h) -> (out, h)`` contract the day-batch
    loop expects; non-recurrent backbones return ``h=None``. All backbones are
    strictly causal over the time axis so a step never sees future ``time_id``s.
    """
    import torch.nn as nn

    if model_type in ("gru", "lstm"):
        rnn_cls = nn.GRU if model_type == "gru" else nn.LSTM
        return _RNNLayer(rnn_cls(input_dim, hidden, num_layers=1, batch_first=True))
    if model_type == "transformer":
        return _TransformerLayer(input_dim, hidden, int(params.get("num_heads", 5)))
    if model_type == "tcn":
        return _TCNLayer(input_dim, hidden, int(params.get("kernel_size", 3)), dilation=2 ** layer_idx)
    raise ValueError(f"Unknown model_type {model_type!r}; expected one of {SEQ_MODEL_TYPES}")


def _seq_layer_classes():
    """Define backbone layer modules lazily (so importing gru.py needs no torch)."""
    import math

    import torch
    import torch.nn as nn

    class RNNLayer(nn.Module):
        def __init__(self, rnn: nn.Module) -> None:
            super().__init__()
            self.rnn = rnn

        def forward(self, x, h=None):
            return self.rnn(x, h)

    class TransformerLayer(nn.Module):
        """Causal self-attention block: project -> sinusoidal PE -> masked encoder."""

        def __init__(self, input_dim: int, hidden: int, num_heads: int) -> None:
            super().__init__()
            if hidden % num_heads != 0:
                raise ValueError(
                    f"transformer hidden={hidden} must be divisible by num_heads={num_heads}"
                )
            self.proj = nn.Linear(input_dim, hidden)
            self.encoder = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=num_heads,
                dim_feedforward=hidden * 2,
                dropout=0.0,
                batch_first=True,
            )

        def _positional_encoding(self, t: int, dim: int, device, dtype):
            pos = torch.arange(t, device=device, dtype=dtype).unsqueeze(1)
            div = torch.exp(
                torch.arange(0, dim, 2, device=device, dtype=dtype)
                * (-math.log(10000.0) / dim)
            )
            pe = torch.zeros(t, dim, device=device, dtype=dtype)
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
            return pe.unsqueeze(0)

        def forward(self, x, h=None):
            z = self.proj(x)
            t = z.shape[1]
            z = z + self._positional_encoding(t, z.shape[-1], z.device, z.dtype)
            mask = torch.triu(
                torch.full((t, t), float("-inf"), device=z.device, dtype=z.dtype),
                diagonal=1,
            )
            return self.encoder(z, src_mask=mask), None

    class TCNLayer(nn.Module):
        """Causal dilated 1-D conv block with a residual projection."""

        def __init__(self, input_dim: int, hidden: int, kernel_size: int, dilation: int) -> None:
            super().__init__()
            self.pad = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(
                input_dim, hidden, kernel_size, padding=self.pad, dilation=dilation
            )
            self.act = nn.ReLU()
            self.res = nn.Linear(input_dim, hidden) if input_dim != hidden else None

        def forward(self, x, h=None):
            z = self.conv(x.transpose(1, 2))
            if self.pad:
                z = z[:, :, : -self.pad]  # chomp right padding to stay causal
            z = self.act(z).transpose(1, 2)
            res = x if self.res is None else self.res(x)
            return z + res, None

    return RNNLayer, TransformerLayer, TCNLayer


_RNNLayer = _TransformerLayer = _TCNLayer = None


def _ensure_layer_classes() -> None:
    global _RNNLayer, _TransformerLayer, _TCNLayer
    if _RNNLayer is None:
        _RNNLayer, _TransformerLayer, _TCNLayer = _seq_layer_classes()


def _build_model(n_features: int, params: dict[str, Any], n_aux: int):
    import torch
    import torch.nn as nn

    _ensure_layer_classes()
    hidden_sizes = list(params["hidden_sizes"])
    dropout_rates = list(params["dropout_rates"])
    hidden_linear = list(params["hidden_sizes_linear"])
    dropout_linear = list(params["dropout_rates_linear"])
    model_type = str(params.get("model_type", "gru"))
    architecture = str(params.get("architecture", "gru_mlp"))
    # Wide/fusion MLP shapes (deep_wide_* only). Fall back to the deep FC head's
    # sizes when omitted so a config can opt in by setting only `architecture`.
    wide_hidden = list(params.get("wide_hidden_sizes") or hidden_linear)
    wide_dropout = list(params.get("wide_dropout_rates") or dropout_linear)
    fusion_hidden = list(params.get("fusion_hidden_sizes") or hidden_linear)
    fusion_dropout = list(params.get("fusion_dropout_rates") or dropout_linear)
    wide_residual_scale = float(params.get("wide_residual_scale", 0.1))

    def _mlp_layers(in_dim: int, hidden: list[int], dropouts: list[float]):
        """Build a [Linear, ReLU, Dropout]* stack; return (layers, output_dim)."""
        layers: list[nn.Module] = []
        prev = in_dim
        for h, dr in zip(hidden, dropouts):
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(float(dr)))
            prev = h
        return layers, prev

    class ModelRBase(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rnns = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            for i, hidden in enumerate(hidden_sizes):
                input_dim = n_features if i == 0 else hidden_sizes[i - 1]
                self.rnns.append(
                    _make_backbone_layer(model_type, input_dim, hidden, i, params)
                )
                self.dropouts.append(nn.Dropout(float(dropout_rates[i])))

            in_dim = hidden_sizes[-1] if hidden_sizes else n_features
            if architecture == "deep_wide_gru":
                # Wide branch: per-timestep MLP over the raw inputs -> wide repr.
                wide_layers, wide_out = _mlp_layers(n_features, wide_hidden, wide_dropout)
                self.wide = nn.Sequential(*wide_layers)
                # Fusion MLP over concat([deep_repr, wide_repr]) -> single pred.
                fusion_layers, fusion_out = _mlp_layers(
                    in_dim + wide_out, fusion_hidden, fusion_dropout
                )
                fusion_layers.append(nn.Linear(fusion_out, 1))
                self.fusion = nn.Sequential(*fusion_layers)
            else:
                # gru_mlp and deep_wide_residual both keep the original deep FC
                # head. Built identically so gru_mlp stays bit-for-bit unchanged.
                fc = []
                for i, hidden in enumerate(hidden_linear):
                    fc.append(nn.Linear(in_dim if i == 0 else hidden_linear[i - 1], hidden))
                    fc.append(nn.ReLU())
                    fc.append(nn.Dropout(float(dropout_linear[i])))
                fc.append(nn.Linear(hidden_linear[-1] if hidden_linear else in_dim, 1))
                self.fc = nn.Sequential(*fc)
                if architecture == "deep_wide_residual":
                    # Wide branch outputs a single prediction added to the deep one.
                    wide_layers, wide_out = _mlp_layers(
                        n_features, wide_hidden, wide_dropout
                    )
                    wide_layers.append(nn.Linear(wide_out, 1))
                    self.wide = nn.Sequential(*wide_layers)

        def forward(self, x, hidden=None):
            d, t, _ = x.shape
            if hidden is None:
                hidden = [None] * len(self.rnns)
            out = x
            next_hidden = []
            for i, layer in enumerate(self.rnns):
                out, h = layer(out, hidden[i])
                out = self.dropouts[i](out)
                next_hidden.append(h)
            if architecture == "deep_wide_gru":
                deep_flat = out.reshape(d * t, -1)
                wide_flat = self.wide(x.reshape(d * t, -1))
                fused = torch.cat([deep_flat, wide_flat], dim=1)
                return self.fusion(fused).reshape(d, t), next_hidden
            out = self.fc(out.reshape(d * t, -1)).reshape(d, t)
            if architecture == "deep_wide_residual":
                wide = self.wide(x.reshape(d * t, -1)).reshape(d, t)
                out = out + wide_residual_scale * wide
            return out, next_hidden

    class ModelR(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.heads = nn.ModuleList([ModelRBase() for _ in range(n_aux)])
            self.out = nn.Linear(n_aux, 1)

        def forward(self, x, hidden=None):
            if hidden is None:
                hidden = [None] * len(self.heads)
            aux_preds = []
            next_hidden = []
            for i, head in enumerate(self.heads):
                pred, h = head(x, hidden[i])
                aux_preds.append(pred.reshape(-1, 1))
                next_hidden.append(h)
            aux = __import__("torch").cat(aux_preds, dim=1)
            y = self.out(aux).reshape(x.shape[0], x.shape[1])
            aux = aux.reshape(x.shape[0], x.shape[1], n_aux)
            return y, aux, next_hidden

    return ModelR()


GRU_DEFAULT_PARAMS: dict[str, Any] = {
    "model_type": "gru",
    "num_heads": 5,        # transformer: attention heads (hidden must divide evenly)
    "kernel_size": 3,      # tcn: causal conv kernel width
    "architecture": "gru_mlp",  # gru_mlp | deep_wide_gru | deep_wide_residual
    "wide_hidden_sizes": None,   # deep_wide_*: wide MLP sizes (None -> linear sizes)
    "wide_dropout_rates": None,
    "fusion_hidden_sizes": None, # deep_wide_gru: fusion MLP sizes (None -> linear sizes)
    "fusion_dropout_rates": None,
    "wide_residual_scale": 0.1,  # deep_wide_residual: scale on the wide prediction
    "hidden_sizes": [500],
    "dropout_rates": [0.3],
    "hidden_sizes_linear": [500, 300],
    "dropout_rates_linear": [0.2, 0.1],
    "lr": 5e-4,
    "lr_refit": 3e-4,
    "epochs": 1000,
    "early_stopping_patience": 1,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "use_amp": False,      # bf16 autocast (+TF32) on CUDA; ~1.5x faster, off by default.
}


class GRUEstimator:
    """Day-batch GRU with auxiliary responders and online fine-tuning."""

    def __init__(
        self,
        feature_cols: list[str],
        *,
        target_col: str = TARGET_COLUMN,
        weight_col: str = WEIGHT_COLUMN,
        params: dict[str, Any] | None = None,
        aux_cols: list[str] | None = None,
        random_state: int = 42,
        device: str = "auto",
    ) -> None:
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.weight_col = weight_col
        # Auxiliary responder columns trained as extra heads; defaults to the
        # public-solution base4 set so callers that omit it keep prior behaviour.
        self.aux_cols = list(aux_cols) if aux_cols is not None else list(GRU_AUX_COLUMNS)
        self.params = {**GRU_DEFAULT_PARAMS, **(params or {})}
        self.random_state = int(random_state)
        self.device = device
        self._model = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self.best_epoch: int = 0

    def _torch_device(self):
        import torch

        if self.device == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _autocast(self):
        """bf16 autocast on CUDA when ``use_amp`` is set, else a no-op context."""
        import contextlib

        import torch

        if self.params.get("use_amp") and self._torch_device().type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _seed(self) -> None:
        import torch

        torch.manual_seed(self.random_state)
        torch.cuda.manual_seed_all(self.random_state)
        np.random.seed(self.random_state)

    @staticmethod
    def _iter_days(df: pl.DataFrame):
        df = df.sort(["date_id", "time_id", "symbol_id"])
        for _, df_day in df.group_by("date_id", maintain_order=True):
            yield df_day

    def _day(self, df_day: pl.DataFrame):
        return _day_tensor(
            df_day,
            self.feature_cols,
            self._mean,
            self._std,
            target_col=self.target_col,
            weight_col=self.weight_col,
            aux_cols=self.aux_cols,
        )

    def _train_day(self, df_day: pl.DataFrame, optimizer) -> float:
        import torch

        device = self._torch_device()
        x, aux, y, w = self._day(df_day)
        xb = torch.from_numpy(x).to(device)
        auxb = torch.from_numpy(aux).to(device)
        yb = torch.from_numpy(y).to(device)
        wb = torch.from_numpy(w).to(device)
        self._model.train()
        optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            out_y, out_aux, _ = self._model(xb, None)
            loss = _weighted_r2_loss(out_y.flatten(), yb.flatten(), wb.flatten())
            for i in range(len(self.aux_cols)):
                loss = loss + _weighted_r2_loss(
                    out_aux[:, :, i].flatten(), auxb[:, :, i].flatten(), wb.flatten()
                )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self._model.parameters(), max_norm=float(self.params["grad_clip"])
        )
        optimizer.step()
        return float(loss.detach().cpu())

    def _predict_day_with_model(self, model, df_day: pl.DataFrame) -> np.ndarray:
        import torch

        device = self._torch_device()
        x, _, _, _ = self._day(df_day)
        xb = torch.from_numpy(x).to(device)
        model.eval()
        with torch.no_grad(), self._autocast():
            preds, _, _ = model(xb, None)
        return preds.float().swapaxes(0, 1).reshape(-1).detach().cpu().numpy().astype(np.float64)

    def _score(self, df: pl.DataFrame, *, online_update: bool) -> float:
        import torch

        model = deepcopy(self._model) if online_update else self._model
        device = self._torch_device()
        model.to(device)
        optimizer = None
        if online_update and float(self.params["lr_refit"]) > 0:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(self.params["lr_refit"]),
                weight_decay=float(self.params["weight_decay"]),
            )

        preds_parts = []
        y_parts = []
        w_parts = []
        for df_day in self._iter_days(df):
            preds_parts.append(self._predict_day_with_model(model, df_day))
            y_parts.append(df_day.get_column(self.target_col).to_numpy().astype(np.float64))
            w_parts.append(df_day.get_column(self.weight_col).to_numpy().astype(np.float64))
            if optimizer is not None:
                # Online validation update uses responder_6 loss only, matching the
                # public solution's inference update.
                x, _, y, w = self._day(df_day)
                xb = torch.from_numpy(x).to(device)
                yb = torch.from_numpy(y).to(device)
                wb = torch.from_numpy(w).to(device)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    out_y, _, _ = model(xb, None)
                    loss = _weighted_r2_loss(out_y.flatten(), yb.flatten(), wb.flatten())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=float(self.params["grad_clip"])
                )
                optimizer.step()
        return weighted_zero_mean_r2(
            np.concatenate(y_parts), np.concatenate(preds_parts), np.concatenate(w_parts)
        )

    def fit(self, df_train: pl.DataFrame, df_valid: pl.DataFrame | None = None) -> "GRUEstimator":
        import torch

        self._seed()
        if self.params.get("use_amp"):
            # TF32 matmuls/cuDNN are a free win alongside bf16 autocast on Ampere+.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self._mean, self._std = fit_feature_standardizer(df_train, self.feature_cols)
        self._model = _build_model(
            len(self.feature_cols), self.params, len(self.aux_cols)
        ).to(self._torch_device())
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=float(self.params["lr"]),
            weight_decay=float(self.params["weight_decay"]),
        )

        best_score = -np.inf
        best_state = None
        bad = 0
        patience = int(self.params["early_stopping_patience"])
        for epoch in range(1, int(self.params["epochs"]) + 1):
            epoch_t0 = time.perf_counter()
            losses = [self._train_day(df_day, optimizer) for df_day in self._iter_days(df_train)]
            train_secs = time.perf_counter() - epoch_t0
            if df_valid is not None and df_valid.height > 0:
                score = self._score(df_valid, online_update=True)
            else:
                score = -float(np.mean(losses))
            epoch_secs = time.perf_counter() - epoch_t0
            train_loss = float(np.mean(losses))
            print(
                f"[js2024] {self.params['model_type']} epoch={epoch} "
                f"loss={train_loss:.6f} valid_R2={score:.6f} "
                f"time={epoch_secs:.1f}s"
            )
            tracking.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "valid_R2": score,
                    "epoch_secs": epoch_secs,
                    "train_secs": train_secs,
                },
                step=epoch,
            )
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().clone() for k, v in self._model.state_dict().items()}
                self.best_epoch = epoch
                bad = 0
            else:
                bad += 1
                if bad >= patience + 1:
                    break
        if best_state is not None:
            self._model.load_state_dict(best_state)
        return self

    def update(self, df_new: pl.DataFrame) -> "GRUEstimator":
        if self._model is None:
            raise RuntimeError("GRUEstimator.update called before fit().")
        if df_new.height == 0 or float(self.params["lr_refit"]) <= 0:
            return self
        import torch

        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=float(self.params["lr_refit"]),
            weight_decay=float(self.params["weight_decay"]),
        )
        for df_day in self._iter_days(df_new):
            x, _, y, w = self._day(df_day)
            xb = torch.from_numpy(x).to(self._torch_device())
            yb = torch.from_numpy(y).to(self._torch_device())
            wb = torch.from_numpy(w).to(self._torch_device())
            self._model.train()
            optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                out_y, _, _ = self._model(xb, None)
                loss = _weighted_r2_loss(out_y.flatten(), yb.flatten(), wb.flatten())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self._model.parameters(), max_norm=float(self.params["grad_clip"])
            )
            optimizer.step()
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("GRUEstimator.predict called before fit().")
        parts = [self._predict_day_with_model(self._model, df_day) for df_day in self._iter_days(df)]
        return np.concatenate(parts)
