"""GRU model closer to evgeniavolkova/kagglejanestreet.

This implementation keeps the public solution's core neural-net protocol:
one date is one batch, rows are reshaped into ``symbols x time_id x features``,
four auxiliary responder heads are trained jointly with the final target, and
online fine-tuning uses a smaller learning rate.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import polars as pl

from ..data.data import FEATURE_COLUMNS, TARGET_COLUMN, WEIGHT_COLUMN
from .features import fit_feature_standardizer
from .metrics import weighted_zero_mean_r2

EVGENIAVOLKOVA_AUX_COLUMNS = ["responder_10", "responder_9", "responder_8", "responder_7"]


def get_gru_evgeniavolkova_feature_columns(include_time: bool = True) -> list[str]:
    """Public-solution-style raw inputs: all features except 09-11, plus time_id."""
    cols = [c for c in FEATURE_COLUMNS if c not in {"feature_09", "feature_10", "feature_11"}]
    if include_time:
        cols.append("time_id")
    return cols


def add_gru_evgeniavolkova_aux_targets(df: pl.DataFrame) -> pl.DataFrame:
    """Add responder_9/10 auxiliary targets described in the public solution."""
    needed = {"responder_6", "responder_7", "responder_8", "symbol_id"}
    missing = sorted(needed.difference(df.columns))
    if missing:
        raise ValueError(f"Missing columns for GRU evgeniavolkova auxiliaries: {missing}")
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


class _GRUBase:
    pass


def _build_model(n_features: int, params: dict[str, Any]):
    import torch.nn as nn

    hidden_sizes = list(params["hidden_sizes"])
    dropout_rates = list(params["dropout_rates"])
    hidden_linear = list(params["hidden_sizes_linear"])
    dropout_linear = list(params["dropout_rates_linear"])
    model_type = str(params.get("model_type", "gru"))

    class ModelRBase(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rnns = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            for i, hidden in enumerate(hidden_sizes):
                input_dim = n_features if i == 0 else hidden_sizes[i - 1]
                if model_type == "gru":
                    layer = nn.GRU(input_dim, hidden, num_layers=1, batch_first=True)
                elif model_type == "lstm":
                    layer = nn.LSTM(input_dim, hidden, num_layers=1, batch_first=True)
                else:
                    raise ValueError(f"Unknown model_type {model_type!r}")
                self.rnns.append(layer)
                self.dropouts.append(nn.Dropout(float(dropout_rates[i])))

            in_dim = hidden_sizes[-1] if hidden_sizes else n_features
            fc = []
            for i, hidden in enumerate(hidden_linear):
                fc.append(nn.Linear(in_dim if i == 0 else hidden_linear[i - 1], hidden))
                fc.append(nn.ReLU())
                fc.append(nn.Dropout(float(dropout_linear[i])))
            fc.append(nn.Linear(hidden_linear[-1] if hidden_linear else in_dim, 1))
            self.fc = nn.Sequential(*fc)

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
            out = self.fc(out.reshape(d * t, -1)).reshape(d, t)
            return out, next_hidden

    class ModelR(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.heads = nn.ModuleList([ModelRBase() for _ in EVGENIAVOLKOVA_AUX_COLUMNS])
            self.out = nn.Linear(len(EVGENIAVOLKOVA_AUX_COLUMNS), 1)

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
            aux = aux.reshape(x.shape[0], x.shape[1], len(EVGENIAVOLKOVA_AUX_COLUMNS))
            return y, aux, next_hidden

    return ModelR()


GRU_EVGENIAVOLKOVA_DEFAULT_PARAMS: dict[str, Any] = {
    "model_type": "gru",
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
}


class GRUEvgeniavolkovaEstimator:
    """Day-batch GRU with auxiliary responders and online fine-tuning."""

    def __init__(
        self,
        feature_cols: list[str],
        *,
        target_col: str = TARGET_COLUMN,
        weight_col: str = WEIGHT_COLUMN,
        params: dict[str, Any] | None = None,
        random_state: int = 42,
        device: str = "auto",
    ) -> None:
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.weight_col = weight_col
        self.params = {**GRU_EVGENIAVOLKOVA_DEFAULT_PARAMS, **(params or {})}
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
            aux_cols=EVGENIAVOLKOVA_AUX_COLUMNS,
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
        out_y, out_aux, _ = self._model(xb, None)
        loss = _weighted_r2_loss(out_y.flatten(), yb.flatten(), wb.flatten())
        for i in range(len(EVGENIAVOLKOVA_AUX_COLUMNS)):
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
        with torch.no_grad():
            preds, _, _ = model(xb, None)
        return preds.swapaxes(0, 1).reshape(-1).detach().cpu().numpy().astype(np.float64)

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

    def fit(self, df_train: pl.DataFrame, df_valid: pl.DataFrame | None = None) -> "GRUEvgeniavolkovaEstimator":
        import torch

        self._seed()
        self._mean, self._std = fit_feature_standardizer(df_train, self.feature_cols)
        self._model = _build_model(len(self.feature_cols), self.params).to(self._torch_device())
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
            losses = [self._train_day(df_day, optimizer) for df_day in self._iter_days(df_train)]
            if df_valid is not None and df_valid.height > 0:
                score = self._score(df_valid, online_update=True)
            else:
                score = -float(np.mean(losses))
            print(
                f"[js2024] gru_evgeniavolkova epoch={epoch} "
                f"loss={np.mean(losses):.6f} valid_R2={score:.6f}"
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

    def update(self, df_new: pl.DataFrame) -> "GRUEvgeniavolkovaEstimator":
        if self._model is None:
            raise RuntimeError("GRUEvgeniavolkovaEstimator.update called before fit().")
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
            raise RuntimeError("GRUEvgeniavolkovaEstimator.predict called before fit().")
        parts = [self._predict_day_with_model(self._model, df_day) for df_day in self._iter_days(df)]
        return np.concatenate(parts)
