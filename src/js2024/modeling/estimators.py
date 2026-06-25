"""Model-agnostic estimator interface for walk-forward evaluation.

The :class:`Estimator` protocol deliberately mirrors the method names used by the
``evgeniavolkova/kagglejanestreet`` pipeline (``fit`` / ``update`` / ``predict``) so
that the :mod:`js2024.walk_forward` engine never needs to know whether it is driving
a LightGBM model, a future GRU, or a test double. Each method takes/produces polars
frames + numpy arrays, leaving per-model feature/sequence prep to the estimator.

V0 ships :class:`LGBMEstimator`, whose ``update`` is a LightGBM *leaf-value refit*
(``Booster.refit``) — the cheap, structure-preserving analog of a one-pass online
weight update — and :class:`GRUEstimator`, a sequence model over per-``symbol_id``
lookback windows whose ``update`` is a few fine-tuning gradient steps on the
newly-revealed labelled rows.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import polars as pl

from .features import (
    build_symbol_windows,
    fit_feature_standardizer,
    prepare_lgbm_frame,
    standardized_symbol_tails,
)
from .metrics import weighted_zero_mean_r2


@runtime_checkable
class Estimator(Protocol):
    """A fit / update / predict estimator driven by the walk-forward engine.

    - ``fit(df_train, df_valid)``: initial offline training. ``df_valid`` (optional)
      is a held-out frame for early stopping; it must never overlap the test block.
    - ``update(df_new)``: incremental update from newly-revealed labelled rows.
    - ``predict(df)``: return a 1-d array of predictions aligned to ``df``'s rows.
    """

    def fit(self, df_train: pl.DataFrame, df_valid: pl.DataFrame | None = None) -> "Estimator":
        ...

    def update(self, df_new: pl.DataFrame) -> "Estimator":
        ...

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        ...


UPDATE_METHODS = ("refit", "continue", "retrain")


class LGBMEstimator:
    """LightGBM estimator with three selectable ``update`` strategies.

    ``update_method``:

    - ``"refit"``    — ``Booster.refit`` leaf values on the newly-revealed rows
      (keeps tree structure; ``refit_decay`` blends old/new leaf outputs).
    - ``"continue"`` — continued boosting: add ``continue_rounds`` trees trained on
      the new rows via ``init_model`` (grows the ensemble over time).
    - ``"retrain"``  — expanding retrain from scratch on **all** labelled data seen
      so far (initial train region + every revealed chunk), at the initial fit's
      ``best_iteration`` rounds. The gold standard; use a coarse cadence.

    Parameters
    ----------
    feature_cols, target_col, weight_col
        Model input columns, label, and sample-weight column names.
    params
        Hyperparameters forwarded to :class:`lightgbm.LGBMRegressor`.
    early_stopping_rounds
        Early-stopping patience used during :meth:`fit` when a ``df_valid`` is given.
    update_method
        One of :data:`UPDATE_METHODS`.
    refit_decay
        ``Booster.refit`` decay (only used by ``"refit"``).
    continue_rounds
        Trees added per ``"continue"`` update.
    """

    def __init__(
        self,
        feature_cols: list[str],
        target_col: str,
        weight_col: str,
        params: dict[str, Any],
        *,
        early_stopping_rounds: int = 100,
        update_method: str = "refit",
        refit_decay: float = 0.9,
        continue_rounds: int = 10,
    ) -> None:
        if update_method not in UPDATE_METHODS:
            raise ValueError(
                f"update_method must be one of {UPDATE_METHODS}, got {update_method!r}"
            )
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.weight_col = weight_col
        self.params = dict(params)
        self.early_stopping_rounds = early_stopping_rounds
        self.update_method = update_method
        self.refit_decay = refit_decay
        self.continue_rounds = continue_rounds
        self._booster = None  # set in fit()
        self.best_iteration: int = 0
        # For "retrain": the full labelled history seen so far (frames are concatenated
        # at update time). For "continue": native params for lgb.train(init_model=...).
        self._history: list[pl.DataFrame] = []
        self._retrain_rounds: int = 0

    def _xyw(self, df: pl.DataFrame):
        return prepare_lgbm_frame(
            df, self.feature_cols, self.target_col, self.weight_col
        )

    def _native_params(self) -> dict[str, Any]:
        """Translate the sklearn-style params to native lgb.train params."""
        p = self.params
        return {
            "objective": "regression",
            "learning_rate": p.get("learning_rate", 0.03),
            "num_leaves": p.get("num_leaves", 31),
            "bagging_fraction": p.get("subsample", 1.0),
            "bagging_freq": 1,
            "feature_fraction": p.get("colsample_bytree", 1.0),
            "device_type": p.get("device_type", "cpu"),
            "max_bin": p.get("max_bin", 255),
            "gpu_use_dp": p.get("gpu_use_dp", False),
            "seed": p.get("random_state", 42),
            "verbose": -1,
        }

    def fit(
        self, df_train: pl.DataFrame, df_valid: pl.DataFrame | None = None
    ) -> "LGBMEstimator":
        import lightgbm as lgb

        X, y, w = self._xyw(df_train)
        reg = lgb.LGBMRegressor(**self.params, n_jobs=-1)

        fit_kwargs: dict[str, Any] = {"sample_weight": w}
        if df_valid is not None and df_valid.height > 0:
            Xv, yv, wv = self._xyw(df_valid)
            fit_kwargs["eval_set"] = [(Xv, yv)]
            fit_kwargs["eval_sample_weight"] = [wv]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(self.early_stopping_rounds),
                lgb.log_evaluation(period=0),
            ]

        reg.fit(X, y, **fit_kwargs)
        self._booster = reg.booster_
        self.best_iteration = int(getattr(reg, "best_iteration_", 0) or 0)

        if self.update_method == "retrain":
            # Expanding retrain trains on ALL labelled data so far. Seed the history
            # with the full labelled region (train + the early-stopping holdout) and
            # fix the round count from the initial fit's best_iteration.
            self._history = [df_train]
            if df_valid is not None and df_valid.height > 0:
                self._history.append(df_valid)
            self._retrain_rounds = self.best_iteration or int(
                self.params.get("n_estimators", 100)
            )
        return self

    def update(self, df_new: pl.DataFrame) -> "LGBMEstimator":
        """Incorporate freshly-revealed labelled rows per ``update_method``."""
        if self._booster is None:
            raise RuntimeError("LGBMEstimator.update called before fit().")
        if df_new.height == 0:
            return self
        import lightgbm as lgb

        if self.update_method == "refit":
            X, y, _ = self._xyw(df_new)
            self._booster = self._booster.refit(X, y, decay_rate=self.refit_decay)
        elif self.update_method == "continue":
            X, y, w = self._xyw(df_new)
            dtrain = lgb.Dataset(X, label=y, weight=w)
            self._booster = lgb.train(
                self._native_params(),
                dtrain,
                num_boost_round=self.continue_rounds,
                init_model=self._booster,
                keep_training_booster=True,
            )
        elif self.update_method == "retrain":
            self._history.append(df_new)
            df_all = pl.concat(self._history, how="vertical")
            X, y, w = self._xyw(df_all)
            reg = lgb.LGBMRegressor(
                **{**self.params, "n_estimators": self._retrain_rounds}, n_jobs=-1
            )
            reg.fit(X, y, sample_weight=w)
            self._booster = reg.booster_
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        if self._booster is None:
            raise RuntimeError("LGBMEstimator.predict called before fit().")
        X, _, _ = self._xyw(df)
        return np.asarray(self._booster.predict(X), dtype=np.float64)


# Default GRU hyperparameters, merged under any user-supplied ``params``.
GRU_DEFAULT_PARAMS: dict[str, Any] = {
    "seq_len": 16,
    "hidden_size": 64,
    "num_layers": 1,
    "dropout": 0.0,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "epochs": 20,
    "batch_size": 1024,
    "early_stopping_rounds": 5,
    "finetune_epochs": 1,
    "finetune_lr": 5e-4,
    # Materializing every lookback window is fine for smoke tests but not for the
    # full Jane Street frame. In "auto" mode, fit() streams by date once the
    # estimated window tensor would exceed this many GiB.
    "train_mode": "auto",
    "max_materialized_windows_gib": 8.0,
}


def _build_gru_module(n_features: int, params: dict[str, Any]):
    """Construct the internal GRU regressor ``nn.Module`` (torch imported lazily)."""
    import torch.nn as nn

    hidden = int(params["hidden_size"])
    num_layers = int(params["num_layers"])
    dropout = float(params["dropout"])

    class _GRURegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gru = nn.GRU(
                input_size=n_features,
                hidden_size=hidden,
                num_layers=num_layers,
                batch_first=True,
                # PyTorch only applies GRU dropout between stacked layers.
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):  # x: (B, L, F)
            out, _ = self.gru(x)
            last = out[:, -1, :]  # last time step
            return self.head(last).squeeze(-1)  # (B,)

    return _GRURegressor()


class GRUEstimator:
    """GRU sequence estimator over per-``symbol_id`` lookback windows.

    Conforms to the :class:`Estimator` protocol so the model-agnostic
    :func:`js2024.modeling.walk_forward.walk_forward_evaluate` engine drives it
    exactly like :class:`LGBMEstimator`.

    Each row's prediction is produced from a length-``seq_len`` window of that
    symbol's feature vectors ordered by ``(date_id, time_id)`` and ending at the
    row. A per-symbol context buffer (the last ``seq_len-1`` standardized rows)
    is carried across calls so windows span day boundaries; it is advanced in
    :meth:`predict` using **features only** (no labels — leakage-clean), which
    also keeps ``mode="full"`` windows correct even though the engine never calls
    :meth:`update` in that mode.

    Parameters
    ----------
    feature_cols, target_col, weight_col
        Model input columns, label, and sample-weight column names.
    params
        Hyperparameters merged over :data:`GRU_DEFAULT_PARAMS`.
    random_state
        Seeds ``torch``/NumPy for reproducible fits.
    device
        Torch device string. ``"auto"`` (default) uses CUDA when available, else
        CPU; pass ``"cpu"`` / ``"cuda"`` to force a device.
    """

    def __init__(
        self,
        feature_cols: list[str],
        target_col: str,
        weight_col: str,
        params: dict[str, Any] | None = None,
        *,
        random_state: int = 42,
        device: str = "auto",
    ) -> None:
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.weight_col = weight_col
        self.params = {**GRU_DEFAULT_PARAMS, **(params or {})}
        self.random_state = int(random_state)
        self.device = device
        self.seq_len = int(self.params["seq_len"])
        if self.seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {self.seq_len}")
        self._model = None  # set in fit()
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        # Cross-day context: last (seq_len-1) standardized feature rows per symbol.
        self._buffer: dict[int, np.ndarray] = {}

    # --- internals ---------------------------------------------------------
    def _torch_device(self):
        import torch

        name = self.device
        if name == "auto":
            name = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(name)

    def _seed(self) -> None:
        import torch

        torch.manual_seed(self.random_state)
        torch.cuda.manual_seed_all(self.random_state)
        np.random.seed(self.random_state)

    def _windows(self, df: pl.DataFrame, *, labelled: bool, use_buffer: bool):
        return build_symbol_windows(
            df,
            self.feature_cols,
            self._mean,
            self._std,
            self.seq_len,
            target_col=self.target_col if labelled else None,
            weight_col=self.weight_col if labelled else None,
            history=self._buffer if use_buffer else None,
        )

    def _train_loop(self, windows, y, w, *, epochs: int, lr: float, df_valid):
        import torch

        model = self._model
        device = self._torch_device()
        model.to(device)
        opt = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=float(self.params["weight_decay"])
        )
        batch_size = int(self.params["batch_size"])
        n = windows.shape[0]

        Xt = torch.from_numpy(windows)
        yt = torch.from_numpy(y)
        wt = torch.from_numpy(w)

        best_score = -np.inf
        best_state = None
        patience = int(self.params["early_stopping_rounds"])
        bad = 0

        for _ in range(int(epochs)):
            model.train()
            perm = torch.randperm(n)
            for s in range(0, n, batch_size):
                idx = perm[s : s + batch_size]
                xb = Xt[idx].to(device)
                yb = yt[idx].to(device)
                wb = wt[idx].to(device)
                opt.zero_grad()
                pred = model(xb)
                # Weighted MSE: sum(w*(y-yhat)^2) / sum(w).
                denom = wb.sum().clamp_min(1e-12)
                loss = (wb * (pred - yb) ** 2).sum() / denom
                loss.backward()
                opt.step()

            if df_valid is not None and df_valid.height > 0:
                score = self._score(df_valid)
                if score > best_score:
                    best_score = score
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience:
                        break

        if best_state is not None:
            model.load_state_dict(best_state)

    def _train_pack_once(self, windows, y, w, *, lr: float, opt=None) -> Any:
        """Train one epoch over a pre-built window pack, optionally reusing optimizer."""
        import torch

        model = self._model
        device = self._torch_device()
        model.to(device)
        if opt is None:
            opt = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=float(self.params["weight_decay"])
            )
        batch_size = int(self.params["batch_size"])
        n = windows.shape[0]
        if n == 0:
            return opt

        Xt = torch.from_numpy(windows)
        yt = torch.from_numpy(y)
        wt = torch.from_numpy(w)
        model.train()
        perm = torch.randperm(n)
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            xb = Xt[idx].to(device)
            yb = yt[idx].to(device)
            wb = wt[idx].to(device)
            opt.zero_grad()
            pred = model(xb)
            denom = wb.sum().clamp_min(1e-12)
            loss = (wb * (pred - yb) ** 2).sum() / denom
            loss.backward()
            opt.step()
        return opt

    def _estimated_window_gib(self, df: pl.DataFrame) -> float:
        bytes_ = df.height * self.seq_len * len(self.feature_cols) * np.dtype(np.float32).itemsize
        return bytes_ / (1024**3)

    @staticmethod
    def _iter_date_groups(df: pl.DataFrame):
        """Yield date groups without repeatedly filtering the whole frame."""
        for _, df_day in df.group_by("date_id", maintain_order=True):
            yield df_day

    def _should_stream_fit(self, df_train: pl.DataFrame) -> bool:
        mode = str(self.params.get("train_mode", "auto")).lower()
        if mode not in {"auto", "materialize", "stream"}:
            raise ValueError(
                "GRU train_mode must be one of {'auto', 'materialize', 'stream'}, "
                f"got {mode!r}"
            )
        if mode == "stream":
            return True
        if mode == "materialize":
            return False
        limit = float(self.params.get("max_materialized_windows_gib", 8.0))
        return self._estimated_window_gib(df_train) > limit

    def _train_loop_stream_by_date(self, df_train: pl.DataFrame, *, epochs: int, lr: float, df_valid):
        """Memory-light GRU training: build lookback windows one date at a time."""
        import torch

        model = self._model
        device = self._torch_device()
        model.to(device)
        opt = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=float(self.params["weight_decay"])
        )

        best_score = -np.inf
        best_state = None
        patience = int(self.params["early_stopping_rounds"])
        bad = 0

        for _ in range(int(epochs)):
            train_buffer: dict[int, np.ndarray] = {}
            for df_day in self._iter_date_groups(df_train):
                pack = build_symbol_windows(
                    df_day,
                    self.feature_cols,
                    self._mean,
                    self._std,
                    self.seq_len,
                    target_col=self.target_col,
                    weight_col=self.weight_col,
                    history=train_buffer,
                )
                opt = self._train_pack_once(
                    pack["windows"], pack["y"], pack["w"], lr=lr, opt=opt
                )
                tails = standardized_symbol_tails(
                    df_day, self.feature_cols, self._mean, self._std, self.seq_len - 1
                )
                train_buffer.update(tails)

            if df_valid is not None and df_valid.height > 0:
                self._buffer = standardized_symbol_tails(
                    df_train, self.feature_cols, self._mean, self._std, self.seq_len - 1
                )
                score = self._score(df_valid)
                if score > best_score:
                    best_score = score
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience:
                        break

        if best_state is not None:
            model.load_state_dict(best_state)

    def _score(self, df: pl.DataFrame) -> float:
        if self._should_stream_fit(df):
            original_buffer = {k: v.copy() for k, v in self._buffer.items()}
            preds_parts = []
            y_parts = []
            w_parts = []
            for df_day in self._iter_date_groups(df):
                preds_parts.append(self.predict(df_day, _advance_buffer=True))
                y_parts.append(df_day.get_column(self.target_col).to_numpy().astype(np.float64))
                w_parts.append(df_day.get_column(self.weight_col).to_numpy().astype(np.float64))
            preds = np.concatenate(preds_parts)
            y = np.concatenate(y_parts)
            w = np.concatenate(w_parts)
            self._buffer = original_buffer
        else:
            preds = self.predict(df, _advance_buffer=False)
            y = df.get_column(self.target_col).to_numpy().astype(np.float64)
            w = df.get_column(self.weight_col).to_numpy().astype(np.float64)
        try:
            return weighted_zero_mean_r2(y, preds, w)
        except ValueError:
            # Degenerate validation block (e.g. all-zero targets) — fall back to
            # negative weighted MSE so early stopping still has a usable signal.
            return -float(np.average((y - preds) ** 2, weights=w))

    def _advance_buffer(self, df: pl.DataFrame) -> None:
        keep = self.seq_len - 1
        if keep <= 0:
            return
        new_tails = standardized_symbol_tails(
            df, self.feature_cols, self._mean, self._std, keep
        )
        for sym, rows in new_tails.items():
            if rows.shape[0] >= keep:
                self._buffer[sym] = rows[-keep:]
            else:
                prev = self._buffer.get(sym)
                combined = rows if prev is None else np.concatenate([prev, rows], axis=0)
                self._buffer[sym] = combined[-keep:]

    # --- protocol ----------------------------------------------------------
    def fit(
        self, df_train: pl.DataFrame, df_valid: pl.DataFrame | None = None
    ) -> "GRUEstimator":
        self._seed()
        self._mean, self._std = fit_feature_standardizer(df_train, self.feature_cols)
        self._model = _build_gru_module(len(self.feature_cols), self.params)
        # No cross-day buffer during the initial in-frame training pass.
        self._buffer = {}
        if self._should_stream_fit(df_train):
            self._train_loop_stream_by_date(
                df_train,
                epochs=int(self.params["epochs"]),
                lr=float(self.params["lr"]),
                df_valid=df_valid,
            )
        else:
            pack = self._windows(df_train, labelled=True, use_buffer=False)
            self._train_loop(
                pack["windows"],
                pack["y"],
                pack["w"],
                epochs=int(self.params["epochs"]),
                lr=float(self.params["lr"]),
                df_valid=df_valid,
            )
        # Seed the buffer from the train tail (chronologically the latest rows).
        self._buffer = standardized_symbol_tails(
            df_train, self.feature_cols, self._mean, self._std, self.seq_len - 1
        )
        return self

    def update(self, df_new: pl.DataFrame) -> "GRUEstimator":
        """Fine-tune on freshly-revealed labelled rows (buffer advanced by predict)."""
        if self._model is None:
            raise RuntimeError("GRUEstimator.update called before fit().")
        if df_new.height == 0:
            return self
        pack = self._windows(df_new, labelled=True, use_buffer=True)
        self._train_loop(
            pack["windows"],
            pack["y"],
            pack["w"],
            epochs=int(self.params["finetune_epochs"]),
            lr=float(self.params["finetune_lr"]),
            df_valid=None,
        )
        return self

    def predict(self, df: pl.DataFrame, *, _advance_buffer: bool = True) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("GRUEstimator.predict called before fit().")
        import torch

        pack = self._windows(df, labelled=False, use_buffer=True)
        model = self._model
        device = self._torch_device()
        model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(pack["windows"]).to(device)
            preds = model(xb).detach().cpu().numpy().astype(np.float64)
        # Advance the cross-day context with this frame's features (no labels).
        if _advance_buffer:
            self._advance_buffer(df)
        return preds
