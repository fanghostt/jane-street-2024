"""Model-agnostic estimator interface for walk-forward evaluation.

The :class:`Estimator` protocol deliberately mirrors the method names used by the
``evgeniavolkova/kagglejanestreet`` pipeline (``fit`` / ``update`` / ``predict``) so
that the :mod:`js2024.walk_forward` engine never needs to know whether it is driving
a LightGBM model, a future GRU, or a test double. Each method takes/produces polars
frames + numpy arrays, leaving per-model feature/sequence prep to the estimator.

V0 ships a single concrete estimator, :class:`LGBMEstimator`, whose ``update`` is a
LightGBM *leaf-value refit* (``Booster.refit``) — the cheap, structure-preserving
analog of a one-pass online weight update.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import polars as pl

from .features import prepare_lgbm_frame


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
