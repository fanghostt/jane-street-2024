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


class LGBMEstimator:
    """LightGBM estimator with a leaf-value-refit ``update``.

    Parameters
    ----------
    feature_cols
        Model input columns.
    target_col, weight_col
        Label and sample-weight column names.
    params
        Hyperparameters forwarded to :class:`lightgbm.LGBMRegressor` (e.g.
        ``n_estimators``, ``learning_rate``, ``num_leaves`` …).
    early_stopping_rounds
        Early-stopping patience used during :meth:`fit` when a ``df_valid`` is given.
    refit_decay
        LightGBM ``Booster.refit`` ``decay_rate``: the new leaf output is
        ``decay_rate * old + (1 - decay_rate) * new``. Default ``0.9`` keeps most of
        the existing fit and nudges it toward freshly-revealed data.
    """

    def __init__(
        self,
        feature_cols: list[str],
        target_col: str,
        weight_col: str,
        params: dict[str, Any],
        *,
        early_stopping_rounds: int = 100,
        refit_decay: float = 0.9,
    ) -> None:
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.weight_col = weight_col
        self.params = dict(params)
        self.early_stopping_rounds = early_stopping_rounds
        self.refit_decay = refit_decay
        self._booster = None  # set in fit()
        self.best_iteration: int = 0

    def _xyw(self, df: pl.DataFrame):
        return prepare_lgbm_frame(
            df, self.feature_cols, self.target_col, self.weight_col
        )

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
        return self

    def update(self, df_new: pl.DataFrame) -> "LGBMEstimator":
        """Refit leaf values on newly-revealed rows (keeps tree structure)."""
        if self._booster is None:
            raise RuntimeError("LGBMEstimator.update called before fit().")
        if df_new.height == 0:
            return self
        X, y, _ = self._xyw(df_new)
        self._booster = self._booster.refit(
            X, y, decay_rate=self.refit_decay
        )
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        if self._booster is None:
            raise RuntimeError("LGBMEstimator.predict called before fit().")
        X, _, _ = self._xyw(df)
        return np.asarray(self._booster.predict(X), dtype=np.float64)
