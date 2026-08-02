"""Supervised fraud-like scorers (HGB + optional XGBoost/LightGBM)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from buildml.anomaly.extras import require_lightgbm, require_xgboost
from buildml.anomaly.types import SupervisedAnomalyMethod
from buildml.core.errors import ValidationError


def build_supervised_estimator(
    *,
    method: SupervisedAnomalyMethod,
    random_state: int | None,
) -> Any:
    if method == "supervised_hgb":
        return HistGradientBoostingClassifier(random_state=random_state)
    if method == "supervised_xgb":
        xgb = require_xgboost(feature="XGBoost supervised anomaly scorer")
        return xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            eval_metric="logloss",
            tree_method="hist",
        )
    if method == "supervised_lgbm":
        lgb = require_lightgbm(feature="LightGBM supervised anomaly scorer")
        return lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=random_state,
            verbose=-1,
        )
    raise ValidationError(f"Unsupported supervised anomaly method '{method}'")


def supervised_anomaly_scores(
    estimator: Any,
    *,
    method: str,
    x: np.ndarray,
) -> np.ndarray:
    proba = np.asarray(estimator.predict_proba(x), dtype=float)
    classes = list(getattr(estimator, "classes_", [0, 1]))
    if len(classes) == 1:
        return np.zeros(shape=(x.shape[0],), dtype=float)
    if 1 in classes:
        idx = classes.index(1)
    else:
        idx = len(classes) - 1
    return proba[:, idx]
