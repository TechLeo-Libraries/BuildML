"""Industry GBDT pseudo-label semi-supervised adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from buildml.core.errors import ValidationError
from buildml.semisupervised.extras import require_lightgbm, require_xgboost
from buildml.semisupervised.types import IndustrySemiSupervisedMethod, SKLEARN_UNLABELED


@dataclass
class PseudoLabelGBDTClassifier:
    """Iterative pseudo-labeling with XGBoost or LightGBM."""

    method: IndustrySemiSupervisedMethod = "pseudo_label_xgb"
    threshold: float = 0.75
    max_iter: int = 10
    random_state: int | None = 0
    classes_: np.ndarray = field(default_factory=lambda: np.array([]))
    estimator_: Any = field(default=None, repr=False)
    n_pseudo_labels_: int = 0
    iterations_run_: int = 0

    def fit(self, x: np.ndarray, y: np.ndarray) -> PseudoLabelGBDTClassifier:
        """Run fit on input data using the fitted internal state.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
x:
    Feature matrix input rows.
y:
    Target vector or series aligned with ``x``.

Returns
-------
PseudoLabelGBDTClassifier
    Return value (PseudoLabelGBDTClassifier) produced by this operation.

Raises
------
ValidationError
    When preconditions for this operation are not met.
        """
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=int)
        labeled = y_arr != SKLEARN_UNLABELED
        if labeled.sum() < 2:
            raise ValidationError(
                "Pseudo-label GBDT needs at least 2 labeled train rows."
            )
        classes = np.unique(y_arr[labeled])
        if len(classes) < 2:
            raise ValidationError(
                "Pseudo-label GBDT needs at least 2 classes among labeled rows."
            )
        self.classes_ = classes
        y_work = y_arr.copy()
        estimator = self._build_estimator()
        n_pseudo = 0
        for iteration in range(int(self.max_iter)):
            train_mask = y_work != SKLEARN_UNLABELED
            if train_mask.sum() < 2:
                break
            estimator = self._build_estimator()
            estimator.fit(x_arr[train_mask], y_work[train_mask])
            unlabeled = ~train_mask
            if not unlabeled.any():
                break
            proba = np.asarray(estimator.predict_proba(x_arr[unlabeled]), dtype=float)
            preds = np.asarray(estimator.classes_, dtype=int)[np.argmax(proba, axis=1)]
            conf = proba.max(axis=1)
            accept = conf >= float(self.threshold)
            if not accept.any():
                break
            idx = np.flatnonzero(unlabeled)
            y_work[idx[accept]] = preds[accept]
            n_pseudo += int(accept.sum())
            self.iterations_run_ = iteration + 1
        train_mask = y_work != SKLEARN_UNLABELED
        final = self._build_estimator()
        final.fit(x_arr[train_mask], y_work[train_mask])
        self.estimator_ = final
        self.n_pseudo_labels_ = n_pseudo
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Run predict on input data using the fitted internal state.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
x:
    Feature matrix input rows.

Returns
-------
np.ndarray
    NumPy array aligned with input rows.

Raises
------
ValidationError
    When preconditions for this operation are not met.
        """
        if self.estimator_ is None:
            raise ValidationError("PseudoLabelGBDTClassifier is not fitted.")
        return np.asarray(self.estimator_.predict(x), dtype=int)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Perform predict proba for the Session-facing workflow step.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
x:
    Feature matrix input rows.

Returns
-------
np.ndarray
    NumPy array aligned with input rows.

Raises
------
ValidationError
    When preconditions for this operation are not met.
        """
        if self.estimator_ is None:
            raise ValidationError("PseudoLabelGBDTClassifier is not fitted.")
        return np.asarray(self.estimator_.predict_proba(x), dtype=float)

    def _build_estimator(self) -> Any:
        if self.method == "pseudo_label_xgb":
            xgb = require_xgboost()
            return xgb.XGBClassifier(
                n_estimators=120,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=self.random_state,
                eval_metric="mlogloss",
                tree_method="hist",
            )
        if self.method == "pseudo_label_lgbm":
            lgb = require_lightgbm()
            return lgb.LGBMClassifier(
                n_estimators=120,
                learning_rate=0.08,
                num_leaves=31,
                random_state=self.random_state,
                verbose=-1,
            )
        raise ValidationError(f"Unsupported industry pseudo-label method '{self.method}'")


def build_industry_estimator(
    *,
    method: IndustrySemiSupervisedMethod,
    threshold: float,
    max_iter: int,
    random_state: int | None,
) -> PseudoLabelGBDTClassifier:
    """Construct a industry estimator ready for fit or scoring.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
method:
    Method or strategy identifier for the resolved backend.
threshold:
    Decision threshold applied to anomaly scores.
max_iter:
    max iter (int).
random_state:
    Seed for stochastic steps (sampling, initialization, bagging).

Returns
-------
PseudoLabelGBDTClassifier
    Return value (PseudoLabelGBDTClassifier) produced by this operation.
    """
    return PseudoLabelGBDTClassifier(
        method=method,
        threshold=float(threshold),
        max_iter=int(max_iter),
        random_state=random_state,
    )
