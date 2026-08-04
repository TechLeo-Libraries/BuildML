"""River streaming online estimator adapter (buildml[online-industry])."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from buildml.core.errors import ValidationError
from buildml.online.extras import require_river
from buildml.online.types import OnlineTask

_RIVER_CLASSIFIERS = {"river_logistic", "river_hoeffding", "river_pa"}
_RIVER_REGRESSORS = {"river_linear_regression", "river_hoeffding_regressor"}


def resolve_river_task(estimator: str, task: OnlineTask | None) -> OnlineTask:
    """Resolve the task type for a River streaming estimator.

    Maps River classifier and regressor keys to the appropriate task and
    validates explicit ``task`` overrides.

    Parameters
    ----------
    estimator:
        River estimator key (``river_logistic``, ``river_hoeffding``, etc.).
    task:
        Optional explicit task override.

    Returns
    -------
    OnlineTask
        Resolved ``classification`` or ``regression`` task.

    Raises
    ------
    ValidationError
        When the estimator is unknown or incompatible with ``task``.
    """
    if estimator in _RIVER_CLASSIFIERS:
        if task == "regression":
            raise ValidationError(
                f"Estimator {estimator!r} is a classifier; task cannot be 'regression'."
            )
        return "classification"
    if estimator in _RIVER_REGRESSORS:
        if task == "classification":
            raise ValidationError(
                f"Estimator {estimator!r} is a regressor; task cannot be "
                "'classification'."
            )
        return "regression"
    raise ValidationError(
        f"Unknown River online estimator={estimator!r}. "
        f"Supported: {sorted(_RIVER_CLASSIFIERS | _RIVER_REGRESSORS)}"
    )


def build_river_estimator(
    name: str,
    *,
    random_state: int | None = 0,
    drift_detector: str = "adwin",
    n_features: int = 0,
) -> RiverOnlineWrapper:
    """Build a sklearn-compatible River streaming estimator wrapper.

    Requires ``buildml[online-industry]``; lazy-initializes the River model on
    first ``partial_fit``.

    Parameters
    ----------
    name:
        River estimator key (``river_logistic``, ``river_hoeffding``, etc.).
    random_state:
        Seed for stochastic River models.
    drift_detector:
        Drift detector for update/evaluate disclosure (``adwin``,
        ``page_hinkley``, or ``mean_shift``).
    n_features:
        Feature dimensionality hint for lazy model init.

    Returns
    -------
    RiverOnlineWrapper
        Wrapper with ``partial_fit`` and ``predict`` sklearn interface.
    """
    return RiverOnlineWrapper(
        estimator_name=name,
        random_state=random_state,
        drift_detector=drift_detector,
        n_features=n_features,
    )


@dataclass
class RiverOnlineWrapper:
    """Sklearn-compatible partial_fit wrapper over a River streaming model."""

    estimator_name: str
    random_state: int | None = 0
    drift_detector: str = "adwin"
    n_features: int = 0
    task: str = "classification"
    model_: Any = field(default=None, repr=False)
    adwin_: Any = field(default=None, repr=False)
    page_hinkley_: Any = field(default=None, repr=False)
    n_seen_: int = 0
    drift_events_: list[dict[str, Any]] = field(default_factory=list)

    def _ensure_model(self, n_features: int) -> None:
        if self.model_ is not None:
            return
        require_river()
        from river import drift, linear_model, tree

        self.n_features = int(n_features)
        seed = self.random_state

        def _river_ctor(factory: Any, **kwargs: Any) -> Any:
            """Construct a River model, dropping unsupported kwargs (e.g. seed)."""
            try:
                return factory(**kwargs)
            except TypeError:
                kwargs.pop("seed", None)
                return factory(**kwargs)

        if self.estimator_name == "river_logistic":
            self.model_ = _river_ctor(linear_model.LogisticRegression, seed=seed)
            self.task = "classification"
        elif self.estimator_name == "river_hoeffding":
            self.model_ = _river_ctor(tree.HoeffdingTreeClassifier, seed=seed)
            self.task = "classification"
        elif self.estimator_name == "river_pa":
            self.model_ = _river_ctor(linear_model.PAClassifier, seed=seed)
            self.task = "classification"
        elif self.estimator_name == "river_linear_regression":
            self.model_ = linear_model.LinearRegression()
            self.task = "regression"
        elif self.estimator_name == "river_hoeffding_regressor":
            self.model_ = _river_ctor(tree.HoeffdingTreeRegressor, seed=seed)
            self.task = "regression"
        else:
            raise ValidationError(f"Unsupported River estimator {self.estimator_name!r}.")

        if self.drift_detector == "adwin":
            self.adwin_ = drift.ADWIN()
        elif self.drift_detector == "page_hinkley":
            self.page_hinkley_ = drift.PageHinkley()

    def _row_dict(self, row: np.ndarray) -> dict[str, float]:
        return {f"f{i}": float(row[i]) for i in range(len(row))}

    def _update_drift(self, signal: float) -> list[str]:
        notes: list[str] = []
        if self.drift_detector == "adwin" and self.adwin_ is not None:
            self.adwin_.update(float(signal))
            if self.adwin_.drift_detected:
                msg = f"River ADWIN drift signal on update (n_seen={self.n_seen_})."
                notes.append(msg)
                self.drift_events_.append(
                    {"detector": "adwin", "n_seen": self.n_seen_, "signal": float(signal)}
                )
        if self.drift_detector == "page_hinkley" and self.page_hinkley_ is not None:
            self.page_hinkley_.update(float(signal))
            if self.page_hinkley_.drift_detected:
                msg = (
                    f"River Page-Hinkley drift signal on update "
                    f"(n_seen={self.n_seen_})."
                )
                notes.append(msg)
                self.drift_events_.append(
                    {
                        "detector": "page_hinkley",
                        "n_seen": self.n_seen_,
                        "signal": float(signal),
                    }
                )
        return notes

    def partial_fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        classes: Sequence[Any] | None = None,
    ) -> RiverOnlineWrapper:
        """Incrementally fit the River model one row at a time.

        Updates drift detectors on per-row prediction error without requiring
        batch refits.

        Parameters
        ----------
        x:
            2-D float feature matrix.
        y:
            Target array aligned with ``x`` rows.
        classes:
            Ignored for River models (accepted for sklearn API compatibility).

        Returns
        -------
        RiverOnlineWrapper
            ``self`` for chaining.

        Raises
        ------
        ValidationError
            When ``x`` is not 2-D or the estimator name is unsupported.
        """
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y)
        if x_arr.ndim != 2:
            raise ValidationError("River partial_fit expects a 2D feature matrix.")
        self._ensure_model(x_arr.shape[1])
        drift_notes: list[str] = []
        for i in range(len(x_arr)):
            row = self._row_dict(x_arr[i])
            target = int(y_arr[i]) if self.task == "classification" else float(y_arr[i])
            pred = self.model_.predict_one(row)
            if self.task == "classification":
                err = 0.0 if pred == target else 1.0
            else:
                err = abs(float(pred) - float(target)) if pred is not None else 0.0
            drift_notes.extend(self._update_drift(err))
            self.model_.learn_one(row, target)
            self.n_seen_ += 1
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict targets for a batch of rows using the River model.

        Runs ``predict_one`` per row without updating drift detectors or the
        underlying River learner.

        Parameters
        ----------
        x:
            2-D float feature matrix.

        Returns
        -------
        numpy.ndarray
            Predicted class codes or regression values.

        Raises
        ------
        ValidationError
            When the wrapper has not been fitted yet.
        """
        if self.model_ is None:
            raise ValidationError("RiverOnlineWrapper is not fitted.")
        x_arr = np.asarray(x, dtype=float)
        out: list[int | float] = []
        for i in range(len(x_arr)):
            row = self._row_dict(x_arr[i])
            pred = self.model_.predict_one(row)
            if self.task == "classification":
                out.append(int(pred) if pred is not None else 0)
            else:
                out.append(float(pred) if pred is not None else 0.0)
        return np.asarray(out)

    def evaluate_drift_stream(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple[bool, list[str]]:
        """Score holdout rows through River drift detectors without model update.

        Feeds prediction errors into ADWIN or Page-Hinkley detectors for
        evaluation-time drift disclosure; does not call ``learn_one``.

        Parameters
        ----------
        x:
            2-D float feature matrix for holdout rows.
        y:
            True targets aligned with ``x`` rows.

        Returns
        -------
        tuple[bool, list[str]]
            ``(drift_detected, disclosure_notes)``.
        """
        if self.model_ is None:
            return False, []
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y)
        detected = False
        notes: list[str] = []
        for i in range(len(x_arr)):
            row = self._row_dict(x_arr[i])
            target = int(y_arr[i]) if self.task == "classification" else float(y_arr[i])
            pred = self.model_.predict_one(row)
            if self.task == "classification":
                err = 0.0 if pred == target else 1.0
            else:
                err = abs(float(pred) - float(target)) if pred is not None else 0.0
            chunk_notes = self._update_drift(err)
            if chunk_notes:
                detected = True
                notes.extend(chunk_notes)
        if detected:
            notes.insert(
                0,
                "River drift detector(s) fired on the holdout error stream during "
                "evaluate_online (evaluation rows were not used for learn_one).",
            )
        return detected, notes
