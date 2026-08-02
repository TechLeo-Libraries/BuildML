"""MAPIE conformal prediction adapter (regression + classification)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from buildml.core.errors import ValidationError
from buildml.probabilistic.extras import require_mapie

MapieMethod = Literal["split", "cv_plus", "jackknife_plus"]


@dataclass(slots=True)
class MapieWrapper:
    """Thin handle around fitted MAPIE objects (v1 legacy or v1.4+ API)."""

    task: str
    method: str
    alpha: float
    estimator: Any
    api: str = "modern"
    label_encoder_classes: tuple[Any, ...] | None = None

    @property
    def confidence_level(self) -> float:
        return 1.0 - float(self.alpha)


def _legacy_mapie_available() -> bool:
    try:
        from mapie.regression import MapieRegressor  # noqa: F401

        return True
    except ImportError:
        return False


def _modern_mapie_available() -> bool:
    try:
        from mapie.regression import SplitConformalRegressor  # noqa: F401

        return True
    except ImportError:
        return False


def _base_regressor():
    from sklearn.linear_model import Ridge

    return Ridge(alpha=1.0)


def _base_classifier():
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=500)


def fit_mapie(
    *,
    method: MapieMethod,
    task: str,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_cal: np.ndarray | None = None,
    y_cal: np.ndarray | None = None,
    random_state: int | None = 0,
    alpha: float = 0.1,
) -> tuple[MapieWrapper, list[str]]:
    """Fit MAPIE conformal regression/classification on Session train only."""
    require_mapie(feature=f"MAPIE {method} conformal ({task})")
    disclosures: list[str] = [
        f"MAPIE backend method={method}, task={task}.",
        "Fit uses Session train only; validation/test never enter calibration.",
    ]

    if _modern_mapie_available():
        wrapper, extra = _fit_modern_mapie(
            method=method,
            task=task,
            x_fit=x_fit,
            y_fit=y_fit,
            x_cal=x_cal,
            y_cal=y_cal,
            random_state=random_state,
            alpha=alpha,
        )
        disclosures.extend(extra)
        return wrapper, disclosures

    if _legacy_mapie_available():
        wrapper, extra = _fit_legacy_mapie(
            method=method,
            task=task,
            x_fit=x_fit,
            y_fit=y_fit,
            x_cal=x_cal,
            y_cal=y_cal,
            random_state=random_state,
            alpha=alpha,
        )
        disclosures.extend(extra)
        disclosures.append("Using legacy MAPIE MapieRegressor/MapieClassifier API.")
        return wrapper, disclosures

    raise ValidationError(
        "Installed mapie package lacks SplitConformalRegressor and MapieRegressor."
    )


def _fit_modern_mapie(
    *,
    method: MapieMethod,
    task: str,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_cal: np.ndarray | None,
    y_cal: np.ndarray | None,
    random_state: int | None,
    alpha: float,
) -> tuple[MapieWrapper, list[str]]:
    disclosures: list[str] = []
    confidence = 1.0 - float(alpha)

    if task == "regression":
        from mapie.regression import (
            CrossConformalRegressor,
            JackknifeAfterBootstrapRegressor,
            SplitConformalRegressor,
        )

        if method == "split":
            if x_cal is None or y_cal is None:
                raise ValidationError(
                    "MAPIE split conformal requires a train-only calibration carve."
                )
            base = _base_regressor().fit(x_fit, y_fit)
            model = SplitConformalRegressor(
                estimator=base,
                prefit=True,
                confidence_level=confidence,
            )
            model.conformalize(x_cal, y_cal)
            disclosures.append(
                "MAPIE SplitConformalRegressor: base on fit-carve, conformalize on calib-carve."
            )
        elif method == "cv_plus":
            model = CrossConformalRegressor(
                estimator=_base_regressor(),
                confidence_level=confidence,
                method="plus",
                cv=5,
                random_state=random_state,
            )
            model.fit_conformalize(x_fit, y_fit)
            disclosures.append("MAPIE CrossConformalRegressor (CV+) on Session train.")
        elif method == "jackknife_plus":
            model = JackknifeAfterBootstrapRegressor(
                estimator=_base_regressor(),
                confidence_level=confidence,
                method="plus",
                cv=-1,
                random_state=random_state,
            )
            model.fit_conformalize(x_fit, y_fit)
            disclosures.append(
                "MAPIE JackknifeAfterBootstrapRegressor (jackknife+) on Session train."
            )
        else:
            raise ValidationError(f"Unknown MAPIE method '{method}'.")
    else:
        from mapie.classification import CrossConformalClassifier, SplitConformalClassifier

        if method == "split":
            if x_cal is None or y_cal is None:
                raise ValidationError(
                    "MAPIE split conformal requires a train-only calibration carve."
                )
            base = _base_classifier().fit(x_fit, y_fit)
            model = SplitConformalClassifier(
                estimator=base,
                prefit=True,
                confidence_level=confidence,
            )
            model.conformalize(x_cal, y_cal)
            disclosures.append(
                "MAPIE SplitConformalClassifier: base on fit-carve, conformalize on calib-carve."
            )
        elif method == "cv_plus":
            model = CrossConformalClassifier(
                estimator=_base_classifier(),
                confidence_level=confidence,
                cv=5,
                random_state=random_state,
            )
            model.fit_conformalize(x_fit, y_fit)
            disclosures.append("MAPIE CrossConformalClassifier (CV+) on Session train.")
        elif method == "jackknife_plus":
            from mapie.classification import CrossConformalClassifier

            model = CrossConformalClassifier(
                estimator=_base_classifier(),
                confidence_level=confidence,
                cv=-1,
                random_state=random_state,
            )
            model.fit_conformalize(x_fit, y_fit)
            disclosures.append(
                "MAPIE CrossConformalClassifier (jackknife+, cv=-1) on Session train."
            )
        else:
            raise ValidationError(f"Unknown MAPIE method '{method}'.")

    return (
        MapieWrapper(task=task, method=method, alpha=alpha, estimator=model, api="modern"),
        disclosures,
    )


def _fit_legacy_mapie(
    *,
    method: MapieMethod,
    task: str,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_cal: np.ndarray | None,
    y_cal: np.ndarray | None,
    random_state: int | None,
    alpha: float,
) -> tuple[MapieWrapper, list[str]]:
    disclosures: list[str] = []
    if task == "regression":
        from mapie.regression import MapieRegressor

        base = _base_regressor()
        if method == "split":
            if x_cal is None or y_cal is None:
                raise ValidationError(
                    "MAPIE split conformal requires a train-only calibration carve."
                )
            base.fit(x_fit, y_fit)
            model = MapieRegressor(estimator=base, cv="prefit", method="base")
            model = model.fit(x_cal, y_cal)
            disclosures.append("Legacy MAPIE split (prefit/base) on train carve.")
        elif method == "cv_plus":
            model = MapieRegressor(
                estimator=_base_regressor(),
                cv=5,
                method="plus",
                random_state=random_state,
            )
            model.fit(x_fit, y_fit)
            disclosures.append("Legacy MAPIE CV+ on Session train.")
        elif method == "jackknife_plus":
            model = MapieRegressor(
                estimator=_base_regressor(),
                cv=-1,
                method="plus",
                random_state=random_state,
            )
            model.fit(x_fit, y_fit)
            disclosures.append("Legacy MAPIE jackknife+ on Session train.")
        else:
            raise ValidationError(f"Unknown MAPIE method '{method}'.")
    else:
        from mapie.classification import MapieClassifier

        base = _base_classifier()
        if method == "split":
            if x_cal is None or y_cal is None:
                raise ValidationError(
                    "MAPIE split conformal requires a train-only calibration carve."
                )
            base.fit(x_fit, y_fit)
            model = MapieClassifier(estimator=base, cv="prefit", method="score")
            model = model.fit(x_cal, y_cal)
            disclosures.append("Legacy MAPIE classification split on train carve.")
        elif method in {"cv_plus", "jackknife_plus"}:
            cv = 5 if method == "cv_plus" else -1
            model = MapieClassifier(
                estimator=_base_classifier(),
                cv=cv,
                method="score",
                random_state=random_state,
            )
            model.fit(x_fit, y_fit)
            disclosures.append(f"Legacy MAPIE classification {method} on Session train.")
        else:
            raise ValidationError(f"Unknown MAPIE method '{method}'.")

    return (
        MapieWrapper(task=task, method=method, alpha=alpha, estimator=model, api="legacy"),
        disclosures,
    )


def mapie_predict_interval(
    wrapper: Any,
    x: np.ndarray,
    *,
    task: str,
    alpha: float,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], str]:
    """Return (point, lower, upper, method_label) from a fitted MAPIE wrapper."""
    handle = _as_wrapper(wrapper, task=task, alpha=alpha)
    if handle.api == "modern" and task == "regression":
        y_pred, y_pis = handle.estimator.predict_interval(x)
        point = tuple(float(v) for v in np.asarray(y_pred).ravel())
        arr = np.asarray(y_pis)
        lo = arr[:, 0, 0] if arr.ndim == 3 else arr[:, 0]
        hi = arr[:, 1, 0] if arr.ndim == 3 else arr[:, 1]
        return point, tuple(float(v) for v in lo), tuple(float(v) for v in hi), "mapie"

    if handle.api == "legacy" and task == "regression":
        alpha_arr = np.array([alpha])
        y_pred, y_pis = handle.estimator.predict(x, alpha=alpha_arr)
        point = tuple(float(v) for v in np.asarray(y_pred).ravel())
        lo = np.asarray(y_pis[:, 0, 0], dtype=float)
        hi = np.asarray(y_pis[:, 1, 0], dtype=float)
        return point, tuple(float(v) for v in lo), tuple(float(v) for v in hi), "mapie"

    raise ValidationError("mapie_predict_interval requires regression task for intervals.")


def mapie_predict_sets(
    wrapper: Any,
    x: np.ndarray,
    *,
    alpha: float,
    task: str = "classification",
) -> tuple[tuple[Any, ...], tuple[tuple[Any, ...], ...]]:
    """Classification prediction sets from MAPIE."""
    handle = _as_wrapper(wrapper, task=task, alpha=alpha)
    if handle.api == "modern":
        y_pred, y_sets = handle.estimator.predict_set(x)
        point = tuple(int(v) for v in np.asarray(y_pred).ravel())
        sets_arr = np.asarray(y_sets, dtype=bool)
        if sets_arr.ndim == 3:
            sets_arr = sets_arr[:, :, 0]
        classes = getattr(handle.estimator, "classes_", None)
        if classes is None:
            est = getattr(handle.estimator, "estimator", None)
            classes = getattr(est, "classes_", None)
        sets: list[tuple[Any, ...]] = []
        for i, row in enumerate(sets_arr):
            if classes is not None:
                members = tuple(classes[j] for j in range(len(row)) if row[j])
            else:
                members = tuple(int(j) for j in range(len(row)) if row[j])
            if not members:
                members = (point[i],)
            sets.append(members)
        return point, tuple(sets)

    alpha_arr = np.array([alpha])
    y_pred, y_pis = handle.estimator.predict(x, alpha=alpha_arr)
    point = tuple(int(v) for v in np.asarray(y_pred).ravel())
    sets_arr = np.asarray(y_pis, dtype=bool)
    if sets_arr.ndim == 3:
        sets_arr = sets_arr[:, :, 0]
    classes = getattr(handle.estimator, "classes_", None)
    sets = []
    for i, row in enumerate(sets_arr):
        if classes is not None:
            members = tuple(classes[j] for j in range(len(row)) if row[j])
        else:
            members = tuple(int(j) for j in range(len(row)) if row[j])
        if not members:
            members = (point[i],)
        sets.append(members)
    return point, tuple(sets)


def mapie_supports_return_std() -> bool:
    return False


def mapie_supports_predict_proba(wrapper: Any) -> bool:
    handle = wrapper if isinstance(wrapper, MapieWrapper) else None
    est = None
    if handle is not None:
        est = getattr(handle.estimator, "estimator", handle.estimator)
    elif hasattr(wrapper, "estimator"):
        est = getattr(wrapper, "estimator", wrapper)
    return est is not None and hasattr(est, "predict_proba")


def _as_wrapper(wrapper: Any, *, task: str, alpha: float) -> MapieWrapper:
    if isinstance(wrapper, MapieWrapper):
        return wrapper
    api = "modern" if hasattr(wrapper, "predict_interval") or hasattr(wrapper, "predict_set") else "legacy"
    return MapieWrapper(task=task, method="split", alpha=alpha, estimator=wrapper, api=api)
