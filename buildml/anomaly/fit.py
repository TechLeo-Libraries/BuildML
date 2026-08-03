"""Train-fitted anomaly detectors with leakage-safe score/flag on holdout."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from buildml.anomaly.adapters.pyod import build_pyod_estimator, pyod_anomaly_scores
from buildml.anomaly.adapters.sklearn import (
    build_sklearn_unsupervised_estimator,
    sklearn_anomaly_scores,
)
from buildml.anomaly.adapters.supervised import (
    build_supervised_estimator,
    supervised_anomaly_scores,
)
from buildml.anomaly.adapters.torch_ae import build_torch_autoencoder, torch_ae_anomaly_scores
from buildml.anomaly.catalog import resolve_backend_method
from buildml.anomaly.features import matrix_from_frame, resolve_anomaly_columns
from buildml.anomaly.results import AnomalyFitResult, AnomalyPlan
from buildml.anomaly.types import (
    AnomalyBackend,
    AnomalyConfig,
    AnomalyMethod,
    AnomalyMode,
    ThresholdPolicy,
)
from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition, frame_for_partition

SUPERVISED_METHODS = {"supervised_hgb", "supervised_xgb", "supervised_lgbm"}
SKLEARN_METHODS = {"isolation_forest", "lof", "one_class_svm"}
PYOD_METHODS = {"hbos", "copod", "ecod", "deepsvdd"}
TORCH_METHODS = {"autoencoder"}


def fit_detector(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    backend: AnomalyBackend | None = None,
    method: AnomalyMethod = "isolation_forest",
    mode: AnomalyMode = "unsupervised",
    columns: list[str] | None = None,
    random_state: int | None = 0,
    contamination: float = 0.05,
    threshold_policy: ThresholdPolicy = "contamination",
    score_threshold: float | None = None,
    quantile: float | None = None,
    n_estimators: int = 100,
    max_samples: str | int | float = "auto",
    n_neighbors: int = 20,
    nu: float = 0.05,
    kernel: str = "rbf",
    gamma: str | float = "scale",
    latent_dim: int = 8,
    ae_epochs: int = 40,
    ae_batch_size: int = 64,
    normal_label_column: str | None = None,
    normal_label_value: Any = 0,
    positive_label: Any = 1,
    prefer_reduce_components: bool = True,
    reduce_plan: Any | None = None,
    flag_column: str = "is_anomaly",
    score_column: str = "anomaly_score",
) -> tuple[AnomalyPlan, AnomalyFitResult]:
    """Fit an anomaly detector on the train partition only.

Backends
--------
sklearn (default):
    IsolationForest, LOF, One-Class SVM — core dependency.
pyod (``buildml[anomaly-industry]``):
    HBOS, COPOD, ECOD, DeepSVDD industry detectors.
torch (``buildml[torch]``):
    Tabular autoencoder reconstruction-error scoring.
Supervised fraud scorers (``mode='supervised'``):
    ``supervised_hgb`` (core), ``supervised_xgb`` / ``supervised_lgbm``
    when ``buildml[anomaly-industry]`` is installed.
Score convention: higher ``anomaly_score`` = more anomalous. Thresholds and
train alert rates are always disclosed on the plan.

Parameters
----------
dataset:
    BuildML dataset with features, target, and role metadata.
split_plan:
    Train/validation/test split; fit uses train partition only.
backend:
    Optional backend override (see capability matrix for identifiers).
method:
    Method or strategy identifier for the resolved backend.
mode:
    Anomaly detection mode (``unsupervised`` or ``supervised``).
columns:
    Optional explicit feature column list; ``None`` auto-selects numerics.
random_state:
    Seed for stochastic steps (sampling, initialization, bagging).
contamination:
    Expected outlier fraction for sklearn-style detectors.
threshold_policy:
    How the decision threshold is chosen from train scores.
score_threshold:
    Fixed score cutoff when threshold policy is ``fixed``.
quantile:
    Quantile for score threshold when policy is ``quantile``.
n_estimators:
    n estimators (int).
max_samples:
    max samples (str | int | float).
n_neighbors:
    n neighbors (int).
nu:
    nu (float).
kernel:
    kernel (str).
gamma:
    gamma (str | float).
latent_dim:
    latent dim (int).
ae_epochs:
    ae epochs (int).
ae_batch_size:
    ae batch size (int).
normal_label_column:
    normal label column (str | None).
normal_label_value:
    normal label value (Any).
positive_label:
    positive label (Any).
prefer_reduce_components:
    Prefer reduced component columns when a reduce plan exists.
reduce_plan:
    Optional preprocess reduce plan from Session.
flag_column:
    flag column (str).
score_column:
    score column (str).

Returns
-------
tuple[AnomalyPlan, AnomalyFitResult]
    Tuple of results (tuple[AnomalyPlan, AnomalyFitResult]) for downstream Session steps.

Raises
------
ValidationError
    When preconditions for this operation are not met.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None
    _validate_names(flag_column, score_column)
    if not 0.0 < float(contamination) < 0.5:
        raise ValidationError("contamination must be in (0, 0.5)")
    if threshold_policy == "validation_tuned":
        raise ValidationError(
            "threshold_policy='validation_tuned' is set by tune_anomaly_threshold "
            "after fit. Fit with contamination/quantile/score_threshold first."
        )

    mode = _resolve_mode(method, mode)
    resolved_backend, resolved_method = resolve_backend_method(
        backend=backend, method=method, mode=mode
    )
    _validate_mode_method(mode, resolved_method)

    train = frame_for_partition(dataset, split_plan, "train")
    n_train = int(len(train))
    disclosures: list[str] = [
        "Anomaly scores are oriented so higher means more anomalous.",
        "Thresholds and alert rates are explicit; this is not a streaming fraud platform "
        "and makes no causal fraud claims.",
        "EDA IsolationForest screens and Session.handle_outliers fences are not this API.",
        f"Backend={resolved_backend}, method={resolved_method}.",
    ]
    disclosures.extend(_score_calibration_disclosures(resolved_backend, resolved_method))
    warnings: list[str] = []

    fit_frame, fit_disclosures, fit_warnings = _select_fit_rows(
        dataset,
        train,
        mode=mode,
        method=resolved_method,
        normal_label_column=normal_label_column,
        normal_label_value=normal_label_value,
        positive_label=positive_label,
    )
    disclosures.extend(fit_disclosures)
    warnings.extend(fit_warnings)
    n_fit = int(len(fit_frame))
    if n_fit < 5:
        raise ValidationError(
            f"Need at least 5 rows to fit an anomaly detector; got n_fit_rows={n_fit}."
        )

    exclude = {c for c in (normal_label_column, flag_column, score_column) if c}
    cols, used_reduce, col_disclosures = resolve_anomaly_columns(
        dataset,
        fit_frame,
        columns,
        reduce_plan=reduce_plan,
        prefer_reduce_components=prefer_reduce_components,
        extra_exclude=exclude,
    )
    disclosures.extend(col_disclosures)
    x_fit = matrix_from_frame(fit_frame, cols)

    if mode == "supervised":
        y_fit = _binary_labels(
            fit_frame,
            dataset,
            positive_label=positive_label,
            label_column=None,
        )
        estimator = build_supervised_estimator(
            method=resolved_method,  # type: ignore[arg-type]
            random_state=random_state,
        )
        estimator.fit(x_fit, y_fit)
        disclosures.append(
            f"Supervised mode fits {resolved_method} on train labels only. "
            "Prefer PR-AUC / precision@k / recall@k under class imbalance; accuracy alone "
            "is often misleading."
        )
        x_train = matrix_from_frame(train, cols)
        train_scores = anomaly_scores(
            estimator,
            backend=resolved_backend,
            method=resolved_method,
            x=x_train,
        )
    else:
        estimator = _build_unsupervised_estimator(
            backend=resolved_backend,
            method=resolved_method,
            x_fit=x_fit,
            contamination=contamination,
            random_state=random_state,
            n_estimators=n_estimators,
            max_samples=max_samples,
            n_neighbors=n_neighbors,
            nu=nu,
            kernel=kernel,
            gamma=gamma,
            latent_dim=latent_dim,
            ae_epochs=ae_epochs,
            ae_batch_size=ae_batch_size,
        )
        x_train = matrix_from_frame(train, cols)
        train_scores = anomaly_scores(
            estimator,
            backend=resolved_backend,
            method=resolved_method,
            x=x_train,
        )

    threshold, thr_disclosures = _resolve_threshold(
        train_scores,
        policy=threshold_policy,
        contamination=contamination,
        score_threshold=score_threshold,
        quantile=quantile,
        method=resolved_method,
        mode=mode,
    )
    disclosures.extend(thr_disclosures)
    train_flags = train_scores >= threshold
    train_alert_rate = float(train_flags.mean()) if len(train_flags) else 0.0
    score_stats = _score_stats(train_scores)

    if abs(train_alert_rate - float(contamination)) > 0.15 and threshold_policy == "contamination":
        warnings.append(
            f"Train alert rate {train_alert_rate:.3f} differs from requested "
            f"contamination={contamination:.3f}; score ties or discrete score mass can "
            "shift realized rates."
        )

    config = AnomalyConfig(
        method=resolved_method,  # type: ignore[arg-type]
        backend=resolved_backend,
        mode=mode,
        columns=tuple(cols),
        random_state=random_state,
        contamination=contamination,
        threshold_policy=threshold_policy,
        score_threshold=score_threshold,
        quantile=quantile,
        n_estimators=n_estimators,
        max_samples=max_samples,
        n_neighbors=n_neighbors,
        nu=nu,
        kernel=kernel,
        gamma=gamma,
        latent_dim=latent_dim,
        ae_epochs=ae_epochs,
        ae_batch_size=ae_batch_size,
        normal_label_column=normal_label_column,
        normal_label_value=normal_label_value,
        positive_label=positive_label,
        prefer_reduce_components=prefer_reduce_components,
        flag_column=flag_column,
        score_column=score_column,
    )
    plan = AnomalyPlan(
        method=resolved_method,
        mode=mode,
        backend=resolved_backend,
        columns=tuple(cols),
        n_train_rows=n_train,
        n_fit_rows=n_fit,
        threshold_policy=threshold_policy,
        threshold_=float(threshold),
        contamination=float(contamination),
        train_alert_rate_=train_alert_rate,
        train_score_stats_=score_stats,
        flag_column=flag_column,
        score_column=score_column,
        estimator_=estimator,
        positive_label=positive_label,
        normal_label_column=normal_label_column,
        normal_label_value=normal_label_value,
        disclosures=tuple(dict.fromkeys(disclosures)),
        warnings=tuple(warnings),
        used_reduce_components=used_reduce,
        config=config.to_dict(),
    )
    result = AnomalyFitResult(
        method=resolved_method,
        mode=mode,
        backend=resolved_backend,
        n_train_rows=n_train,
        n_fit_rows=n_fit,
        columns=tuple(cols),
        threshold_policy=threshold_policy,
        threshold=float(threshold),
        contamination=float(contamination),
        train_alert_rate=train_alert_rate,
        train_score_stats=score_stats,
        used_reduce_components=used_reduce,
        disclosures=tuple(dict.fromkeys(disclosures)),
        warnings=tuple(warnings),
    )
    return plan, result


def anomaly_scores(
    plan_or_estimator: AnomalyPlan | Any,
    x: np.ndarray | None = None,
    *,
    backend: str | None = None,
    method: str | None = None,
) -> np.ndarray:
    """Compute higher-is-more-anomalous scores from a frozen plan or raw estimator.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
plan_or_estimator:
    plan or estimator (AnomalyPlan | Any).
x:
    Feature matrix input rows.
backend:
    Optional backend override (see capability matrix for identifiers).
method:
    Method or strategy identifier for the resolved backend.

Returns
-------
np.ndarray
    NumPy array aligned with input rows.

Raises
------
ValidationError
    When preconditions for this operation are not met.
    """
    if isinstance(plan_or_estimator, AnomalyPlan):
        plan = plan_or_estimator
        if x is None:
            raise ValidationError("x is required when passing an AnomalyPlan")
        return _anomaly_scores(
            plan.estimator_,
            backend=plan.backend,
            method=plan.method,
            x=x,
        )
    if x is None:
        raise ValidationError("x is required")
    if backend is None or method is None:
        raise ValidationError("backend and method are required for raw estimator scoring")
    return _anomaly_scores(plan_or_estimator, backend=backend, method=method, x=x)


def _anomaly_scores(
    estimator: Any,
    *,
    backend: str,
    method: str,
    x: np.ndarray,
) -> np.ndarray:
    if method in SUPERVISED_METHODS:
        return supervised_anomaly_scores(estimator, method=method, x=x)
    if backend == "sklearn":
        return sklearn_anomaly_scores(estimator, method=method, x=x)
    if backend == "pyod":
        return pyod_anomaly_scores(estimator, method=method, x=x)
    if backend == "torch" and method == "autoencoder":
        return torch_ae_anomaly_scores(estimator, x=x)
    raise ValidationError(
        f"Unsupported anomaly scoring backend='{backend}' method='{method}'"
    )


def _build_unsupervised_estimator(
    *,
    backend: str,
    method: str,
    x_fit: np.ndarray,
    contamination: float,
    random_state: int | None,
    n_estimators: int,
    max_samples: str | int | float,
    n_neighbors: int,
    nu: float,
    kernel: str,
    gamma: str | float,
    latent_dim: int,
    ae_epochs: int,
    ae_batch_size: int,
) -> Any:
    if backend == "sklearn":
        est = build_sklearn_unsupervised_estimator(
            method=method,  # type: ignore[arg-type]
            contamination=contamination,
            random_state=random_state,
            n_estimators=n_estimators,
            max_samples=max_samples,
            n_neighbors=n_neighbors,
            nu=nu,
            kernel=kernel,
            gamma=gamma,
        )
        est.fit(x_fit)
        return est
    if backend == "pyod":
        est = build_pyod_estimator(
            method=method,  # type: ignore[arg-type]
            contamination=contamination,
            random_state=random_state,
            n_neighbors=n_neighbors,
            n_features=int(x_fit.shape[1]),
        )
        est.fit(x_fit)
        return est
    if backend == "torch" and method == "autoencoder":
        return build_torch_autoencoder(
            x_fit,
            latent_dim=latent_dim,
            epochs=ae_epochs,
            batch_size=ae_batch_size,
            random_state=random_state,
        )
    raise ValidationError(f"Unsupported backend='{backend}' method='{method}'")


def _score_calibration_disclosures(backend: str, method: str) -> list[str]:
    if backend == "sklearn" and method in SKLEARN_METHODS:
        return [
            "Score calibration (sklearn): inverted score_samples / decision_function "
            "so higher anomaly_score = more anomalous."
        ]
    if backend == "pyod":
        return [
            "Score calibration (PyOD): decision_function scores; higher = more anomalous. "
            "Holdout score scale may differ from sklearn detectors — compare via ranking metrics."
        ]
    if backend == "torch" and method == "autoencoder":
        return [
            "Score calibration (torch AE): per-row MSE reconstruction error on train-fitted "
            "autoencoder; not calibrated to probability — use validation threshold tuning."
        ]
    if method in SUPERVISED_METHODS:
        return [
            f"Score calibration ({method}): positive-class probability; "
            "not guaranteed well-calibrated under extreme imbalance."
        ]
    return []


def _resolve_threshold(
    scores: np.ndarray,
    *,
    policy: ThresholdPolicy,
    contamination: float,
    score_threshold: float | None,
    quantile: float | None,
    method: str,
    mode: str,
) -> tuple[float, list[str]]:
    disclosures: list[str] = []
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        raise ValidationError("Cannot resolve threshold from empty score vector")

    if policy == "score_threshold":
        if score_threshold is None:
            raise ValidationError(
                "threshold_policy='score_threshold' requires score_threshold=..."
            )
        disclosures.append(
            f"Absolute score_threshold={float(score_threshold)} applied to "
            "higher-is-more-anomalous scores (flag when score >= threshold)."
        )
        return float(score_threshold), disclosures

    if policy == "decision_zero":
        if method != "one_class_svm":
            raise ValidationError(
                "threshold_policy='decision_zero' is only valid for one_class_svm "
                "(maps to anomaly_score threshold 0)."
            )
        disclosures.append(
            "decision_zero: flag when -decision_function(x) >= 0 "
            "(sklearn One-Class SVM outlier side)."
        )
        return 0.0, disclosures

    if policy == "quantile":
        q = float(contamination if quantile is None else quantile)
        if not 0.0 < q < 1.0:
            raise ValidationError("quantile must be in (0, 1)")
        thr = float(np.quantile(scores, 1.0 - q))
        disclosures.append(
            f"Quantile threshold at top {q:.4f} of train anomaly scores "
            f"(threshold={thr:.6g}). Alert rate is partition-relative to the "
            "score distribution used for calibration (train)."
        )
        return thr, disclosures

    if policy == "contamination":
        thr = float(np.quantile(scores, 1.0 - float(contamination)))
        disclosures.append(
            f"Contamination prior={float(contamination):.4f}: threshold set to the "
            f"train score quantile so ~that fraction of train rows flag "
            f"(threshold={thr:.6g}). Holdout alert rates may differ."
        )
        if mode == "novelty":
            disclosures.append(
                "Novelty fit used normal-only rows; contamination still calibrates "
                "the score threshold on the full train partition scores."
            )
        return thr, disclosures

    raise ValidationError(f"Unsupported threshold_policy '{policy}'")


def _select_fit_rows(
    dataset: Dataset,
    train: pd.DataFrame,
    *,
    mode: AnomalyMode,
    method: str,
    normal_label_column: str | None,
    normal_label_value: Any,
    positive_label: Any,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    disclosures: list[str] = []
    warnings: list[str] = []

    if mode == "unsupervised":
        disclosures.append(
            "Unsupervised mode fits on all train rows (may include anomalies). "
            "Do not treat this as novelty detection on a clean normal class."
        )
        return train, disclosures, warnings

    if mode == "novelty":
        label_col = normal_label_column
        if label_col is None:
            targets = dataset.role_columns(ColumnRole.TARGET)
            if len(targets) == 1:
                label_col = targets[0]
                disclosures.append(
                    f"novelty mode: using target role column '{label_col}' as "
                    "normal_label_column (rows with normal_label_value are fit-only)."
                )
            else:
                raise ValidationError(
                    "novelty mode requires normal_label_column (or a single target "
                    "role) defining the normal-only train subset."
                )
        if label_col not in train.columns:
            raise ValidationError(
                f"normal_label_column '{label_col}' not found on the train partition"
            )
        mask = train[label_col] == normal_label_value
        if mask.isna().any():
            raise ValidationError(
                "normal_label_column contains nulls; drop or impute before novelty fit"
            )
        fit_frame = train.loc[mask]
        n_excluded = int((~mask).sum())
        disclosures.append(
            f"Novelty / normal-only fit: kept {int(mask.sum())} train rows where "
            f"{label_col}=={normal_label_value!r}; excluded {n_excluded} row(s). "
            "Holdout scoring uses the frozen detector (no refit)."
        )
        if method in SUPERVISED_METHODS:
            raise ValidationError(
                "novelty mode is for unsupervised detectors; use mode='supervised' "
                "with a supervised scorer."
            )
        if int(mask.sum()) < 5:
            n_unique = int(pd.Series(train[label_col]).nunique(dropna=True))
            hint = ""
            if n_unique > 2:
                hint = (
                    f" Column '{label_col}' currently has {n_unique} unique values — "
                    "if it was scaled/encoded, set it as a target role (excluded from "
                    "scale) or supply an untransformed normal_label_column."
                )
            raise ValidationError(
                "novelty mode retained fewer than 5 normal train rows; "
                "check normal_label_value or collect more normal data."
                + hint
            )
        return fit_frame, disclosures, warnings

    if mode == "supervised":
        if method not in SUPERVISED_METHODS:
            raise ValidationError(
                f"supervised mode requires a supervised scorer; got '{method}'."
            )
        targets = dataset.role_columns(ColumnRole.TARGET)
        if not targets:
            raise ValidationError(
                "supervised anomaly mode requires a binary target role. "
                "Set roles with a target, or use unsupervised/novelty modes."
            )
        disclosures.append(
            "Supervised anomaly/fraud mode reuses classical binary classification "
            "patterns with imbalance-honest metrics on evaluate_anomaly. "
            "It is not graph fraud, online streaming, or causal attribution."
        )
        return train, disclosures, warnings

    raise ValidationError(f"Unsupported anomaly mode '{mode}'")


def _binary_labels(
    frame: pd.DataFrame,
    dataset: Dataset,
    *,
    positive_label: Any,
    label_column: str | None,
) -> np.ndarray:
    if label_column is not None:
        col = label_column
    else:
        targets = dataset.role_columns(ColumnRole.TARGET)
        if len(targets) != 1:
            raise ValidationError("supervised mode requires exactly one target role column")
        col = targets[0]
    if col not in frame.columns:
        raise ValidationError(f"label column '{col}' missing from frame")
    series = frame[col]
    if series.isna().any():
        raise ValidationError("target/label column contains nulls; drop or impute first")
    values = series.to_numpy()
    uniq = pd.unique(series)
    if len(uniq) != 2:
        raise ValidationError(
            f"supervised anomaly mode requires a binary label; found {len(uniq)} unique "
            f"value(s) in '{col}'."
        )
    y = (values == positive_label).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        raise ValidationError(
            f"positive_label={positive_label!r} does not create both classes on train. "
            "Check positive_label against the target values."
        )
    return y


def _resolve_mode(method: str, mode: AnomalyMode) -> AnomalyMode:
    if method in SUPERVISED_METHODS and mode != "supervised":
        return "supervised"
    return mode


def _validate_mode_method(mode: AnomalyMode, method: str) -> None:
    unsupervised_methods = SKLEARN_METHODS | PYOD_METHODS | TORCH_METHODS
    if mode in {"unsupervised", "novelty"} and method not in unsupervised_methods:
        raise ValidationError(
            f"mode='{mode}' supports unsupervised/novelty methods; got '{method}'."
        )
    if mode == "supervised" and method not in SUPERVISED_METHODS:
        raise ValidationError(
            f"mode='supervised' requires a supervised scorer; got '{method}'."
        )


def _score_stats(scores: np.ndarray) -> dict[str, float]:
    arr = np.asarray(scores, dtype=float)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
    }


def _validate_names(flag_column: str, score_column: str) -> None:
    for name, label in ((flag_column, "flag_column"), (score_column, "score_column")):
        if not name or not str(name).replace("_", "").isalnum():
            raise ValidationError(f"{label} must be a non-empty alphanumeric token")
    if flag_column == score_column:
        raise ValidationError("flag_column and score_column must differ")
