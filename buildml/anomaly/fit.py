"""Train-fitted anomaly detectors with leakage-safe score/flag on holdout."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    IsolationForest,
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from buildml.anomaly.features import matrix_from_frame, resolve_anomaly_columns
from buildml.anomaly.results import AnomalyFitResult, AnomalyPlan
from buildml.anomaly.types import AnomalyConfig, AnomalyMethod, AnomalyMode, ThresholdPolicy
from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition, frame_for_partition


def fit_detector(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
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
    normal_label_column: str | None = None,
    normal_label_value: Any = 0,
    positive_label: Any = 1,
    prefer_reduce_components: bool = True,
    reduce_plan: Any | None = None,
    flag_column: str = "is_anomaly",
    score_column: str = "anomaly_score",
) -> tuple[AnomalyPlan, AnomalyFitResult]:
    """Fit an anomaly detector on the train partition only.

    Modes
    -----
    unsupervised:
        Fit on all train rows (may contain anomalies). Typical for IsolationForest
        with a contamination prior.
    novelty:
        Fit on a **normal-only** train subset defined by ``normal_label_column``
        / ``normal_label_value``. This is semi-supervised novelty detection —
        not unlabeled clustering and not a full fraud platform.
    supervised:
        Fit a binary classifier (``supervised_hgb``) when a binary target role
        exists. Scores are positive-class probabilities; classical imbalance
        metrics belong on ``evaluate_anomaly``.

    Score convention: higher ``anomaly_score`` = more anomalous. Thresholds and
    train alert rates are always disclosed on the plan.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None
    _validate_names(flag_column, score_column)
    if not 0.0 < float(contamination) < 0.5:
        raise ValidationError("contamination must be in (0, 0.5)")

    mode = _resolve_mode(method, mode)
    _validate_mode_method(mode, method)

    train = frame_for_partition(dataset, split_plan, "train")
    n_train = int(len(train))
    disclosures: list[str] = [
        "Anomaly scores are oriented so higher means more anomalous.",
        "Thresholds and alert rates are explicit; this is not a streaming fraud platform "
        "and makes no causal fraud claims.",
        "EDA IsolationForest screens and Session.handle_outliers fences are not this API.",
    ]
    warnings: list[str] = []

    fit_frame, fit_disclosures, fit_warnings = _select_fit_rows(
        dataset,
        train,
        mode=mode,
        method=method,
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
        estimator = HistGradientBoostingClassifier(random_state=random_state)
        estimator.fit(x_fit, y_fit)
        disclosures.append(
            "Supervised mode fits HistGradientBoostingClassifier on train labels only. "
            "Prefer PR-AUC / precision@k / recall@k under class imbalance; accuracy alone "
            "is often misleading."
        )
        # Score the full train partition for threshold calibration disclosure.
        x_train = matrix_from_frame(train, cols)
        train_scores = _anomaly_scores(estimator, method="supervised_hgb", x=x_train)
    else:
        estimator = _build_unsupervised_estimator(
            method=method,
            contamination=contamination,
            random_state=random_state,
            n_estimators=n_estimators,
            max_samples=max_samples,
            n_neighbors=n_neighbors,
            nu=nu,
            kernel=kernel,
            gamma=gamma,
        )
        estimator.fit(x_fit)
        x_train = matrix_from_frame(train, cols)
        train_scores = _anomaly_scores(estimator, method=method, x=x_train)

    threshold, thr_disclosures = _resolve_threshold(
        train_scores,
        policy=threshold_policy,
        contamination=contamination,
        score_threshold=score_threshold,
        quantile=quantile,
        method=method,
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
        method=method,
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
        normal_label_column=normal_label_column,
        normal_label_value=normal_label_value,
        positive_label=positive_label,
        prefer_reduce_components=prefer_reduce_components,
        flag_column=flag_column,
        score_column=score_column,
    )
    plan = AnomalyPlan(
        method=method,
        mode=mode,
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
        method=method,
        mode=mode,
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


def anomaly_scores(plan: AnomalyPlan, x: np.ndarray) -> np.ndarray:
    """Compute higher-is-more-anomalous scores from a frozen plan."""
    return _anomaly_scores(plan.estimator_, method=plan.method, x=x)


def _anomaly_scores(estimator: Any, *, method: str, x: np.ndarray) -> np.ndarray:
    if method == "supervised_hgb":
        proba = np.asarray(estimator.predict_proba(x), dtype=float)
        # Positive class is index 1 when classes_ are sorted [neg, pos] — use classes_.
        classes = list(getattr(estimator, "classes_", [0, 1]))
        if len(classes) == 1:
            # Degenerate single-class train — all probability mass on that class.
            return np.zeros(shape=(x.shape[0],), dtype=float)
        # Prefer class label 1 when present; else last class.
        if 1 in classes:
            idx = classes.index(1)
        else:
            idx = len(classes) - 1
        return proba[:, idx]
    if method in {"isolation_forest", "lof"}:
        # sklearn: lower score_samples => more abnormal
        return -np.asarray(estimator.score_samples(x), dtype=float)
    if method == "one_class_svm":
        # sklearn: negative decision_function => outlier
        return -np.asarray(estimator.decision_function(x), dtype=float)
    raise ValidationError(f"Unsupported anomaly method '{method}'")


def _build_unsupervised_estimator(
    *,
    method: AnomalyMethod,
    contamination: float,
    random_state: int | None,
    n_estimators: int,
    max_samples: str | int | float,
    n_neighbors: int,
    nu: float,
    kernel: str,
    gamma: str | float,
) -> Any:
    if method == "isolation_forest":
        return IsolationForest(
            n_estimators=int(n_estimators),
            max_samples=max_samples,
            contamination=float(contamination),
            random_state=random_state,
        )
    if method == "lof":
        if n_neighbors < 2:
            raise ValidationError("lof n_neighbors must be >= 2")
        return LocalOutlierFactor(
            n_neighbors=int(n_neighbors),
            contamination=float(contamination),
            novelty=True,
        )
    if method == "one_class_svm":
        if not 0.0 < float(nu) <= 1.0:
            raise ValidationError("one_class_svm nu must be in (0, 1]")
        return OneClassSVM(kernel=kernel, gamma=gamma, nu=float(nu))
    raise ValidationError(
        f"Method '{method}' is not an unsupervised/novelty detector. "
        "Use mode='supervised' with method='supervised_hgb'."
    )


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
        # Top-q fraction: threshold at (1-q) quantile of scores
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
    method: AnomalyMethod,
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
            # Allow target role as the normal/anomaly label source.
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
        if method == "supervised_hgb":
            raise ValidationError(
                "novelty mode is for unsupervised detectors; use mode='supervised' "
                "with method='supervised_hgb'."
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
        if method != "supervised_hgb":
            raise ValidationError(
                "supervised mode requires method='supervised_hgb' "
                "(binary classifier scores as anomaly probabilities)."
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
        # Fit on all labeled train rows (standard supervised contract).
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


def _resolve_mode(method: AnomalyMethod, mode: AnomalyMode) -> AnomalyMode:
    if method == "supervised_hgb" and mode != "supervised":
        return "supervised"
    return mode


def _validate_mode_method(mode: AnomalyMode, method: AnomalyMethod) -> None:
    unsupervised_methods = {"isolation_forest", "lof", "one_class_svm"}
    if mode in {"unsupervised", "novelty"} and method not in unsupervised_methods:
        raise ValidationError(
            f"mode='{mode}' supports methods {sorted(unsupervised_methods)}; "
            f"got '{method}'."
        )
    if mode == "supervised" and method != "supervised_hgb":
        raise ValidationError("mode='supervised' requires method='supervised_hgb'")


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
