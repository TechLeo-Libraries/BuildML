"""Thin Session facades over buildml.anomaly."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from buildml.anomaly.catalog import anomaly_capability_matrix
from buildml.anomaly.checkpoint import load_anomaly_bundle, save_anomaly_bundle
from buildml.anomaly.evaluate import evaluate_anomaly
from buildml.anomaly.explain_hooks import (
    eval_result_summary,
    fit_result_summary,
    score_result_summary,
)
from buildml.anomaly.fit import fit_detector
from buildml.anomaly.score import score_anomalies
from buildml.anomaly.threshold import apply_threshold_tune, tune_anomaly_threshold
from buildml.anomaly.types import (
    AnomalyBackend,
    AnomalyMethod,
    AnomalyMode,
    ThresholdPolicy,
    ThresholdTuningMetric,
)
from buildml.core.errors import ValidationError
from buildml.data.splits import PartitionName

PartitionOrAll = PartitionName | Literal["all"]


def fit_anomaly(
    session,
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
    flag_column: str = "is_anomaly",
    score_column: str = "anomaly_score",
) -> Any:
    """Fit an anomaly detector on the train partition only.

    Delegates to :func:`buildml.anomaly.fit.fit_detector`, stores the
    :class:`~buildml.anomaly.results.AnomalyPlan` on Session, and records
    the fit. ``backend`` selects sklearn (core), pyod
    (``buildml[anomaly-industry]``), or torch (``buildml[torch]``). ``method``
    must belong to the backend catalog — see :func:`anomaly_capability_matrix_op`.

    Parameters
    ----------
    session:
        Active Session with dataset and split plan attached.
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
        Number of trees for forest-based detectors.
    max_samples:
        Subsample size for isolation forest (``auto``, int, or float).
    n_neighbors:
        Neighborhood size for LOF and similar methods.
    nu:
        Upper bound on training-error fraction for one-class SVM.
    kernel:
        Kernel type for one-class SVM.
    gamma:
        Kernel coefficient for one-class SVM.
    latent_dim:
        Bottleneck width for torch autoencoder backend.
    ae_epochs:
        Training epochs for torch autoencoder backend.
    ae_batch_size:
        Minibatch size for torch autoencoder backend.
    normal_label_column:
        Optional column marking normal rows in semi-supervised setups.
    normal_label_value:
        Value in ``normal_label_column`` indicating normal rows.
    positive_label:
        Positive class label for supervised fraud scorers.
    prefer_reduce_components:
        Prefer reduced component columns when a reduce plan exists on Session.
    flag_column:
        Output column name for boolean anomaly flags when attaching scores.
    score_column:
        Output column name for continuous anomaly scores when attaching.

    Returns
    -------
    AnomalyFitResult
        Serializable fit summary including threshold and alert-rate disclosures.
        Follow with :func:`score_anomalies_op` or :func:`tune_anomaly_threshold_op`.
    """
    session.assert_can_fit("train")
    plan, result = fit_detector(
        session.dataset,
        session._split_plan,
        backend=backend,
        method=method,
        mode=mode,
        columns=columns,
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
        reduce_plan=getattr(session, "_reduce_plan", None),
        flag_column=flag_column,
        score_column=score_column,
    )
    session._anomaly_plan = plan
    session._anomaly_fit_result = result
    session._anomaly_score_result = None
    session._anomaly_eval_result = None
    session._anomaly_threshold_tune_result = None
    session._record(
        "fit_anomaly",
        {
            "backend": backend,
            "method": method,
            "mode": mode,
            "columns": columns,
            "contamination": contamination,
            "threshold_policy": threshold_policy,
            "score_threshold": score_threshold,
            "quantile": quantile,
            "normal_label_column": normal_label_column,
            "normal_label_value": normal_label_value,
            "positive_label": positive_label,
            "prefer_reduce_components": prefer_reduce_components,
            "flag_column": flag_column,
            "score_column": score_column,
        },
        warnings=tuple(result.warnings),
        result_summary=fit_result_summary(result),
    )
    return result


def tune_anomaly_threshold_op(
    session,
    *,
    partition: PartitionName = "validation",
    label_column: str | None = None,
    positive_label: Any | None = None,
    metric: ThresholdTuningMetric = "f1",
    fbeta: float = 2.0,
    allow_test_tuning: bool = False,
    update_plan: bool = True,
) -> Any:
    """Tune the anomaly decision threshold on validation labels without refitting.

    Delegates to :func:`buildml.anomaly.threshold.tune_anomaly_threshold` and
    optionally writes the tuned threshold back to the Session plan. Test
    tuning requires explicit ``allow_test_tuning=True``.

    Parameters
    ----------
    session:
        Active Session with an anomaly plan from :func:`fit_anomaly`.
    partition:
        Labeled holdout partition for threshold search (default ``validation``).
    label_column:
        Optional explicit label column; defaults to target role column.
    positive_label:
        Positive/anomaly label value for supervised tuning metrics.
    metric:
        Metric to optimize (``f1``, ``fbeta``, ``precision``, ``recall``).
    fbeta:
        Beta for F-beta when ``metric='fbeta'``.
    allow_test_tuning:
        When False, refuse tuning on the test partition.
    update_plan:
        When True, apply the tuned threshold to the active Session plan.

    Returns
    -------
    AnomalyThresholdTuneResult
        Tuned threshold, metric value, and partition used for search.

    Raises
    ------
    ValidationError
        When no anomaly plan exists or tuning preconditions fail.
    """
    plan = getattr(session, "_anomaly_plan", None)
    if plan is None:
        raise ValidationError("No anomaly plan. Call fit_anomaly(...) first.")
    resolved = partition
    split = session._split_plan
    if (
        partition == "validation"
        and split is not None
        and not split.validation_indices
    ):
        resolved = "train"
    result = tune_anomaly_threshold(
        session.dataset,
        plan,
        session._split_plan,
        partition=resolved,
        label_column=label_column,
        positive_label=positive_label,
        metric=metric,
        fbeta=fbeta,
        allow_test_tuning=allow_test_tuning,
    )
    if update_plan:
        apply_threshold_tune(plan, result)
    session._anomaly_threshold_tune_result = result
    session._anomaly_score_result = None
    session._anomaly_eval_result = None
    session._record(
        "tune_anomaly_threshold",
        {
            "partition": resolved,
            "metric": metric,
            "update_plan": update_plan,
            "threshold": result.threshold,
        },
        result_summary=result.to_dict(),
    )
    return result


def anomaly_capability_matrix_op() -> dict[str, Any]:
    """Return the anomaly backend/method capability matrix for this install.

    Delegates to :func:`buildml.anomaly.catalog.anomaly_capability_matrix`.
    Use before :func:`fit_anomaly` to confirm ``backend`` and ``method`` pairs
    available with current extras.

    Returns
    -------
    dict
        Nested map of backend identifiers to supported methods and modes.
    """
    return anomaly_capability_matrix()


def score_anomalies_op(
    session,
    *,
    partition: PartitionOrAll = "test",
    attach: bool = False,
    override_threshold: float | None = None,
) -> Any:
    """Score and flag rows with the train-fitted anomaly plan without refitting.

    Delegates to :func:`buildml.anomaly.score.score_anomalies`. When
    ``attach=True``, score and flag columns are merged into Session dataset.

    Parameters
    ----------
    session:
        Active Session with an anomaly plan from :func:`fit_anomaly`.
    partition:
        Partition to score (``train``, ``validation``, ``test``, or ``all``).
    attach:
        When True, attach score and flag columns to the Session dataset frame.
    override_threshold:
        Optional score cutoff overriding the plan threshold for this call.

    Returns
    -------
    AnomalyScoreResult
        Scores, flags, and optional alert-rate summary for the partition.

    Raises
    ------
    ValidationError
        When no anomaly plan exists on the Session.
    """
    plan = getattr(session, "_anomaly_plan", None)
    if plan is None:
        raise ValidationError("No anomaly plan. Call fit_anomaly(...) first.")
    new_dataset, result = score_anomalies(
        session.dataset,
        plan,
        session._split_plan,
        partition=partition,
        attach=attach,
        override_threshold=override_threshold,
    )
    if new_dataset is not None:
        session._dataset = new_dataset
    session._anomaly_score_result = result
    session._record(
        "score_anomalies",
        {
            "partition": partition,
            "attach": attach,
            "override_threshold": override_threshold,
        },
        result_summary=score_result_summary(result),
    )
    return result


def evaluate_anomaly_op(
    session,
    *,
    partition: PartitionOrAll = "validation",
    label_column: str | None = None,
    positive_label: Any | None = None,
    k: int | None = None,
    override_threshold: float | None = None,
) -> Any:
    """Evaluate train-fitted anomaly scores on a labeled holdout partition.

    Delegates to :func:`buildml.anomaly.evaluate.evaluate_anomaly`. Requires
    labels on the holdout partition; does not refit the detector.

    Parameters
    ----------
    session:
        Active Session with an anomaly plan from :func:`fit_anomaly`.
    partition:
        Holdout partition to score. Validation falls back to test when absent.
    label_column:
        Optional explicit label column; defaults to target role column.
    positive_label:
        Positive/anomaly label value for supervised metrics.
    k:
        Optional top-k for precision@k / recall@k style metrics.
    override_threshold:
        Optional score cutoff overriding the plan threshold for this evaluation.

    Returns
    -------
    AnomalyEvalResult
        Holdout classification metrics and ranking diagnostics when labels exist.

    Raises
    ------
    ValidationError
        When no anomaly plan exists on the Session.
    """
    plan = getattr(session, "_anomaly_plan", None)
    if plan is None:
        raise ValidationError("No anomaly plan. Call fit_anomaly(...) first.")
    resolved: PartitionOrAll = partition
    split = session._split_plan
    if (
        partition == "validation"
        and split is not None
        and not split.validation_indices
    ):
        resolved = "test"
    result = evaluate_anomaly(
        session.dataset,
        plan,
        session._split_plan,
        partition=resolved,
        label_column=label_column,
        positive_label=positive_label,
        k=k,
        override_threshold=override_threshold,
    )
    session._anomaly_eval_result = result
    session._record(
        "evaluate_anomaly",
        {
            "partition": resolved,
            "label_column": label_column,
            "positive_label": positive_label,
            "k": k,
            "override_threshold": override_threshold,
        },
        warnings=tuple(result.warnings),
        result_summary=eval_result_summary(result),
    )
    return result


def save_anomaly_bundle_op(session, path: str | Path) -> Path:
    """Persist the active anomaly plan as ``buildml.anomaly_bundle.v1``.

    Delegates to :func:`buildml.anomaly.checkpoint.save_anomaly_bundle`.
    Reload with :func:`load_anomaly_bundle_op`.

    Parameters
    ----------
    session:
        Active Session with an anomaly plan from :func:`fit_anomaly`.
    path:
        Destination directory for the bundle (created if missing).

    Returns
    -------
    pathlib.Path
        Resolved bundle directory path.

    Raises
    ------
    ValidationError
        When no anomaly plan exists on the Session.
    """
    plan = getattr(session, "_anomaly_plan", None)
    if plan is None:
        raise ValidationError("No anomaly plan. Call fit_anomaly(...) first.")
    out = save_anomaly_bundle(
        path,
        plan,
        fit_result=getattr(session, "_anomaly_fit_result", None),
        eval_result=getattr(session, "_anomaly_eval_result", None),
    )
    session._record(
        "save_anomaly_bundle",
        {"path": str(out)},
        result_summary={
            "path": str(out),
            "method": plan.method,
            "backend": plan.backend,
            "mode": plan.mode,
            "threshold": plan.threshold_,
        },
    )
    return out


def load_anomaly_bundle_op(session, path: str | Path) -> Any:
    """Load an anomaly bundle into this Session.

    Delegates to :func:`buildml.anomaly.checkpoint.load_anomaly_bundle` and
    clears prior score/eval/threshold-tune results.

    Parameters
    ----------
    session:
        Session instance to populate with the loaded anomaly plan.
    path:
        Path to a ``buildml.anomaly_bundle.v1`` directory.

    Returns
    -------
    Session
        ``session`` with anomaly plan attached for chaining.
    """
    plan = load_anomaly_bundle(path)
    session._anomaly_plan = plan
    session._anomaly_fit_result = None
    session._anomaly_score_result = None
    session._anomaly_eval_result = None
    session._anomaly_threshold_tune_result = None
    session._record(
        "load_anomaly_bundle",
        {
            "path": str(path),
            "method": plan.method,
            "backend": plan.backend,
            "mode": plan.mode,
        },
        result_summary=plan.to_dict(),
    )
    return session
