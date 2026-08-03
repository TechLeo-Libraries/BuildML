"""Thin Session facades over buildml.cbr."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence

import pandas as pd

from buildml.cbr.checkpoint import load_cbr_bundle, save_cbr_bundle
from buildml.cbr.evaluate import evaluate_cbr
from buildml.cbr.explain_hooks import (
    eval_result_summary,
    fit_result_summary,
    predict_result_summary,
    retain_result_summary,
    retrieve_result_summary,
)
from buildml.cbr.fit import fit_cbr
from buildml.cbr.predict import predict_cbr
from buildml.cbr.retain import retain_cbr, retain_from_indices
from buildml.cbr.retrieve import retrieve_cases
from buildml.cbr.types import CbrAdaptMode, CbrMetric, CbrReuseMode, CbrTask
from buildml.core.errors import ValidationError
from buildml.data.splits import PartitionName

PartitionOrAll = PartitionName | Literal["all"]


def fit_cbr_op(
    session,
    *,
    backend: str | None = None,
    task: CbrTask | None = None,
    metric: CbrMetric = "euclidean",
    reuse: CbrReuseMode = "distance_weighted",
    adapt: CbrAdaptMode = "none",
    k: int = 5,
    columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    text_columns: list[str] | None = None,
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    standardize: bool = True,
    distance_eps: float = 1e-8,
    random_state: int | None = 0,
    prefer_reduce_components: bool = True,
    torch_epochs: int = 40,
    device: str = "cpu",
) -> Any:
    """Build a case base from Session train.

    Delegates to :func:`buildml.cbr.fit.fit_cbr`, stores the
    :class:`~buildml.cbr.results.CbrPlan` on Session, and records the fit.
    Follow with :func:`retrieve_cases_op` or :func:`predict_cbr_op`.

    Parameters
    ----------
    session:
        Active Session with dataset and split plan attached.
    backend:
        Optional backend override (see CBR capability matrix).
    task:
        Optional task override (classification/regression).
    metric:
        Case distance metric (``euclidean`` by default).
    reuse:
        Reuse mode for combining retrieved cases.
    adapt:
        Adaptation mode applied after retrieval.
    k:
        Default number of neighbors to retrieve.
    columns:
        Optional explicit feature column list.
    categorical_columns:
        Optional categorical feature columns for mixed distances.
    text_columns:
        Optional text columns for embedding-based retrieval.
    text_model_name:
        Sentence-transformer model for text columns.
    standardize:
        When True, standardize numeric features on train.
    distance_eps:
        Epsilon added to distances for numerical stability.
    random_state:
        Seed for stochastic steps.
    prefer_reduce_components:
        Prefer reduced component columns when a reduce plan exists.
    torch_epochs:
        Training epochs for torch embedding backend.
    device:
        Torch device string for embedding backend.

    Returns
    -------
    CbrFitResult
        Serializable fit summary including case-base size and disclosures.

    Notes
    -----
    **Leakage:** Requires a split. Case memory uses train only. Honesty:
    tabular case→solution CBR — not RAG document retrieval.
    """
    session.assert_can_fit("train")
    plan, result = fit_cbr(
        session.dataset,
        session._split_plan,
        backend=backend,  # type: ignore[arg-type]
        task=task,
        metric=metric,
        reuse=reuse,
        adapt=adapt,
        k=k,
        columns=columns,
        categorical_columns=categorical_columns,
        text_columns=text_columns,
        text_model_name=text_model_name,
        standardize=standardize,
        distance_eps=distance_eps,
        random_state=random_state,
        prefer_reduce_components=prefer_reduce_components,
        reduce_plan=getattr(session, "_reduce_plan", None),
        torch_epochs=torch_epochs,
        device=device,
    )
    session._cbr_plan = plan
    session._cbr_fit_result = result
    session._cbr_eval_result = None
    session._cbr_predict_result = None
    session._cbr_retrieve_result = None
    session._cbr_retain_result = None
    session._record(
        "fit_cbr",
        {
            "backend": backend,
            "task": task,
            "metric": metric,
            "reuse": reuse,
            "adapt": adapt,
            "k": k,
            "columns": columns,
            "categorical_columns": categorical_columns,
            "text_columns": text_columns,
            "text_model_name": text_model_name,
            "standardize": standardize,
            "distance_eps": distance_eps,
            "random_state": random_state,
            "prefer_reduce_components": prefer_reduce_components,
            "torch_epochs": torch_epochs,
            "device": device,
        },
        warnings=tuple(result.warnings),
        result_summary=fit_result_summary(result),
    )
    return result


def retrieve_cases_op(
    session,
    *,
    partition: PartitionOrAll = "test",
    k: int | None = None,
    backend: str | None = None,
) -> Any:
    """Retrieve k nearest cases for a partition (no reuse).

    Delegates to :func:`buildml.cbr.retrieve.retrieve_cases` for inspection
    without applying a reuse/adapt policy.

    Parameters
    ----------
    session:
        Active Session with a CBR plan from :func:`fit_cbr_op`.
    partition:
        Partition to retrieve against (``test`` by default).
    k:
        Optional neighbor override; uses plan default when omitted.
    backend:
        Optional backend override for retrieval.

    Returns
    -------
    CbrRetrieveResult
        Retrieved cases and distance traces for each query row.

    Raises
    ------
    ValidationError
        When no CBR plan exists on the Session.
    """
    plan = getattr(session, "_cbr_plan", None)
    if plan is None:
        raise ValidationError("No CBR plan. Call fit_cbr(...) first.")
    result = retrieve_cases(
        session.dataset,
        plan,
        session._split_plan,
        partition=partition,
        k=k,
        backend=backend,
    )
    session._cbr_retrieve_result = result
    session._record(
        "retrieve_cases",
        {"partition": partition, "k": k, "backend": backend},
        warnings=tuple(result.warnings),
        result_summary=retrieve_result_summary(result),
    )
    return result


def predict_cbr_op(
    session,
    *,
    partition: PartitionOrAll = "test",
    k: int | None = None,
    return_traces: bool = True,
    backend: str | None = None,
) -> Any:
    """Predict via retrieve + reuse (no case-base update).

    Delegates to :func:`buildml.cbr.predict.predict_cbr` using the fitted
    reuse/adapt policy without modifying the case base.

    Parameters
    ----------
    session:
        Active Session with a CBR plan from :func:`fit_cbr_op`.
    partition:
        Partition to predict on (``test`` by default).
    k:
        Optional neighbor override; uses plan default when omitted.
    return_traces:
        When True, include retrieval/reuse traces in the result.
    backend:
        Optional backend override for prediction.

    Returns
    -------
    CbrPredictResult
        Predictions and optional retrieval traces for the partition.

    Raises
    ------
    ValidationError
        When no CBR plan exists on the Session.
    """
    plan = getattr(session, "_cbr_plan", None)
    if plan is None:
        raise ValidationError("No CBR plan. Call fit_cbr(...) first.")
    result = predict_cbr(
        session.dataset,
        plan,
        session._split_plan,
        partition=partition,
        k=k,
        return_traces=return_traces,
        backend=backend,
    )
    session._cbr_predict_result = result
    session._record(
        "predict_cbr",
        {
            "partition": partition,
            "k": k,
            "return_traces": return_traces,
            "backend": backend,
        },
        warnings=tuple(result.warnings),
        result_summary=predict_result_summary(result),
    )
    return result


def evaluate_cbr_op(
    session,
    *,
    partition: PartitionOrAll = "validation",
    k: int | None = None,
) -> Any:
    """Evaluate CBR on a holdout partition.

    Delegates to :func:`buildml.cbr.evaluate.evaluate_cbr` using the frozen
    train case base. Falls back to ``test`` when validation is empty.

    Parameters
    ----------
    session:
        Active Session with a CBR plan from :func:`fit_cbr_op`.
    partition:
        Holdout partition for evaluation (``validation`` by default).
    k:
        Optional neighbor override; uses plan default when omitted.

    Returns
    -------
    CbrEvalResult
        Holdout metrics and retrieval disclosures.

    Raises
    ------
    ValidationError
        When no CBR plan exists on the Session.
    """
    plan = getattr(session, "_cbr_plan", None)
    if plan is None:
        raise ValidationError("No CBR plan. Call fit_cbr(...) first.")
    resolved: PartitionOrAll = partition
    split = session._split_plan
    if (
        partition == "validation"
        and split is not None
        and not split.validation_indices
    ):
        resolved = "test"
    result = evaluate_cbr(
        session.dataset,
        plan,
        session._split_plan,
        partition=resolved,
        k=k,
    )
    session._cbr_eval_result = result
    session._record(
        "evaluate_cbr",
        {"partition": resolved, "k": k},
        warnings=tuple(result.warnings),
        result_summary=eval_result_summary(result),
    )
    return result


def retain_cbr_op(
    session,
    *,
    labeled_frame: pd.DataFrame | None = None,
    row_indices: Sequence[Any] | None = None,
    solution_column: str | None = None,
    source_disclosure: str,
    allow_overlap_with_train: bool = True,
) -> Any:
    """Retain new labeled cases (refuses Session validation/test indices).

    Delegates to :func:`buildml.cbr.retain.retain_cbr` or
    :func:`buildml.cbr.retain.retain_from_indices` to grow the case base
    with explicit source disclosure.

    Parameters
    ----------
    session:
        Active Session with a CBR plan from :func:`fit_cbr_op`.
    labeled_frame:
        Optional frame of new labeled rows to retain.
    row_indices:
        Optional dataset row indices to retain (mutually exclusive with
        ``labeled_frame``).
    solution_column:
        Solution column when ``labeled_frame`` is supplied.
    source_disclosure:
        Required provenance string for retained cases.
    allow_overlap_with_train:
        When True, permit overlap between retained rows and train indices.

    Returns
    -------
    CbrRetainResult
        Retain summary including updated case-base size.

    Raises
    ------
    ValidationError
        When no CBR plan exists or retain inputs are invalid.
    """
    plan = getattr(session, "_cbr_plan", None)
    if plan is None:
        raise ValidationError("No CBR plan. Call fit_cbr(...) first.")
    if labeled_frame is None and row_indices is None:
        raise ValidationError(
            "retain_cbr requires labeled_frame=... or row_indices=..."
        )
    if labeled_frame is not None and row_indices is not None:
        raise ValidationError(
            "Pass only one of labeled_frame or row_indices."
        )
    if row_indices is not None:
        new_plan, result = retain_from_indices(
            session.dataset,
            plan,
            session._split_plan,
            row_indices=row_indices,
            source_disclosure=source_disclosure,
        )
    else:
        assert labeled_frame is not None
        new_plan, result = retain_cbr(
            session.dataset,
            plan,
            session._split_plan,
            labeled_frame=labeled_frame,
            solution_column=solution_column,
            source_disclosure=source_disclosure,
            allow_overlap_with_train=allow_overlap_with_train,
        )
    session._cbr_plan = new_plan
    session._cbr_retain_result = result
    session._cbr_eval_result = None
    session._cbr_predict_result = None
    session._record(
        "retain_cbr",
        {
            "n_labeled_frame_rows": (
                None if labeled_frame is None else len(labeled_frame)
            ),
            "n_row_indices": None if row_indices is None else len(list(row_indices)),
            "solution_column": solution_column,
            "source_disclosure": source_disclosure,
            "allow_overlap_with_train": allow_overlap_with_train,
        },
        warnings=tuple(result.warnings),
        result_summary=retain_result_summary(result),
    )
    return result


def save_cbr_bundle_op(session, path: str | Path) -> Path:
    """Persist the active CbrPlan.

    Delegates to :func:`buildml.cbr.checkpoint.save_cbr_bundle`.
    Reload with :func:`load_cbr_bundle_op`.

    Parameters
    ----------
    session:
        Active Session with a CBR plan from :func:`fit_cbr_op`.
    path:
        Destination directory for the bundle (created if missing).

    Returns
    -------
    pathlib.Path
        Resolved bundle directory path.

    Raises
    ------
    ValidationError
        When no CBR plan exists on the Session.
    """
    plan = getattr(session, "_cbr_plan", None)
    if plan is None:
        raise ValidationError("No CBR plan. Call fit_cbr(...) first.")
    out = save_cbr_bundle(
        path,
        plan,
        fit_result=getattr(session, "_cbr_fit_result", None),
        eval_result=getattr(session, "_cbr_eval_result", None),
    )
    session._record(
        "save_cbr_bundle",
        {"path": str(out)},
        result_summary={"path": str(out), "format": "buildml.cbr_bundle.v1"},
    )
    return out


def load_cbr_bundle_op(session, path: str | Path):
    """Load a CBR bundle into this Session.

    Delegates to :func:`buildml.cbr.checkpoint.load_cbr_bundle` and clears
    prior eval/predict/retrieve/retain results.

    Parameters
    ----------
    session:
        Session instance to populate with the loaded CBR plan.
    path:
        Path to a ``buildml.cbr_bundle.v1`` directory.

    Returns
    -------
    Session
        ``session`` with CBR plan attached for chaining.
    """
    plan = load_cbr_bundle(path)
    session._cbr_plan = plan
    session._cbr_fit_result = None
    session._cbr_eval_result = None
    session._cbr_predict_result = None
    session._cbr_retrieve_result = None
    session._cbr_retain_result = None
    session._record(
        "load_cbr_bundle",
        {"path": str(path)},
        result_summary={
            "path": str(path),
            "kind": "cbr",
            "metric": plan.metric,
            "n_cases": plan.case_base.n_cases,
        },
    )
    return session
