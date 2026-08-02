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
    task: CbrTask | None = None,
    metric: CbrMetric = "euclidean",
    reuse: CbrReuseMode = "distance_weighted",
    adapt: CbrAdaptMode = "none",
    k: int = 5,
    columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    standardize: bool = True,
    distance_eps: float = 1e-8,
    random_state: int | None = 0,
    prefer_reduce_components: bool = True,
) -> Any:
    """Build a case base from Session train.

    Notes
    -----
    **Leakage:** Requires a split. Case memory uses train only. Honesty:
    tabular case→solution CBR — not RAG document retrieval.
    """
    session.assert_can_fit("train")
    plan, result = fit_cbr(
        session.dataset,
        session._split_plan,
        task=task,
        metric=metric,
        reuse=reuse,
        adapt=adapt,
        k=k,
        columns=columns,
        categorical_columns=categorical_columns,
        standardize=standardize,
        distance_eps=distance_eps,
        random_state=random_state,
        prefer_reduce_components=prefer_reduce_components,
        reduce_plan=getattr(session, "_reduce_plan", None),
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
            "task": task,
            "metric": metric,
            "reuse": reuse,
            "adapt": adapt,
            "k": k,
            "columns": columns,
            "categorical_columns": categorical_columns,
            "standardize": standardize,
            "distance_eps": distance_eps,
            "random_state": random_state,
            "prefer_reduce_components": prefer_reduce_components,
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
) -> Any:
    """Retrieve k nearest cases for a partition (no reuse)."""
    plan = getattr(session, "_cbr_plan", None)
    if plan is None:
        raise ValidationError("No CBR plan. Call fit_cbr(...) first.")
    result = retrieve_cases(
        session.dataset,
        plan,
        session._split_plan,
        partition=partition,
        k=k,
    )
    session._cbr_retrieve_result = result
    session._record(
        "retrieve_cases",
        {"partition": partition, "k": k},
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
) -> Any:
    """Predict via retrieve + reuse (no case-base update)."""
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
    )
    session._cbr_predict_result = result
    session._record(
        "predict_cbr",
        {
            "partition": partition,
            "k": k,
            "return_traces": return_traces,
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
    """Evaluate CBR on a holdout partition."""
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
    """Retain new labeled cases (refuses Session validation/test indices)."""
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
    """Persist the active CbrPlan."""
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
    """Load a CBR bundle into this Session."""
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
