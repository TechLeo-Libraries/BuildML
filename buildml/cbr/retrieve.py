"""Retrieve k nearest cases from a train-built case memory."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from buildml.cbr.cases import (
    CaseTrace,
    encode_categoricals,
)
from buildml.cbr.features import matrix_from_frame, standardize_apply
from buildml.cbr.results import CbrPlan, CbrRetrieveResult
from buildml.cbr.retrieval_engine import retrieve_neighbor_batches
from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import PartitionName, SplitPlan, frame_for_partition

PartitionOrAll = PartitionName | Literal["all"]


def retrieve_cases(
    dataset: Dataset,
    plan: CbrPlan,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll = "test",
    k: int | None = None,
    backend: str | None = None,
) -> CbrRetrieveResult:
    """Retrieve k nearest cases for each query row (no reuse / no refit).

    Holdout queries never update the case base. Distances use the metric and
    train-fit transforms stored on ``CbrPlan.case_base``.
    """
    frame, indices = _partition_frame(dataset, split_plan, partition)
    kk = int(plan.k if k is None else k)
    if kk < 1:
        raise ValidationError("k must be >= 1.")

    q_num, q_cat = encode_query_features(frame, plan)
    memory = plan.case_base
    active_plan = _plan_with_backend(plan, backend)
    orders, drows = retrieve_neighbor_batches(
        active_plan,
        q_num,
        q_cat,
        k=kk,
        query_frame=frame if active_plan.backend == "embedding" else None,
    )
    traces: list[CaseTrace] = []
    for i in range(len(frame)):
        order = orders[i]
        neighbors = [memory.cases[j] for j in order]
        dvals = tuple(float(drows[i][j]) for j in range(len(order)))
        traces.append(
            CaseTrace(
                query_index=indices[i],
                neighbor_case_ids=tuple(c.case_id for c in neighbors),
                neighbor_row_indices=tuple(c.row_index for c in neighbors),
                distances=dvals,
                weights=(),
                neighbor_solutions=tuple(c.solution for c in neighbors),
                prediction=None,
                reuse_mode="retrieve_only",
                adapt_mode="none",
                notes=("retrieve_cases: neighbors only; no reuse applied.",),
            )
        )
    return CbrRetrieveResult(
        partition=str(partition),
        k=kk,
        metric=plan.metric,
        n_queries=len(frame),
        backend=str(active_plan.backend),
        traces=tuple(traces),
        disclosures=(
            "Retrieval is score-only against the train-built case memory.",
            *plan.disclosures[:2],
        ),
        warnings=(),
    )


def encode_query_features(
    frame: pd.DataFrame, plan: CbrPlan
) -> tuple[np.ndarray, np.ndarray]:
    """Transform a query frame with train-fit numeric/cat encodings."""
    cols = list(plan.columns)
    cat_cols = list(plan.categorical_columns)
    memory = plan.case_base

    if cols:
        x = matrix_from_frame(frame, cols)
        if (
            plan.standardize
            and plan.metric != "mixed"
            and memory.numeric_mean_ is not None
            and memory.numeric_scale_ is not None
        ):
            x = standardize_apply(x, memory.numeric_mean_, memory.numeric_scale_)
    else:
        x = np.zeros((len(frame), 0), dtype=float)

    if cat_cols:
        codes = []
        for c, vocab in zip(cat_cols, memory.cat_vocabularies_, strict=True):
            if c not in frame.columns:
                raise ValidationError(
                    f"Query frame missing categorical column {c!r}."
                )
            if frame[c].isna().any():
                raise ValidationError(
                    f"Query categorical column {c!r} has nulls."
                )
            codes.append(encode_categoricals(frame[c].tolist(), vocab))
        q_cat = np.column_stack(codes)
    else:
        q_cat = np.zeros((len(frame), 0), dtype=int)
    return x, q_cat


def neighbor_pack_for_row(
    plan: CbrPlan,
    order: np.ndarray,
    dvals: np.ndarray,
) -> tuple[list[Any], np.ndarray, np.ndarray]:
    """Return neighbor Case list, distance vector, and index order."""
    neighbors = [plan.case_base.cases[j] for j in order]
    return neighbors, dvals, order


def _plan_with_backend(plan: CbrPlan, backend: str | None) -> CbrPlan:
    if backend is None or str(backend) == str(plan.backend):
        return plan
    from buildml.cbr.catalog import resolve_backend_metric

    resolved, metric = resolve_backend_metric(
        backend=backend,  # type: ignore[arg-type]
        metric=plan.metric,
        text_columns=list(plan.text_columns) if plan.text_columns else None,
    )
    return CbrPlan(
        task=plan.task,
        backend=resolved,
        metric=metric,
        reuse=plan.reuse,
        adapt=plan.adapt,
        k=plan.k,
        columns=plan.columns,
        categorical_columns=plan.categorical_columns,
        text_columns=plan.text_columns,
        text_model_name=plan.text_model_name,
        target_column=plan.target_column,
        n_train_rows=plan.n_train_rows,
        case_base=plan.case_base,
        classes_=plan.classes_,
        label_encoder_=plan.label_encoder_,
        distance_eps=plan.distance_eps,
        standardize=plan.standardize,
        disclosures=plan.disclosures,
        warnings=plan.warnings,
        used_reduce_components=plan.used_reduce_components,
        config=plan.config,
    )


def _partition_frame(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    partition: PartitionOrAll,
) -> tuple[pd.DataFrame, list[Any]]:
    if partition == "all":
        frame = dataset._ensure_pandas()
        return frame, list(frame.index)
    if split_plan is None:
        raise ValidationError(
            "retrieve_cases / predict_cbr require a SplitPlan unless "
            "partition='all'."
        )
    frame = frame_for_partition(dataset, split_plan, partition)
    return frame, list(frame.index)
