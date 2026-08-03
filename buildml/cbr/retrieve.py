"""Find the most similar past cases, and stop there.

Retrieval is the first step of the case-based reasoning cycle, and exposing it
on its own is deliberate. Before you trust predictions, look at what the
reasoner considers similar: pull the neighbours for a handful of rows and ask
whether you would have picked them yourself. A feature set that retrieves
unconvincing neighbours will produce unconvincing predictions, and this is much
the cheaper place to discover it.

Queries never change the case base. The transforms applied to a query: the
standardisation, the categorical vocabularies: are the ones fitted on train and
are applied, never refitted. A holdout row is encoded in the training data's
terms, which is precisely what makes its neighbours meaningful.

See Also
--------
buildml.cbr.predict.predict_cbr : Retrieval plus reuse.
buildml.cbr.cases.pairwise_distances : The distance functions themselves.
"""

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
    """Return the nearest cases for every row in a partition, without predicting.

    For each query row, the ``k`` closest cases in memory with their distances
    and outcomes, wrapped in traces. Nothing is combined and nothing is stored.

    Parameters
    ----------
    dataset:
        The query data. Must supply the plan's feature columns.
    plan:
        The fitted reasoner whose memory is searched.
    split_plan:
        Partition membership. Required unless ``partition='all'``.
    partition:
        Which rows to use as queries.
    k:
        Override the plan's neighbour count for this call.
    backend:
        Override the retrieval backend, which is how you compare exact against
        approximate search on the same memory.

    Returns
    -------
    CbrRetrieveResult
        One trace per query, nearest first, with ``prediction`` unset.

    Raises
    ------
    ValidationError
        If ``k`` is below one, no split plan was supplied for a partition query,
        a feature column is missing, or a categorical column contains nulls.

    Notes
    -----
    **Neighbours are always returned, however far away they are.** There is no
    similarity threshold. A query unlike anything in memory still yields ``k``
    cases, and only the distances reveal that.

    **Distances are comparable across queries within one call**, since the
    metric and scaling are fixed. Rows whose nearest neighbour is unusually far
    are the ones the reasoner has least basis for.

    **Retrieving the train partition returns each row itself first**, at
    distance zero. Useful for confirming the encoding round-trips; useless as a
    measure of anything.

    Examples
    --------
    Inspect what the reasoner thinks is similar::

        result = retrieve_cases(dataset, plan, split_plan, partition="test", k=5)
        for trace in result.traces[:3]:
            print(trace.query_index, trace.distances, trace.neighbor_solutions)

    See Also
    --------
    buildml.cbr.results.CbrRetrieveResult : What comes back.
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
    """Encode query rows into the same space the case base lives in.

    Applies the train-fitted standardisation and categorical vocabularies to a
    query frame. Applies, never refits: that distinction is the whole point.
    A query standardised by its own mean and variance would sit in a different
    coordinate system from memory, and the resulting distances would be
    arithmetic on incomparable numbers.

    Parameters
    ----------
    frame:
        The query rows. Must contain the plan's feature columns.
    plan:
        The fitted reasoner supplying the column contract and transforms.

    Returns
    -------
    tuple
        ``(numeric, categorical)`` arrays aligned with the case base's column
        order, ready for :func:`~buildml.cbr.cases.pairwise_distances`.

    Raises
    ------
    ValidationError
        If a categorical column is missing or contains nulls.

    Notes
    -----
    **Standardisation is skipped for the mixed metric**, which normalises by
    per-feature range instead. Applying both would scale twice.

    **Unseen categories encode to ``-1``**, which the mixed metric treats as
    different from every known value. A query full of unseen categories is
    maximally distant on those columns, which is honest and worth noticing.

    **Nulls in categorical columns are refused rather than coded.** There is no
    defensible distance between a missing value and a present one; impute
    before querying.
    """
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
    """Resolve neighbour indices into the cases themselves.

    A small adapter between the retrieval backends, which speak in indices, and
    trace construction, which needs the case objects. Kept as a named function
    so prediction and retrieval build their traces the same way.

    Parameters
    ----------
    plan:
        The fitted reasoner holding the case base.
    order:
        Neighbour indices, nearest first.
    dvals:
        Their distances, aligned with ``order``.

    Returns
    -------
    tuple
        ``(cases, distances, order)``: the resolved cases plus the inputs
        passed through, so callers can keep the indices for feature lookups.
    """
    neighbors = [plan.case_base.cases[j] for j in order]
    return neighbors, dvals, order


def _plan_with_backend(plan: CbrPlan, backend: str | None) -> CbrPlan:
    """Return a plan copy that searches with a different backend.

    Lets one call override the retrieval backend without refitting, which is how
    exact and approximate search get compared on identical memory. The original
    plan is returned unchanged when no override is asked for.

    Parameters
    ----------
    plan:
        The fitted reasoner.
    backend:
        The backend to use, or ``None`` to keep the plan's own.

    Returns
    -------
    CbrPlan
        The original plan, or a copy with the backend and metric re-resolved.

    Notes
    -----
    **The case base is shared, not copied.** Only the backend and metric fields
    differ, so this is cheap regardless of memory size.

    **The requested backend may still be substituted** if its dependency is
    absent; the resolver falls back rather than raising.
    """
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
    """Select the rows to query and remember their original indices.

    The indices matter: traces are keyed by them, so a prediction can be joined
    back to the source row it belongs to even after partitioning.

    Parameters
    ----------
    dataset:
        The source data.
    split_plan:
        Partition membership, or ``None`` when querying everything.
    partition:
        ``'train'``, ``'validation'``, ``'test'``, or ``'all'``.

    Returns
    -------
    tuple
        ``(frame, indices)``: the selected rows and their original index
        labels.

    Raises
    ------
    ValidationError
        If a named partition was requested without a split plan.

    Notes
    -----
    **``partition='all'`` materialises the whole dataset**, including rows that
    are already in the case base. That is legitimate for exploration and is not
    an evaluation.
    """
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
