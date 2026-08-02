"""Thin Session facades over buildml.tda."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

from buildml.core.errors import ValidationError
from buildml.data.splits import PartitionName
from buildml.tda.catalog import tda_capability_matrix
from buildml.tda.checkpoint import load_tda_bundle, save_tda_bundle
from buildml.tda.evaluate import evaluate_tda
from buildml.tda.explain_hooks import (
    eval_result_summary,
    fit_result_summary,
    predict_result_summary,
    transform_result_summary,
)
from buildml.tda.fit import fit_tda
from buildml.tda.predict import predict_tda
from buildml.tda.transform import transform_tda
from buildml.tda.types import DiagramDistanceMetric, SubsampleStrategy, TdaBackend, TdaHead, TdaTask, Vectorization

PartitionOrAll = PartitionName | Literal["all"]


def fit_tda_op(
    session,
    *,
    backend: TdaBackend | None = None,
    vectorization: Vectorization = "persistence_image",
    homology_dims: Sequence[int] = (0, 1),
    knn: int = 16,
    maxdim: int = 1,
    thresh: float | None = None,
    n_bins: int = 20,
    n_layers: int = 3,
    pixel_size: float | None = None,
    standardize: bool = True,
    head: TdaHead = "logistic_regression",
    task: TdaTask | None = None,
    columns: list[str] | None = None,
    random_state: int | None = 0,
    prefer_reduce_components: bool = True,
    max_points_guard: int = 4000,
    subsample_strategy: SubsampleStrategy = "error",
    mapper: bool = False,
):
    """Fit TDA features (+ optional head) on Session train only.

    Notes
    -----
    **Leakage:** Requires a split. NN index, vectorizer ranges, and head use
    train only. Requires ``buildml[tda]`` (native) or ``buildml[tda-industry]``
    (giotto backend).
    """
    session.assert_can_fit("train")
    plan, result = fit_tda(
        session.dataset,
        session._split_plan,
        backend=backend,
        vectorization=vectorization,
        homology_dims=homology_dims,
        knn=knn,
        maxdim=maxdim,
        thresh=thresh,
        n_bins=n_bins,
        n_layers=n_layers,
        pixel_size=pixel_size,
        standardize=standardize,
        head=head,
        task=task,
        columns=columns,
        random_state=random_state,
        prefer_reduce_components=prefer_reduce_components,
        reduce_plan=getattr(session, "_reduce_plan", None),
        max_points_guard=max_points_guard,
        subsample_strategy=subsample_strategy,
        mapper=mapper,
    )
    session._tda_plan = plan
    session._tda_fit_result = result
    session._tda_eval_result = None
    session._tda_transform_result = None
    session._tda_predict_result = None
    session._record(
        "fit_tda",
        {
            "backend": backend,
            "vectorization": vectorization,
            "homology_dims": list(homology_dims),
            "knn": knn,
            "maxdim": maxdim,
            "thresh": thresh,
            "n_bins": n_bins,
            "n_layers": n_layers,
            "pixel_size": pixel_size,
            "standardize": standardize,
            "head": head,
            "task": task,
            "columns": columns,
            "random_state": random_state,
            "prefer_reduce_components": prefer_reduce_components,
            "max_points_guard": max_points_guard,
            "subsample_strategy": subsample_strategy,
            "mapper": mapper,
        },
        warnings=tuple(result.warnings),
        result_summary=fit_result_summary(result),
    )
    return result


def transform_tda_op(
    session,
    *,
    partition: PartitionOrAll = "test",
    backend: TdaBackend | None = None,
):
    """Transform a partition with the frozen train-fitted TDA pipeline."""
    plan = getattr(session, "_tda_plan", None)
    if plan is None:
        raise ValidationError("No TdaPlan. Call fit_tda(...) first.")
    result = transform_tda(
        session.dataset,
        plan,
        session._split_plan,
        partition=partition,
        backend=backend,
    )
    session._tda_transform_result = result
    session._record(
        "transform_tda",
        {"partition": partition, "backend": backend},
        result_summary=transform_result_summary(result),
    )
    return result


def predict_tda_op(session, *, partition: PartitionOrAll = "test"):
    """Predict with the optional TDA supervised head."""
    plan = getattr(session, "_tda_plan", None)
    if plan is None:
        raise ValidationError("No TdaPlan. Call fit_tda(...) first.")
    result = predict_tda(
        session.dataset, plan, session._split_plan, partition=partition
    )
    session._tda_predict_result = result
    session._record(
        "predict_tda",
        {"partition": partition},
        result_summary=predict_result_summary(result),
    )
    return result


def evaluate_tda_op(
    session,
    *,
    partition: PartitionOrAll = "validation",
    backend: TdaBackend | None = None,
    compare_diagram_distances: bool = False,
    diagram_distance_metric: DiagramDistanceMetric = "wasserstein",
    diagram_distance_dim: int = 1,
):
    """Evaluate the TDA head on a holdout partition."""
    plan = getattr(session, "_tda_plan", None)
    if plan is None:
        raise ValidationError("No TdaPlan. Call fit_tda(...) first.")
    result = evaluate_tda(
        session.dataset,
        plan,
        session._split_plan,
        partition=partition,
        backend=backend,
        compare_diagram_distances=compare_diagram_distances,
        diagram_distance_metric=diagram_distance_metric,
        diagram_distance_dim=diagram_distance_dim,
    )
    session._tda_eval_result = result
    session._record(
        "evaluate_tda",
        {
            "partition": partition,
            "backend": backend,
            "compare_diagram_distances": compare_diagram_distances,
            "diagram_distance_metric": diagram_distance_metric,
        },
        result_summary=eval_result_summary(result),
    )
    return result


def tda_capability_matrix_op() -> dict:
    return tda_capability_matrix()


def save_tda_bundle_op(session, path: str | Path) -> Path:
    plan = getattr(session, "_tda_plan", None)
    if plan is None:
        raise ValidationError("No TdaPlan. Call fit_tda(...) first.")
    out = save_tda_bundle(
        path,
        plan,
        fit_result=getattr(session, "_tda_fit_result", None),
        eval_result=getattr(session, "_tda_eval_result", None),
    )
    session._record(
        "save_tda_bundle",
        {"path": str(path)},
        result_summary={"path": str(out), "format": "buildml.tda_bundle.v2"},
    )
    return out


def load_tda_bundle_op(session, path: str | Path):
    plan = load_tda_bundle(path)
    session._tda_plan = plan
    session._tda_fit_result = None
    session._tda_eval_result = None
    session._tda_transform_result = None
    session._tda_predict_result = None
    session._record(
        "load_tda_bundle",
        {"path": str(path)},
        result_summary={"path": str(path), "format": "buildml.tda_bundle.v2"},
    )
    return session
