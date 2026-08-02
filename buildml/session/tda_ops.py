"""Thin Session facades over buildml.tda."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

from buildml.core.errors import ValidationError
from buildml.data.splits import PartitionName
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
from buildml.tda.types import TdaHead, TdaTask, Vectorization

PartitionOrAll = PartitionName | Literal["all"]


def fit_tda_op(
    session,
    *,
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
):
    """Fit TDA features (+ optional head) on Session train only.

    Notes
    -----
    **Leakage:** Requires a split. NN index, vectorizer ranges, and head use
    train only. Requires ``buildml[tda]`` (ripser + persim).
    """
    session.assert_can_fit("train")
    plan, result = fit_tda(
        session.dataset,
        session._split_plan,
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
    )
    session._tda_plan = plan
    session._tda_fit_result = result
    session._tda_eval_result = None
    session._tda_transform_result = None
    session._tda_predict_result = None
    session._record(
        "fit_tda",
        {
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
        },
        warnings=tuple(result.warnings),
        result_summary=fit_result_summary(result),
    )
    return result


def transform_tda_op(session, *, partition: PartitionOrAll = "test"):
    """Transform a partition with the frozen train-fitted TDA pipeline."""
    plan = getattr(session, "_tda_plan", None)
    if plan is None:
        raise ValidationError("No TdaPlan. Call fit_tda(...) first.")
    result = transform_tda(
        session.dataset, plan, session._split_plan, partition=partition
    )
    session._tda_transform_result = result
    session._record(
        "transform_tda",
        {"partition": partition},
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


def evaluate_tda_op(session, *, partition: PartitionOrAll = "validation"):
    """Evaluate the TDA head on a holdout partition."""
    plan = getattr(session, "_tda_plan", None)
    if plan is None:
        raise ValidationError("No TdaPlan. Call fit_tda(...) first.")
    result = evaluate_tda(
        session.dataset, plan, session._split_plan, partition=partition
    )
    session._tda_eval_result = result
    session._record(
        "evaluate_tda",
        {"partition": partition},
        result_summary=eval_result_summary(result),
    )
    return result


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
        result_summary={"path": str(out), "format": "buildml.tda_bundle.v1"},
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
        result_summary={"path": str(path), "format": "buildml.tda_bundle.v1"},
    )
    return session
