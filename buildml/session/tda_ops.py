"""Thin Session facades over buildml.tda."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence, cast

if TYPE_CHECKING:
    from buildml.session.session import Session

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
from buildml.tda.types import (
    DiagramDistanceMetric,
    SubsampleStrategy,
    TdaBackend,
    TdaHead,
    TdaTask,
    Vectorization,
)

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
    """Fit TDA features and optional supervised head on Session train only.

    Delegates to :func:`buildml.tda.fit.fit_tda`, stores the
    :class:`~buildml.tda.results.TdaPlan` on Session, and records the fit.
    Follow with :func:`transform_tda_op`, :func:`predict_tda_op`, or
    :func:`evaluate_tda_op`.

    Parameters
    ----------
    session:
        Active Session with numeric features and a split plan.
    backend:
        Optional backend override (``native`` or ``giotto``).
    vectorization:
        Persistence diagram vectorization method.
    homology_dims:
        Homology dimensions to compute (e.g. H0, H1).
    knn:
        Neighborhood size for Vietoris-Rips / kNN graph construction.
    maxdim:
        Maximum homology dimension for persistent homology.
    thresh:
        Optional distance threshold for filtration truncation.
    n_bins:
        Bin count for persistence images and landscapes.
    n_layers:
        Layer count for multi-scale vectorizations.
    pixel_size:
        Pixel size for persistence images.
    standardize:
        Standardize vectorized features before the optional head.
    head:
        Optional supervised classifier/regressor head on TDA features.
    task:
        Task override when head is supervised (classification/regression).
    columns:
        Explicit feature columns; ``None`` auto-selects numerics.
    random_state:
        Seed for subsampling and stochastic steps.
    prefer_reduce_components:
        Prefer reduced component columns when a reduce plan exists on Session.
    max_points_guard:
        Maximum point count before subsample/error guard triggers.
    subsample_strategy:
        Behavior when point count exceeds ``max_points_guard``.
    mapper:
        When True, also compute a Mapper graph summary.

    Returns
    -------
    TdaFitResult
        Serializable fit summary including homology and vectorizer state.

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
    """Transform a partition with the frozen train-fitted TDA pipeline.

    Delegates to :func:`buildml.tda.transform.transform_tda` using the plan
    from :func:`fit_tda_op`. No refit occurs on holdout partitions.

    Parameters
    ----------
    session:
        Active Session with a TdaPlan from :func:`fit_tda_op`.
    partition:
        Split partition to transform (default ``test``).
    backend:
        Optional backend override for transform step.

    Returns
    -------
    TdaTransformResult
        Vectorized persistence features for the requested partition.

    Raises
    ------
    ValidationError
        When no TdaPlan exists on the Session.
    """
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
    """Predict with the optional TDA supervised head on a partition.

    Delegates to :func:`buildml.tda.predict.predict_tda`. Requires a
    supervised head fitted during :func:`fit_tda_op`.

    Parameters
    ----------
    session:
        Active Session with a TdaPlan from :func:`fit_tda_op`.
    partition:
        Split partition to predict on (default ``test``).

    Returns
    -------
    TdaPredictResult
        Predictions and optional probabilities from the TDA head.

    Raises
    ------
    ValidationError
        When no TdaPlan exists on the Session.
    """
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
    """Evaluate the TDA head on a holdout partition.

    Delegates to :func:`buildml.tda.evaluate.evaluate_tda` and optionally
    compares persistence diagram distances between partitions.

    Parameters
    ----------
    session:
        Active Session with a TdaPlan from :func:`fit_tda_op`.
    partition:
        Holdout partition for evaluation (default ``validation``).
    backend:
        Optional backend override for evaluation.
    compare_diagram_distances:
        When True, compute diagram distance metrics between partitions.
    diagram_distance_metric:
        Distance metric for persistence diagrams (e.g. Wasserstein).
    diagram_distance_dim:
        Homology dimension for diagram distance comparison.

    Returns
    -------
    TdaEvalResult
        Holdout metrics for the supervised TDA head and optional distances.

    Raises
    ------
    ValidationError
        When no TdaPlan exists on the Session.
    """
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
    """Return the TDA backend and vectorization capability matrix.

    Delegates to :func:`buildml.tda.catalog.tda_capability_matrix`.
    Use before :func:`fit_tda_op` to confirm backend and method availability.

    Returns
    -------
    dict
        Nested map of backend identifiers to supported vectorizations.
    """
    return tda_capability_matrix()


def save_tda_bundle_op(session, path: str | Path) -> Path:
    """Persist the active TdaPlan as ``buildml.tda_bundle.v2``.

    Delegates to :func:`buildml.tda.checkpoint.save_tda_bundle`.
    Reload with :func:`load_tda_bundle_op`.

    Parameters
    ----------
    session:
        Active Session with a TdaPlan from :func:`fit_tda_op`.
    path:
        Destination directory for the bundle (created if missing).

    Returns
    -------
    pathlib.Path
        Resolved bundle directory path.

    Raises
    ------
    ValidationError
        When no TdaPlan exists on the Session.
    """
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


def load_tda_bundle_op(session, path: str | Path, *, trusted: bool = False):
    """Load a TDA bundle into this Session.

    Delegates to :func:`buildml.tda.checkpoint.load_tda_bundle` and clears
    prior transform/predict/eval results.

    Parameters
    ----------
    session:
        Session instance to populate with the loaded TdaPlan.
    path:
        Path to a ``buildml.tda_bundle.v2`` directory.
    trusted:
        Must be ``True`` to deserialize pickle/joblib/torch payloads. Pass
        only for artifacts you created or fully trust. Defaults to ``False``.

    Returns
    -------
    Session
        ``session`` with TdaPlan attached for chaining.
    """
    plan = load_tda_bundle(path, trusted=trusted)
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
    return cast("Session", session)