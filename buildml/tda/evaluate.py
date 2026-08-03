"""Evaluate a TDA supervised head on holdout topological features."""

from __future__ import annotations

import numpy as np

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.tda.features import (
    classification_metrics,
    partition_frame,
    regression_metrics,
)
from buildml.tda.homology import compute_rips_diagrams, local_point_cloud
from buildml.tda.predict import predict_tda
from buildml.tda.results import TdaEvalResult, TdaPlan
from buildml.tda.types import DiagramDistanceMetric


def evaluate_tda(
    dataset: Dataset,
    plan: TdaPlan,
    split_plan: SplitPlan | None,
    *,
    partition: str = "validation",
    backend: str | None = None,
    compare_diagram_distances: bool = False,
    diagram_distance_metric: DiagramDistanceMetric = "wasserstein",
    diagram_distance_dim: int = 1,
) -> TdaEvalResult:
    """Score the train-fitted TDA head on a holdout partition.

    Transform and head stay frozen from train; this only scores labeled rows.
    Optional diagram-distance diagnostics compare train versus holdout local
    clouds when persim is available.

    Parameters
    ----------
    dataset:
        Session dataset with target column present on the partition.
    plan:
        Train-fitted :class:`~buildml.tda.results.TdaPlan` with a supervised head.
    split_plan:
        Split plan defining the holdout partition.
    partition:
        ``validation``, ``test``, or ``all``.
    backend:
        Optional backend check: must match ``plan.backend`` when set.
    compare_diagram_distances:
        When True, report mean Wasserstein or bottleneck distance between sampled
        train and holdout local diagrams (diagnostic stability signal).
    diagram_distance_metric:
        ``wasserstein`` or ``bottleneck`` (requires persim).
    diagram_distance_dim:
        Homology dimension for diagram-distance comparison.

    Returns
    -------
    TdaEvalResult
        Holdout metrics plus optional diagram-distance summaries and disclosures.

    Raises
    ------
    ValidationError
        When no head is fitted, backend mismatches, or targets contain nulls.
    """
    if backend is not None and str(backend) != str(getattr(plan, "backend", "native")):
        raise ValidationError(
            f"backend={backend!r} does not match fitted plan backend="
            f"{getattr(plan, 'backend', 'native')!r}."
        )
    if plan.head_estimator_ is None or plan.head == "none":
        raise ValidationError(
            "evaluate_tda requires a supervised head. Refit with head!='none'."
        )
    if plan.task is None:
        raise ValidationError("TdaPlan.task is missing; cannot evaluate.")

    target = dataset.require_target()
    frame = partition_frame(dataset, split_plan, partition)
    if target not in frame.columns:
        raise ValidationError(f"Target column {target!r} missing from partition.")
    if frame[target].isna().any():
        raise ValidationError(
            f"Target column {target!r} has nulls on partition={partition!r}."
        )

    pred = predict_tda(dataset, plan, split_plan, partition=partition)
    y_true = frame[target]
    if plan.task == "classification":
        metrics = classification_metrics(list(y_true), list(pred.predictions))
    else:
        metrics = regression_metrics(
            y_true.to_numpy(dtype=float),
            np.asarray(pred.predictions, dtype=float),
        )

    diagram_distances: dict[str, float] = {}
    disclosures = [
        "Holdout scored with frozen train TDA transformer + head (no refit).",
        *plan.disclosures[:3],
    ]
    if compare_diagram_distances:
        dist = _mean_diagram_distance(
            dataset,
            plan,
            split_plan,
            partition=partition,
            metric=diagram_distance_metric,
            homology_dim=int(diagram_distance_dim),
        )
        if dist is not None:
            key = f"{diagram_distance_metric}_H{diagram_distance_dim}"
            diagram_distances[key] = float(dist)
            disclosures.append(
                f"Mean {diagram_distance_metric} distance (train vs {partition} "
                f"local H{diagram_distance_dim} diagrams): {dist:.6f}."
            )

    return TdaEvalResult(
        partition=str(partition),
        task=plan.task,
        n_rows=int(len(y_true)),
        metrics=metrics,
        diagram_distances=diagram_distances,
        vectorization=plan.vectorization,
        backend=str(getattr(plan, "backend", "native")),
        disclosures=tuple(disclosures),
        warnings=tuple(plan.warnings),
    )


def _mean_diagram_distance(
    dataset: Dataset,
    plan: TdaPlan,
    split_plan: SplitPlan | None,
    *,
    partition: str,
    metric: str,
    homology_dim: int,
    max_samples: int = 40,
) -> float | None:
    """Compare aggregate train vs holdout diagram distances (persim)."""
    try:
        from buildml.tda.extras import require_persim

        persim = require_persim(feature="diagram distance comparison")
    except Exception:
        return None

    if plan.nn_ is None or plan.train_x_ is None:
        return None

    from buildml.tda.adapters.giotto import transform_diagrams_giotto
    from buildml.tda.features import matrix_from_frame, partition_frame, standardize_apply

    plan_backend = str(getattr(plan, "backend", "native"))
    holdout = partition_frame(dataset, split_plan, partition)
    x_raw = matrix_from_frame(holdout, list(plan.columns))
    if plan.standardize and plan.mean_ is not None and plan.scale_ is not None:
        x_hold = standardize_apply(x_raw, plan.mean_, plan.scale_)
    else:
        x_hold = x_raw

    n_train = min(int(plan.n_train_rows), max_samples)
    n_hold = min(len(x_hold), max_samples)
    if n_train < 2 or n_hold < 2:
        return None

    train_x = np.asarray(plan.train_x_, dtype=float)
    rng = np.random.default_rng(0)
    train_pick = rng.choice(len(train_x), size=n_train, replace=False)
    hold_pick = rng.choice(len(x_hold), size=n_hold, replace=False)

    def _diagram_at(row: np.ndarray) -> np.ndarray:
        cloud = local_point_cloud(row, plan.nn_, train_x, knn=plan.knn)
        if plan_backend == "giotto":
            dgms = transform_diagrams_giotto(cloud, plan=plan)
        else:
            dgms = compute_rips_diagrams(cloud, maxdim=plan.maxdim, thresh=plan.thresh)
        d = int(homology_dim)
        if d < len(dgms):
            return np.asarray(dgms[d], dtype=float)
        return np.zeros((0, 2), dtype=float)

    train_dgms = [_diagram_at(train_x[i]) for i in train_pick]
    hold_dgms = [_diagram_at(x_hold[i]) for i in hold_pick]

    if metric == "bottleneck":
        dist_fn = persim.bottleneck
    else:
        dist_fn = persim.wasserstein

    distances: list[float] = []
    for td, hd in zip(train_dgms, hold_dgms, strict=False):
        if td.size == 0 and hd.size == 0:
            distances.append(0.0)
            continue
        if td.size == 0 or hd.size == 0:
            continue
        try:
            distances.append(float(dist_fn(td, hd)))
        except Exception:
            continue
    if not distances:
        return None
    return float(np.mean(distances))
