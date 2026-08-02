"""Transform partitions with a frozen train-fitted TDA pipeline."""

from __future__ import annotations

import numpy as np

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.tda.extras import require_tda_stack
from buildml.tda.features import matrix_from_frame, partition_frame, standardize_apply
from buildml.tda.homology import compute_rips_diagrams, local_point_cloud
from buildml.tda.results import TdaPlan, TdaTransformResult
from buildml.tda.vectorize import vectorize_diagrams


def transform_tda(
    dataset: Dataset,
    plan: TdaPlan,
    split_plan: SplitPlan | None,
    *,
    partition: str = "test",
) -> TdaTransformResult:
    """Apply the train-fitted TDA transformer to a partition (no refit).

    Leakage discipline: NearestNeighbors index, scaler, and vectorizer ranges
    stay frozen from train. Holdout rows never update the PH pipeline.
    """
    require_tda_stack(feature="transform_tda")
    if plan is None:
        raise ValidationError("No TdaPlan. Call fit_tda(...) first.")
    if plan.nn_ is None or plan.train_x_ is None or not plan.vectorizer_state_:
        raise ValidationError("TdaPlan is incomplete (missing NN / train_x / vectorizer).")

    frame = partition_frame(dataset, split_plan, partition)
    cols = list(plan.columns)
    for col in cols:
        if col not in frame.columns:
            raise ValidationError(f"TDA feature column {col!r} missing from partition.")
    x_raw = matrix_from_frame(frame, cols)
    if plan.standardize:
        if plan.mean_ is None or plan.scale_ is None:
            raise ValidationError("TdaPlan.standardize=True but mean_/scale_ missing.")
        x = standardize_apply(x_raw, plan.mean_, plan.scale_)
    else:
        x = x_raw

    rows: list[np.ndarray] = []
    for i in range(len(x)):
        cloud = local_point_cloud(x[i], plan.nn_, plan.train_x_, knn=plan.knn)
        dgms = compute_rips_diagrams(cloud, maxdim=plan.maxdim, thresh=plan.thresh)
        rows.append(vectorize_diagrams(dgms, plan.vectorizer_state_))
    features = np.vstack(rows) if rows else np.zeros((0, plan.feature_dim), dtype=float)

    return TdaTransformResult(
        partition=str(partition),
        n_rows=int(features.shape[0]),
        feature_dim=int(features.shape[1]),
        feature_names=tuple(plan.feature_names),
        features=features,
        vectorization=plan.vectorization,
        disclosures=(
            "Transform used frozen train NN index + vectorizer ranges (no refit).",
        ),
    )
