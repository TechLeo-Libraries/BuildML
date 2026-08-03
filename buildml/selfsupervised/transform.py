"""Export SSL representations from a frozen pretext plan (no refit)."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset
from buildml.data.splits import PartitionName, SplitPlan, frame_for_partition
from buildml.ingest.detect import schema_from_dataframe
from buildml.selfsupervised.features import matrix_from_frame
from buildml.selfsupervised.results import (
    SelfSupervisedPlan,
    SelfSupervisedTransformResult,
)

PartitionOrAll = PartitionName | Literal["all"]


def transform_ssl(
    dataset: Dataset,
    plan: SelfSupervisedPlan,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll = "train",
    attach: bool = False,
) -> tuple[Dataset | None, SelfSupervisedTransformResult, np.ndarray]:
    """Project features through the frozen SSL encoder without refit.

    Encodes the requested partition and optionally attaches representation
    columns to a new Session dataset when ``attach=True``.

    Parameters
    ----------
    dataset:
        Session dataset containing SSL input columns.
    plan:
        Train-fitted :class:`~buildml.selfsupervised.results.SelfSupervisedPlan`.
    split_plan:
        Split plan defining partitions; required unless ``partition='all'``.
    partition:
        Partition name or ``'all'`` for the full frame.
    attach:
        When True, return a new dataset with embedding columns added (requires
        ``partition='all'``).

    Returns
    -------
    tuple[Dataset | None, SelfSupervisedTransformResult, numpy.ndarray]
        Optional attached dataset, transform report, and embedding matrix.

    Raises
    ------
    ValidationError
        When the plan is missing, partition requirements fail, columns are
        absent, or attach would overwrite existing columns.
    """
    if plan is None:
        raise ValidationError("No SelfSupervisedPlan. Call fit_ssl_pretext first.")
    if partition == "all":
        frame = dataset._ensure_pandas()
        part_name = "all"
    else:
        if split_plan is None:
            raise ValidationError(
                f"partition='{partition}' requires a SplitPlan. Call session.split(...)."
            )
        frame = frame_for_partition(dataset, split_plan, partition)
        part_name = str(partition)

    emb = _encode_frame(frame, plan)
    if emb.shape[1] != len(plan.representation_columns):
        raise ValidationError(
            "SSL encoder latent width drifted from plan.representation_columns."
        )

    disclosures = [
        "SSL transform reuses the train-fitted SelfSupervisedPlan (no pretext refit).",
        "Exported columns are encoder representations (not reconstructions).",
    ]
    new_dataset: Dataset | None = None
    if attach:
        if partition != "all":
            raise ValidationError(
                "attach=True requires partition='all' so representation columns "
                "align with the full Session frame."
            )
        full = dataset._ensure_pandas().copy()
        roles = dict(dataset.roles)
        for i, col in enumerate(plan.representation_columns):
            if col in full.columns:
                raise ValidationError(
                    f"Representation column '{col}' already exists; choose a different "
                    "representation_prefix or drop the existing column."
                )
            full[col] = emb[:, i]
            roles[col] = ColumnRole.FEATURE
        new_dataset = Dataset.from_transformed(
            dataset,
            full,
            schema=schema_from_dataframe(full),
            roles=roles,
        )
        disclosures.append(
            f"Attached {len(plan.representation_columns)} SSL embedding columns "
            f"with prefix {plan.representation_prefix!r}."
        )

    result = SelfSupervisedTransformResult(
        partition=part_name,
        n_rows=int(emb.shape[0]),
        method=plan.method,
        representation_columns=plan.representation_columns,
        attached=attach,
        disclosures=tuple(disclosures),
    )
    return new_dataset, result, emb


def _encode_frame(frame: pd.DataFrame, plan: SelfSupervisedPlan) -> np.ndarray:
    modality = getattr(plan, "modality", "tabular")
    if modality == "text":
        col = plan.columns[0]
        if col not in frame.columns:
            raise ValidationError(f"Missing text column {col!r} for SSL transform.")
        texts = frame[col].astype(str).tolist()
        return np.asarray(plan.encoder_.transform(texts), dtype=float)
    if modality == "vision":
        col = plan.columns[0]
        if col not in frame.columns:
            raise ValidationError(f"Missing image column {col!r} for SSL transform.")
        images = frame[col].tolist()
        return np.asarray(plan.encoder_.transform(images), dtype=float)
    missing = [c for c in plan.columns if c not in frame.columns]
    if missing:
        raise ValidationError(f"Missing feature columns for SSL transform: {missing}")
    x = matrix_from_frame(frame, list(plan.columns))
    return np.asarray(plan.encoder_.transform(x), dtype=float)
