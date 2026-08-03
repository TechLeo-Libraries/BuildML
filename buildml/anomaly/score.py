"""Score / flag partitions with a train-fitted AnomalyPlan (no refit)."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from buildml.anomaly.features import matrix_from_frame
from buildml.anomaly.fit import anomaly_scores
from buildml.anomaly.results import AnomalyPlan, AnomalyScoreResult
from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset
from buildml.data.splits import PartitionName, SplitPlan, frame_for_partition
from buildml.ingest.detect import schema_from_dataframe

PartitionOrAll = PartitionName | Literal["all"]


def score_anomalies(
    dataset: Dataset,
    plan: AnomalyPlan,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll = "test",
    attach: bool = False,
    override_threshold: float | None = None,
) -> tuple[Dataset | None, AnomalyScoreResult]:
    """Score and flag rows with a frozen anomaly plan (no refit).

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
partition:
    ``train``, ``validation``, ``test``, or ``all``.
attach:
    When True, requires ``partition='all'`` and writes ``score_column`` /
    ``flag_column`` onto a copy of the dataset.
override_threshold:
    Optional absolute threshold on higher-is-more-anomalous scores. When set,
    it is disclosed and does not mutate the stored plan threshold.
dataset:
    BuildML dataset with features, target, and role metadata.
plan:
    Fitted plan object carrying model state and feature contract.
split_plan:
    Train/validation/test split; fit uses train partition only.

Returns
-------
tuple[Dataset | None, AnomalyScoreResult]
    Tuple of results (tuple[Dataset | None, AnomalyScoreResult]) for downstream Session steps.

Raises
------
ValidationError
    When preconditions for this operation are not met.
    """
    frame, part_name = _frame_for_score(dataset, split_plan, partition)
    missing = [c for c in plan.columns if c not in frame.columns]
    if missing:
        raise ValidationError(f"Anomaly plan columns missing from dataset: {missing}")
    x = matrix_from_frame(frame, list(plan.columns))
    scores = anomaly_scores(plan, x)
    threshold = float(plan.threshold_ if override_threshold is None else override_threshold)
    flags = (scores >= threshold).astype(int)
    n_flagged = int(flags.sum())
    n_rows = int(len(flags))
    alert_rate = float(n_flagged) / float(max(n_rows, 1))
    disclosures = list(plan.disclosures)
    disclosures.append(
        f"Scored partition='{part_name}' with frozen plan "
        f"(method={plan.method}, mode={plan.mode}); threshold={threshold:.6g} "
        f"({plan.threshold_policy if override_threshold is None else 'override'}); "
        f"alert_rate={alert_rate:.4f} ({n_flagged}/{n_rows})."
    )
    if override_threshold is not None:
        disclosures.append(
            f"override_threshold={float(override_threshold)} used for this score call; "
            "AnomalyPlan.threshold_ was not mutated."
        )
    if part_name != "train":
        disclosures.append(
            "Holdout alert rates are not guaranteed to match train contamination; "
            "report alert_rate beside every operational claim."
        )

    attached = False
    new_dataset: Dataset | None = None
    if attach:
        if partition != "all":
            raise ValidationError(
                "attach=True requires partition='all' so score/flag columns stay "
                "aligned with the Session frame."
            )
        out = dataset._ensure_pandas().copy()
        for col in (plan.score_column, plan.flag_column):
            if col in out.columns:
                raise ValidationError(
                    f"Column '{col}' already exists on the dataset; choose different "
                    "score_column/flag_column or drop the existing column."
                )
        out[plan.score_column] = np.asarray(scores, dtype=float)
        out[plan.flag_column] = np.asarray(flags, dtype=int)
        roles = dict(dataset.roles)
        roles[plan.score_column] = ColumnRole.FEATURE
        roles[plan.flag_column] = ColumnRole.FEATURE
        new_dataset = Dataset.from_transformed(
            dataset,
            out,
            schema=schema_from_dataframe(out),
            roles=roles,
        )
        attached = True

    result = AnomalyScoreResult(
        partition=part_name,
        method=plan.method,
        mode=plan.mode,
        n_rows=n_rows,
        n_flagged=n_flagged,
        alert_rate=alert_rate,
        threshold=threshold,
        threshold_policy=(
            plan.threshold_policy if override_threshold is None else "override"
        ),
        scores=tuple(float(v) for v in scores.tolist()),
        flags=tuple(int(v) for v in flags.tolist()),
        score_stats=_score_stats(scores),
        attached=attached,
        disclosures=tuple(dict.fromkeys(disclosures)),
    )
    return new_dataset, result


def _frame_for_score(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    partition: PartitionOrAll,
) -> tuple[pd.DataFrame, str]:
    if partition == "all":
        return dataset._ensure_pandas(), "all"
    if split_plan is None:
        raise ValidationError(
            f"partition='{partition}' requires a SplitPlan. "
            "Call session.split(...) first, or use partition='all'."
        )
    return frame_for_partition(dataset, split_plan, partition), str(partition)


def _score_stats(scores: np.ndarray) -> dict[str, float]:
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return {}
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
    }
