"""Feature scaling with train-only fit semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition, frame_for_partition
from buildml.ingest.detect import schema_from_dataframe
from buildml.preprocess.columns import resolve_transform_columns

ScaleMethod = Literal["standard", "minmax"]


@dataclass(slots=True)
class ScalePlan:
    """Fitted scaling plan."""

    columns: tuple[str, ...]
    method: ScaleMethod
    scaler: Any

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "columns": list(self.columns),
            "method": self.method,
        }
        if hasattr(self.scaler, "mean_"):
            payload["mean_"] = [float(x) for x in np.asarray(self.scaler.mean_)]
        if hasattr(self.scaler, "scale_"):
            payload["scale_"] = [float(x) for x in np.asarray(self.scaler.scale_)]
        if hasattr(self.scaler, "data_min_"):
            payload["data_min_"] = [float(x) for x in np.asarray(self.scaler.data_min_)]
        if hasattr(self.scaler, "data_max_"):
            payload["data_max_"] = [float(x) for x in np.asarray(self.scaler.data_max_)]
        return payload


def fit_scaler(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    columns: list[str] | None = None,
    method: ScaleMethod = "standard",
) -> ScalePlan:
    """Fit a scaler on the train partition only.

    By default only numeric ``feature``-role columns are scaled (``ignore``,
    ``id``, ``target``, ``group``, ``time``, and ``weight`` are skipped).
    Pass ``columns=[...]`` explicitly to force-include any column.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None

    train = frame_for_partition(dataset, split_plan, "train")
    cols = resolve_transform_columns(
        dataset,
        train,
        columns,
        kind="numeric",
        empty_message=(
            "No numeric feature columns available for scaling. "
            "Pass columns=... explicitly to include ignore/id roles."
        ),
    )
    scaler: Any
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValidationError(f"Unsupported scale method '{method}'")

    scaler.fit(train[list(cols)])
    return ScalePlan(columns=tuple(cols), method=method, scaler=scaler)


def transform_scaler(
    dataset: Dataset,
    plan: ScalePlan,
    *,
    hard_limit_bytes: int | None = None,
) -> Dataset:
    """Apply a fitted scale plan to the full dataset.

    Soft/hard materialization gates run on the columns being scaled before the
    dense transform so large frames are not rewritten without disclosure.
    """
    from buildml.ingest.detect import MEMORY_HARD_LIMIT, check_materialization

    missing = [c for c in plan.columns if c not in dataset.columns]
    if missing:
        raise ValidationError(f"Scale plan columns missing from dataset: {missing}")

    base = dataset._ensure_pandas()
    scale_frame = base[list(plan.columns)]
    check_materialization(
        scale_frame,
        context="transform_scaler (design columns)",
        hard_limit_bytes=(
            MEMORY_HARD_LIMIT if hard_limit_bytes is None else hard_limit_bytes
        ),
    )
    frame = base.copy()
    scaled = plan.scaler.transform(frame[list(plan.columns)])
    scaled_df = pd.DataFrame(scaled, columns=list(plan.columns), index=frame.index)
    for column in plan.columns:
        frame[column] = scaled_df[column].astype(float)
    return Dataset.from_transformed(
        dataset,
        frame,
        schema=schema_from_dataframe(frame),
    )


