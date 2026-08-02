"""Fast adapt a meta-learner to one task's labeled support set."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import PartitionName, SplitPlan, frame_for_partition
from buildml.metalearning.features import (
    compute_prototypes,
    encode_labels,
    frame_for_task,
    matrix_from_frame,
)
from buildml.metalearning.results import MetaAdaptResult, MetaLearningPlan


def adapt_to_task(
    dataset: Dataset,
    plan: MetaLearningPlan,
    split_plan: SplitPlan | None,
    *,
    task_id: Any | None = None,
    partition: PartitionName = "train",
    support_frame: pd.DataFrame | None = None,
    max_support_per_class: int | None = None,
    random_state: int | None = 0,
) -> MetaAdaptResult:
    """Adapt the meta-learner to one task using a labeled support set.

    Provide either ``support_frame`` (explicit rows) or ``task_id`` to pull
    rows for that task from ``partition``. Holdout partitions may supply
    support for novel-task adaptation; the meta-train plan itself is never
    refit here.
    """
    if plan is None:
        raise ValidationError("No MetaLearningPlan. Call fit_metalearning first.")

    disclosures: list[str] = [
        "adapt_to_task freezes the meta-train plan and fits only on the "
        "provided support set (fast adapt).",
        f"method={plan.method}.",
    ]
    warnings: list[str] = []

    if support_frame is not None:
        support = support_frame.copy()
        resolved_task = task_id
        if resolved_task is None and plan.task_column in support.columns:
            ids = support[plan.task_column].dropna().unique().tolist()
            if len(ids) == 1:
                resolved_task = ids[0]
        disclosures.append(
            f"Support taken from explicit support_frame (n={len(support)})."
        )
    else:
        if task_id is None:
            raise ValidationError(
                "adapt_to_task requires task_id= when support_frame is omitted."
            )
        if split_plan is None and partition != "train":
            # frame_for_partition needs a split for named partitions
            raise ValidationError(
                f"partition={partition!r} requires a SplitPlan."
            )
        if split_plan is None:
            frame = dataset._ensure_pandas()
        else:
            frame = frame_for_partition(dataset, split_plan, partition)
        support = frame_for_task(frame, plan.task_column, task_id)
        resolved_task = task_id
        disclosures.append(
            f"Support taken from partition={partition!r} for "
            f"task_id={task_id!r} (n={len(support)})."
        )

    if len(support) < 1:
        raise ValidationError(
            f"Empty support set for task_id={resolved_task!r}."
        )
    missing = [c for c in plan.columns if c not in support.columns]
    if missing:
        raise ValidationError(f"Support frame missing feature columns: {missing}")
    if plan.target_column not in support.columns:
        raise ValidationError(
            f"Support frame missing target column {plan.target_column!r}."
        )

    if max_support_per_class is not None:
        if int(max_support_per_class) < 1:
            raise ValidationError("max_support_per_class must be >= 1.")
        import numpy as np

        rng = np.random.default_rng(random_state)
        parts: list[pd.DataFrame] = []
        for label, group in support.groupby(plan.target_column, sort=False):
            if len(group) <= int(max_support_per_class):
                parts.append(group)
            else:
                idx = rng.choice(
                    len(group), size=int(max_support_per_class), replace=False
                )
                parts.append(group.iloc[idx])
            _ = label
        support = pd.concat(parts, axis=0)
        disclosures.append(
            f"Capped support at max_support_per_class={max_support_per_class}."
        )

    x = matrix_from_frame(support, list(plan.columns))
    y_codes, _, classes = encode_labels(
        support[plan.target_column], label_encoder=plan.label_encoder_
    )

    prototypes: dict[Any, tuple[float, ...]] | None = None
    adapted_estimator: Any = None

    if plan.method == "prototypical":
        proto_map = compute_prototypes(x, y_codes)
        # Map codes → original labels for the result payload
        label_by_code = {
            i: _coerce(plan.label_encoder_.classes_[i])
            for i in range(len(plan.label_encoder_.classes_))
        }
        prototypes = {
            label_by_code[code]: tuple(float(v) for v in vec)
            for code, vec in proto_map.items()
            if code in label_by_code
        }
        # Keep code-keyed prototypes on a private attr for evaluate/predict
        adapted_estimator = {"_prototypes_by_code": proto_map}
        disclosures.append(
            f"Built {len(proto_map)} class prototype(s) from support "
            "(tabular nearest-centroid)."
        )
    elif plan.method == "prototypical_torch":
        if plan.meta_learner_ is None:
            raise ValidationError(
                "Torch prototypical plan has no meta_learner_. Refit with "
                "method='prototypical_torch'."
            )
        emb = plan.meta_learner_.embed(x)
        proto_map = compute_prototypes(emb, y_codes)
        label_by_code = {
            i: _coerce(plan.label_encoder_.classes_[i])
            for i in range(len(plan.label_encoder_.classes_))
        }
        prototypes = {
            label_by_code[code]: tuple(float(v) for v in vec)
            for code, vec in proto_map.items()
            if code in label_by_code
        }
        adapted_estimator = plan.meta_learner_
        disclosures.append(
            f"Torch prototypical adapt: {len(proto_map)} embedding-space "
            "prototype(s) from support."
        )
    elif plan.method in {"maml", "reptile"}:
        if plan.meta_learner_ is None:
            raise ValidationError(
                f"{plan.method} plan has no meta_learner_. Refit with "
                f"method={plan.method!r}."
            )
        adapted_estimator = plan.meta_learner_
        disclosures.append(
            f"Industry {plan.method} adapt: inner-loop refit on support "
            f"(inner_steps={getattr(plan.meta_learner_, 'inner_steps', None)})."
        )
    elif plan.method == "warm_start":
        if plan.init_estimator_ is None:
            raise ValidationError(
                "Warm-start plan has no init_estimator_. Refit with "
                "method='warm_start'."
            )
        adapted_estimator = deepcopy(plan.init_estimator_)
        try:
            adapted_estimator.fit(x, y_codes)
        except Exception as exc:  # noqa: BLE001
            raise ValidationError(
                f"adapt_to_task warm-start refit failed: {exc}"
            ) from exc
        disclosures.append(
            "Cloned meta-init estimator and refit on the support set."
        )
    else:
        raise ValidationError(f"Unknown plan.method={plan.method!r}.")

    if resolved_task in plan.train_task_ids:
        warnings.append(
            f"task_id={resolved_task!r} appeared in meta-train task ids; "
            "adaptation is still valid but is not a novel-task few-shot test."
        )

    return MetaAdaptResult(
        method=plan.method,
        task_id=resolved_task,
        n_support=int(len(support)),
        n_classes_adapted=len(classes),
        classes_=classes,
        prototypes_=prototypes,
        adapted_estimator_=adapted_estimator,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )


def _coerce(value: Any) -> Any:
    text = str(value)
    if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
        return int(text)
    return text
