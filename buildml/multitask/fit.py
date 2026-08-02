"""Train-only multi-task / multi-output fit (sklearn MultiOutput / Chain)."""

from __future__ import annotations

from typing import Any, Sequence

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multioutput import (
    ClassifierChain,
    MultiOutputClassifier,
    MultiOutputRegressor,
    RegressorChain,
)

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition, frame_for_partition
from buildml.multitask.features import (
    encode_multitask_y,
    infer_task_type,
    matrix_from_frame,
    resolve_multitask_columns,
    resolve_target_columns,
)
from buildml.multitask.results import MultiTaskFitResult, MultiTaskPlan
from buildml.multitask.types import (
    MultiTaskBaseEstimator,
    MultiTaskConfig,
    MultiTaskMethod,
    MultiTaskTask,
)

_CLS_BASES = {
    "logistic_regression": lambda rs: LogisticRegression(
        max_iter=500, random_state=rs
    ),
    "hist_gradient_boosting": lambda rs: HistGradientBoostingClassifier(
        random_state=rs
    ),
}
_REG_BASES = {
    "ridge": lambda rs: Ridge(random_state=rs),
    "hist_gradient_boosting_regressor": lambda rs: HistGradientBoostingRegressor(
        random_state=rs
    ),
}


def fit_multitask(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    method: MultiTaskMethod = "multi_output",
    task: MultiTaskTask = "auto",
    targets: Sequence[str] | None = None,
    columns: list[str] | None = None,
    base_estimator: MultiTaskBaseEstimator | str = "logistic_regression",
    random_state: int | None = 0,
    order: Sequence[str] | None = None,
    prefer_reduce_components: bool = True,
    prediction_prefix: str = "multitask_pred",
    reduce_plan: Any | None = None,
) -> tuple[MultiTaskPlan, MultiTaskFitResult]:
    """Fit a multi-target estimator on the train partition only.

    Honesty
    -------
    Uses sklearn ``MultiOutput*`` / ``*Chain`` façades on shared features with
    multiple targets. Same-type tasks only (all classification or all
    regression). Mixed classification+regression is refused. Classical
    ``Session.fit`` remains single-target. This is not a deep multi-head MTL
    research platform.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None

    method_key = str(method).lower().replace("-", "_")
    if method_key not in {"multi_output", "classifier_chain", "regressor_chain"}:
        raise ValidationError(
            f"Unknown multi-task method={method!r}. Supported: "
            "'multi_output', 'classifier_chain', 'regressor_chain'."
        )

    target_cols, disclosures = resolve_target_columns(dataset, targets)
    train = frame_for_partition(dataset, split_plan, "train")
    resolved_task, task_notes = infer_task_type(train, target_cols, task)
    disclosures.extend(task_notes)

    if method_key == "classifier_chain" and resolved_task != "classification":
        raise ValidationError(
            "method='classifier_chain' requires classification targets "
            f"(resolved task={resolved_task!r})."
        )
    if method_key == "regressor_chain" and resolved_task != "regression":
        raise ValidationError(
            "method='regressor_chain' requires regression targets "
            f"(resolved task={resolved_task!r})."
        )

    cols, used_reduce, col_notes = resolve_multitask_columns(
        dataset,
        train,
        columns,
        reduce_plan=reduce_plan,
        prefer_reduce_components=prefer_reduce_components,
        target_columns=target_cols,
    )
    disclosures.extend(col_notes)

    x = matrix_from_frame(train, cols)
    y, encoders, classes = encode_multitask_y(
        train, target_cols, task=resolved_task
    )

    chain_order = _resolve_chain_order(target_cols, order) if order is not None else None
    if order is not None:
        disclosures.append(
            f"Chain order (target names): {[target_cols[i] for i in chain_order]}."
            if chain_order is not None
            else f"Chain order requested: {list(order)}."
        )

    estimator = _build_estimator(
        method_key,
        resolved_task,
        str(base_estimator),
        random_state,
        order_indices=chain_order,
    )
    try:
        estimator.fit(x, y)
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(
            f"Multi-task fit failed for method={method_key!r}, "
            f"base_estimator={base_estimator!r}: {exc}"
        ) from exc

    disclosures.extend(
        [
            "Multi-task learning fits sklearn MultiOutput / Chain estimators on "
            "shared features with multiple targets (train partition only).",
            "Validation/test partitions are never used for fitting.",
            "Same-type tasks only; mixed classification+regression is refused.",
            "Not a deep multi-head MTL research platform or multi-label binary "
            "relevance zoo.",
            f"n_tasks={len(target_cols)}, method={method_key}, "
            f"task={resolved_task}, n_train_rows={len(train)}.",
        ]
    )

    config = MultiTaskConfig(
        method=method_key,  # type: ignore[arg-type]
        task=resolved_task,  # type: ignore[arg-type]
        targets=tuple(target_cols),
        columns=tuple(cols),
        base_estimator=str(base_estimator),  # type: ignore[arg-type]
        random_state=random_state,
        order=None if order is None else tuple(order),
        prefer_reduce_components=prefer_reduce_components,
        prediction_prefix=prediction_prefix,
    )
    plan = MultiTaskPlan(
        method=method_key,
        task=resolved_task,
        columns=tuple(cols),
        target_columns=tuple(target_cols),
        n_train_rows=int(len(train)),
        classes_per_task_=classes,
        estimator_=estimator,
        label_encoders_=encoders,
        disclosures=tuple(disclosures),
        warnings=(),
        used_reduce_components=used_reduce,
        config=config.to_dict(),
    )
    result = MultiTaskFitResult(
        method=method_key,
        task=resolved_task,
        n_train_rows=int(len(train)),
        columns=tuple(cols),
        target_columns=tuple(target_cols),
        n_tasks=len(target_cols),
        used_reduce_components=used_reduce,
        disclosures=tuple(disclosures),
        warnings=(),
    )
    return plan, result


def _resolve_chain_order(
    target_columns: list[str], order: Sequence[str]
) -> list[int]:
    names = list(order)
    if sorted(names) != sorted(target_columns):
        raise ValidationError(
            f"order= must be a permutation of target columns {target_columns}; "
            f"got {names}."
        )
    index = {name: i for i, name in enumerate(target_columns)}
    return [index[name] for name in names]


def _build_estimator(
    method: str,
    task: str,
    base_estimator: str,
    random_state: int | None,
    *,
    order_indices: list[int] | None,
) -> Any:
    base_key = str(base_estimator).lower().replace("-", "_")
    if task == "classification":
        if base_key not in _CLS_BASES:
            # Allow regression bases to fail clearly
            if base_key in _REG_BASES:
                raise ValidationError(
                    f"base_estimator={base_estimator!r} is a regressor; "
                    "classification multi-task needs "
                    f"{sorted(_CLS_BASES)}."
                )
            raise ValidationError(
                f"Unknown classification base_estimator={base_estimator!r}. "
                f"Supported: {sorted(_CLS_BASES)}"
            )
        base = _CLS_BASES[base_key](random_state)
        if method == "multi_output":
            return MultiOutputClassifier(base)
        return ClassifierChain(base, order=order_indices, random_state=random_state)

    if base_key not in _REG_BASES:
        if base_key in _CLS_BASES:
            raise ValidationError(
                f"base_estimator={base_estimator!r} is a classifier; "
                f"regression multi-task needs {sorted(_REG_BASES)}."
            )
        raise ValidationError(
            f"Unknown regression base_estimator={base_estimator!r}. "
            f"Supported: {sorted(_REG_BASES)}"
        )
    base = _REG_BASES[base_key](random_state)
    if method == "multi_output":
        return MultiOutputRegressor(base)
    return RegressorChain(base, order=order_indices, random_state=random_state)
