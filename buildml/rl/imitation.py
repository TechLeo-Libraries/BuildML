"""Behavioral cloning (imitation learning) from demonstration tables."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import (
    PartitionName,
    SplitPlan,
    assert_fit_partition,
    frame_for_partition,
)
from buildml.rl.features import (
    classification_metrics,
    continuous_actions,
    decode_discrete_actions,
    encode_discrete_actions,
    infer_imitation_task,
    matrix_from_frame,
    regression_metrics,
    resolve_rl_columns,
)
from buildml.rl.results import (
    ImitationEvalResult,
    ImitationFitResult,
    ImitationPlan,
    ImitationPredictResult,
)
from buildml.rl.types import ImitationConfig, ImitationEstimator, ImitationTask

PartitionOrAll = PartitionName | Literal["all"]

_CLF = {
    "logistic_regression": lambda rs: LogisticRegression(
        max_iter=500, random_state=rs
    ),
    "hist_gradient_boosting": lambda rs: HistGradientBoostingClassifier(
        random_state=rs
    ),
}
_REG = {
    "ridge": lambda rs: Ridge(alpha=1.0),
    "hist_gradient_boosting_regressor": lambda rs: HistGradientBoostingRegressor(
        random_state=rs
    ),
}


def fit_imitation(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    task: ImitationTask | None = None,
    estimator: ImitationEstimator | None = None,
    columns: list[str] | None = None,
    action_column: str | None = None,
    random_state: int | None = 0,
    prefer_reduce_components: bool = True,
    reduce_plan: Any | None = None,
) -> tuple[ImitationPlan, ImitationFitResult]:
    """Fit a behavioral cloning policy on Session train demonstration rows.

    Demonstrations are ``(state features → action)``. When ``action_column`` is
    omitted, the Dataset target is treated as the demonstrated action.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None

    target = dataset.require_target()
    action_col = action_column or target
    if action_col not in dataset.columns:
        raise ValidationError(
            f"action_column={action_col!r} is not present on the dataset."
        )

    train = frame_for_partition(dataset, split_plan, "train")
    if action_col not in train.columns:
        raise ValidationError(
            f"action_column={action_col!r} missing from the train partition."
        )

    resolved_task = task or infer_imitation_task(train[action_col])  # type: ignore[arg-type]
    resolved_estimator = estimator or (
        "logistic_regression"
        if resolved_task == "classification"
        else "ridge"
    )
    _validate_estimator_for_task(resolved_task, resolved_estimator)

    cols, used_reduce, disclosures = resolve_rl_columns(
        dataset,
        train,
        columns,
        reduce_plan=reduce_plan,
        prefer_reduce_components=prefer_reduce_components,
        target_column=target,
        exclude_columns=(action_col,) if action_col != target else (),
    )
    x = matrix_from_frame(train, cols)
    n_train = int(x.shape[0])
    warnings: list[str] = []
    disclosures.extend(
        [
            "Behavioral cloning fits a supervised policy on train demonstrations only.",
            "Validation/test are never used to fit the cloning policy.",
            "Honesty: BC from tables — not inverse RL, not DAgger, not a robotics stack.",
            f"Action column={action_col!r}; task={resolved_task}; "
            f"estimator={resolved_estimator}.",
        ]
    )

    classes: tuple[Any, ...] | None = None
    label_encoder = None
    train_score: float | None = None
    if resolved_task == "classification":
        y_codes, label_encoder, classes = encode_discrete_actions(train[action_col])
        model = _build_estimator(resolved_estimator, random_state, task="classification")
        try:
            model.fit(x, y_codes)
        except Exception as exc:  # noqa: BLE001
            raise ValidationError(
                f"Imitation (BC) fit failed for estimator={resolved_estimator!r}: {exc}"
            ) from exc
        pred = model.predict(x)
        train_score = float(np.mean(pred == y_codes))
    else:
        y = continuous_actions(train[action_col])
        model = _build_estimator(resolved_estimator, random_state, task="regression")
        try:
            model.fit(x, y)
        except Exception as exc:  # noqa: BLE001
            raise ValidationError(
                f"Imitation (BC) fit failed for estimator={resolved_estimator!r}: {exc}"
            ) from exc
        pred = np.asarray(model.predict(x), dtype=float)
        train_score = float(1.0 - np.mean(np.abs(pred - y)) / (np.std(y) + 1e-8))

    config = ImitationConfig(
        task=resolved_task,  # type: ignore[arg-type]
        estimator=resolved_estimator,  # type: ignore[arg-type]
        columns=tuple(cols),
        action_column=action_col,
        random_state=random_state,
        prefer_reduce_components=prefer_reduce_components,
    )
    plan = ImitationPlan(
        task=resolved_task,
        estimator=resolved_estimator,
        columns=tuple(cols),
        action_column=action_col,
        n_train_rows=n_train,
        classes_=classes,
        label_encoder_=label_encoder,
        estimator_=model,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
        used_reduce_components=used_reduce,
        config=config.to_dict(),
        train_score=train_score,
    )
    result = ImitationFitResult(
        task=resolved_task,
        estimator=resolved_estimator,
        n_train_rows=n_train,
        columns=tuple(cols),
        action_column=action_col,
        classes=classes,
        train_score=train_score,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
    return plan, result


def predict_imitation_action(
    dataset: Dataset,
    plan: ImitationPlan,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll = "test",
) -> ImitationPredictResult:
    """Predict actions for a partition under the fitted BC policy."""
    frame = _frame_for(dataset, split_plan, partition)
    if frame.empty:
        return ImitationPredictResult(
            partition=str(partition),
            task=plan.task,
            n_rows=0,
            actions=(),
            disclosures=("Empty partition; no actions predicted.",),
        )
    x = matrix_from_frame(frame, list(plan.columns))
    raw = plan.estimator_.predict(x)
    if plan.task == "classification":
        actions = tuple(decode_discrete_actions(np.asarray(raw), plan.label_encoder_))
    else:
        actions = tuple(float(v) for v in np.asarray(raw, dtype=float).tolist())
    return ImitationPredictResult(
        partition=str(partition),
        task=plan.task,
        n_rows=int(x.shape[0]),
        actions=actions,
        disclosures=(
            "Actions predicted by a train-fitted behavioral cloning policy.",
        ),
    )


def evaluate_imitation(
    dataset: Dataset,
    plan: ImitationPlan,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll = "validation",
) -> ImitationEvalResult:
    """Compare predicted actions to held-out demonstration actions."""
    frame = _frame_for(dataset, split_plan, partition)
    if plan.action_column not in frame.columns:
        raise ValidationError(
            f"action_column={plan.action_column!r} missing from partition={partition!r}."
        )
    if frame.empty:
        return ImitationEvalResult(
            partition=str(partition),
            task=plan.task,
            n_rows=0,
            metrics={},
            disclosures=("Empty partition; no imitation metrics.",),
            warnings=("Empty evaluation partition.",),
        )
    pred = predict_imitation_action(
        dataset, plan, split_plan, partition=partition
    )
    y_true = frame[plan.action_column]
    disclosures = [
        "Imitation metrics compare policy actions to held-out demonstration actions.",
        "Holdout rows are never used to fit the cloning policy.",
    ]
    warnings: list[str] = []
    if plan.task == "classification":
        if y_true.isna().any():
            raise ValidationError(
                "Imitation classification eval requires non-null demonstration actions."
            )
        metrics = classification_metrics(list(y_true), list(pred.actions))
    else:
        yt = continuous_actions(y_true)
        yp = np.asarray(pred.actions, dtype=float)
        metrics = regression_metrics(yt, yp)
    return ImitationEvalResult(
        partition=str(partition),
        task=plan.task,
        n_rows=int(len(frame)),
        metrics=metrics,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )


def _frame_for(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    partition: PartitionOrAll,
) -> pd.DataFrame:
    if partition == "all":
        return dataset._ensure_pandas()
    if split_plan is None:
        raise ValidationError(
            "A SplitPlan is required for partition-scoped imitation operations."
        )
    return frame_for_partition(dataset, split_plan, partition)  # type: ignore[arg-type]


def _validate_estimator_for_task(task: str, estimator: str) -> None:
    if task == "classification" and estimator not in _CLF:
        raise ValidationError(
            f"estimator={estimator!r} is not valid for classification BC. "
            f"Supported: {sorted(_CLF)}"
        )
    if task == "regression" and estimator not in _REG:
        raise ValidationError(
            f"estimator={estimator!r} is not valid for regression BC. "
            f"Supported: {sorted(_REG)}"
        )


def _build_estimator(name: str, random_state: int | None, *, task: str) -> Any:
    table = _CLF if task == "classification" else _REG
    key = str(name).lower().replace("-", "_")
    if key not in table:
        raise ValidationError(f"Unknown imitation estimator={name!r}.")
    return table[key](random_state)
