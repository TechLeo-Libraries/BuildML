"""Learn a policy by copying demonstrated decisions.

Behavioural cloning is the most direct answer to "I have a record of what a
person did in each situation, and I want a model that does the same". You give
it a table where each row is a situation plus the action taken, and it fits a
supervised model mapping situation to action. Discrete actions make it a
classification problem, continuous ones a regression problem.

The framing matters more than the machinery. **Cloning learns to reproduce the
demonstrations, not to succeed.** If the demonstrator was mediocre, the clone is
mediocre in the same way, and no amount of accuracy on holdout rows will reveal
that: high agreement with a poor demonstrator is still a poor policy. Accuracy
here answers "does it act like the demonstrator", never "does it act well".

The second limit is subtler and specific to policies. A cloned policy acting in
the real world drives itself into situations the demonstrator never reached,
because its small errors compound: a slightly-off action produces a slightly
unfamiliar state, where the policy is less reliable still. Interactive methods
such as DAgger address this by querying the demonstrator on the states the
policy actually visits. This module does not: it fits offline from a fixed
table, and holdout rows come from the same demonstrator distribution as train.

Fitting uses train rows only, so :func:`evaluate_imitation` measures agreement
on demonstrations the policy has not seen.

See Also
--------
buildml.rl.fit : Learning from rewards rather than from demonstrations.
buildml.rl.features : Column resolution and the metrics used here.
"""

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
from buildml.rl.catalog import resolve_imitation_backend_method
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
from buildml.rl.types import ImitationBackend, ImitationConfig, ImitationEstimator, ImitationMethod, ImitationTask

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
    backend: ImitationBackend | None = None,
    task: ImitationTask | None = None,
    estimator: ImitationEstimator | None = None,
    method: ImitationMethod | None = None,
    columns: list[str] | None = None,
    action_column: str | None = None,
    env_id: str | None = None,
    n_epochs: int = 40,
    random_state: int | None = 0,
    prefer_reduce_components: bool = True,
    reduce_plan: Any | None = None,
) -> tuple[ImitationPlan, ImitationFitResult]:
    """Learn to reproduce demonstrated actions from a table of examples.

    Reads the training partition as demonstrations: each row a situation and
    the action taken in it: and fits a model that predicts the action from the
    situation. Whether that is a classifier or a regressor follows from the
    action column: labelled or few-valued actions give classification, numeric
    ones give regression.

    Parameters
    ----------
    dataset:
        The demonstration table.
    split_plan:
        Required. Fitting on all rows would leave nothing to measure agreement
        against.
    backend:
        ``'sklearn'`` (default) fits a scikit-learn model and always works.
        ``'industry'`` fits a neural policy and needs ``buildml[rl-industry]``.
        Reach for the neural path when the mapping from state to action is
        genuinely non-linear and there are enough demonstrations to support it;
        on tabular data the boosted-tree default is usually competitive.
    task:
        Override the inferred task. Useful when actions are integer-coded
        categories that would otherwise look continuous.
    estimator:
        The scikit-learn model. ``'logistic_regression'`` and ``'ridge'`` are
        linear, fast, and inspectable; ``'hist_gradient_boosting'`` and
        ``'hist_gradient_boosting_regressor'`` capture interactions between
        state features at the cost of transparency. Must match the task.
    method:
        The industry method: ``'bc_mlp'`` for a plain neural policy, or
        ``'gail_lite'`` for adversarial imitation, which also needs ``env_id``.
    columns:
        The state features. Defaults to the usable columns of the dataset with
        the action column excluded.
    action_column:
        The demonstrated action. Defaults to the Dataset target.
    env_id:
        The Gymnasium environment, required by ``'gail_lite'``.
    n_epochs:
        Neural training passes. Ignored on the scikit-learn path.
    random_state:
        Seed for reproducibility.
    prefer_reduce_components:
        When ``True`` and a reduction is attached, its components are used as
        state features rather than the raw columns.
    reduce_plan:
        An explicit reduction plan, overriding whatever is attached.

    Returns
    -------
    ImitationPlan
        The fitted policy. Pass this to :func:`predict_imitation_action` and
        :func:`evaluate_imitation`.
    ImitationFitResult
        What the fit saw: rows, columns, action classes, and the in-sample
        agreement score.

    Raises
    ------
    LeakageError
        If ``split_plan`` is ``None`` or defines no train partition.
    ValidationError
        If the action column is absent from the dataset or from train, if the
        estimator does not suit the task, if ``'gail_lite'`` is requested
        without ``env_id``, or if the underlying fit fails.

    Notes
    -----
    **``train_score`` measures agreement with the demonstrator, in-sample.** It
    is not a measure of whether the policy is any good: a clone that perfectly
    reproduces bad decisions scores 1.0. Judge the demonstrator separately;
    cloning can only inherit its quality.

    **The regression score is not R².** It is ``1 - MAE / std(y)``, a scale-free
    agreement measure that stays interpretable when actions are bounded. Do not
    compare it against R² values from elsewhere in BuildML.

    Examples
    --------
    >>> plan, result = fit_imitation(dataset, split_plan)  # doctest: +SKIP
    >>> result.task, result.n_train_rows  # doctest: +SKIP
    ('classification', 800)
    >>> evaluate_imitation(dataset, plan, split_plan).metrics  # doctest: +SKIP
    {'accuracy': 0.86, 'macro_f1': 0.71}

    See Also
    --------
    evaluate_imitation : Agreement on demonstrations the policy never saw.
    buildml.rl.fit.fit_rl : Learn from rewards when demonstrations are absent
        or the demonstrator is not worth copying.
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
    resolved_backend, resolved_method = resolve_imitation_backend_method(
        backend=backend,
        estimator=estimator,
        method=method,
        task=resolved_task,
    )

    if resolved_backend == "industry":
        return _fit_imitation_industry(
            dataset,
            split_plan,
            train=train,
            action_col=action_col,
            target=target,
            task=resolved_task,
            method=resolved_method,  # type: ignore[arg-type]
            columns=columns,
            env_id=env_id,
            n_epochs=n_epochs,
            random_state=random_state,
            prefer_reduce_components=prefer_reduce_components,
            reduce_plan=reduce_plan,
        )

    resolved_estimator = resolved_method  # sklearn estimator key
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
            "Honesty: BC from tables: not inverse RL, not DAgger, not a robotics stack.",
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
        backend="sklearn",
        estimator=resolved_estimator,  # type: ignore[arg-type]
        columns=tuple(cols),
        action_column=action_col,
        random_state=random_state,
        prefer_reduce_components=prefer_reduce_components,
    )
    plan = ImitationPlan(
        task=resolved_task,
        backend="sklearn",
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
        backend="sklearn",
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


def _fit_imitation_industry(
    dataset: Dataset,
    split_plan: SplitPlan,
    *,
    train: pd.DataFrame,
    action_col: str,
    target: str,
    task: str,
    method: str,
    columns: list[str] | None,
    env_id: str | None,
    n_epochs: int,
    random_state: int | None,
    prefer_reduce_components: bool,
    reduce_plan: Any | None,
) -> tuple[ImitationPlan, ImitationFitResult]:
    from buildml.rl.adapters.imitation_industry import (
        fit_tabular_bc_mlp,
        fit_tabular_gail_lite,
    )

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
    y_codes, label_encoder, classes = encode_discrete_actions(train[action_col])
    n_actions = len(classes)
    warnings: list[str] = []

    if method == "bc_mlp":
        policy, train_score, ind_disclosures, ind_warnings = fit_tabular_bc_mlp(
            x,
            y_codes,
            n_actions=n_actions,
            n_epochs=n_epochs,
            random_state=random_state,
        )
    elif method == "gail_lite":
        if env_id is None:
            raise ValidationError(
                "gail_lite requires env_id=... with env-compatible demonstration rows."
            )
        policy, train_score, ind_disclosures, ind_warnings = fit_tabular_gail_lite(
            x,
            y_codes,
            env_id=env_id,
            n_actions=n_actions,
            random_state=random_state,
        )
    else:
        raise ValidationError(f"Unknown industry imitation method={method!r}.")

    disclosures.extend(ind_disclosures)
    warnings.extend(ind_warnings)

    config = ImitationConfig(
        task=task,  # type: ignore[arg-type]
        backend="industry",
        method=method,  # type: ignore[arg-type]
        columns=tuple(cols),
        action_column=action_col,
        env_id=env_id,
        n_epochs=n_epochs,
        random_state=random_state,
        prefer_reduce_components=prefer_reduce_components,
    )
    plan = ImitationPlan(
        task=task,
        backend="industry",
        estimator=method,
        method=method,
        columns=tuple(cols),
        action_column=action_col,
        n_train_rows=n_train,
        classes_=classes,
        label_encoder_=label_encoder,
        estimator_=policy,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
        used_reduce_components=used_reduce,
        config=config.to_dict(),
        train_score=train_score,
    )
    result = ImitationFitResult(
        task=task,
        backend="industry",
        estimator=method,
        method=method,
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
    """Ask the cloned policy what it would do in each row's situation.

    Applies the fitted policy to a partition and returns one action per row. No
    demonstrated action is needed: this is what you call in production, where
    the right answer is not known.

    Parameters
    ----------
    dataset:
        A dataset carrying the state columns the plan was fitted on.
    plan:
        The fitted policy from :func:`fit_imitation`.
    split_plan:
        Required unless ``partition='all'``.
    partition:
        Which rows to act on. ``'all'`` scores every row.

    Returns
    -------
    ImitationPredictResult
        The chosen actions in row order. Classification actions come back as
        the original labels, not the internal integer codes; regression actions
        as floats.

    Raises
    ------
    ValidationError
        If the state columns are missing, or a partition is requested without a
        split plan.

    Notes
    -----
    An empty partition returns an empty result rather than raising, so a
    pipeline that scores several partitions is not derailed by one being empty.

    See Also
    --------
    evaluate_imitation : The same predictions, scored against known actions.
    """
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
    estimator = plan.estimator_
    if hasattr(estimator, "predict") and hasattr(estimator, "method"):
        raw = estimator.predict(x)
    else:
        raw = estimator.predict(x)
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
    """Measure how often the clone agrees with the demonstrator on unseen rows.

    Predicts actions for a holdout partition and compares them against the
    actions actually demonstrated there. Because the policy was fitted on train
    alone, this is an honest measure of agreement.

    Parameters
    ----------
    dataset:
        A dataset carrying both the state columns and the action column.
    plan:
        The fitted policy from :func:`fit_imitation`.
    split_plan:
        Required unless ``partition='all'``.
    partition:
        Which rows to score. Defaults to ``'validation'``; keep test in reserve
        until the policy is settled.

    Returns
    -------
    ImitationEvalResult
        ``accuracy`` and ``macro_f1`` for discrete actions, or ``rmse``,
        ``mae``, and ``r2`` for continuous ones.

    Raises
    ------
    ValidationError
        If the action column is missing from the partition, or if
        classification actions contain nulls: there is no defensible way to
        score a prediction against an unknown action.

    Notes
    -----
    **These metrics measure imitation, not performance.** A score of 0.95 says
    the clone acts like the demonstrator 95% of the time. Whether that is
    desirable depends entirely on the demonstrator, which no metric here can
    assess.

    **Read ``macro_f1`` before ``accuracy`` when actions are imbalanced.** If the
    demonstrator chose one action 90% of the time, a policy that always chooses
    it scores 0.90 accuracy while having learned nothing. Macro F1 averages over
    actions rather than rows, so the ignored actions drag it down.

    Examples
    --------
    >>> result = evaluate_imitation(dataset, plan, split_plan)  # doctest: +SKIP
    >>> result.metrics  # doctest: +SKIP
    {'accuracy': 0.86, 'macro_f1': 0.71}

    See Also
    --------
    predict_imitation_action : Actions without a comparison.
    """
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
