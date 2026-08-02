"""Leakage-safe native ensemble fit helpers (voting / stacking / blending)."""

from __future__ import annotations

from typing import Any, Literal, Mapping, Sequence

from sklearn.base import clone
from sklearn.ensemble import (
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition, frame_for_partition
from buildml.ensemble.blending import HoldoutBlendClassifier, HoldoutBlendRegressor
from buildml.ensemble.results import EnsembleFitResult, EnsemblePlan
from buildml.ensemble.types import BlendMethod, EnsembleConfig, VotingMethod
from buildml.model.supervised import FitResult, _infer_task, fit_estimator

TaskType = Literal["classification", "regression", "auto"]
EstimatorMap = Mapping[str, Any] | Sequence[tuple[str, Any]]


def _named_estimators(estimators: EstimatorMap) -> list[tuple[str, Any]]:
    if isinstance(estimators, Mapping):
        named = [(str(name), est) for name, est in estimators.items()]
    else:
        named = [(str(name), est) for name, est in estimators]
    if len(named) < 2:
        raise ValidationError("Native ensembles require at least two named base estimators.")
    names = [n for n, _ in named]
    if len(set(names)) != len(names):
        raise ValidationError(f"Duplicate estimator names are not allowed: {names}")
    for name, est in named:
        if est is None:
            raise ValidationError(f"Estimator {name!r} is None.")
    return named


def _default_final(task: Literal["classification", "regression"]) -> Any:
    if task == "classification":
        return LogisticRegression(max_iter=1000)
    return Ridge()


def _resolve_task_from_train(
    dataset: Dataset,
    split_plan: SplitPlan,
    named: list[tuple[str, Any]],
    task: TaskType,
) -> Literal["classification", "regression"]:
    target = dataset.require_target()
    y = frame_for_partition(dataset, split_plan, "train")[target]
    return _infer_task(y, task, named[0][1])


def build_voting_estimator(
    named: list[tuple[str, Any]],
    *,
    task: Literal["classification", "regression"],
    voting: VotingMethod = "hard",
    weights: Sequence[float] | None = None,
) -> Any:
    """Build an unfitted sklearn VotingClassifier / VotingRegressor."""
    weight_list = None if weights is None else list(weights)
    if weight_list is not None and len(weight_list) != len(named):
        raise ValidationError(
            f"weights length ({len(weight_list)}) must match estimators ({len(named)})."
        )
    clones = [(n, clone(e)) for n, e in named]
    if task == "classification":
        if voting == "soft":
            missing = [n for n, e in clones if not hasattr(e, "predict_proba")]
            if missing:
                raise ValidationError(
                    "Soft voting requires predict_proba on every base estimator; "
                    f"missing for: {missing}"
                )
        return VotingClassifier(estimators=clones, voting=voting, weights=weight_list)
    return VotingRegressor(estimators=clones, weights=weight_list)


def build_stacking_estimator(
    named: list[tuple[str, Any]],
    *,
    task: Literal["classification", "regression"],
    final_estimator: Any | None = None,
    cv: int = 5,
    passthrough: bool = False,
    stack_method: str = "auto",
) -> Any:
    """Build an unfitted sklearn StackingClassifier / StackingRegressor."""
    if cv < 2:
        raise ValidationError("Stacking cv must be >= 2 (out-of-fold meta features).")
    clones = [(n, clone(e)) for n, e in named]
    final = clone(final_estimator) if final_estimator is not None else _default_final(task)
    if task == "classification":
        kwargs: dict[str, Any] = {
            "estimators": clones,
            "final_estimator": final,
            "cv": cv,
            "passthrough": passthrough,
        }
        if stack_method and stack_method != "auto":
            kwargs["stack_method"] = stack_method
        try:
            return StackingClassifier(**kwargs)
        except TypeError:
            kwargs.pop("stack_method", None)
            return StackingClassifier(**kwargs)
    return StackingRegressor(
        estimators=clones,
        final_estimator=final,
        cv=cv,
        passthrough=passthrough,
    )


def build_blending_estimator(
    named: list[tuple[str, Any]],
    *,
    task: Literal["classification", "regression"],
    final_estimator: Any | None = None,
    holdout_fraction: float = 0.2,
    blend_method: BlendMethod = "predict_proba",
    random_state: int | None = 0,
    refit_bases_on_full_train: bool = True,
    passthrough: bool = False,
) -> Any:
    """Build an unfitted holdout-blend estimator (train-inner holdout only)."""
    clones = [(n, clone(e)) for n, e in named]
    final = clone(final_estimator) if final_estimator is not None else None
    if task == "classification":
        method: BlendMethod = blend_method
        if method == "predict_proba":
            missing = [n for n, e in clones if not hasattr(e, "predict_proba")]
            if missing:
                method = "predict"
        return HoldoutBlendClassifier(
            estimators=clones,
            final_estimator=final,
            holdout_fraction=holdout_fraction,
            blend_method=method,
            random_state=random_state,
            refit_bases_on_full_train=refit_bases_on_full_train,
            passthrough=passthrough,
        )
    return HoldoutBlendRegressor(
        estimators=clones,
        final_estimator=final,
        holdout_fraction=holdout_fraction,
        blend_method="predict",
        random_state=random_state,
        refit_bases_on_full_train=refit_bases_on_full_train,
        passthrough=passthrough,
    )


def _disclosures_for(
    strategy: str,
    *,
    task: str,
    named: list[tuple[str, Any]],
    voting: str | None = None,
    cv: int | None = None,
    holdout_fraction: float | None = None,
    blend_method: str | None = None,
    refit_bases: bool = True,
    passthrough: bool = False,
) -> tuple[str, ...]:
    notes = [
        f"Native {strategy} ensemble fitted on the Session train partition only.",
        f"Base estimators: {', '.join(n for n, _ in named)}.",
        "Session test / validation rows are never used to fit bases or the meta-learner.",
    ]
    if strategy == "voting":
        notes.append(f"Voting mode={voting or 'hard'} for task={task}.")
        notes.append(
            "Passing RandomForest / GradientBoosting to Session.fit remains a single "
            "estimator; this API builds VotingClassifier/VotingRegressor."
        )
    elif strategy == "stacking":
        notes.append(
            f"Stacking uses {cv}-fold out-of-fold predictions inside train "
            "(sklearn Stacking*); Session test is held out."
        )
        if passthrough:
            notes.append("Passthrough=True concatenates original features with meta features.")
    else:
        notes.append(
            f"Blending carved holdout_fraction={holdout_fraction} from train only "
            f"(blend_method={blend_method}); not Session validation/test."
        )
        if refit_bases:
            notes.append(
                "Base estimators were refit on the full train partition after meta-learner "
                "fit (standard deploy pattern; disclosed)."
            )
        else:
            notes.append("Base estimators were left as blend-train fits (no full-train refit).")
        if passthrough:
            notes.append(
                "Passthrough=True concatenates original features with blend meta features."
            )
    return tuple(notes)


def _package(
    fit_result: FitResult,
    *,
    strategy: Literal["voting", "stacking", "blending"],
    named: list[tuple[str, Any]],
    config: EnsembleConfig,
) -> tuple[EnsemblePlan, EnsembleFitResult, FitResult]:
    fitted = fit_result.estimator
    final_name = config.final_estimator_name
    if final_name is None and hasattr(fitted, "final_estimator_"):
        final_name = type(getattr(fitted, "final_estimator_")).__name__

    blend_method = config.blend_method if strategy == "blending" else None
    if strategy == "blending" and hasattr(fitted, "blend_method"):
        blend_method = fitted.blend_method

    disclosures = _disclosures_for(
        strategy,
        task=fit_result.task,
        named=named,
        voting=config.voting if strategy == "voting" else None,
        cv=config.cv if strategy == "stacking" else None,
        holdout_fraction=config.holdout_fraction if strategy == "blending" else None,
        blend_method=blend_method,
        refit_bases=config.refit_bases_on_full_train,
        passthrough=config.passthrough,
    )
    warnings: list[str] = []
    if strategy == "blending" and hasattr(fitted, "blend_holdout_rows_"):
        if int(fitted.blend_holdout_rows_) < 10:
            warnings.append(
                "Blend holdout has fewer than 10 rows; meta-learner estimates may be unstable."
            )

    plan = EnsemblePlan(
        strategy=strategy,
        task=fit_result.task,
        estimator_names=tuple(n for n, _ in named),
        feature_columns=fit_result.feature_columns,
        target_column=fit_result.target_column,
        n_train_rows=fit_result.n_train_rows,
        estimator_=fitted,
        final_estimator_name=final_name,
        voting=config.voting if strategy == "voting" else None,
        cv=config.cv if strategy == "stacking" else None,
        passthrough=config.passthrough,
        holdout_fraction=config.holdout_fraction if strategy == "blending" else None,
        blend_method=blend_method,
        refit_bases_on_full_train=config.refit_bases_on_full_train,
        disclosures=disclosures,
        warnings=tuple(warnings),
        config=config.to_dict(),
    )
    result = EnsembleFitResult(
        strategy=strategy,
        task=fit_result.task,
        estimator_names=plan.estimator_names,
        n_train_rows=plan.n_train_rows,
        feature_columns=plan.feature_columns,
        target_column=plan.target_column,
        final_estimator_name=final_name,
        voting=plan.voting,
        cv=plan.cv,
        holdout_fraction=plan.holdout_fraction,
        blend_method=plan.blend_method,
        disclosures=disclosures,
        warnings=tuple(warnings),
    )
    return plan, result, fit_result


def fit_voting_ensemble(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    estimators: EstimatorMap,
    *,
    voting: VotingMethod = "hard",
    weights: Sequence[float] | None = None,
    task: TaskType = "auto",
) -> tuple[EnsemblePlan, EnsembleFitResult, FitResult]:
    """Fit a VotingClassifier / VotingRegressor on the train partition only."""
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None
    named = _named_estimators(estimators)
    resolved = _resolve_task_from_train(dataset, split_plan, named, task)
    config = EnsembleConfig(
        strategy="voting",
        estimator_names=tuple(n for n, _ in named),
        task=task,
        voting=voting if resolved == "classification" else "hard",
        weights=None if weights is None else tuple(float(w) for w in weights),
    )
    estimator = build_voting_estimator(
        named, task=resolved, voting=config.voting, weights=config.weights
    )
    fit_result = fit_estimator(dataset, split_plan, estimator, task=resolved)
    return _package(fit_result, strategy="voting", named=named, config=config)


def fit_stacking_ensemble(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    estimators: EstimatorMap,
    *,
    final_estimator: Any | None = None,
    cv: int = 5,
    passthrough: bool = False,
    stack_method: str = "auto",
    task: TaskType = "auto",
) -> tuple[EnsemblePlan, EnsembleFitResult, FitResult]:
    """Fit a StackingClassifier / StackingRegressor on the train partition only."""
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None
    named = _named_estimators(estimators)
    resolved = _resolve_task_from_train(dataset, split_plan, named, task)
    final_name = None if final_estimator is None else type(final_estimator).__name__
    config = EnsembleConfig(
        strategy="stacking",
        estimator_names=tuple(n for n, _ in named),
        task=task,
        cv=cv,
        passthrough=passthrough,
        stack_method=stack_method,
        final_estimator_name=final_name,
    )
    estimator = build_stacking_estimator(
        named,
        task=resolved,
        final_estimator=final_estimator,
        cv=cv,
        passthrough=passthrough,
        stack_method=stack_method,
    )
    fit_result = fit_estimator(dataset, split_plan, estimator, task=resolved)
    return _package(fit_result, strategy="stacking", named=named, config=config)


def fit_blending_ensemble(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    estimators: EstimatorMap,
    *,
    final_estimator: Any | None = None,
    holdout_fraction: float = 0.2,
    blend_method: BlendMethod = "predict_proba",
    random_state: int | None = 0,
    refit_bases_on_full_train: bool = True,
    passthrough: bool = False,
    task: TaskType = "auto",
) -> tuple[EnsemblePlan, EnsembleFitResult, FitResult]:
    """Fit a holdout-blend ensemble on the train partition only.

    The blend holdout is carved from **train**. Session validation/test never
    enter meta-learner fitting.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None
    named = _named_estimators(estimators)
    resolved = _resolve_task_from_train(dataset, split_plan, named, task)
    final_name = None if final_estimator is None else type(final_estimator).__name__
    method: BlendMethod = blend_method if resolved == "classification" else "predict"
    config = EnsembleConfig(
        strategy="blending",
        estimator_names=tuple(n for n, _ in named),
        task=task,
        holdout_fraction=holdout_fraction,
        blend_method=method,
        random_state=random_state,
        refit_bases_on_full_train=refit_bases_on_full_train,
        passthrough=passthrough,
        final_estimator_name=final_name,
    )
    estimator = build_blending_estimator(
        named,
        task=resolved,
        final_estimator=final_estimator,
        holdout_fraction=holdout_fraction,
        blend_method=method,
        random_state=random_state,
        refit_bases_on_full_train=refit_bases_on_full_train,
        passthrough=passthrough,
    )
    fit_result = fit_estimator(dataset, split_plan, estimator, task=resolved)
    return _package(fit_result, strategy="blending", named=named, config=config)
