"""Combine several models into one, without letting them see the holdout.

Different models make different mistakes. A linear model misses interactions a
tree captures; a tree extrapolates badly where the linear model does not. When
their errors are uncorrelated, combining them beats any one of them: which is
why ensembles win competitions and why almost every production system is one.

Three ways to combine, in increasing order of power and of risk.

*Voting* averages predictions, or takes a majority. There is nothing to learn,
so there is nothing to overfit, and it is the right first attempt.

*Stacking* trains a second model on the base models' predictions, learning that
one model is more trustworthy in some regions than another. The danger is
obvious once stated: if the meta-learner trains on predictions the base models
made about rows they were fitted on, it learns from memorised answers and the
whole thing collapses. scikit-learn avoids this with out-of-fold predictions,
which is why stacking costs a full cross-validation.

*Blending* does the same thing more cheaply, by carving a holdout out of train,
fitting bases on the rest, and fitting the meta-learner on their predictions
over that holdout. One split instead of `cv` folds, at the cost of a
meta-learner trained on less data.

The discipline this module adds is that every one of these fits on the Session
train partition and nothing else. Blending's inner holdout comes out of train,
never out of validation or test: an easy mistake to make by hand, and one that
produces a beautiful score and a model that does not work.

See Also
--------
buildml.ensemble.blending : The holdout-blend estimators.
buildml.model.compare : Deciding whether the ensemble actually beat its parts.
"""

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
    """Assemble an unfitted voting estimator, checking what soft voting needs.

    Voting is the ensemble with nothing to learn: predictions are averaged for
    regression, and either majority-voted or probability-averaged for
    classification. Because no meta-model is fitted, there is no way for it to
    overfit the combination, which makes it the safest place to start.

    Soft voting averages probabilities rather than counting votes, and is
    usually better because it keeps the information in *how* confident each
    model was: a model that says 0.51 and a model that says 0.99 should not
    count equally. It requires every base to expose ``predict_proba``, and a
    base that does not is caught here rather than at fit time.

    Parameters
    ----------
    named:
        Base estimators as ``(name, estimator)`` pairs. Cloned, so the caller's
        objects are left unfitted.
    task:
        ``'classification'`` or ``'regression'``, selecting the estimator class.
    voting:
        ``'hard'`` for majority vote, ``'soft'`` for averaged probabilities.
        Ignored for regression, which always averages.
    weights:
        Relative influence per estimator, in the same order as ``named``.
        ``None`` weights them equally.

    Returns
    -------
    Any
        An unfitted ``VotingClassifier`` or ``VotingRegressor``.

    Raises
    ------
    ValidationError
        If ``weights`` has a different length than ``named``: a mismatch would
        otherwise silently misalign weights with models: or if soft voting was
        asked for and some base cannot produce probabilities.

    Notes
    -----
    **Weight the models by measured performance, not by intuition.** Weights are
    another thing to tune, and tuning them against the test partition is a way
    to overfit an ensemble that had no other way to overfit.

    **Voting helps most when the bases disagree.** Three variants of the same
    gradient-boosting model will average to approximately themselves; a tree, a
    linear model, and a nearest-neighbour model will not.

    See Also
    --------
    build_stacking_estimator : Learning the combination instead of fixing it.
    """
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
    """Assemble an unfitted stacking estimator over out-of-fold predictions.

    Stacking learns *how* to combine rather than fixing it in advance. The
    meta-learner sees each base model's prediction as a feature and works out
    which to trust, which can capture that one model is reliable on short
    tenures and another on long ones.

    The reason ``cv`` exists, and cannot be avoided, is that the meta-learner
    must never see a base model's prediction for a row that base was fitted on.
    Those predictions are memorised rather than earned, and a meta-learner
    trained on them learns to trust whichever base overfits hardest. Fitting
    each base ``cv`` times on folds and predicting the held-out fold each time
    is what produces honest meta-features: and is why stacking costs roughly
    ``cv`` times a plain fit.

    Parameters
    ----------
    named:
        Base estimators as ``(name, estimator)`` pairs. Cloned.
    task:
        ``'classification'`` or ``'regression'``.
    final_estimator:
        The meta-learner. Defaults to logistic regression or ridge: a simple,
        regularised model is the standard choice, because a complex meta-learner
        on a handful of prediction columns overfits readily.
    cv:
        Folds for the out-of-fold predictions. Must be at least 2. More folds
        give the meta-learner cleaner features at proportionally more compute.
    passthrough:
        Also give the meta-learner the original features, not just the base
        predictions. Occasionally helps; it also multiplies the meta-learner's
        input width and its chance of overfitting.
    stack_method:
        What the bases contribute: ``'auto'`` picks probabilities where
        available, and ``'predict'``, ``'predict_proba'``, or
        ``'decision_function'`` force it. Regression ignores this.

    Returns
    -------
    Any
        An unfitted ``StackingClassifier`` or ``StackingRegressor``.

    Raises
    ------
    ValidationError
        If ``cv`` is below 2. One fold cannot produce out-of-fold predictions,
        so the meta-learner would train on in-sample answers: the exact failure
        stacking exists to avoid.

    Notes
    -----
    **Stacking is expensive.** Each base is fitted ``cv`` times for the
    meta-features and once more on the full data, so five folds and four bases
    means twenty-four fits.

    **Compare against voting before keeping it.** Stacking often wins by very
    little over soft voting while costing far more and being harder to explain;
    :mod:`buildml.model.compare` is how you find out which happened here.

    See Also
    --------
    build_blending_estimator : The cheaper approximation of this.
    """
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
    """Assemble an unfitted blend estimator using one inner split, not ``cv`` folds.

    Blending is stacking's cheaper cousin. Instead of ``cv`` rounds of fitting to
    produce out-of-fold predictions, it splits train once: bases are fitted on
    the larger part, predict the smaller part, and the meta-learner trains on
    those predictions. One extra fit rather than ``cv`` of them.

    What you pay for the speed is a meta-learner trained on a single holdout
    rather than on every training row. With a small dataset that holdout may be
    a few hundred rows, and a meta-learner fitted on a few hundred noisy
    predictions is itself noisy.

    Parameters
    ----------
    named:
        Base estimators as ``(name, estimator)`` pairs. Cloned.
    task:
        ``'classification'`` or ``'regression'``.
    final_estimator:
        The meta-learner. A sensible default is chosen when omitted.
    holdout_fraction:
        How much of train to reserve for the meta-learner. Too small and the
        meta-learner is unstable; too large and the bases are undertrained.
    blend_method:
        ``'predict_proba'`` or ``'predict'``. Probabilities carry more
        information, and are downgraded automatically when a base cannot supply
        them: quietly, so check ``blend_method`` on the resulting plan if it
        matters.
    random_state:
        Seed for the inner split, so a blend is reproducible.
    refit_bases_on_full_train:
        After the meta-learner is fitted, refit the bases on all of train. This
        is the standard deployment pattern and generally right: the bases get
        the full data: but it does mean the deployed bases are not quite the
        ones the meta-learner was calibrated against.
    passthrough:
        Also give the meta-learner the original features.

    Returns
    -------
    Any
        An unfitted :class:`~buildml.ensemble.blending.HoldoutBlendClassifier`
        or :class:`~buildml.ensemble.blending.HoldoutBlendRegressor`.

    Notes
    -----
    **Regression always blends on ``'predict'``**, since there are no
    probabilities to average.

    **The inner holdout comes out of train, always.** Carving it from validation
    or test is the mistake blending invites, and it does not happen here.

    **Prefer stacking when you can afford it.** Blending exists for when ``cv``
    full fits is too slow; with the compute available, out-of-fold meta-features
    are strictly better.

    See Also
    --------
    build_stacking_estimator : The more thorough version.
    """
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
    """Fit a voting ensemble on train, and record how it was built.

    The safest ensemble and the right first one to try. Nothing is learned about
    how to combine, so nothing about the combination can overfit; any gain over
    the best single model is real.

    Returns three objects because they answer different questions. The plan is
    what you save and reload. The ensemble result is the disclosure record :
    which bases, which mode, what was and was not used. The fit result is the
    ordinary one every other BuildML model produces, so the ensemble drops into
    ``evaluate``, ``compare``, and the diagnostics unchanged.

    Parameters
    ----------
    dataset:
        The data, with roles assigned.
    split_plan:
        Partition membership. Required: an ensemble without a holdout cannot be
        honestly evaluated, and the point of an ensemble is the comparison.
    estimators:
        At least two base estimators, as a mapping or ``(name, estimator)``
        pairs. Names must be unique, since they identify the models in the plan
        and the disclosures.
    voting:
        ``'hard'`` for majority vote, ``'soft'`` for averaged probabilities.
        Forced to ``'hard'`` for regression.
    weights:
        Relative influence, in estimator order.
    task:
        ``'auto'`` infers from the target and the first estimator.

    Returns
    -------
    tuple
        ``(EnsemblePlan, EnsembleFitResult, FitResult)``: the reloadable plan,
        the disclosure record, and the standard fit result.

    Raises
    ------
    ValidationError
        If fewer than two estimators are given, if names repeat, if an estimator
        is ``None``, if the weights do not line up, if soft voting is asked of a
        base without probabilities, or if the split is missing.

    Notes
    -----
    **Two estimators is the minimum, and the point.** Averaging one model with
    itself is just that model.

    **Diverse bases matter more than strong ones.** Three tuned variants of the
    same algorithm average to approximately themselves.

    **Soft voting usually beats hard.** A majority vote discards how confident
    each model was, and that is often the most useful thing they said.

    Examples
    --------
    ::

        plan, ensemble, fit = fit_voting_ensemble(
            dataset, split_plan,
            {"forest": RandomForestClassifier(),
             "linear": LogisticRegression(max_iter=1000),
             "boost": GradientBoostingClassifier()},
            voting="soft",
        )
        for note in ensemble.disclosures:
            print(note)

    See Also
    --------
    fit_stacking_ensemble : Learning the combination instead of fixing it.
    """
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
    """Fit a stacking ensemble on train, using out-of-fold meta-features.

    The cross-validation happens entirely inside the train partition, so the
    meta-learner never sees a base model's prediction for a row that base was
    fitted on, and never sees a validation or test row at all.

    Parameters
    ----------
    dataset:
        The data, with roles assigned.
    split_plan:
        Partition membership. Required.
    estimators:
        At least two base estimators, uniquely named.
    final_estimator:
        The meta-learner. Defaults to logistic regression or ridge.
    cv:
        Folds for the out-of-fold predictions, at least 2. Five is the usual
        compromise between clean meta-features and runtime.
    passthrough:
        Also give the meta-learner the original features.
    stack_method:
        What the bases contribute; ``'auto'`` prefers probabilities.
    task:
        ``'auto'`` infers from the target and the first estimator.

    Returns
    -------
    tuple
        ``(EnsemblePlan, EnsembleFitResult, FitResult)``.

    Raises
    ------
    ValidationError
        If fewer than two estimators are given, if names repeat, if an estimator
        is ``None``, if ``cv`` is below 2, or if the split is missing.

    Notes
    -----
    **Budget for roughly ``cv`` times the cost of the slowest base**, times the
    number of bases, plus a final full fit each. Stacking four models with five
    folds is twenty-four fits.

    **Use a simple meta-learner.** It sees only a handful of columns, and a
    complex model on a handful of noisy columns overfits.

    **Measure the gain before keeping it.** Stacking frequently beats soft
    voting by an amount smaller than the fold-to-fold variation, at several
    times the cost and with a model that is much harder to explain.

    See Also
    --------
    fit_blending_ensemble : The cheaper approximation.
    buildml.model.compare.compare_estimators : Checking the gain is real.
    """
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
    """Fit a blend ensemble using one inner split of train, not ``cv`` folds.

    The blend holdout is carved from **train**. Session validation and test rows
    never reach the bases or the meta-learner: which is the whole reason to use
    this rather than hand-rolling the same idea, since the natural way to write
    blending by hand is to blend on the validation set, and that quietly makes
    every subsequent validation score meaningless.

    Parameters
    ----------
    dataset:
        The data, with roles assigned.
    split_plan:
        Partition membership. Required.
    estimators:
        At least two base estimators, uniquely named.
    final_estimator:
        The meta-learner.
    holdout_fraction:
        Share of train reserved for the meta-learner. On a small training set
        this is the parameter to watch; a warning is attached when the holdout
        falls below ten rows, at which point the meta-learner is fitting noise.
    blend_method:
        ``'predict_proba'`` or ``'predict'``. Downgraded automatically when a
        base has no probabilities; the plan records what was actually used.
    random_state:
        Seed for the inner split.
    refit_bases_on_full_train:
        Refit the bases on all of train once the meta-learner is fitted.
    passthrough:
        Also give the meta-learner the original features.
    task:
        ``'auto'`` infers from the target and the first estimator.

    Returns
    -------
    tuple
        ``(EnsemblePlan, EnsembleFitResult, FitResult)``. The plan's
        ``warnings`` carry the small-holdout caution when it applies.

    Raises
    ------
    ValidationError
        If fewer than two estimators are given, if names repeat, if an estimator
        is ``None``, or if the split is missing.

    Notes
    -----
    **Check ``warnings`` before trusting a blend on a small dataset.** A
    meta-learner fitted on a few dozen predictions will look fine in training
    and vary wildly between seeds.

    **The refit is disclosed because it changes the model.** After
    ``refit_bases_on_full_train``, the deployed bases are not the ones the
    meta-learner was calibrated against: usually an improvement, occasionally
    not, and always worth knowing.

    **Reach for stacking when the compute allows.** Blending exists for when
    ``cv`` full fits is too slow.

    See Also
    --------
    fit_stacking_ensemble : Out-of-fold meta-features instead of one holdout.
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
