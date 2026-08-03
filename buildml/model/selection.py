"""Cross-validation and hyperparameter search that do not quietly lie to you.

A single train/test split gives one number, and that number moves: sometimes a
lot: depending on which rows happened to land where. Cross-validation replaces
it with several: the train partition is divided into folds, each fold is scored
by a model trained on the others, and you get a mean *and* a spread. The spread
is the part people skip and the part that matters, because it tells you whether
a difference between two models is a real difference or resampling noise.

Hyperparameter search then uses those scores to choose a configuration. That
introduces its own problem: the moment you pick the best of fifty configurations
by cross-validated score, that score is optimistically biased. You selected on
it, so it partly measures which configuration got lucky on these folds. Nested
cross-validation is the honest answer: an inner loop chooses, an outer loop
scores what was chosen, and the outer score is unbiased because the rows it
scores on took no part in the choosing.

Two forms of leakage are guarded rather than documented-and-hoped-for:

*Partition leakage.* Everything here runs on the train partition alone. Test and
validation rows never enter fold membership, never get scored, and never
influence a ranking. Overlapping partitions raise rather than proceed.

*Preprocessing leakage.* Scaling or imputing across the whole train partition
before cross-validating means every fold's held-out rows helped compute the
statistics used to transform them. Scores come out optimistic and the run looks
completely normal. Pass a fold-local :class:`~buildml.preprocess.fold.PreprocessRecipe`
instead and it is refitted inside each fold. If Session-global preprocessing has
already run, these functions refuse with a
:class:`~buildml.core.errors.LeakageError` rather than producing a number that
cannot be trusted.

Four search strategies are available. Grid search is exhaustive and predictable,
and its cost multiplies with each parameter. Randomized search samples a fixed
budget, which usually finds a comparable configuration far sooner. Optuna's TPE
sampler learns from earlier trials and concentrates on promising regions.
Evolutionary search runs a genetic algorithm, which suits rugged spaces where
parameters interact.

See Also
--------
buildml.model.supervised : Fitting and evaluating a single configuration.
buildml.preprocess.fold : Fold-local recipes that avoid preprocessing leakage.
buildml.model.compare : Comparing estimators once configured.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    ParameterSampler,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
    check_cv,
)
from sklearn.pipeline import Pipeline as SkPipeline

from buildml.core.errors import LeakageError, ValidationError
from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition, frame_for_partition
from buildml.model.supervised import (
    FitResult,
    TaskType,
    _feature_target_frames,
    _infer_task,
    fit_estimator,
    fit_kwargs_for_sample_weight,
    validate_sample_weights,
    weight_column,
)
from buildml.preprocess.fold import (
    SAFE_RECIPE_KNOBS,
    PreprocessRecipe,
    build_fold_preprocessor,
    transform_fold_features,
)

CvStrategy = Literal["auto", "kfold", "stratified", "group", "stratified_group", "time"]
SearchMethod = Literal["grid", "randomized", "optuna", "evolutionary"]
InnerSearchMethod = Literal["auto", "grid", "randomized", "optuna", "evolutionary"]
OptunaSpace = Callable[[Any], dict[str, Any]] | dict[str, Any]
EvolutionarySpace = dict[str, Any]

_LOWER_IS_BETTER = {"mae", "mse", "rmse", "log_loss", "median_ae", "mape"}


@dataclass(slots=True)
class FoldScore:
    """What one fold scored, and how much data it had to work with.

    Folds are kept individually rather than only averaged because the spread
    across them is diagnostic. One fold far below the others usually means that
    fold's held-out rows differ from the rest: a rare class concentrated in one
    place, a time period with different behaviour, a group that does not
    resemble its neighbours. Averaging hides that; listing the folds does not.

    Attributes
    ----------
    fold:
        Zero-based fold number, in the order the splitter produced them. For
        time-series splits this order is chronological and meaningful.
    n_train:
        Rows the fold's model trained on.
    n_eval:
        Rows it was scored on. Small values make that fold's metrics noisy, and
        a per-class metric on a small fold can rest on a handful of rows.
    metrics:
        This fold's scores.

    See Also
    --------
    CVScoreResult : The aggregate these roll up into.
    """

    fold: int
    n_train: int
    n_eval: int
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the fold record to plain data.

        Used when serialising a whole cross-validation result for history or a
        report.

        Returns
        -------
        dict
            ``fold``, ``n_train``, ``n_eval``, and a copy of ``metrics``.
        """
        return {
            "fold": self.fold,
            "n_train": self.n_train,
            "n_eval": self.n_eval,
            "metrics": dict(self.metrics),
        }


@dataclass(slots=True)
class CVScoreResult:
    """Fold-by-fold scores, their spread, and an honest account of the limits.

    The result carries three kinds of thing. The numbers: per-fold metrics and
    their mean and standard deviation. The context needed to interpret them :
    which population was used, what was held out, whether preprocessing ran
    inside the folds. And the caveats, spelled out rather than left implicit.

    The standard deviation deserves as much attention as the mean. Two models
    at 0.84 ± 0.01 and 0.86 ± 0.09 are not ranked by their means; the second is
    less reliable, and on a different split it could easily come out lower. Any
    difference smaller than the fold spread is not yet a difference.

    Attributes
    ----------
    task:
        ``'classification'`` or ``'regression'``.
    scoring_metric:
        The headline metric. Defaults to F1-weighted for classification and R²
        for regression.
    cv_strategy:
        Which folding scheme ran: k-fold, stratified, group, stratified-group,
        or time. Worth checking: an inappropriate strategy silently inflates
        every fold, and this field is where that becomes visible.
    n_splits:
        How many folds. More folds means more training data per fold and a
        noisier per-fold estimate, since each is scored on less.
    folds:
        The individual fold results.
    mean_metrics:
        Metrics averaged across folds.
    std_metrics:
        Standard deviation across folds: the estimate's stability.
    population:
        Which rows the folds were drawn from. Always ``'train'``.
    held_out_partitions:
        What was excluded entirely, so a later confirmation score on it is
        genuinely independent.
    fold_preprocess:
        The fold-local recipe, if one was used. ``None`` means no preprocessing
        ran inside the folds: which is fine if the data needed none, and a leak
        if it was applied globally beforehand.
    limitations:
        What this run cannot tell you, including any leakage that was explicitly
        permitted.
    interpretation:
        What the numbers appear to say.
    recommendations:
        Suggested next steps.
    params:
        The estimator parameters and recipe knobs used, so a result read later
        is self-describing.

    Notes
    -----
    **A cross-validated mean is not a holdout score.** It estimates how this
    procedure performs on data like the training data. Confirm once on the
    held-out partition before believing it.

    **Fold standard deviation above about 20% of the mean is a warning.** The
    estimate is unstable: usually too little data, too many parameters, or a
    folding scheme that does not match the data's structure.

    See Also
    --------
    cv_score : Producing this result.
    NestedCVResult : The unbiased estimate when selection is involved.
    """

    task: Literal["classification", "regression"]
    scoring_metric: str
    cv_strategy: str
    n_splits: int
    folds: list[FoldScore] = field(default_factory=list)
    mean_metrics: dict[str, float] = field(default_factory=dict)
    std_metrics: dict[str, float] = field(default_factory=dict)
    population: str = "train"
    held_out_partitions: tuple[str, ...] = ()
    fold_preprocess: dict[str, Any] | None = None
    limitations: list[str] = field(default_factory=list)
    interpretation: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        """Lay the folds out as a table, one row per fold.

        The convenient shape for looking at the spread directly: sorting by the
        metric, plotting it, or spotting the one fold that dropped.

        Returns
        -------
        pandas.DataFrame
            Columns ``fold``, ``n_train``, ``n_eval``, and one per metric.

        Notes
        -----
        **This is where an unusual fold shows itself.** A single low row is more
        informative than the standard deviation summarising it, because you can
        go and look at which rows that fold held out.
        """
        rows = [
            {"fold": fold.fold, "n_train": fold.n_train, "n_eval": fold.n_eval, **fold.metrics}
            for fold in self.folds
        ]
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        """Convert the whole card to plain data for history and reports.

        Includes the limitations and recommendations, not just the numbers, so a
        serialised result stays as qualified as the object was.

        Returns
        -------
        dict
            Every field, with collections copied.
        """
        return {
            "task": self.task,
            "scoring_metric": self.scoring_metric,
            "cv_strategy": self.cv_strategy,
            "n_splits": self.n_splits,
            "folds": [fold.to_dict() for fold in self.folds],
            "mean_metrics": dict(self.mean_metrics),
            "std_metrics": dict(self.std_metrics),
            "population": self.population,
            "held_out_partitions": list(self.held_out_partitions),
            "fold_preprocess": self.fold_preprocess,
            "limitations": list(self.limitations),
            "interpretation": list(self.interpretation),
            "recommendations": list(self.recommendations),
            "params": dict(self.params),
        }

    def show(self) -> None:
        """Print the metrics as mean ± standard deviation, then the tips.

        Every metric is printed with its spread rather than alone, because a
        mean read without one invites treating a 0.01 difference as meaningful
        when the folds vary by 0.09.

        Notes
        -----
        **Limitations are not printed**, only recommendations. Read
        ``limitations`` directly before reporting a number: that is where an
        acknowledged leak or an ill-fitting fold strategy is recorded.
        """
        metric = self.scoring_metric
        mean = self.mean_metrics.get(metric)
        std = self.std_metrics.get(metric)
        print(
            f"CVScore · {self.task} · {self.cv_strategy} · {self.n_splits}-fold · "
            f"population={self.population}"
        )
        if mean is not None and std is not None:
            print(f"  {metric}: {mean:.6f} ± {std:.6f}")
        for key, value in self.mean_metrics.items():
            if key == metric:
                continue
            print(f"  {key}: {value:.6f} ± {self.std_metrics.get(key, float('nan')):.6f}")
        for tip in self.recommendations[:8]:
            print(f"  - {tip}")


@dataclass(slots=True)
class SearchTrial:
    """One configuration that was tried, and the cross-validation behind it.

    Each trial keeps its full cross-validation result, not just the score it was
    ranked by. That is what lets you ask whether the winner actually won: if the
    top two differ by less than the leading trial's fold standard deviation, the
    ranking is a coin toss dressed as a result.

    Attributes
    ----------
    trial:
        The trial's position in the ranking after sorting, so ``0`` is the
        winner.
    params:
        The estimator parameters tried.
    mean_score:
        Cross-validated mean of the ranking metric: the number that decided
        this trial's position.
    std_score:
        Fold spread for that metric. The scale against which any gap to the next
        trial should be judged.
    mean_metrics, std_metrics:
        All metrics, not only the ranking one. A configuration can win on F1 and
        lose badly on recall, and this is where that shows.
    recipe_knobs:
        Fold-local preprocessing settings tried, such as ``select_k``.
    cv:
        The full cross-validation result, including per-fold scores.

    Notes
    -----
    **A trial's score is optimistic in exactly the way selection makes it.** It
    was chosen for being high, so part of its height is luck on these folds.
    Nested CV is what removes that bias.

    See Also
    --------
    SearchResult : The ranked collection of trials.
    """

    trial: int
    params: dict[str, Any]
    mean_score: float
    std_score: float
    mean_metrics: dict[str, float] = field(default_factory=dict)
    std_metrics: dict[str, float] = field(default_factory=dict)
    recipe_knobs: dict[str, Any] = field(default_factory=dict)
    cv: CVScoreResult | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the trial to plain data, without its full CV result.

        The nested :class:`CVScoreResult` is omitted because serialising every
        trial's per-fold detail turns a search log into something unreadable.
        Read ``cv`` off the object when the detail is wanted.

        Returns
        -------
        dict
            ``trial``, ``params``, ``recipe_knobs``, ``mean_score``,
            ``std_score``, and the metric dictionaries.
        """
        return {
            "trial": self.trial,
            "params": dict(self.params),
            "recipe_knobs": dict(self.recipe_knobs),
            "mean_score": self.mean_score,
            "std_score": self.std_score,
            "mean_metrics": dict(self.mean_metrics),
            "std_metrics": dict(self.std_metrics),
        }


@dataclass(slots=True)
class SearchResult:
    """Every configuration tried, ranked, with the winner and its caveats.

    A search returns a ranking rather than an answer. The winner is the
    configuration that scored best on these folds, which is a weaker claim than
    "the best configuration": and the difference matters most when the top few
    are close together, which is the common case once a search is well specified.

    The result therefore keeps all the trials, not just the best, and states in
    ``interpretation`` when the gap between the top two is smaller than the
    leading trial's fold spread.

    Attributes
    ----------
    method:
        Which strategy ran: grid, randomized, optuna, or evolutionary.
    task:
        ``'classification'`` or ``'regression'``.
    ranking_metric:
        What trials were ordered by. Loss-like metrics are minimised, everything
        else maximised.
    trials:
        Every trial, best first.
    best_params:
        The winning estimator parameters.
    best_recipe_knobs:
        The winning fold-local preprocessing settings.
    best_score, best_std:
        The winner's cross-validated mean and fold spread.
    best_cv:
        The winner's full cross-validation result.
    refit_result:
        The winning configuration refitted on the whole train partition, when
        ``refit=True``. This is the model to deploy: more training data than
        any single fold saw. ``None`` when refitting was skipped.
    interpretation:
        What the ranking appears to say, including whether the top gap is
        meaningful.
    recommendations:
        Suggested next steps.
    limitations:
        What this search cannot tell you.
    study:
        Backend-specific search state: the Optuna study, or the evolutionary
        run's generation history. ``None`` for grid and randomized search.

    Notes
    -----
    **``best_score`` is not an estimate of future performance.** It was selected
    for being the maximum over many trials, so it is biased upward. Use nested
    cross-validation for an unbiased number, or confirm once on the held-out
    partition.

    **A refit model is not the model that produced ``best_score``.** It was
    trained on more data with the same settings, which usually helps and is not
    guaranteed to.

    See Also
    --------
    nested_cv_score : Estimating performance without selection bias.
    grid_search, randomized_search, optuna_search, evolutionary_search
    """

    method: SearchMethod
    task: Literal["classification", "regression"]
    ranking_metric: str
    trials: list[SearchTrial] = field(default_factory=list)
    best_params: dict[str, Any] = field(default_factory=dict)
    best_recipe_knobs: dict[str, Any] = field(default_factory=dict)
    best_score: float | None = None
    best_std: float | None = None
    best_cv: CVScoreResult | None = None
    refit_result: FitResult | None = None
    interpretation: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    study: Any | None = None

    def to_frame(self) -> pd.DataFrame:
        """Lay the trials out as a table, one row per configuration.

        Parameters are flattened into ``param_``- and ``recipe_``-prefixed
        columns, which makes the table sortable and easy to plot: score against
        one parameter shows immediately whether that parameter mattered at all.

        Returns
        -------
        pandas.DataFrame
            Columns ``trial``, ``mean_score``, ``std_score``, and one per
            parameter and recipe knob.

        Notes
        -----
        **A parameter whose column shows no relationship to score is not doing
        anything.** Drop it from the space and spend the budget on one that is.
        """
        rows = [
            {
                "trial": trial.trial,
                "mean_score": trial.mean_score,
                "std_score": trial.std_score,
                **{f"param_{k}": v for k, v in trial.params.items()},
                **{f"recipe_{k}": v for k, v in trial.recipe_knobs.items()},
            }
            for trial in self.trials
        ]
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        """Convert the search to plain data for history and reports.

        The refit model is reduced to its :meth:`FitResult.to_dict` summary and
        the backend ``study`` is dropped, since neither is serialisable.

        Returns
        -------
        dict
            Method, task, ranking metric, every trial, the winner, and the
            interpretation, recommendations, and limitations.
        """
        return {
            "method": self.method,
            "task": self.task,
            "ranking_metric": self.ranking_metric,
            "trials": [trial.to_dict() for trial in self.trials],
            "best_params": dict(self.best_params),
            "best_recipe_knobs": dict(self.best_recipe_knobs),
            "best_score": self.best_score,
            "best_std": self.best_std,
            "interpretation": list(self.interpretation),
            "recommendations": list(self.recommendations),
            "limitations": list(self.limitations),
            "refit": None if self.refit_result is None else self.refit_result.to_dict(),
        }

    def show(self) -> None:
        """Print the winner and the first eight recommendations.

        The score is shown with its fold spread, so a narrow win over the runner
        up can be recognised as narrow.

        Notes
        -----
        **Only the winner is printed.** Use :meth:`to_frame` to see how close
        the rest came: if several trials sit within a fold standard deviation
        of the top, the choice among them is arbitrary.
        """
        print(f"Search · {self.method} · ranked by {self.ranking_metric}")
        if self.best_score is not None and self.best_std is not None:
            print(f"  best: {self.best_score:.6f} ± {self.best_std:.6f}")
            print(f"  params: {self.best_params}")
        for tip in self.recommendations[:8]:
            print(f"  - {tip}")


@dataclass(slots=True)
class OuterFoldResult:
    """One outer fold: what the inner search chose, and how that choice scored.

    Each outer fold runs a complete search on its own training rows and then
    scores the winner on rows that search never saw. Comparing
    ``inner_best_score`` with ``metrics`` shows the selection bias directly :
    the inner score is usually the higher of the two, and the gap is roughly how
    much optimism selection introduced.

    Attributes
    ----------
    fold:
        Zero-based outer fold number.
    n_train:
        Rows the inner search had, and the winner was refitted on.
    n_eval:
        Rows the winner was scored on, which the inner search never saw.
    best_params:
        What the inner search chose for this fold.
    best_recipe_knobs:
        The fold-local preprocessing settings it chose.
    inner_best_score, inner_best_std:
        The winner's inner cross-validated mean and spread. Selection evidence,
        not a performance estimate.
    inner_n_trials:
        How many configurations the inner search evaluated. More trials means
        more selection bias in ``inner_best_score``.
    metrics:
        The honest score: the winner evaluated on the outer-eval rows.

    Notes
    -----
    **Different outer folds often choose different parameters.** That is not a
    bug; it means the choice is not strongly determined by the data, and any
    single set of "best" parameters is one of several equally plausible ones.

    See Also
    --------
    NestedCVResult : The aggregate over outer folds.
    """

    fold: int
    n_train: int
    n_eval: int
    best_params: dict[str, Any] = field(default_factory=dict)
    best_recipe_knobs: dict[str, Any] = field(default_factory=dict)
    inner_best_score: float | None = None
    inner_best_std: float | None = None
    inner_n_trials: int = 0
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the outer-fold record to plain data.

        Keeps both the inner selection score and the outer evaluation score, so
        the gap between them survives serialisation.

        Returns
        -------
        dict
            Fold number, row counts, chosen parameters and knobs, inner scores,
            trial count, and outer metrics.
        """
        return {
            "fold": self.fold,
            "n_train": self.n_train,
            "n_eval": self.n_eval,
            "best_params": dict(self.best_params),
            "best_recipe_knobs": dict(self.best_recipe_knobs),
            "inner_best_score": self.inner_best_score,
            "inner_best_std": self.inner_best_std,
            "inner_n_trials": self.inner_n_trials,
            "metrics": dict(self.metrics),
        }


@dataclass(slots=True)
class NestedCVResult:
    """An unbiased estimate of what tuning-then-fitting actually achieves.

    The number most people want and few compute. An ordinary search reports the
    best score it found, which is optimistic by construction: pick the maximum
    of fifty noisy estimates and you have picked partly for noise. Nested
    cross-validation separates the two jobs: an inner loop chooses, an outer
    loop scores what was chosen on rows the choosing never touched.

    What this estimates is the *procedure*, not one model: "if I tune this way on
    data like this, this is roughly how well the result performs." That is the
    question worth answering before committing to an approach.

    The cost is real. Nested CV runs a full search inside every outer fold, so
    five outer folds over a fifty-configuration search is 250 searches' worth of
    fitting.

    Attributes
    ----------
    task:
        ``'classification'`` or ``'regression'``.
    scoring_metric:
        The metric the outer loop reports and the inner loop ranks by.
    outer_cv_strategy, inner_cv_strategy:
        The folding schemes used at each level.
    n_outer_splits, n_inner_splits:
        Fold counts at each level. Total fits scale with their product.
    search_method:
        Which strategy the inner loop used.
    outer_folds:
        Per-fold results, each with its own chosen configuration.
    mean_metrics, std_metrics:
        Outer-fold means and spreads. **This is the estimate**: report the mean
        with its spread, never the inner scores.
    inner_selection_summary:
        How consistently the inner searches agreed, including a
        ``param_stability`` rating and the per-fold choices.
    population:
        Always ``'train'``.
    held_out_partitions:
        Never touched by either loop.
    fold_preprocess:
        The fold-local recipe, refitted in both loops.
    limitations, interpretation, recommendations:
        What this cannot tell you, what it appears to say, and what to do next.
    warm_start_studies:
        Whether Optuna state was shared across outer folds, which couples them
        slightly. Recorded because it makes the estimate marginally optimistic.

    Notes
    -----
    **Nested CV does not hand you a model.** It estimates a procedure. Afterward,
    run the search once on the full train partition and use that winner :
    accepting that its inner score is optimistic while the nested estimate
    describes what to expect.

    **Unstable parameter choices across folds are informative.** They mean the
    data does not determine the choice, so treat any single winner as one of
    several defensible options rather than the answer.

    **The inner means are always higher than the outer mean.** That gap is the
    selection bias, made visible. It is why the outer number is the one to
    report.

    See Also
    --------
    nested_cv_score : Producing this result.
    SearchResult : The biased-but-cheaper alternative.
    """

    task: Literal["classification", "regression"]
    scoring_metric: str
    outer_cv_strategy: str
    inner_cv_strategy: str
    n_outer_splits: int
    n_inner_splits: int
    search_method: SearchMethod
    outer_folds: list[OuterFoldResult] = field(default_factory=list)
    mean_metrics: dict[str, float] = field(default_factory=dict)
    std_metrics: dict[str, float] = field(default_factory=dict)
    inner_selection_summary: dict[str, Any] = field(default_factory=dict)
    population: str = "train"
    held_out_partitions: tuple[str, ...] = ()
    fold_preprocess: dict[str, Any] | None = None
    limitations: list[str] = field(default_factory=list)
    interpretation: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    warm_start_studies: bool = False

    def to_frame(self) -> pd.DataFrame:
        """Lay the outer folds out as a table, one row per fold.

        Puts the inner selection score beside the outer evaluation score with
        the chosen parameters, which is the clearest way to see both the
        selection bias and how much the choices varied.

        Returns
        -------
        pandas.DataFrame
            Fold number, row counts, inner scores, outer metrics, and one column
            per chosen parameter and recipe knob.

        Notes
        -----
        **Read the parameter columns down, not across.** Identical values in
        every row mean the choice is well determined; scattered values mean it
        is not, whatever the search reported as best.
        """
        rows = [
            {
                "fold": fold.fold,
                "n_train": fold.n_train,
                "n_eval": fold.n_eval,
                "inner_best_score": fold.inner_best_score,
                "inner_best_std": fold.inner_best_std,
                **fold.metrics,
                **{f"param_{k}": v for k, v in fold.best_params.items()},
                **{f"recipe_{k}": v for k, v in fold.best_recipe_knobs.items()},
            }
            for fold in self.outer_folds
        ]
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        """Convert the nested result to plain data for history and reports.

        Keeps the limitations and the warm-start flag alongside the numbers, so
        a serialised estimate remains as qualified as the object was.

        Returns
        -------
        dict
            Every field, with collections copied.
        """
        return {
            "task": self.task,
            "scoring_metric": self.scoring_metric,
            "outer_cv_strategy": self.outer_cv_strategy,
            "inner_cv_strategy": self.inner_cv_strategy,
            "n_outer_splits": self.n_outer_splits,
            "n_inner_splits": self.n_inner_splits,
            "search_method": self.search_method,
            "outer_folds": [fold.to_dict() for fold in self.outer_folds],
            "mean_metrics": dict(self.mean_metrics),
            "std_metrics": dict(self.std_metrics),
            "inner_selection_summary": dict(self.inner_selection_summary),
            "population": self.population,
            "held_out_partitions": list(self.held_out_partitions),
            "fold_preprocess": self.fold_preprocess,
            "limitations": list(self.limitations),
            "interpretation": list(self.interpretation),
            "recommendations": list(self.recommendations),
            "warm_start_studies": self.warm_start_studies,
        }

    def show(self) -> None:
        """Print the outer-loop estimate and the first eight recommendations.

        Only the outer mean and spread are shown. The inner scores are
        deliberately absent: they are selection evidence, and printing them
        beside the outer number invites reporting the wrong one.

        Notes
        -----
        **Check ``inner_selection_summary`` as well.** A low
        ``param_stability`` rating means the outer estimate is sound while any
        single set of best parameters is not settled.
        """
        metric = self.scoring_metric
        mean = self.mean_metrics.get(metric)
        std = self.std_metrics.get(metric)
        print(
            f"NestedCV · {self.task} · outer={self.outer_cv_strategy}/"
            f"{self.n_outer_splits} · inner={self.search_method}/"
            f"{self.n_inner_splits} · population={self.population}"
        )
        if mean is not None and std is not None:
            print(f"  outer {metric}: {mean:.6f} ± {std:.6f}")
        for tip in self.recommendations[:8]:
            print(f"  - {tip}")


def _refuse_session_global_cv_leakage(
    *,
    session_preprocess_applied: bool,
    preprocess: PreprocessRecipe | None,
    allow_session_global_preprocess: bool,
) -> None:
    """Refuse to cross-validate on data that global preprocessing already spoiled.

    Fitting a scaler or an imputer on the whole train partition and then
    cross-validating is one of the most common ways to get a wrong number. Every
    fold's held-out rows contributed to the statistics used to transform them,
    so each fold has partial knowledge of what it is about to be scored on. The
    result is optimistic and completely unremarkable-looking.

    Passing a fold-local recipe afterwards does not fix it. The recipe operates
    on the frame as it currently is: already transformed: so it cannot rebuild
    from untouched values. The only real fix is to start from data that global
    preprocessing has not yet run on.

    Parameters
    ----------
    session_preprocess_applied:
        Whether Session-global fit-capable plans already ran.
    preprocess:
        The fold-local recipe, if one was passed. Used only to make the error
        message specific about why it is not sufficient.
    allow_session_global_preprocess:
        Explicit opt-in to proceed anyway, accepting a biased score.

    Raises
    ------
    LeakageError
        When global preprocessing has run and the override was not set.

    Notes
    -----
    **This refuses rather than warns because the warning would be ignored.** The
    run otherwise succeeds and produces a plausible number, so nothing later
    forces the issue.

    **The override exists for comparing against a known-biased baseline**, and
    for that only. Every result produced under it carries the fact in its
    limitations.

    See Also
    --------
    buildml.preprocess.fold.PreprocessRecipe : Preprocessing that folds honestly.
    """
    if not session_preprocess_applied:
        return
    if allow_session_global_preprocess:
        return
    recipe_note = ""
    if preprocess is not None and not preprocess.is_empty():
        recipe_note = (
            " A fold-local PreprocessRecipe was provided, but Session data is already "
            "transformed with train-global statistics: the recipe cannot rebuild from "
            "raw/unpoisoned rows. Re-ingest or checkpoint_load an unpoisoned frame, then "
            "use fold-local recipes without Session-global impute/encode/scale/select/…"
            " first."
        )
    else:
        recipe_note = (
            " Pass preprocess=PreprocessRecipe(...) on unpoisoned data for fold-local "
            "refits, or set allow_session_global_preprocess=True to override explicitly "
            "(scores remain leakage-biased)."
        )
    raise LeakageError(
        "Refusing CV/search because Session-global preprocess plans were already "
        "fitted on the full train partition (fold-eval rows influenced those frozen "
        f"statistics).{recipe_note}"
    )


def cv_score(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    estimator: Any,
    *,
    task: TaskType = "auto",
    cv: int | Any = 5,
    cv_strategy: CvStrategy = "auto",
    scoring_metric: str | None = None,
    groups: pd.Series | None = None,
    preprocess: PreprocessRecipe | None = None,
    session_preprocess_applied: bool = False,
    allow_session_global_preprocess: bool = False,
    params: dict[str, Any] | None = None,
    recipe_knobs: dict[str, Any] | None = None,
) -> CVScoreResult:
    """Score an estimator across several folds instead of trusting one split.

    Splits the train partition into folds, trains on all but one and scores on
    the one left out, and repeats until every row has been scored exactly once
    by a model that never saw it. What comes back is a mean and a spread.

    The spread is the reason to do this. A single train/test split gives one
    number whose stability is unknown; cross-validation shows how much that
    number moves when the rows are dealt differently. A model at 0.85 ± 0.02 and
    one at 0.85 ± 0.12 are not equally good, and only the second answer reveals
    that.

    Parameters
    ----------
    dataset:
        The data.
    split_plan:
        The split. Folds are drawn from the train partition only.
    estimator:
        Any scikit-learn-compatible estimator. Cloned for every fold, so folds
        are genuinely independent.
    task:
        ``'classification'``, ``'regression'``, or ``'auto'``.
    cv:
        Fold count, or a scikit-learn splitter for full control.
    cv_strategy:
        How to fold: ``'auto'``, ``'kfold'``, ``'stratified'``, ``'group'``,
        ``'stratified_group'``, or ``'time'``. ``'auto'`` picks from the
        dataset's roles, which is usually right: see the notes for why the
        choice is not cosmetic.
    scoring_metric:
        The headline metric. Defaults to F1-weighted for classification and R²
        for regression.
    groups:
        Explicit group labels, overriding the group-role column.
    preprocess:
        A fold-local recipe, refitted inside each fold. This is how to
        preprocess without leaking.
    session_preprocess_applied:
        Whether Session-global preprocessing already ran, which triggers the
        leakage refusal.
    allow_session_global_preprocess:
        Override that refusal, accepting a biased score.
    params:
        Estimator parameters to set before fitting. Keys prefixed ``recipe__``
        are routed to the recipe instead.
    recipe_knobs:
        Fold-local recipe overrides such as ``select_k`` or ``n_bins``. Only the
        knobs in ``SAFE_RECIPE_KNOBS`` are permitted, because the rest cannot be
        varied per fold without leaking.

    Returns
    -------
    CVScoreResult
        Per-fold scores, their mean and spread, and the limitations of the run.

    Raises
    ------
    ValidationError
        If ``cv`` is under 2, if a recipe knob is not recognised, if knobs are
        passed without a recipe, or if no folds were produced.
    LeakageError
        If partitions overlap, if a fold's train and eval rows overlap, or if
        global preprocessing already ran without the override.

    Notes
    -----
    **The folding strategy is a correctness decision, not a preference.** Plain
    k-fold on grouped data splits a patient's or a customer's rows across the
    boundary, and the model recognises the individual rather than the pattern :
    scores come out high and do not survive contact with a new group. On time
    series, k-fold trains on the future to predict the past. Group and time
    strategies exist to prevent exactly these.

    **Stratification matters most where it is easiest to skip.** With a rare
    class, an unstratified fold can contain almost no positives, making that
    fold's metrics close to meaningless.

    **Cross-validated scores are not holdout scores.** They estimate performance
    on data resembling the training data. Confirm once on the held-out partition
    at the end.

    **More folds is not simply better.** Ten folds gives each model more training
    data and scores it on less, so per-fold estimates get noisier even as the
    mean stabilises. Five is a reasonable default.

    Examples
    --------
    Cross-validate with fold-local preprocessing::

        from buildml.preprocess.fold import PreprocessRecipe

        result = cv_score(
            dataset, split_plan, estimator,
            cv=5,
            preprocess=PreprocessRecipe(impute="median", scale="standard"),
        )
        result.show()
        print(result.to_frame())

    See Also
    --------
    nested_cv_score : When hyperparameters are being chosen too.
    grid_search : Searching over configurations.
    buildml.preprocess.fold.PreprocessRecipe : Leak-free fold preprocessing.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None
    _refuse_session_global_cv_leakage(
        session_preprocess_applied=session_preprocess_applied,
        preprocess=preprocess,
        allow_session_global_preprocess=allow_session_global_preprocess,
    )

    x_train, y_train, _feature_cols, _target, sample_weight = _feature_target_frames(
        dataset, split_plan, "train"
    )
    resolved_task = _infer_task(y_train, task, estimator)
    metric = scoring_metric or ("r2" if resolved_task == "regression" else "f1_weighted")

    held_out: list[str] = ["test"]
    if split_plan.validation_indices:
        held_out.append("validation")

    train_set = set(split_plan.train_indices)
    if train_set & set(split_plan.test_indices):
        raise LeakageError("Train and test partitions overlap; refusing cross-validation")
    if train_set & set(split_plan.validation_indices):
        raise LeakageError("Train and validation partitions overlap; refusing cross-validation")

    est_params, knob_params = _split_trial_params(params or {})
    if recipe_knobs:
        unknown = sorted(set(recipe_knobs) - SAFE_RECIPE_KNOBS)
        if unknown:
            raise ValidationError(
                f"Unsupported recipe knobs: {unknown}. Allowed: {sorted(SAFE_RECIPE_KNOBS)}"
            )
        knob_params = {**knob_params, **dict(recipe_knobs)}
    active_recipe = _recipe_with_knobs(preprocess, knob_params)

    model = clone(estimator)
    if est_params:
        model.set_params(**est_params)
    # Validate weight support once up front (clone keeps the same fit signature).
    fit_kwargs_for_sample_weight(model, sample_weight)

    group_values, strategy_name, splitter, row_order = _resolve_splitter(
        dataset=dataset,
        split_plan=split_plan,
        y_train=y_train,
        cv=cv,
        cv_strategy=cv_strategy,
        groups=groups,
        task=resolved_task,
    )

    x_reset = x_train.reset_index(drop=True)
    y_reset = y_train.reset_index(drop=True)
    w_reset = None if sample_weight is None else sample_weight.reset_index(drop=True)
    if row_order is not None:
        x_reset = x_reset.iloc[row_order].reset_index(drop=True)
        y_reset = y_reset.iloc[row_order].reset_index(drop=True)
        if w_reset is not None:
            w_reset = w_reset.iloc[row_order].reset_index(drop=True)
        if group_values is not None:
            group_values = pd.Series(group_values).iloc[row_order]

    group_reset = None if group_values is None else pd.Series(group_values).reset_index(drop=True)

    folds: list[FoldScore] = []
    metric_rows: list[dict[str, float]] = []
    split_iter = (
        splitter.split(x_reset, y_reset, group_reset)
        if group_reset is not None
        else splitter.split(x_reset, y_reset)
    )

    for fold_id, (train_pos, eval_pos) in enumerate(split_iter):
        if set(train_pos) & set(eval_pos):
            raise LeakageError("CV fold train/eval indices overlap")
        x_fold_train = x_reset.iloc[list(train_pos)]
        y_fold_train = y_reset.iloc[list(train_pos)]
        x_fold_eval = x_reset.iloc[list(eval_pos)]
        y_fold_eval = y_reset.iloc[list(eval_pos)]
        w_fold_train = None if w_reset is None else w_reset.iloc[list(train_pos)]
        w_fold_eval = None if w_reset is None else w_reset.iloc[list(eval_pos)]

        if active_recipe is not None and not active_recipe.is_empty():
            prep = build_fold_preprocessor(x_fold_train, active_recipe, y_fold_train)
            x_fit = transform_fold_features(prep, x_fold_train)
            x_score = transform_fold_features(prep, x_fold_eval)
        else:
            x_fit = x_fold_train
            x_score = x_fold_eval

        fold_model = clone(model)
        fold_model.fit(
            x_fit,
            y_fold_train,
            **fit_kwargs_for_sample_weight(fold_model, w_fold_train),
        )
        y_pred = fold_model.predict(x_score)
        fold_metrics = _score_predictions(
            resolved_task, y_fold_eval, y_pred, sample_weight=w_fold_eval
        )
        folds.append(
            FoldScore(
                fold=fold_id,
                n_train=int(len(train_pos)),
                n_eval=int(len(eval_pos)),
                metrics=fold_metrics,
            )
        )
        metric_rows.append(fold_metrics)

    if not metric_rows:
        raise ValidationError("Cross-validation produced no folds")

    mean_metrics, std_metrics = _aggregate_metrics(metric_rows)
    recorded_params = {**est_params, **{f"recipe__{k}": v for k, v in knob_params.items()}}
    session_global_override = bool(session_preprocess_applied and allow_session_global_preprocess)
    return CVScoreResult(
        task=resolved_task,
        scoring_metric=metric,
        cv_strategy=strategy_name,
        n_splits=len(folds),
        folds=folds,
        mean_metrics=mean_metrics,
        std_metrics=std_metrics,
        population="train",
        held_out_partitions=tuple(held_out),
        fold_preprocess=None if active_recipe is None else active_recipe.to_dict(),
        limitations=_cv_limitations(
            session_preprocess_applied=session_global_override,
            preprocess=active_recipe,
            strategy_name=strategy_name,
            n_folds=len(folds),
        ),
        interpretation=_cv_interpretation(
            metric=metric,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            n_folds=len(folds),
            task=resolved_task,
        ),
        recommendations=_cv_recommendations(
            metric=metric,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            held_out=held_out,
            session_preprocess_applied=session_global_override,
            preprocess=active_recipe,
        ),
        params=recorded_params,
    )


def grid_search(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    estimator: Any,
    param_grid: dict[str, list[Any]] | None = None,
    *,
    recipe_grid: dict[str, list[Any]] | None = None,
    task: TaskType = "auto",
    cv: int | Any = 5,
    cv_strategy: CvStrategy = "auto",
    ranking_metric: str | None = None,
    groups: pd.Series | None = None,
    preprocess: PreprocessRecipe | None = None,
    session_preprocess_applied: bool = False,
    allow_session_global_preprocess: bool = False,
    refit: bool = True,
) -> SearchResult:
    """Try every combination in the grid, cross-validating each one.

    Exhaustive and predictable: the space is enumerated, every point is
    cross-validated, and the results are ranked. Nothing is missed within the
    grid, and the run is fully reproducible.

    The cost is multiplicative. Three parameters with five values each is 125
    combinations, and at 5-fold that is 625 model fits. A fourth parameter makes
    it 3,125. Beyond two or three parameters, :func:`randomized_search` reaches
    a comparable configuration far sooner, because most parameters have little
    effect and grid search spends the same effort on them regardless.

    Parameters
    ----------
    dataset:
        The data.
    split_plan:
        The split. Folds come from train only.
    estimator:
        The estimator to configure.
    param_grid:
        Parameter names to lists of values, every combination of which is tried.
        Keys prefixed ``recipe__`` are routed to the recipe.
    recipe_grid:
        Fold-local recipe knobs to try, such as ``{'select_k': [5, 10, 20]}``.
        Requires ``preprocess``.
    task:
        ``'classification'``, ``'regression'``, or ``'auto'``.
    cv:
        Fold count or splitter for each trial's cross-validation.
    cv_strategy:
        How to fold.
    ranking_metric:
        What to rank by. Defaults to F1-weighted or R².
    groups:
        Explicit group labels.
    preprocess:
        Fold-local recipe, refitted inside every fold of every trial.
    session_preprocess_applied:
        Whether Session-global preprocessing already ran.
    allow_session_global_preprocess:
        Override the resulting leakage refusal.
    refit:
        Refit the winner on the whole train partition. This is the model to
        deploy; set ``False`` inside a nested loop, where the outer loop refits.

    Returns
    -------
    SearchResult
        Every trial ranked, the winner, and the optional refitted model.

    Raises
    ------
    ValidationError
        If both grids are empty, if a recipe knob is unrecognised, or if recipe
        knobs are given without a recipe.
    LeakageError
        If partitions overlap or global preprocessing already ran without the
        override.

    Notes
    -----
    **The best score is optimistic.** It is the maximum over many noisy
    estimates, so it partly measures luck on these folds. Use
    :func:`nested_cv_score` for an unbiased estimate.

    **Check the gap against the fold spread before believing the ranking.** If
    the top two differ by less than the leading trial's standard deviation, the
    order is not a finding.

    **A grid is only as good as its edges.** If the winner sits at the boundary
    of a range, the optimum probably lies outside it: widen and rerun.

    Examples
    --------
    Search two parameters and a fold-local knob::

        result = grid_search(
            dataset, split_plan, estimator,
            {"max_depth": [3, 5, 10], "min_samples_leaf": [1, 5]},
            recipe_grid={"select_k": [10, 20]},
            preprocess=PreprocessRecipe(scale="standard", select="model"),
        )
        print(result.best_params, result.best_score)

    See Also
    --------
    randomized_search : Cheaper on larger spaces.
    optuna_search : Adaptive sampling that learns from earlier trials.
    nested_cv_score : Estimating what the tuning actually buys.
    """
    trials = _expand_grid_trials(param_grid=param_grid, recipe_grid=recipe_grid)
    _require_recipe_for_knobs(preprocess, any(t[1] for t in trials))
    return _run_search(
        dataset,
        split_plan,
        estimator,
        trials,
        method="grid",
        task=task,
        cv=cv,
        cv_strategy=cv_strategy,
        ranking_metric=ranking_metric,
        groups=groups,
        preprocess=preprocess,
        session_preprocess_applied=session_preprocess_applied,
        allow_session_global_preprocess=allow_session_global_preprocess,
        refit=refit,
    )


def randomized_search(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    estimator: Any,
    param_distributions: dict[str, Any] | None = None,
    *,
    recipe_distributions: dict[str, Any] | None = None,
    n_iter: int = 10,
    random_state: int | None = 42,
    task: TaskType = "auto",
    cv: int | Any = 5,
    cv_strategy: CvStrategy = "auto",
    ranking_metric: str | None = None,
    groups: pd.Series | None = None,
    preprocess: PreprocessRecipe | None = None,
    session_preprocess_applied: bool = False,
    allow_session_global_preprocess: bool = False,
    refit: bool = True,
) -> SearchResult:
    """Sample a fixed number of configurations rather than trying them all.

    Usually the better default. The reason is that most hyperparameters barely
    matter and one or two matter a great deal, but you rarely know in advance
    which. Grid search spends the same effort on every axis; random sampling
    tries a different value of the important parameter on *every* draw, so with
    a fixed budget it explores the axis that matters far more thoroughly.

    The budget is set directly by ``n_iter``, so cost does not explode when the
    space grows. Adding a parameter to a grid multiplies the work; adding one
    here does not.

    Parameters
    ----------
    dataset:
        The data.
    split_plan:
        The split. Folds come from train only.
    estimator:
        The estimator to configure.
    param_distributions:
        Parameter names to either a list to choose from or a scipy distribution
        to draw from. Continuous parameters benefit most from a distribution,
        since sampling can land between grid points.
    recipe_distributions:
        Fold-local recipe knobs, in the same forms. Requires ``preprocess``.
    n_iter:
        How many configurations to try. The whole budget.
    random_state:
        Seed, so the sampled configurations reproduce.
    task:
        ``'classification'``, ``'regression'``, or ``'auto'``.
    cv:
        Fold count or splitter for each trial.
    cv_strategy:
        How to fold.
    ranking_metric:
        What to rank by.
    groups:
        Explicit group labels.
    preprocess:
        Fold-local recipe, refitted inside every fold.
    session_preprocess_applied:
        Whether Session-global preprocessing already ran.
    allow_session_global_preprocess:
        Override the resulting leakage refusal.
    refit:
        Refit the winner on the whole train partition.

    Returns
    -------
    SearchResult
        Every sampled trial ranked, the winner, and the optional refitted model.

    Raises
    ------
    ValidationError
        If ``n_iter`` is below 1, if both distributions are empty, or if recipe
        knobs are given without a recipe.
    LeakageError
        If partitions overlap or global preprocessing already ran without the
        override.

    Notes
    -----
    **Use distributions for continuous parameters, lists for discrete ones.** A
    log-uniform distribution over a learning rate covers orders of magnitude
    evenly; a list of five values covers five points.

    **The same configuration can be drawn twice**, particularly from a small
    discrete space. That costs budget without adding information: use
    :func:`grid_search` when the space is small enough to enumerate.

    **A too-small ``n_iter`` finds a decent configuration, not a good one.**
    Twenty to fifty is a reasonable starting range for a handful of parameters.

    Examples
    --------
    Sample from mixed discrete and continuous ranges::

        from scipy.stats import loguniform

        result = randomized_search(
            dataset, split_plan, estimator,
            {"learning_rate": loguniform(1e-3, 1e-1), "max_depth": [3, 5, 7, 10]},
            n_iter=40,
        )
        print(result.to_frame().head())

    See Also
    --------
    grid_search : Exhaustive, for small spaces.
    optuna_search : Adaptive, concentrating on promising regions.
    """
    if n_iter < 1:
        raise ValidationError("n_iter must be >= 1")
    trials = _expand_randomized_trials(
        param_distributions=param_distributions,
        recipe_distributions=recipe_distributions,
        n_iter=n_iter,
        random_state=random_state,
    )
    _require_recipe_for_knobs(preprocess, any(t[1] for t in trials))
    return _run_search(
        dataset,
        split_plan,
        estimator,
        trials,
        method="randomized",
        task=task,
        cv=cv,
        cv_strategy=cv_strategy,
        ranking_metric=ranking_metric,
        groups=groups,
        preprocess=preprocess,
        session_preprocess_applied=session_preprocess_applied,
        allow_session_global_preprocess=allow_session_global_preprocess,
        refit=refit,
    )


def optuna_search(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    estimator: Any,
    *,
    param_space: OptunaSpace | None = None,
    recipe_space: OptunaSpace | None = None,
    n_trials: int = 20,
    random_state: int | None = 42,
    task: TaskType = "auto",
    cv: int | Any = 5,
    cv_strategy: CvStrategy = "auto",
    ranking_metric: str | None = None,
    groups: pd.Series | None = None,
    preprocess: PreprocessRecipe | None = None,
    session_preprocess_applied: bool = False,
    allow_session_global_preprocess: bool = False,
    refit: bool = True,
    study: Any | None = None,
) -> SearchResult:
    """Let each trial learn from the ones before it, instead of sampling blind.

    Random search treats every draw as independent, which means it keeps
    sampling regions it has already found to be poor. Optuna's TPE sampler
    builds a model of which regions produced good scores and concentrates
    subsequent trials there, so a fixed budget goes further: usually
    noticeably so once the space has more than a couple of dimensions.

    The trade-off is that concentrating can also mean converging early on a
    local optimum, and the run is no longer embarrassingly parallel in the way
    random search is.

    Parameters
    ----------
    dataset:
        The data.
    split_plan:
        The split. Folds come from train only.
    estimator:
        The estimator to configure.
    param_space:
        Either a callable taking an Optuna trial and returning parameters, or a
        declare-style dict: ``{'type': 'float', 'low': …, 'high': …, 'log':
        bool}``, ``{'type': 'int', 'low': …, 'high': …}``, ``{'type':
        'categorical', 'choices': [...]}``, or a plain list of choices. Keys may
        use a ``recipe__`` prefix.
    recipe_space:
        The same forms, for fold-local recipe knobs. Requires ``preprocess``.
    n_trials:
        How many configurations to evaluate.
    random_state:
        Seed for the TPE sampler.
    task:
        ``'classification'``, ``'regression'``, or ``'auto'``.
    cv:
        Fold count or splitter for each trial.
    cv_strategy:
        How to fold.
    ranking_metric:
        What to optimise. Loss-like metrics are minimised, everything else
        maximised.
    groups:
        Explicit group labels.
    preprocess:
        Fold-local recipe, refitted inside every fold.
    session_preprocess_applied:
        Whether Session-global preprocessing already ran.
    allow_session_global_preprocess:
        Override the resulting leakage refusal.
    refit:
        Refit the winner on the whole train partition.
    study:
        An existing Optuna study to continue, so a second call builds on the
        first rather than starting over. Its direction must match the metric.

    Returns
    -------
    SearchResult
        Trials ranked, the winner, the optional refitted model, and the Optuna
        study in ``study``.

    Raises
    ------
    MissingExtraError
        If Optuna is not installed. Install with ``pip install
        'buildml[optuna]'``.
    ValidationError
        If ``n_trials`` is below 1, if neither space is given, if a warm-start
        study's direction disagrees with the metric, or if recipe knobs are
        given without a recipe.
    LeakageError
        If partitions overlap or global preprocessing already ran without the
        override.

    Notes
    -----
    **Use ``log=True`` for parameters that span orders of magnitude.** A
    learning rate from 1e-5 to 1e-1 sampled linearly puts almost every draw
    above 0.01; sampled logarithmically it covers each decade evenly.

    **Early trials are effectively random.** TPE needs some observations before
    its model is worth anything, so very small budgets get little benefit over
    random search.

    **Adaptive search intensifies selection bias.** Concentrating on
    high-scoring regions means the best score is more optimistic than random
    search's would be, not less. The case for nested CV is stronger here, not
    weaker.

    Examples
    --------
    Search a declare-style space::

        result = optuna_search(
            dataset, split_plan, estimator,
            param_space={
                "learning_rate": {"type": "float", "low": 1e-3, "high": 0.3, "log": True},
                "max_depth": {"type": "int", "low": 2, "high": 12},
            },
            n_trials=50,
        )

    See Also
    --------
    randomized_search : No extra dependency, easy to parallelise.
    evolutionary_search : Better on rugged spaces with interacting parameters.
    """
    try:
        import optuna
    except ImportError as exc:
        from buildml.core.errors import MissingExtraError

        raise MissingExtraError("optuna", "Optuna hyperparameter search") from exc

    if n_trials < 1:
        raise ValidationError("n_trials must be >= 1")
    if param_space is None and recipe_space is None:
        raise ValidationError("Provide param_space and/or recipe_space")
    _require_recipe_for_knobs(preprocess, recipe_space is not None)

    assert_fit_partition(split_plan, "train")
    assert split_plan is not None
    _x_train, y_train, _feature_cols, _target, _sample_weight = _feature_target_frames(
        dataset, split_plan, "train"
    )
    resolved_task = _infer_task(y_train, task, estimator)
    metric_name = ranking_metric or ("r2" if resolved_task == "regression" else "f1_weighted")
    higher_is_better = metric_name not in _LOWER_IS_BETTER
    direction = "maximize" if higher_is_better else "minimize"

    # Keep Optuna console quiet; BuildML returns a SearchResult card instead.
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if study is None:
        sampler = optuna.samplers.TPESampler(seed=random_state)
        study = optuna.create_study(direction=direction, sampler=sampler)
    else:
        study_direction = getattr(study, "direction", None)
        actual = getattr(study_direction, "name", str(study_direction))
        if direction.upper() not in str(actual).upper():
            raise ValidationError(
                f"warm-start Optuna study direction {actual!r} does not match "
                f"required {direction.upper()!r} for metric {metric_name!r}"
            )
    trial_rows: list[SearchTrial] = []

    def _objective(trial: Any) -> float:
        est_params: dict[str, Any] = {}
        recipe_knobs: dict[str, Any] = {}
        if param_space is not None:
            raw = (
                param_space(trial)
                if callable(param_space)
                else _suggest_from_space(trial, param_space, prefix="param")
            )
            est_params, embedded = _split_trial_params(dict(raw))
            recipe_knobs.update(embedded)
        if recipe_space is not None:
            from_recipe = (
                dict(recipe_space(trial))
                if callable(recipe_space)
                else _suggest_from_space(trial, recipe_space, prefix="recipe")
            )
            unknown = sorted(set(from_recipe) - SAFE_RECIPE_KNOBS)
            if unknown:
                raise ValidationError(
                    f"Unsupported recipe knobs: {unknown}. Allowed: {sorted(SAFE_RECIPE_KNOBS)}"
                )
            recipe_knobs.update(from_recipe)

        cv_result = cv_score(
            dataset,
            split_plan,
            estimator,
            task=resolved_task,
            cv=cv,
            cv_strategy=cv_strategy,
            scoring_metric=metric_name,
            groups=groups,
            preprocess=preprocess,
            session_preprocess_applied=session_preprocess_applied,
            allow_session_global_preprocess=allow_session_global_preprocess,
            params=est_params,
            recipe_knobs=recipe_knobs,
        )
        score = float(cv_result.mean_metrics[metric_name])
        trial_rows.append(
            SearchTrial(
                trial=int(trial.number),
                params=dict(est_params),
                recipe_knobs=dict(recipe_knobs),
                mean_score=score,
                std_score=float(cv_result.std_metrics.get(metric_name, float("nan"))),
                mean_metrics=dict(cv_result.mean_metrics),
                std_metrics=dict(cv_result.std_metrics),
                cv=cv_result,
            )
        )
        return score

    study.optimize(_objective, n_trials=n_trials)

    if not trial_rows:
        raise ValidationError("Optuna search produced no trials")

    trials = sorted(trial_rows, key=lambda item: item.mean_score, reverse=higher_is_better)
    # Re-number display order after ranking while keeping Optuna trial ids in params lineage.
    ranked = [
        SearchTrial(
            trial=i,
            params=dict(t.params),
            recipe_knobs=dict(t.recipe_knobs),
            mean_score=t.mean_score,
            std_score=t.std_score,
            mean_metrics=dict(t.mean_metrics),
            std_metrics=dict(t.std_metrics),
            cv=t.cv,
        )
        for i, t in enumerate(trials)
    ]
    return _finalize_search_result(
        method="optuna",
        resolved_task=resolved_task,
        metric_name=metric_name,
        trials=ranked,
        estimator=estimator,
        dataset=dataset,
        split_plan=split_plan,
        preprocess=preprocess,
        session_preprocess_applied=session_preprocess_applied,
        allow_session_global_preprocess=allow_session_global_preprocess,
        refit=refit,
        study=study,
    )


def evolutionary_search(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    estimator: Any,
    *,
    param_space: EvolutionarySpace | None = None,
    recipe_space: EvolutionarySpace | None = None,
    population_size: int = 12,
    n_generations: int = 5,
    elite_size: int = 2,
    crossover_rate: float = 0.7,
    mutation_rate: float = 0.2,
    tournament_size: int = 3,
    max_evaluations: int | None = None,
    random_state: int | None = 42,
    task: TaskType = "auto",
    cv: int | Any = 5,
    cv_strategy: CvStrategy = "auto",
    ranking_metric: str | None = None,
    groups: pd.Series | None = None,
    preprocess: PreprocessRecipe | None = None,
    session_preprocess_applied: bool = False,
    allow_session_global_preprocess: bool = False,
    refit: bool = True,
) -> SearchResult:
    """Breed good configurations from other good configurations.

    A genetic algorithm. A population of configurations is scored, the better
    ones are more likely to be selected as parents, and children are formed by
    mixing two parents' values and randomly perturbing some of them. Repeat for
    a few generations and the population drifts toward regions that work.

    The reason to prefer this over TPE is parameter *interaction*. TPE models
    each parameter's contribution largely separately, which struggles when a
    high learning rate is only good alongside a shallow tree and terrible
    otherwise. Crossover recombines whole configurations, so a combination that
    only works as a pair can survive and spread as a pair.

    It needs more evaluations than TPE to get going, since early generations are
    close to random, and it has more knobs of its own to set.

    Parameters
    ----------
    dataset:
        The data.
    split_plan:
        The split. Folds come from train only.
    estimator:
        The estimator to configure.
    param_space:
        A declare-style dict: ``{'type': 'float', 'low': …, 'high': …, 'log':
        bool}``, ``{'type': 'int', 'low': …, 'high': …}``, ``{'type':
        'categorical', 'choices': [...]}``, or a plain list. Callables are not
        accepted: the algorithm needs an explicit encoding to mutate and
        recombine. Keys may use a ``recipe__`` prefix.
    recipe_space:
        The same forms, for fold-local recipe knobs. Requires ``preprocess``.
    population_size:
        Configurations alive at once. Larger keeps more diversity and costs more
        per generation.
    n_generations:
        Breeding rounds. More generations means more refinement and more risk of
        the population collapsing onto one region.
    elite_size:
        How many of the best carry over unchanged. Guarantees the best score
        never gets worse; too many and diversity dies out.
    crossover_rate:
        Probability a child is formed by mixing two parents rather than copying
        one.
    mutation_rate:
        Per-parameter probability of a random change. The only source of genuine
        novelty once the population has converged: too low and the search
        stalls, too high and it is random search.
    tournament_size:
        How many candidates compete to be a parent. Larger favours the strong
        more aggressively and converges faster on less exploration.
    max_evaluations:
        Hard ceiling on distinct configurations scored. Defaults to
        ``population_size * n_generations``. Repeats are cached, not rescored.
    random_state:
        Seed for the whole evolutionary process.
    task:
        ``'classification'``, ``'regression'``, or ``'auto'``.
    cv:
        Fold count or splitter for each evaluation.
    cv_strategy:
        How to fold.
    ranking_metric:
        What to optimise. Loss-like metrics are minimised.
    groups:
        Explicit group labels.
    preprocess:
        Fold-local recipe, refitted inside every fold.
    session_preprocess_applied:
        Whether Session-global preprocessing already ran.
    allow_session_global_preprocess:
        Override the resulting leakage refusal.
    refit:
        Refit the winner on the whole train partition.

    Returns
    -------
    SearchResult
        Every distinct configuration evaluated, ranked, with the per-generation
        history in ``study``.

    Raises
    ------
    ValidationError
        If ``population_size`` is under 2, ``n_generations`` under 1,
        ``elite_size`` not between 1 and ``population_size``, a rate outside
        ``[0, 1]``, ``tournament_size`` under 2, ``max_evaluations`` below
        ``population_size``, no space given, or a space passed as a callable.
    LeakageError
        If partitions overlap or global preprocessing already ran without the
        override.

    Notes
    -----
    **Identical configurations are cached, not rescored.** Convergence means the
    population repeats itself, so the evaluation budget is spent on distinct
    genomes only.

    **This is hyperparameter search, not architecture search.** It evolves
    values in a space you define; it does not invent network topologies. The
    implementation needs only NumPy.

    **Watch ``study['generation_best']``.** A best score that stopped improving
    several generations ago means the population has converged and further
    generations will not help: raise the mutation rate or widen the space.

    Examples
    --------
    Evolve over interacting parameters::

        result = evolutionary_search(
            dataset, split_plan, estimator,
            param_space={
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
                "max_depth": {"type": "int", "low": 2, "high": 12},
                "max_features": ["sqrt", "log2", None],
            },
            population_size=16,
            n_generations=6,
        )
        print(result.study["generation_best"])

    See Also
    --------
    optuna_search : Usually stronger when parameters act independently.
    randomized_search : The simplest baseline worth beating.
    """
    if population_size < 2:
        raise ValidationError("population_size must be >= 2")
    if n_generations < 1:
        raise ValidationError("n_generations must be >= 1")
    if elite_size < 1:
        raise ValidationError("elite_size must be >= 1")
    if elite_size >= population_size:
        raise ValidationError("elite_size must be < population_size")
    if not 0.0 <= crossover_rate <= 1.0:
        raise ValidationError("crossover_rate must be in [0, 1]")
    if not 0.0 <= mutation_rate <= 1.0:
        raise ValidationError("mutation_rate must be in [0, 1]")
    if tournament_size < 2:
        raise ValidationError("tournament_size must be >= 2")
    if param_space is None and recipe_space is None:
        raise ValidationError("Provide param_space and/or recipe_space")
    if param_space is not None and not isinstance(param_space, dict):
        raise ValidationError(
            "evolutionary_search param_space must be a declare-style dict "
            "(callables are not supported; use optuna_search for trial callables)"
        )
    if recipe_space is not None and not isinstance(recipe_space, dict):
        raise ValidationError(
            "evolutionary_search recipe_space must be a declare-style dict "
            "(callables are not supported)"
        )

    budget = (
        int(max_evaluations)
        if max_evaluations is not None
        else int(population_size) * int(n_generations)
    )
    if budget < 1:
        raise ValidationError("max_evaluations must be >= 1")
    if budget < population_size:
        raise ValidationError(
            f"max_evaluations ({budget}) must be >= population_size ({population_size})"
        )

    assert_fit_partition(split_plan, "train")
    assert split_plan is not None
    _x_train, y_train, _feature_cols, _target, _sample_weight = _feature_target_frames(
        dataset, split_plan, "train"
    )
    resolved_task = _infer_task(y_train, task, estimator)
    metric_name = ranking_metric or ("r2" if resolved_task == "regression" else "f1_weighted")
    higher_is_better = metric_name not in _LOWER_IS_BETTER

    genes = _parse_evolutionary_genes(param_space=param_space, recipe_space=recipe_space)
    if not genes:
        raise ValidationError("Evolutionary search space produced no genes")
    needs_recipe = any(
        g.name.startswith("recipe__") or g.name in SAFE_RECIPE_KNOBS for g in genes
    )
    _require_recipe_for_knobs(preprocess, needs_recipe)

    rng = np.random.default_rng(random_state)
    score_cache: dict[tuple[tuple[str, Any], ...], SearchTrial] = {}
    trial_rows: list[SearchTrial] = []
    generation_best: list[dict[str, Any]] = []

    def _evaluate(individual: dict[str, Any]) -> SearchTrial:
        key = _genome_key(individual)
        cached = score_cache.get(key)
        if cached is not None:
            return cached
        if len(score_cache) >= budget:
            # Budget exhausted: return a sentinel-like worst score without CV.
            worst = float("-inf") if higher_is_better else float("inf")
            placeholder = SearchTrial(
                trial=-1,
                params={},
                recipe_knobs={},
                mean_score=worst,
                std_score=float("nan"),
            )
            return placeholder

        est_params, recipe_knobs = _split_trial_params(dict(individual))
        cv_result = cv_score(
            dataset,
            split_plan,
            estimator,
            task=resolved_task,
            cv=cv,
            cv_strategy=cv_strategy,
            scoring_metric=metric_name,
            groups=groups,
            preprocess=preprocess,
            session_preprocess_applied=session_preprocess_applied,
            allow_session_global_preprocess=allow_session_global_preprocess,
            params=est_params,
            recipe_knobs=recipe_knobs,
        )
        score = float(cv_result.mean_metrics[metric_name])
        row = SearchTrial(
            trial=len(trial_rows),
            params=dict(est_params),
            recipe_knobs=dict(recipe_knobs),
            mean_score=score,
            std_score=float(cv_result.std_metrics.get(metric_name, float("nan"))),
            mean_metrics=dict(cv_result.mean_metrics),
            std_metrics=dict(cv_result.std_metrics),
            cv=cv_result,
        )
        score_cache[key] = row
        trial_rows.append(row)
        return row

    population = [_sample_evolutionary_individual(genes, rng) for _ in range(population_size)]
    fitness = [_evaluate(ind) for ind in population]

    for generation in range(n_generations):
        ranked_idx = sorted(
            range(len(population)),
            key=lambda i: fitness[i].mean_score,
            reverse=higher_is_better,
        )
        best_i = ranked_idx[0]
        generation_best.append(
            {
                "generation": generation,
                "best_score": float(fitness[best_i].mean_score),
                "best_params": dict(fitness[best_i].params),
                "best_recipe_knobs": dict(fitness[best_i].recipe_knobs),
                "n_evaluations": len(score_cache),
            }
        )
        if generation + 1 >= n_generations or len(score_cache) >= budget:
            break

        next_pop: list[dict[str, Any]] = []
        next_fit: list[SearchTrial] = []
        for elite_rank in ranked_idx[:elite_size]:
            next_pop.append(dict(population[elite_rank]))
            next_fit.append(fitness[elite_rank])

        while len(next_pop) < population_size:
            slot = len(next_pop)
            if len(score_cache) >= budget:
                filler_i = ranked_idx[slot % len(ranked_idx)]
                next_pop.append(dict(population[filler_i]))
                next_fit.append(fitness[filler_i])
                continue
            p1 = _tournament_select(population, fitness, tournament_size, higher_is_better, rng)
            p2 = _tournament_select(population, fitness, tournament_size, higher_is_better, rng)
            if rng.random() < crossover_rate:
                child, _sibling = _uniform_crossover(p1, p2, genes, rng)
            else:
                child = dict(p1)
            child = _mutate_individual(child, genes, mutation_rate, rng)
            next_pop.append(child)
            next_fit.append(_evaluate(child))

        population = next_pop[:population_size]
        fitness = next_fit[:population_size]

    if not trial_rows:
        raise ValidationError("Evolutionary search produced no evaluated trials")

    trials = sorted(trial_rows, key=lambda item: item.mean_score, reverse=higher_is_better)
    ranked = [
        SearchTrial(
            trial=i,
            params=dict(t.params),
            recipe_knobs=dict(t.recipe_knobs),
            mean_score=t.mean_score,
            std_score=t.std_score,
            mean_metrics=dict(t.mean_metrics),
            std_metrics=dict(t.std_metrics),
            cv=t.cv,
        )
        for i, t in enumerate(trials)
    ]
    history = {
        "kind": "evolutionary",
        "population_size": int(population_size),
        "n_generations": int(n_generations),
        "elite_size": int(elite_size),
        "crossover_rate": float(crossover_rate),
        "mutation_rate": float(mutation_rate),
        "tournament_size": int(tournament_size),
        "max_evaluations": int(budget),
        "n_evaluations": len(score_cache),
        "generation_best": generation_best,
    }
    return _finalize_search_result(
        method="evolutionary",
        resolved_task=resolved_task,
        metric_name=metric_name,
        trials=ranked,
        estimator=estimator,
        dataset=dataset,
        split_plan=split_plan,
        preprocess=preprocess,
        session_preprocess_applied=session_preprocess_applied,
        allow_session_global_preprocess=allow_session_global_preprocess,
        refit=refit,
        study=history,
    )


def nested_cv_score(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    estimator: Any,
    *,
    param_grid: dict[str, list[Any]] | None = None,
    param_distributions: dict[str, Any] | None = None,
    recipe_grid: dict[str, list[Any]] | None = None,
    recipe_distributions: dict[str, Any] | None = None,
    param_space: OptunaSpace | None = None,
    recipe_space: OptunaSpace | None = None,
    inner_search: InnerSearchMethod = "auto",
    n_iter: int = 10,
    n_trials: int = 20,
    population_size: int = 8,
    n_generations: int = 3,
    random_state: int | None = 42,
    task: TaskType = "auto",
    outer_cv: int | Any = 5,
    inner_cv: int | Any = 3,
    cv_strategy: CvStrategy = "auto",
    scoring_metric: str | None = None,
    groups: pd.Series | None = None,
    preprocess: PreprocessRecipe | None = None,
    session_preprocess_applied: bool = False,
    allow_session_global_preprocess: bool = False,
    warm_start_studies: bool = False,
) -> NestedCVResult:
    """Estimate what tuning actually achieves, without the selection bias.

    The problem this solves is easy to miss. Run a search, take the best
    cross-validated score, and report it: that number is too high, and not
    because anything went wrong. You picked the maximum of many noisy estimates,
    so you picked partly for noise. With enough trials, a configuration will
    score well on your folds through luck alone.

    Nested cross-validation separates choosing from measuring. The train
    partition is divided into outer folds. Within each, a complete search runs on
    that fold's training rows, its winner is refitted on those rows, and it is
    scored on the outer-eval rows: which took no part in choosing it. Average
    those outer scores and you have an unbiased estimate of the whole
    tune-then-fit procedure.

    What comes back describes the procedure, not a model. It answers "if I tune
    this way on data like this, how well does the result do?" Different outer
    folds may well choose different parameters, and that is itself worth
    knowing: it means the data does not determine the choice.

    Parameters
    ----------
    dataset:
        The data.
    split_plan:
        The split. Both loops stay inside the train partition.
    estimator:
        The estimator to tune.
    param_grid:
        Grid space for grid inner search. Mutually exclusive with
        ``param_distributions``.
    param_distributions:
        Distribution space for randomized inner search.
    recipe_grid:
        Fold-local recipe knobs as a grid. Requires ``preprocess``.
    recipe_distributions:
        Fold-local recipe knobs as distributions. Requires ``preprocess``.
    param_space:
        Declare-style or callable space for Optuna, or declare-style for
        evolutionary.
    recipe_space:
        The same, for recipe knobs.
    inner_search:
        ``'auto'`` to infer from the spaces given, or an explicit ``'grid'``,
        ``'randomized'``, ``'optuna'``, or ``'evolutionary'``.
    n_iter:
        Trials per inner randomized search.
    n_trials:
        Trials per inner Optuna search; also the evaluation ceiling for
        evolutionary.
    population_size:
        Configurations alive at once in an evolutionary inner search.
    n_generations:
        Breeding rounds in an evolutionary inner search.
    random_state:
        Base seed. Offset per outer fold so folds do not search identically.
    task:
        ``'classification'``, ``'regression'``, or ``'auto'``.
    outer_cv:
        Outer fold count or splitter. These produce the estimate.
    inner_cv:
        Inner fold count or splitter. These rank configurations.
    cv_strategy:
        Folding scheme, shared by both loops.
    scoring_metric:
        Reported by the outer loop and ranked by the inner.
    groups:
        Explicit group labels, aligned to the train partition.
    preprocess:
        Fold-local recipe, refitted in both loops.
    session_preprocess_applied:
        Whether Session-global preprocessing already ran.
    allow_session_global_preprocess:
        Override the resulting leakage refusal.
    warm_start_studies:
        Share one Optuna study across outer folds so later folds inherit earlier
        TPE priors. Cheaper, and slightly couples the folds: see the notes.

    Returns
    -------
    NestedCVResult
        Outer-fold scores and their mean and spread, each fold's chosen
        configuration, and a stability summary of those choices.

    Raises
    ------
    ValidationError
        If no search space is given, if mutually exclusive spaces are combined,
        if a space does not match the chosen inner method, if
        ``warm_start_studies`` is set without Optuna, if ``groups`` does not
        match the train partition's length, or if no outer folds were produced.
    LeakageError
        If partitions overlap, if an outer fold intersects a Session holdout, or
        if global preprocessing already ran without the override.

    Notes
    -----
    **The cost is the product of both loops.** Five outer folds around a
    fifty-trial inner search at three inner folds is 750 model fits before the
    outer refits. Start with fewer folds and a smaller space than you would use
    for a plain search.

    **Report the outer mean and spread, never the inner scores.** The inner
    means are selection evidence and are biased upward. The gap between them and
    the outer mean is exactly the bias this procedure exists to expose.

    **Nested CV gives you a number, not a model.** Afterwards, run the search
    once on the full train partition and deploy that winner, reporting the
    nested estimate as what to expect from it.

    **Unstable choices across folds are a real finding.** When
    ``inner_selection_summary['param_stability']`` is low, several
    configurations are effectively tied, and the one a single search picks is
    somewhat arbitrary.

    **Warm-starting couples the outer folds.** Shared study state carries only
    inner-CV scores from earlier outer-train subsets: outer-eval rows and
    Session holdouts never enter any objective: but later folds start from
    priors shaped by earlier ones, which makes the estimate mildly optimistic
    relative to independent studies. It is opt-in for that reason.

    Examples
    --------
    Estimate the procedure, then tune once for real::

        nested = nested_cv_score(
            dataset, split_plan, estimator,
            param_distributions={"max_depth": [3, 5, 8], "min_samples_leaf": [1, 5, 10]},
            outer_cv=5,
            inner_cv=3,
        )
        nested.show()
        print(nested.inner_selection_summary["param_stability"])

        final = randomized_search(dataset, split_plan, estimator, {...})

    See Also
    --------
    cv_score : When there is nothing to select.
    grid_search : The biased-but-cheaper alternative.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None
    search_method = _resolve_inner_search(
        inner_search=inner_search,
        param_grid=param_grid,
        param_distributions=param_distributions,
        recipe_grid=recipe_grid,
        recipe_distributions=recipe_distributions,
        param_space=param_space,
        recipe_space=recipe_space,
    )
    has_est = param_grid is not None or param_distributions is not None or param_space is not None
    has_recipe = (
        recipe_grid is not None or recipe_distributions is not None or recipe_space is not None
    )
    if not has_est and not has_recipe:
        raise ValidationError(
            "nested_cv_score requires an estimator and/or recipe search space "
            "(param_grid/param_distributions/param_space and/or "
            "recipe_grid/recipe_distributions/recipe_space)"
        )
    if param_grid is not None and param_distributions is not None:
        raise ValidationError(
            "nested_cv_score accepts at most one of param_grid or param_distributions"
        )
    if recipe_grid is not None and recipe_distributions is not None:
        raise ValidationError(
            "nested_cv_score accepts at most one of recipe_grid or recipe_distributions"
        )
    if search_method in {"optuna", "evolutionary"}:
        if param_space is None and recipe_space is None:
            raise ValidationError(
                f"inner_search={search_method!r} requires param_space and/or recipe_space"
            )
        if param_grid is not None or param_distributions is not None:
            raise ValidationError(
                f"inner_search={search_method!r} uses param_space; "
                "omit param_grid/param_distributions"
            )
        if recipe_grid is not None or recipe_distributions is not None:
            raise ValidationError(
                f"inner_search={search_method!r} uses recipe_space; "
                "omit recipe_grid/recipe_distributions"
            )
    if warm_start_studies and search_method != "optuna":
        raise ValidationError(
            "warm_start_studies=True requires Optuna inner search "
            "(inner_search='optuna' or auto with param_space/recipe_space)"
        )
    if search_method in {"optuna", "evolutionary"} and n_trials < 1:
        raise ValidationError("n_trials must be >= 1")
    if param_grid is not None and not param_grid:
        raise ValidationError("param_grid must not be empty when provided")
    if param_distributions is not None and not param_distributions:
        raise ValidationError("param_distributions must not be empty when provided")
    if recipe_grid is not None and not recipe_grid:
        raise ValidationError("recipe_grid must not be empty when provided")
    if recipe_distributions is not None and not recipe_distributions:
        raise ValidationError("recipe_distributions must not be empty when provided")
    _require_recipe_for_knobs(preprocess, has_recipe)
    _refuse_session_global_cv_leakage(
        session_preprocess_applied=session_preprocess_applied,
        preprocess=preprocess,
        allow_session_global_preprocess=allow_session_global_preprocess,
    )

    x_train, y_train, _feature_cols, _target, _sample_weight = _feature_target_frames(
        dataset, split_plan, "train"
    )
    resolved_task = _infer_task(y_train, task, estimator)
    metric = scoring_metric or ("r2" if resolved_task == "regression" else "f1_weighted")
    weight_col = weight_column(dataset)

    held_out: list[str] = ["test"]
    if split_plan.validation_indices:
        held_out.append("validation")

    train_set = set(split_plan.train_indices)
    session_test = set(split_plan.test_indices)
    session_valid = set(split_plan.validation_indices)
    if train_set & session_test:
        raise LeakageError("Train and test partitions overlap; refusing nested CV")
    if train_set & session_valid:
        raise LeakageError("Train and validation partitions overlap; refusing nested CV")

    train_positions = list(split_plan.train_indices)
    group_values, outer_strategy, outer_splitter, row_order = _resolve_splitter(
        dataset=dataset,
        split_plan=split_plan,
        y_train=y_train,
        cv=outer_cv,
        cv_strategy=cv_strategy,
        groups=groups,
        task=resolved_task,
    )

    x_reset = x_train.reset_index(drop=True)
    y_reset = y_train.reset_index(drop=True)
    position_map = np.asarray(train_positions, dtype=int)
    if row_order is not None:
        x_reset = x_reset.iloc[row_order].reset_index(drop=True)
        y_reset = y_reset.iloc[row_order].reset_index(drop=True)
        position_map = position_map[row_order]
        if group_values is not None:
            group_values = pd.Series(group_values).iloc[row_order]

    group_reset = None if group_values is None else pd.Series(group_values).reset_index(drop=True)
    split_iter = (
        outer_splitter.split(x_reset, y_reset, group_reset)
        if group_reset is not None
        else outer_splitter.split(x_reset, y_reset)
    )

    outer_folds: list[OuterFoldResult] = []
    metric_rows: list[dict[str, float]] = []
    selected_params: list[dict[str, Any]] = []
    selected_recipe_knobs: list[dict[str, Any]] = []
    inner_strategy_name = outer_strategy
    inner_n_splits = int(inner_cv) if isinstance(inner_cv, int) else 0
    shared_study: Any | None = None

    for fold_id, (outer_train_pos, outer_eval_pos) in enumerate(split_iter):
        if set(outer_train_pos) & set(outer_eval_pos):
            raise LeakageError("Nested CV outer fold train/eval indices overlap")
        outer_train_idx = tuple(int(i) for i in position_map[list(outer_train_pos)])
        outer_eval_idx = tuple(int(i) for i in position_map[list(outer_eval_pos)])
        if set(outer_train_idx) & session_test or set(outer_eval_idx) & session_test:
            raise LeakageError("Nested CV outer fold intersected Session test partition")
        if set(outer_train_idx) & session_valid or set(outer_eval_idx) & session_valid:
            raise LeakageError("Nested CV outer fold intersected Session validation partition")

        inner_plan = SplitPlan(
            kind="nested_inner",
            test_size=None,
            validation_size=None,
            random_state=None,
            stratify_column=None,
            train_indices=outer_train_idx,
            validation_indices=(),
            test_indices=outer_eval_idx,
        )
        # Explicit groups are aligned to Session train-row order; subset to outer-train.
        inner_groups = None
        if groups is not None:
            aligned = pd.Series(groups).reset_index(drop=True)
            if len(aligned) != len(split_plan.train_indices):
                raise ValidationError("groups length must match the train partition")
            train_pos_lookup = {int(idx): i for i, idx in enumerate(split_plan.train_indices)}
            inner_groups = pd.Series(
                [aligned.iloc[train_pos_lookup[idx]] for idx in outer_train_idx],
                dtype=aligned.dtype,
            )

        if search_method == "grid":
            fold_search = grid_search(
                dataset,
                inner_plan,
                estimator,
                param_grid,
                recipe_grid=recipe_grid,
                task=resolved_task,
                cv=inner_cv,
                cv_strategy=cv_strategy,
                ranking_metric=metric,
                groups=inner_groups,
                preprocess=preprocess,
                session_preprocess_applied=session_preprocess_applied,
                allow_session_global_preprocess=allow_session_global_preprocess,
                refit=False,
            )
        elif search_method == "optuna":
            fold_search = optuna_search(
                dataset,
                inner_plan,
                estimator,
                param_space=param_space,
                recipe_space=recipe_space,
                n_trials=n_trials,
                random_state=None if random_state is None else int(random_state) + fold_id,
                task=resolved_task,
                cv=inner_cv,
                cv_strategy=cv_strategy,
                ranking_metric=metric,
                groups=inner_groups,
                preprocess=preprocess,
                session_preprocess_applied=session_preprocess_applied,
                allow_session_global_preprocess=allow_session_global_preprocess,
                refit=False,
                study=shared_study if warm_start_studies else None,
            )
            if warm_start_studies:
                shared_study = fold_search.study
        elif search_method == "evolutionary":
            if not isinstance(param_space, (dict, type(None))) or not isinstance(
                recipe_space, (dict, type(None))
            ):
                raise ValidationError(
                    "inner_search='evolutionary' requires declare-style dict "
                    "param_space/recipe_space (not Optuna trial callables)"
                )
            fold_search = evolutionary_search(
                dataset,
                inner_plan,
                estimator,
                param_space=param_space,
                recipe_space=recipe_space,
                population_size=population_size,
                n_generations=n_generations,
                max_evaluations=n_trials,
                random_state=None if random_state is None else int(random_state) + fold_id,
                task=resolved_task,
                cv=inner_cv,
                cv_strategy=cv_strategy,
                ranking_metric=metric,
                groups=inner_groups,
                preprocess=preprocess,
                session_preprocess_applied=session_preprocess_applied,
                allow_session_global_preprocess=allow_session_global_preprocess,
                refit=False,
            )
        else:
            fold_search = randomized_search(
                dataset,
                inner_plan,
                estimator,
                param_distributions,
                recipe_distributions=recipe_distributions,
                n_iter=n_iter,
                random_state=None if random_state is None else int(random_state) + fold_id,
                task=resolved_task,
                cv=inner_cv,
                cv_strategy=cv_strategy,
                ranking_metric=metric,
                groups=inner_groups,
                preprocess=preprocess,
                session_preprocess_applied=session_preprocess_applied,
                allow_session_global_preprocess=allow_session_global_preprocess,
                refit=False,
            )
        if fold_search.best_cv is not None:
            inner_strategy_name = fold_search.best_cv.cv_strategy
            inner_n_splits = fold_search.best_cv.n_splits

        # Refit winner on outer-train only; score outer-eval.
        model = clone(estimator)
        if fold_search.best_params:
            model.set_params(**fold_search.best_params)
        active_recipe = _recipe_with_knobs(preprocess, fold_search.best_recipe_knobs)
        base = dataset._ensure_pandas()
        x_outer_train = base.iloc[list(outer_train_idx)][list(_feature_cols)]
        y_outer_train = base.iloc[list(outer_train_idx)][_target]
        x_outer_eval = base.iloc[list(outer_eval_idx)][list(_feature_cols)]
        y_outer_eval = base.iloc[list(outer_eval_idx)][_target]
        w_outer_train = None
        w_outer_eval = None
        if weight_col is not None:
            w_outer_train = validate_sample_weights(
                base.iloc[list(outer_train_idx)][weight_col], column=weight_col
            )
            w_outer_eval = validate_sample_weights(
                base.iloc[list(outer_eval_idx)][weight_col], column=weight_col
            )
        if active_recipe is not None and not active_recipe.is_empty():
            prep = build_fold_preprocessor(x_outer_train, active_recipe, y_outer_train)
            x_fit = transform_fold_features(prep, x_outer_train)
            x_score = transform_fold_features(prep, x_outer_eval)
        else:
            x_fit = x_outer_train
            x_score = x_outer_eval
        model.fit(
            x_fit,
            y_outer_train,
            **fit_kwargs_for_sample_weight(model, w_outer_train),
        )
        y_pred = model.predict(x_score)
        fold_metrics = _score_predictions(
            resolved_task, y_outer_eval, y_pred, sample_weight=w_outer_eval
        )
        outer_folds.append(
            OuterFoldResult(
                fold=fold_id,
                n_train=int(len(outer_train_idx)),
                n_eval=int(len(outer_eval_idx)),
                best_params=dict(fold_search.best_params),
                best_recipe_knobs=dict(fold_search.best_recipe_knobs),
                inner_best_score=fold_search.best_score,
                inner_best_std=fold_search.best_std,
                inner_n_trials=len(fold_search.trials),
                metrics=fold_metrics,
            )
        )
        metric_rows.append(fold_metrics)
        selected_params.append(dict(fold_search.best_params))
        selected_recipe_knobs.append(dict(fold_search.best_recipe_knobs))

    if not metric_rows:
        raise ValidationError("Nested cross-validation produced no outer folds")

    mean_metrics, std_metrics = _aggregate_metrics(metric_rows)
    summary = _inner_selection_summary(
        selected_params, outer_folds, metric, selected_recipe_knobs=selected_recipe_knobs
    )
    session_global_override = bool(session_preprocess_applied and allow_session_global_preprocess)
    limitations = _nested_limitations(
        session_preprocess_applied=session_global_override,
        preprocess=preprocess,
        outer_strategy=outer_strategy,
        inner_strategy=inner_strategy_name,
        n_outer=len(outer_folds),
        n_inner=inner_n_splits,
        held_out=held_out,
    )
    if any(selected_recipe_knobs):
        limitations.append(
            "Inner search chose fold-local recipe knobs "
            f"({sorted({k for knobs in selected_recipe_knobs for k in knobs})}); "
            "outer-eval rows never contributed to those choices."
        )
    if warm_start_studies:
        limitations.append(
            "warm_start_studies=True shared one Optuna study across outer folds. "
            "Trial objectives still scored only each outer-train subset via inner "
            "CV (no Session test/validation or outer-eval peeking), but search "
            "priors couple outer folds and can be mildly optimistic versus "
            "independent studies."
        )
    interpretation = [
        (
            f"Outer-loop mean {metric}={mean_metrics[metric]:.6f} ± "
            f"{std_metrics.get(metric, 0.0):.6f} across {len(outer_folds)} folds "
            f"after inner {search_method} selection ({resolved_task})."
        )
    ]
    if summary.get("param_stability") == "low":
        interpretation.append(
            "Inner search selected different parameter sets across outer folds: "
            "treat a single full-train refit as one of several plausible winners."
        )
    recommendations = [
        (
            f"Report the outer mean±std of '{metric}' as the post-selection estimate; "
            "do not substitute inner CV means."
        ),
        (
            "After nested CV, refit the chosen recipe on full train and confirm "
            f"once on {held_out[0]}."
        ),
    ]
    if session_global_override:
        recommendations.append(
            "allow_session_global_preprocess=True was set; Session-global preprocess "
            "poisoned folds. Re-ingest unpoisoned data before fold-local "
            "PreprocessRecipe CV next time."
        )
    elif preprocess is not None and not preprocess.is_empty():
        recommendations.append(
            "Fold-local preprocess was refit inside outer and inner loops on their "
            "respective training rows only."
        )
    if any(selected_recipe_knobs):
        recommendations.append(
            "Inspect outer_folds[*].best_recipe_knobs and "
            "inner_selection_summary.selected_recipe_knobs_by_fold before freezing "
            "a single production recipe."
        )

    return NestedCVResult(
        task=resolved_task,
        scoring_metric=metric,
        outer_cv_strategy=outer_strategy,
        inner_cv_strategy=inner_strategy_name,
        n_outer_splits=len(outer_folds),
        n_inner_splits=inner_n_splits,
        search_method=search_method,
        outer_folds=outer_folds,
        mean_metrics=mean_metrics,
        std_metrics=std_metrics,
        inner_selection_summary=summary,
        population="train",
        held_out_partitions=tuple(held_out),
        fold_preprocess=None if preprocess is None else preprocess.to_dict(),
        limitations=limitations,
        interpretation=interpretation,
        recommendations=recommendations,
        warm_start_studies=bool(warm_start_studies),
    )


def _inner_selection_summary(
    selected_params: list[dict[str, Any]],
    outer_folds: list[OuterFoldResult],
    metric: str,
    *,
    selected_recipe_knobs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    from collections import Counter

    recipe_knobs = selected_recipe_knobs or [dict(f.best_recipe_knobs) for f in outer_folds]
    frozen: list[tuple[tuple[str, str], ...]] = []
    for params, knobs in zip(selected_params, recipe_knobs, strict=True):
        merged = dict(params)
        merged.update({f"recipe__{rk}": rv for rk, rv in knobs.items()})
        frozen.append(tuple(sorted((k, repr(v)) for k, v in merged.items())))
    counts = Counter(frozen)
    unique = len(counts)
    top = counts.most_common(1)[0] if counts else ((), 0)
    if unique == 1:
        stability = "high"
    elif unique <= max(2, len(frozen) // 2):
        stability = "medium"
    else:
        stability = "low"
    inner_means = [f.inner_best_score for f in outer_folds if f.inner_best_score is not None]
    return {
        "n_outer_folds": len(outer_folds),
        "n_unique_param_sets": unique,
        "most_common_params": dict(selected_params[0]) if selected_params and unique == 1 else None,
        "most_common_recipe_knobs": (
            dict(recipe_knobs[0]) if recipe_knobs and unique == 1 else None
        ),
        "most_common_count": int(top[1]),
        "param_stability": stability,
        "inner_best_score_mean": float(np.mean(inner_means)) if inner_means else None,
        "inner_best_score_std": (
            float(np.std(inner_means, ddof=1)) if len(inner_means) > 1 else 0.0
        ),
        "outer_metric": metric,
        "selected_params_by_fold": [dict(p) for p in selected_params],
        "selected_recipe_knobs_by_fold": [dict(k) for k in recipe_knobs],
    }


def _nested_limitations(
    *,
    session_preprocess_applied: bool,
    preprocess: PreprocessRecipe | None,
    outer_strategy: str,
    inner_strategy: str,
    n_outer: int,
    n_inner: int,
    held_out: list[str],
) -> list[str]:
    tips = [
        (
            f"Outer scores summarize {n_outer} folds drawn only from the train partition; "
            f"inner search used {n_inner}-fold CV on each outer-train subset."
        ),
        (
            "Outer-eval rows never enter inner CV membership or inner ranking "
            f"(outer={outer_strategy}, inner={inner_strategy})."
        ),
        (f"Session held-out partition(s) stay untouched during nested CV: {', '.join(held_out)}."),
        (
            "Inner CV means are selection evidence only; the outer mean±std is the "
            "post-selection estimate."
        ),
    ]
    tips.extend(
        _cv_limitations(
            session_preprocess_applied=session_preprocess_applied,
            preprocess=preprocess,
            strategy_name=outer_strategy,
            n_folds=n_outer,
        )
    )
    return tips


def _run_search(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    estimator: Any,
    combos: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    method: SearchMethod,
    task: TaskType,
    cv: int | Any,
    cv_strategy: CvStrategy,
    ranking_metric: str | None,
    groups: pd.Series | None,
    preprocess: PreprocessRecipe | None,
    session_preprocess_applied: bool,
    allow_session_global_preprocess: bool,
    refit: bool,
) -> SearchResult:
    trials: list[SearchTrial] = []
    resolved_task: Literal["classification", "regression"] | None = None
    metric_name: str | None = ranking_metric

    for idx, (est_params, recipe_knobs) in enumerate(combos):
        cv_result = cv_score(
            dataset,
            split_plan,
            estimator,
            task=task,
            cv=cv,
            cv_strategy=cv_strategy,
            scoring_metric=metric_name,
            groups=groups,
            preprocess=preprocess,
            session_preprocess_applied=session_preprocess_applied,
            allow_session_global_preprocess=allow_session_global_preprocess,
            params=est_params,
            recipe_knobs=recipe_knobs,
        )
        resolved_task = cv_result.task
        metric_name = cv_result.scoring_metric
        score = cv_result.mean_metrics[metric_name]
        trials.append(
            SearchTrial(
                trial=idx,
                params=dict(est_params),
                recipe_knobs=dict(recipe_knobs),
                mean_score=score,
                std_score=cv_result.std_metrics.get(metric_name, float("nan")),
                mean_metrics=dict(cv_result.mean_metrics),
                std_metrics=dict(cv_result.std_metrics),
                cv=cv_result,
            )
        )

    assert resolved_task is not None and metric_name is not None
    higher_is_better = metric_name not in _LOWER_IS_BETTER
    trials.sort(key=lambda item: item.mean_score, reverse=higher_is_better)
    return _finalize_search_result(
        method=method,
        resolved_task=resolved_task,
        metric_name=metric_name,
        trials=trials,
        estimator=estimator,
        dataset=dataset,
        split_plan=split_plan,
        preprocess=preprocess,
        session_preprocess_applied=session_preprocess_applied,
        allow_session_global_preprocess=allow_session_global_preprocess,
        refit=refit,
    )


def _finalize_search_result(
    *,
    method: SearchMethod,
    resolved_task: Literal["classification", "regression"],
    metric_name: str,
    trials: list[SearchTrial],
    estimator: Any,
    dataset: Dataset,
    split_plan: SplitPlan | None,
    preprocess: PreprocessRecipe | None,
    session_preprocess_applied: bool,
    refit: bool,
    allow_session_global_preprocess: bool = False,
    study: Any | None = None,
) -> SearchResult:
    best = trials[0]
    session_global_override = bool(session_preprocess_applied and allow_session_global_preprocess)

    refit_result = None
    if refit:
        model = clone(estimator)
        if best.params:
            model.set_params(**best.params)
        active_recipe = _recipe_with_knobs(preprocess, best.recipe_knobs)
        if active_recipe is not None and not active_recipe.is_empty():
            x_train, y_train, feature_cols, target, sample_weight = _feature_target_frames(
                dataset,
                split_plan,
                "train",  # type: ignore[arg-type]
            )
            prep = build_fold_preprocessor(x_train, active_recipe, y_train)
            x_fit = transform_fold_features(prep, x_train)
            fitted = clone(model)
            fitted.fit(
                x_fit,
                y_train,
                **fit_kwargs_for_sample_weight(fitted, sample_weight),
            )
            bundled = SkPipeline([("preprocess", prep), ("model", fitted)])
            refit_result = FitResult(
                estimator=bundled,
                task=resolved_task,
                feature_columns=tuple(feature_cols),
                target_column=target,
                n_train_rows=int(len(x_train)),
                weight_column=weight_column(dataset),
            )
        else:
            refit_result = fit_estimator(dataset, split_plan, model, task=resolved_task)

    interpretation = [
        (
            f"Best {metric_name} over {len(trials)} {method} trial(s): "
            f"{best.mean_score:.6f} ± {best.std_score:.6f} on train-fold CV."
        )
    ]
    if best.recipe_knobs:
        interpretation.append(f"Best recipe knobs: {best.recipe_knobs}.")
    if len(trials) >= 2:
        gap = abs(trials[0].mean_score - trials[1].mean_score)
        interpretation.append(
            f"Top-2 mean {metric_name} gap is {gap:.6f}; "
            f"second-best std is {trials[1].std_score:.6f}."
        )
        if gap < max(trials[0].std_score, 1e-12):
            interpretation.append(
                "Top-2 gap is within the leading trial's fold standard deviation: "
                "treat rank as fragile without a confirmation holdout."
            )

    held = list(best.cv.held_out_partitions) if best.cv is not None else ["test"]
    recommendations = [
        f"Selected params by mean {metric_name} across CV folds on the train population.",
        f"Confirm the winner once on {held[0]} after search.",
    ]
    if session_global_override:
        recommendations.append(
            "allow_session_global_preprocess=True was set; Session preprocess was "
            "train-global. Re-ingest unpoisoned data, then use fold-local "
            "PreprocessRecipe without Session.impute/scale before search."
        )
    if best.std_score > abs(best.mean_score) * 0.15 and abs(best.mean_score) > 1e-9:
        recommendations.append(
            "Fold spread is large relative to the mean: prefer simpler params or more data."
        )
    if method == "optuna":
        recommendations.append(
            "Optuna TPE sampled this budget; raise n_trials only while fold std still "
            "informs whether gaps are real."
        )
    if method == "evolutionary":
        recommendations.append(
            "Evolutionary search used a real GA (population, selection, crossover/mutation, "
            "elitism) under a CV budget: not random search renamed. Raise population_size / "
            "n_generations only while fold std still informs whether gaps are real."
        )

    limitations = list(best.cv.limitations) if best.cv is not None else []
    limitations.append(
        "Search ranks configurations by nested train-fold CV; it does not peek at held-out "
        f"partition(s): {', '.join(held)}."
    )

    return SearchResult(
        method=method,
        task=resolved_task,
        ranking_metric=metric_name,
        trials=trials,
        best_params=dict(best.params),
        best_recipe_knobs=dict(best.recipe_knobs),
        best_score=best.mean_score,
        best_std=best.std_score,
        best_cv=best.cv,
        refit_result=refit_result,
        interpretation=interpretation,
        recommendations=recommendations,
        limitations=limitations,
        study=study,
    )


def _suggest_from_space(
    trial: Any,
    space: dict[str, Any],
    *,
    prefix: str,
) -> dict[str, Any]:
    """Suggest values from a declare-style Optuna space mapping.

    Supported value forms:

    - ``{"type": "float", "low": ..., "high": ..., "log": bool}``
    - ``{"type": "int", "low": ..., "high": ...}``
    - ``{"type": "categorical", "choices": [...]}``
    - plain list/tuple → categorical choices
    """
    out: dict[str, Any] = {}
    for name, spec in space.items():
        key = f"{prefix}__{name}"
        if isinstance(spec, (list, tuple)):
            out[name] = trial.suggest_categorical(key, list(spec))
            continue
        if not isinstance(spec, dict):
            raise ValidationError(
                f"Optuna space entry '{name}' must be a dict spec or a list of choices"
            )
        kind = str(spec.get("type", "")).lower()
        if kind == "float":
            out[name] = trial.suggest_float(
                key,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        elif kind == "int":
            out[name] = trial.suggest_int(key, int(spec["low"]), int(spec["high"]))
        elif kind == "categorical":
            choices = list(spec.get("choices") or [])
            if not choices:
                raise ValidationError(f"Optuna categorical space '{name}' needs non-empty choices")
            out[name] = trial.suggest_categorical(key, choices)
        else:
            raise ValidationError(
                f"Unsupported Optuna space type '{kind}' for '{name}'. "
                "Use float, int, or categorical."
            )
    return out


def _split_trial_params(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a flat trial dict into estimator params and recipe knobs."""
    est: dict[str, Any] = {}
    recipe: dict[str, Any] = {}
    for key, value in params.items():
        if key.startswith("recipe__"):
            recipe[key[len("recipe__") :]] = value
        elif key in SAFE_RECIPE_KNOBS:
            recipe[key] = value
        else:
            est[key] = value
    unknown = sorted(set(recipe) - SAFE_RECIPE_KNOBS)
    if unknown:
        raise ValidationError(
            f"Unsupported recipe knobs: {unknown}. Allowed: {sorted(SAFE_RECIPE_KNOBS)}"
        )
    return est, recipe


def _recipe_with_knobs(
    preprocess: PreprocessRecipe | None,
    knobs: dict[str, Any] | None,
) -> PreprocessRecipe | None:
    if preprocess is None:
        if knobs:
            raise ValidationError("recipe knobs require a base PreprocessRecipe")
        return None
    if not knobs:
        return preprocess
    return preprocess.with_knobs(knobs)


def _resolve_inner_search(
    *,
    inner_search: InnerSearchMethod,
    param_grid: dict[str, list[Any]] | None,
    param_distributions: dict[str, Any] | None,
    recipe_grid: dict[str, list[Any]] | None,
    recipe_distributions: dict[str, Any] | None,
    param_space: OptunaSpace | None,
    recipe_space: OptunaSpace | None,
) -> SearchMethod:
    """Resolve nested-CV inner search method from explicit choice or spaces."""
    has_space = param_space is not None or recipe_space is not None
    has_grid = param_grid is not None or recipe_grid is not None
    has_random = param_distributions is not None or recipe_distributions is not None

    if inner_search == "optuna":
        return "optuna"
    if inner_search == "evolutionary":
        return "evolutionary"
    if inner_search == "grid":
        if has_space:
            raise ValidationError(
                "inner_search='grid' cannot be combined with param_space/recipe_space"
            )
        if has_random and not has_grid:
            raise ValidationError("inner_search='grid' requires param_grid and/or recipe_grid")
        return "grid"
    if inner_search == "randomized":
        if has_space:
            raise ValidationError(
                "inner_search='randomized' cannot be combined with param_space/recipe_space"
            )
        return "randomized"
    # auto: declare/callable spaces default to Optuna (not evolutionary).
    if has_space:
        if has_grid or has_random:
            raise ValidationError(
                "Provide either Optuna/evolutionary spaces (param_space/recipe_space) or "
                "grid/randomized spaces, not both; or set inner_search explicitly"
            )
        return "optuna"
    if has_grid and not has_random:
        return "grid"
    if has_random:
        return "randomized"
    if has_grid:
        return "grid"
    raise ValidationError(
        "nested_cv_score requires an estimator and/or recipe search space "
        "(param_grid/param_distributions/param_space and/or "
        "recipe_grid/recipe_distributions/recipe_space)"
    )


@dataclass(frozen=True, slots=True)
class _GeneSpec:
    """One searchable gene in an evolutionary HPO genome."""

    name: str
    kind: Literal["float", "int", "categorical"]
    low: float | None = None
    high: float | None = None
    log: bool = False
    choices: tuple[Any, ...] | None = None


def _parse_evolutionary_genes(
    *,
    param_space: EvolutionarySpace | None,
    recipe_space: EvolutionarySpace | None,
) -> list[_GeneSpec]:
    genes: list[_GeneSpec] = []
    if param_space:
        for name, spec in param_space.items():
            key = name if name.startswith("recipe__") else name
            genes.append(_gene_from_spec(key, spec))
    if recipe_space:
        for name, spec in recipe_space.items():
            if name in SAFE_RECIPE_KNOBS:
                gene_name = name
            elif name.startswith("recipe__"):
                gene_name = name
            else:
                gene_name = f"recipe__{name}"
            genes.append(_gene_from_spec(gene_name, spec))
    # De-dupe by name (recipe_space wins on collision).
    by_name: dict[str, _GeneSpec] = {g.name: g for g in genes}
    return list(by_name.values())


def _gene_from_spec(name: str, spec: Any) -> _GeneSpec:
    if isinstance(spec, (list, tuple)):
        choices = tuple(spec)
        if not choices:
            raise ValidationError(f"Evolutionary categorical space '{name}' needs non-empty choices")
        return _GeneSpec(name=name, kind="categorical", choices=choices)
    if not isinstance(spec, dict):
        raise ValidationError(
            f"Evolutionary space entry '{name}' must be a dict spec or a list of choices"
        )
    kind = str(spec.get("type", "")).lower()
    if kind == "float":
        low = float(spec["low"])
        high = float(spec["high"])
        if high < low:
            raise ValidationError(f"Evolutionary float space '{name}' has high < low")
        return _GeneSpec(
            name=name,
            kind="float",
            low=low,
            high=high,
            log=bool(spec.get("log", False)),
        )
    if kind == "int":
        low_i = int(spec["low"])
        high_i = int(spec["high"])
        if high_i < low_i:
            raise ValidationError(f"Evolutionary int space '{name}' has high < low")
        return _GeneSpec(name=name, kind="int", low=float(low_i), high=float(high_i))
    if kind == "categorical":
        choices = tuple(spec.get("choices") or [])
        if not choices:
            raise ValidationError(f"Evolutionary categorical space '{name}' needs non-empty choices")
        return _GeneSpec(name=name, kind="categorical", choices=choices)
    raise ValidationError(
        f"Unsupported evolutionary space type '{kind}' for '{name}'. "
        "Use float, int, or categorical."
    )


def _sample_evolutionary_individual(
    genes: list[_GeneSpec],
    rng: np.random.Generator,
) -> dict[str, Any]:
    individual: dict[str, Any] = {}
    for gene in genes:
        individual[gene.name] = _sample_gene(gene, rng)
    return individual


def _sample_gene(gene: _GeneSpec, rng: np.random.Generator) -> Any:
    if gene.kind == "categorical":
        assert gene.choices is not None
        return gene.choices[int(rng.integers(0, len(gene.choices)))]
    assert gene.low is not None and gene.high is not None
    if gene.kind == "int":
        return int(rng.integers(int(gene.low), int(gene.high) + 1))
    if gene.log:
        if gene.low <= 0 or gene.high <= 0:
            raise ValidationError(f"Log-float gene '{gene.name}' requires low/high > 0")
        log_low = float(np.log(gene.low))
        log_high = float(np.log(gene.high))
        return float(np.exp(rng.uniform(log_low, log_high)))
    return float(rng.uniform(gene.low, gene.high))


def _mutate_individual(
    individual: dict[str, Any],
    genes: list[_GeneSpec],
    mutation_rate: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    child = dict(individual)
    for gene in genes:
        if rng.random() >= mutation_rate:
            continue
        if gene.kind == "categorical":
            child[gene.name] = _sample_gene(gene, rng)
            continue
        assert gene.low is not None and gene.high is not None
        if gene.kind == "int":
            # Small integer step, else full resample.
            current = int(child.get(gene.name, int(gene.low)))
            if rng.random() < 0.5:
                step = int(rng.choice([-2, -1, 1, 2]))
                child[gene.name] = int(np.clip(current + step, int(gene.low), int(gene.high)))
            else:
                child[gene.name] = _sample_gene(gene, rng)
            continue
        # Float: multiplicative/additive jitter in-range, else resample.
        current_f = float(child.get(gene.name, gene.low))
        if gene.log and current_f > 0:
            factor = float(np.exp(rng.normal(0.0, 0.25)))
            mutated = float(np.clip(current_f * factor, gene.low, gene.high))
        else:
            span = gene.high - gene.low
            mutated = float(np.clip(current_f + rng.normal(0.0, 0.1 * span), gene.low, gene.high))
        child[gene.name] = mutated
    return child


def _uniform_crossover(
    parent1: dict[str, Any],
    parent2: dict[str, Any],
    genes: list[_GeneSpec],
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, Any]]:
    child1 = dict(parent1)
    child2 = dict(parent2)
    for gene in genes:
        if rng.random() < 0.5:
            child1[gene.name], child2[gene.name] = parent2[gene.name], parent1[gene.name]
    return child1, child2


def _tournament_select(
    population: list[dict[str, Any]],
    fitness: list[SearchTrial],
    tournament_size: int,
    higher_is_better: bool,
    rng: np.random.Generator,
) -> dict[str, Any]:
    n = len(population)
    k = min(tournament_size, n)
    idxs = [int(i) for i in rng.choice(n, size=k, replace=False)]
    best = idxs[0]
    for idx in idxs[1:]:
        if higher_is_better:
            if fitness[idx].mean_score > fitness[best].mean_score:
                best = idx
        elif fitness[idx].mean_score < fitness[best].mean_score:
            best = idx
    return dict(population[best])


def _genome_key(individual: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    items: list[tuple[str, Any]] = []
    for key in sorted(individual):
        value = individual[key]
        if isinstance(value, (float, np.floating)):
            items.append((key, round(float(value), 12)))
        elif isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            items.append((key, int(value)))
        else:
            items.append((key, value))
    return tuple(items)


def _require_recipe_for_knobs(preprocess: PreprocessRecipe | None, needs_recipe: bool) -> None:
    if needs_recipe and preprocess is None:
        raise ValidationError(
            "recipe_grid/recipe_distributions/recipe_space require preprocess=PreprocessRecipe(...)"
        )


def _grid_dicts(space: dict[str, list[Any]] | None) -> list[dict[str, Any]]:
    if not space:
        return [{}]
    keys = list(space)
    return [dict(zip(keys, values, strict=True)) for values in product(*(space[k] for k in keys))]


def _expand_grid_trials(
    *,
    param_grid: dict[str, list[Any]] | None,
    recipe_grid: dict[str, list[Any]] | None,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if (param_grid is None or not param_grid) and (recipe_grid is None or not recipe_grid):
        raise ValidationError("Provide a non-empty param_grid and/or recipe_grid")
    if recipe_grid:
        unknown = sorted(set(recipe_grid) - SAFE_RECIPE_KNOBS)
        if unknown:
            raise ValidationError(
                f"Unsupported recipe_grid knobs: {unknown}. Allowed: {sorted(SAFE_RECIPE_KNOBS)}"
            )
    est_raw = _grid_dicts(param_grid)
    # Allow recipe__ keys inside param_grid for convenience.
    est_combos: list[dict[str, Any]] = []
    recipe_from_params: list[dict[str, Any]] = []
    for raw in est_raw:
        est, recipe = _split_trial_params(raw)
        est_combos.append(est)
        recipe_from_params.append(recipe)
    recipe_combos = _grid_dicts(recipe_grid)
    trials: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for est, embedded in zip(est_combos, recipe_from_params, strict=True):
        for recipe in recipe_combos:
            merged = {**embedded, **recipe}
            trials.append((dict(est), merged))
    return trials


def _expand_randomized_trials(
    *,
    param_distributions: dict[str, Any] | None,
    recipe_distributions: dict[str, Any] | None,
    n_iter: int,
    random_state: int | None,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if (param_distributions is None or not param_distributions) and (
        recipe_distributions is None or not recipe_distributions
    ):
        raise ValidationError("Provide a non-empty param_distributions and/or recipe_distributions")
    if recipe_distributions:
        unknown = sorted(set(recipe_distributions) - SAFE_RECIPE_KNOBS)
        if unknown:
            raise ValidationError(
                f"Unsupported recipe_distributions knobs: {unknown}. "
                f"Allowed: {sorted(SAFE_RECIPE_KNOBS)}"
            )
    # Sample a joint space so n_iter bounds total trials.
    joint: dict[str, Any] = {}
    if param_distributions:
        joint.update(param_distributions)
    if recipe_distributions:
        for key, values in recipe_distributions.items():
            joint[f"recipe__{key}"] = values
    sampler = ParameterSampler(joint, n_iter=n_iter, random_state=random_state)
    return [_split_trial_params(dict(params)) for params in sampler]


def _resolve_splitter(
    *,
    dataset: Dataset,
    split_plan: SplitPlan,
    y_train: pd.Series,
    cv: int | Any,
    cv_strategy: CvStrategy,
    groups: pd.Series | None,
    task: Literal["classification", "regression"],
) -> tuple[pd.Series | None, str, Any, np.ndarray | None]:
    if not isinstance(cv, int):
        splitter = check_cv(cv, y=y_train, classifier=task == "classification")
        return groups, type(splitter).__name__, splitter, None

    if cv < 2:
        raise ValidationError("cv must be an integer >= 2 or a CV splitter")

    strategy = cv_strategy
    if strategy == "auto":
        if groups is not None or dataset.role_columns(ColumnRole.GROUP):
            strategy = "stratified_group" if task == "classification" else "group"
        elif dataset.role_columns(ColumnRole.TIME):
            strategy = "time"
        elif task == "classification":
            strategy = "stratified"
        else:
            strategy = "kfold"

    group_values = groups
    if strategy in {"group", "stratified_group"}:
        if group_values is None:
            group_cols = dataset.role_columns(ColumnRole.GROUP)
            if not group_cols:
                raise ValidationError(
                    "Group CV requires a column with role 'group' or an explicit groups series"
                )
            if len(group_cols) != 1:
                raise ValidationError("Group CV expects exactly one group-role column")
            train_frame = frame_for_partition(dataset, split_plan, "train")
            group_values = train_frame[group_cols[0]]
        n_groups = int(pd.Series(group_values).nunique(dropna=False))
        if n_groups < cv:
            raise ValidationError(
                f"Need at least {cv} distinct groups for {cv}-fold group CV; found {n_groups}"
            )
        if strategy == "stratified_group":
            if task != "classification":
                raise ValidationError("stratified_group CV is only valid for classification")
            return (
                group_values,
                "stratified_group",
                StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=42),
                None,
            )
        return group_values, "group", GroupKFold(n_splits=cv), None

    if strategy == "time":
        time_cols = dataset.role_columns(ColumnRole.TIME)
        if not time_cols:
            raise ValidationError("Time CV requires a column with role 'time'")
        train_frame = frame_for_partition(dataset, split_plan, "train")
        stamps = pd.to_datetime(train_frame[time_cols[0]], errors="coerce")
        if stamps.isna().any():
            raise ValidationError("Time CV requires parseable values in the time-role column")
        order = np.argsort(stamps.to_numpy(), kind="mergesort")
        return None, "time", TimeSeriesSplit(n_splits=cv), order

    if strategy == "stratified":
        if task != "classification":
            raise ValidationError("stratified CV is only valid for classification")
        return None, "stratified", StratifiedKFold(n_splits=cv, shuffle=True, random_state=42), None

    if strategy == "kfold":
        return None, "kfold", KFold(n_splits=cv, shuffle=True, random_state=42), None

    raise ValidationError(f"Unknown cv_strategy '{cv_strategy}'")


def _score_predictions(
    task: Literal["classification", "regression"],
    y_true: pd.Series,
    y_pred: Any,
    *,
    sample_weight: pd.Series | None = None,
) -> dict[str, float]:
    sw = None if sample_weight is None else sample_weight.to_numpy(dtype=float)
    if task == "regression":
        mse = float(mean_squared_error(y_true, y_pred, sample_weight=sw))
        return {
            "mae": float(mean_absolute_error(y_true, y_pred, sample_weight=sw)),
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(y_true, y_pred, sample_weight=sw)),
        }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred, sample_weight=sw)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred, sample_weight=sw)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0, sample_weight=sw)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0, sample_weight=sw)
        ),
    }


def _aggregate_metrics(
    rows: list[dict[str, float]],
) -> tuple[dict[str, float], dict[str, float]]:
    keys = sorted({key for row in rows for key in row})
    mean_metrics: dict[str, float] = {}
    std_metrics: dict[str, float] = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows if key in row], dtype=float)
        mean_metrics[key] = float(np.mean(values))
        std_metrics[key] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean_metrics, std_metrics


def _cv_limitations(
    *,
    session_preprocess_applied: bool,
    preprocess: PreprocessRecipe | None,
    strategy_name: str,
    n_folds: int,
) -> list[str]:
    tips = [
        f"Scores summarize {n_folds} folds drawn only from the train partition.",
        "The Session test partition is not used for fold membership or fold scoring.",
    ]
    if session_preprocess_applied:
        tips.append(
            "allow_session_global_preprocess=True: Session-global preprocess plans "
            "(impute/encode/scale/outliers/binning/feature_select/dates/text/reduce/"
            "resample) were fitted on the full train partition before CV, so "
            "fold-eval rows influenced those frozen statistics."
        )
        tips.append(
            "Session-global target encoding uses out-of-fold values on train, but still "
            "freezes full-train category maps before CV; prefer fold-local "
            "PreprocessRecipe(encode='target') on unpoisoned data when selection itself "
            "uses CV."
        )
        if preprocess is not None and not preprocess.is_empty():
            tips.append(
                "A fold-local PreprocessRecipe was also provided, but Session data was "
                "already transformed with train-global statistics: the recipe does not "
                "rebuild from raw/unpoisoned rows."
            )
    if preprocess is not None and not preprocess.is_empty() and not session_preprocess_applied:
        tips.append(
            "Fold-local PreprocessRecipe statistics were refit on each fold's training rows only."
        )
        if preprocess.encode == "target":
            tips.append(
                "Fold-local target encoding fits category means on fold-train labels only; "
                "fold-eval rows receive those frozen means and never contribute label stats."
            )
        if preprocess.encode == "infrequent":
            tips.append("Infrequent-level maps are learned from fold-train category counts only.")
        if preprocess.select is not None:
            tips.append(
                "Fold-local feature selection fits on fold-train transformed features only "
                "(variance, univariate, and model-based SelectFromModel)."
            )
        if preprocess.outliers is not None:
            tips.append(
                "Fold-local outlier fences are fit on fold-train only and applied to "
                "fold-eval with frozen bounds (detect/cap; no row drops inside CV)."
            )
        if preprocess.binning is not None:
            tips.append("Fold-local bin edges are learned from fold-train finite values only.")
        if preprocess.dates:
            tips.append(
                "Fold-local date expansion is row-wise deterministic; including it in "
                "the recipe avoids Session-global extract_dates before CV."
            )
        if preprocess.text is not None:
            tips.append(
                "Fold-local text vectorizers fit vocabulary/IDF on fold-train documents "
                "only; fold-eval rows use the frozen mapping."
            )
        if preprocess.reduce is not None:
            tips.append(
                "Fold-local PCA fits the rotation on fold-train numeric columns only; "
                "fold-eval rows use the frozen components."
            )
    if strategy_name == "time":
        tips.append(
            "Time-series folds respect row order by the time-role column within train; "
            "they do not invent a calendar-aware embargo."
        )
    return tips


def _cv_interpretation(
    *,
    metric: str,
    mean_metrics: dict[str, float],
    std_metrics: dict[str, float],
    n_folds: int,
    task: str,
) -> list[str]:
    mean = mean_metrics.get(metric)
    std = std_metrics.get(metric, 0.0)
    lines = [
        (
            f"Observed mean {metric}={mean:.6f} with fold std={std:.6f} "
            f"across {n_folds} folds ({task})."
        )
    ]
    if mean is not None and std > 0 and abs(mean) > 1e-12 and (std / abs(mean)) > 0.2:
        lines.append(
            "Fold coefficient of variation exceeds 0.2: estimate instability is material."
        )
    if task == "classification" and "balanced_accuracy" in mean_metrics:
        gap = mean_metrics.get("accuracy", 0.0) - mean_metrics["balanced_accuracy"]
        if gap > 0.05:
            lines.append(
                f"Mean accuracy exceeds mean balanced accuracy by {gap:.3f}; "
                "class imbalance may dominate the primary score."
            )
    return lines


def _cv_recommendations(
    *,
    metric: str,
    mean_metrics: dict[str, float],
    std_metrics: dict[str, float],
    held_out: list[str],
    session_preprocess_applied: bool,
    preprocess: PreprocessRecipe | None,
) -> list[str]:
    tips = [
        f"Use mean±std of '{metric}' for model comparison; report fold count and population=train.",
        f"After selection, evaluate once on {held_out[0]} (held out from CV).",
    ]
    std = std_metrics.get(metric, 0.0)
    if std > 0.05:
        tips.append("Consider more folds, grouped/time-aware splits, or a simpler estimator.")
    if session_preprocess_applied:
        tips.append(
            "This run used allow_session_global_preprocess=True. For honest selection, "
            "re-ingest or reattach unpoisoned data, pass preprocess=PreprocessRecipe(...), "
            "and avoid Session-global impute/encode/scale/select/outliers/text/reduce "
            "before CV."
        )
    elif preprocess is not None:
        tips.append(
            "Refit the winning preprocess+estimator on the full train partition before deployment."
        )
        if (
            preprocess.encode in {"target", "infrequent"}
            or preprocess.select is not None
            or preprocess.text is not None
            or preprocess.reduce is not None
        ):
            tips.append(
                "Fold-local encode/select/text/reduce inside CV is leakage-safer than "
                "Session-global equivalents for model selection; persist the final "
                "Session plans via save_pipeline after the confirmed refit."
            )
    return tips
