"""Search Torch hyperparameters without the search inflating the score.

Trying many configurations and reporting the best one's score overstates it,
and by more than most people expect. With enough configurations, one of them
will fit the evaluation rows well by luck, and that luck does not transfer. The
selected score measures the selection as much as the model.

Nested cross-validation is the standard fix, and :func:`nested_cv_torch`
implements it. An outer loop holds out a fold; the entire search runs inside on
the remaining rows; the winner is scored once on the held-out fold. The outer
score never influenced the choice, so it stays honest. The cost is that the
search runs once per outer fold, which multiplies the work.

:func:`search_torch` is the cheaper option: a single search with no outer loop.
It finds good hyperparameters at a fraction of the cost, and its best score is
optimistic. Use it to choose, then get your estimate from a partition the search
never touched.

Both refuse to touch Session validation and test partitions, verify that
training and evaluation indices do not overlap, and fit normalisation statistics
fold-locally.

See Also
--------
buildml.dl.cv.cross_validate_torch : Evaluation without selection.
buildml.model.tuning : The classical equivalent.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from buildml.core.errors import LeakageError, ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.dl.cv import TorchFoldScore, _eval_accuracy_or_mse, _make_fold_indices, _partition_xy
from buildml.dl.dataset import arrays_to_tensor_dataset, infer_task, resolve_feature_target
from buildml.dl.extras import require_torch
from buildml.dl.labels import encode_class_targets, fit_class_labels, n_classes_from_labels
from buildml.dl.models import build_tabular_mlp
from buildml.dl.transforms import apply_standardize, fit_standardize
from buildml.dl.types import FeatureContract, TrainConfig

ModuleFactory = Callable[[int, str, int, Mapping[str, Any]], Any]
SearchMethod = Literal["grid", "randomized", "auto"]


@dataclass(slots=True)
class TorchSearchTrial:
    """One configuration and how it scored across the folds that tested it.

    Attributes
    ----------
    params:
        The hyperparameters tried.
    mean_metrics:
        Average of each metric across the folds.
    std_metrics:
        Spread across folds. Two configurations with similar means but very
        different spreads are not equally good, and the ranking cannot see
        that.
    n_inner_folds:
        How many folds contributed.
    rank_metric:
        Which metric decided the ordering.
    rank_value:
        This configuration's value for that metric.

    Notes
    -----
    **Compare ``std_metrics`` between the top few trials before trusting the
    ranking.** When the gap between first and second is smaller than either
    one's spread, the ordering is noise, and picking the simpler or cheaper
    configuration is the better call.

    See Also
    --------
    search_torch : Produces these.
    """

    params: dict[str, Any]
    mean_metrics: dict[str, float]
    std_metrics: dict[str, float]
    n_inner_folds: int
    rank_metric: str
    rank_value: float

    def to_dict(self) -> dict[str, Any]:
        """Return this trial as JSON-safe values.

        Includes the per-fold spread alongside the mean, so a recorded search
        can still be second-guessed later.

        Returns
        -------
        dict
            Parameters, mean and standard-deviation metrics, fold count,
            ranking metric, and ranking value.
        """
        return {
            "params": dict(self.params),
            "mean_metrics": dict(self.mean_metrics),
            "std_metrics": dict(self.std_metrics),
            "n_inner_folds": self.n_inner_folds,
            "rank_metric": self.rank_metric,
            "rank_value": self.rank_value,
        }


@dataclass(slots=True)
class TorchOuterFoldResult:
    """One outer fold: what the inner search chose, and how it then scored.

    Attributes
    ----------
    fold:
        Zero-based outer fold index.
    train_size, eval_size:
        Row counts. The training side ran the whole inner search; the
        evaluation side was scored once.
    best_params:
        What the inner search selected on this fold's training rows.
    inner_best:
        The winning trial's full inner-fold evidence.
    outer_metrics:
        How the refit winner scored on the held-out rows. **This** is the
        honest number.
    n_inner_trials:
        How many configurations the inner search compared.
    warnings:
        Anything notable about this fold.

    Notes
    -----
    **``outer_metrics`` will usually be worse than ``inner_best``, and that gap
    is the point.** The inner score was chosen as the maximum over many trials;
    the outer score was measured once on rows that had no influence on the
    choice. The difference is how much the selection was flattering itself.

    See Also
    --------
    nested_cv_torch : Produces these.
    """

    fold: int
    train_size: int
    eval_size: int
    best_params: dict[str, Any]
    inner_best: TorchSearchTrial | None
    outer_metrics: dict[str, float]
    n_inner_trials: int
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return this outer fold as JSON-safe values.

        Keeps the inner winner's evidence alongside the outer score, so the
        gap between selection and generalisation stays visible in the record.

        Returns
        -------
        dict
            Fold index, both row counts, selected parameters, the nested inner
            trial, outer metrics, trial count, and warnings.
        """
        return {
            "fold": self.fold,
            "train_size": self.train_size,
            "eval_size": self.eval_size,
            "best_params": dict(self.best_params),
            "inner_best": None if self.inner_best is None else self.inner_best.to_dict(),
            "outer_metrics": dict(self.outer_metrics),
            "n_inner_trials": self.n_inner_trials,
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class TorchNestedCVResult:
    """The nested cross-validation outcome: an honest estimate, plus what won.

    Attributes
    ----------
    n_outer_folds, n_inner_folds:
        Loop sizes. Total training runs is roughly outer times inner times the
        number of configurations.
    task:
        The resolved task.
    search_method:
        ``'grid'`` or ``'randomized'``.
    scoring_metric:
        What ranked the configurations.
    outer_folds:
        Each outer fold's record.
    mean_metrics:
        Average outer-fold performance. The headline, and the honest one.
    std_metrics:
        Spread across outer folds.
    best_params_per_fold:
        What each fold's inner search chose. Disagreement here is informative.
    consensus_params:
        A modal-and-median summary across the fold winners. A starting point,
        not a validated configuration.
    held_out_partitions:
        Session partitions kept out of the whole procedure.
    disclosures, limitations, warnings:
        How the run was arranged, what it does not establish, anything notable.
    config:
        The settings used.

    Notes
    -----
    **``mean_metrics`` estimates the whole procedure, not one model.** It says
    "searching this space this way, on data like this, yields models that score
    around here": which is the useful claim, and is not the same as a
    guarantee about any particular fitted model.

    **When ``best_params_per_fold`` disagrees, the search space is not clearly
    separated.** Different folds preferring different configurations means the
    differences between them are within noise, and ``consensus_params`` is
    papering over that rather than resolving it.

    See Also
    --------
    nested_cv_torch : Produces this.
    """

    n_outer_folds: int
    n_inner_folds: int
    task: str
    search_method: str
    scoring_metric: str
    outer_folds: tuple[TorchOuterFoldResult, ...]
    mean_metrics: dict[str, float]
    std_metrics: dict[str, float]
    best_params_per_fold: tuple[dict[str, Any], ...]
    consensus_params: dict[str, Any] | None
    held_out_partitions: tuple[str, ...] = ()
    disclosures: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the whole nested run as JSON-safe values.

        Includes every outer fold's record and its nested inner winner :
        keeping the per-fold disagreement visible, which a summary of the means
        alone would hide.

        Returns
        -------
        dict
            Loop sizes, task, method, scoring metric, outer fold records, mean
            and standard-deviation metrics, per-fold winners, consensus
            parameters, held-out partitions, the three prose lists, and config.
        """
        return {
            "n_outer_folds": self.n_outer_folds,
            "n_inner_folds": self.n_inner_folds,
            "task": self.task,
            "search_method": self.search_method,
            "scoring_metric": self.scoring_metric,
            "outer_folds": [f.to_dict() for f in self.outer_folds],
            "mean_metrics": dict(self.mean_metrics),
            "std_metrics": dict(self.std_metrics),
            "best_params_per_fold": [dict(p) for p in self.best_params_per_fold],
            "consensus_params": None
            if self.consensus_params is None
            else dict(self.consensus_params),
            "held_out_partitions": list(self.held_out_partitions),
            "disclosures": list(self.disclosures),
            "limitations": list(self.limitations),
            "warnings": list(self.warnings),
            "config": dict(self.config),
        }


@dataclass(slots=True)
class TorchSearchResult:
    """A hyperparameter search outcome, with the caveat that its score is high.

    Attributes
    ----------
    task:
        The resolved task.
    search_method:
        ``'grid'`` or ``'randomized'``.
    scoring_metric:
        What ranked the trials.
    n_folds:
        How many folds each trial was scored on.
    trials:
        Every configuration tried, ordered best first.
    best_params:
        The top configuration. This is what the search is for.
    best_metrics:
        Its cross-validated scores. **Optimistic**: the maximum over many
        trials, which is not the same as a typical outcome.
    held_out_partitions:
        Session partitions the search never touched. Score there for an honest
        number.
    disclosures, limitations, warnings:
        How the search ran, what it does not establish, anything notable.
    config:
        The settings used.

    Notes
    -----
    **Take ``best_params`` and leave ``best_metrics``.** The selected score is
    biased upward by the selection itself. Refit with the chosen configuration
    and evaluate on a partition that played no part in the search: that is the
    number to report.

    See Also
    --------
    search_torch : Produces this.
    nested_cv_torch : When you need the estimate from the same run.
    """

    task: str
    search_method: str
    scoring_metric: str
    n_folds: int
    trials: tuple[TorchSearchTrial, ...]
    best_params: dict[str, Any]
    best_metrics: dict[str, float]
    held_out_partitions: tuple[str, ...] = ()
    disclosures: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the search as JSON-safe values.

        Every trial is included, not just the winner: the runners-up are what
        tell you whether the winner won by a margin or by noise.

        Returns
        -------
        dict
            Task, method, scoring metric, fold count, all trials, best
            parameters and metrics, held-out partitions, the three prose lists,
            and config.
        """
        return {
            "task": self.task,
            "search_method": self.search_method,
            "scoring_metric": self.scoring_metric,
            "n_folds": self.n_folds,
            "trials": [t.to_dict() for t in self.trials],
            "best_params": dict(self.best_params),
            "best_metrics": dict(self.best_metrics),
            "held_out_partitions": list(self.held_out_partitions),
            "disclosures": list(self.disclosures),
            "limitations": list(self.limitations),
            "warnings": list(self.warnings),
            "config": dict(self.config),
        }


_SEARCHABLE = frozenset(
    {"learning_rate", "hidden", "dropout", "batch_size", "epochs", "weight_decay"}
)


def _default_factory(
    in_features: int, task: str, n_classes: int, params: Mapping[str, Any]
) -> Any:
    hidden = params.get("hidden", (32, 16))
    if isinstance(hidden, list):
        hidden = tuple(int(h) for h in hidden)
    elif isinstance(hidden, tuple):
        hidden = tuple(int(h) for h in hidden)
    else:
        hidden = (int(hidden),)
    return build_tabular_mlp(
        in_features,
        task=task,
        n_classes=max(n_classes, 2),
        hidden=hidden,
        dropout=float(params.get("dropout", 0.1)),
    )


def _expand_grid(param_grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    if not param_grid:
        raise ValidationError("param_grid must be a non-empty mapping of lists")
    keys = list(param_grid)
    unknown = [k for k in keys if k not in _SEARCHABLE]
    if unknown:
        raise ValidationError(
            f"Unsupported Torch search keys: {unknown}. Allowed: {sorted(_SEARCHABLE)}"
        )
    values = [list(param_grid[k]) for k in keys]
    if any(len(v) == 0 for v in values):
        raise ValidationError("param_grid values must be non-empty sequences")
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]


def _sample_distributions(
    param_distributions: Mapping[str, Any],
    *,
    n_iter: int,
    seed: int,
) -> list[dict[str, Any]]:
    if n_iter < 1:
        raise ValidationError("n_iter must be >= 1")
    if not param_distributions:
        raise ValidationError("param_distributions must be a non-empty mapping")
    unknown = [k for k in param_distributions if k not in _SEARCHABLE]
    if unknown:
        raise ValidationError(
            f"Unsupported Torch search keys: {unknown}. Allowed: {sorted(_SEARCHABLE)}"
        )
    rng = np.random.default_rng(seed)
    trials: list[dict[str, Any]] = []
    for _ in range(n_iter):
        params: dict[str, Any] = {}
        for key, dist in param_distributions.items():
            if isinstance(dist, (list, tuple)):
                if not dist:
                    raise ValidationError(f"Empty distribution for {key!r}")
                params[key] = dist[int(rng.integers(0, len(dist)))]
            elif hasattr(dist, "rvs"):
                # scipy.stats prefers an int seed / RandomState; Generator often fails.
                seed_i = int(rng.integers(0, 2**31 - 1))
                try:
                    params[key] = dist.rvs(random_state=seed_i)
                except TypeError:
                    params[key] = dist.rvs()
            else:
                raise ValidationError(
                    f"Distribution for {key!r} must be a sequence or scipy-like rvs object"
                )
        trials.append(params)
    return trials


def _resolve_method(
    *,
    inner_search: SearchMethod,
    param_grid: Mapping[str, Sequence[Any]] | None,
    param_distributions: Mapping[str, Any] | None,
) -> str:
    if param_grid is not None and param_distributions is not None:
        raise ValidationError("Provide at most one of param_grid or param_distributions")
    if param_grid is None and param_distributions is None:
        raise ValidationError(
            "Torch search requires param_grid or param_distributions "
            f"(allowed keys: {sorted(_SEARCHABLE)})"
        )
    if inner_search == "auto":
        return "grid" if param_grid is not None else "randomized"
    if inner_search == "grid" and param_grid is None:
        raise ValidationError("inner_search='grid' requires param_grid")
    if inner_search == "randomized" and param_distributions is None:
        raise ValidationError("inner_search='randomized' requires param_distributions")
    return inner_search


def _rank_metric_for_task(task: str, scoring_metric: str | None) -> str:
    if scoring_metric is not None:
        return scoring_metric
    return "accuracy" if task == "classification" else "mse"


def _is_better(value: float, best: float | None, *, maximize: bool) -> bool:
    if best is None or (isinstance(best, float) and math.isnan(best)):
        return True
    return value > best if maximize else value < best


def _train_universe_indices(
    dataset: Dataset,
    split_plan: SplitPlan | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return row indices eligible for CV / search and held-out partition names."""
    frame = dataset._ensure_pandas()
    n = len(frame)
    if split_plan is None:
        return np.arange(n), ()
    train_idx = np.asarray(list(split_plan.indices_for("train")), dtype=np.int64)
    if train_idx.size == 0:
        raise ValidationError("Train partition is empty; cannot run Torch search")
    held: list[str] = []
    for name in ("validation", "test"):
        idx = list(split_plan.indices_for(name))  # type: ignore[arg-type]
        if idx:
            held.append(name)
            overlap = set(train_idx.tolist()) & set(idx)
            if overlap:
                raise LeakageError(
                    f"Train and {name} partitions overlap; refusing Torch nested search"
                )
    return train_idx, tuple(held)


def _fit_score_params(
    *,
    frame: pd.DataFrame,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    feature_cols: Sequence[str],
    target: str,
    task: str,
    params: Mapping[str, Any],
    normalize: bool,
    device: str,
    seed: int,
    module_factory: ModuleFactory,
    base_epochs: int,
    base_batch_size: int,
    base_lr: float,
) -> tuple[dict[str, float], int, str]:
    from buildml.dl.results import LoaderReport, TorchLoaderBundle
    from buildml.dl.train import train_supervised_module

    torch = require_torch(feature="Torch nested search")
    x_train, y_train = _partition_xy(frame, train_idx, feature_cols, target)
    x_eval, y_eval = _partition_xy(frame, eval_idx, feature_cols, target)
    mean = std = None
    if normalize:
        mean, std = fit_standardize(x_train)
        x_train = apply_standardize(x_train, mean, std)
        x_eval = apply_standardize(x_eval, mean, std)

    if task == "classification":
        class_labels = fit_class_labels(y_train)
        n_classes = n_classes_from_labels(class_labels)
        y_train = encode_class_targets(y_train, class_labels).astype(np.float64, copy=False)
        y_eval = encode_class_targets(y_eval, class_labels).astype(np.float64, copy=False)
    else:
        class_labels = ()
        n_classes = 1
    module = module_factory(len(feature_cols), task, n_classes, params)
    epochs = int(params.get("epochs", base_epochs))
    batch_size = int(params.get("batch_size", base_batch_size))
    lr = float(params.get("learning_rate", base_lr))
    cfg = TrainConfig(
        epochs=max(1, epochs),
        learning_rate=lr,
        batch_size=max(1, batch_size),
        device=device,  # type: ignore[arg-type]
        normalize=normalize,
        seed=seed,
        mixed_precision=False,
    )
    contract = FeatureContract(
        feature_columns=tuple(feature_cols),
        target_column=target,
        task=task,  # type: ignore[arg-type]
        class_labels=class_labels,
        normalize_mean=None if mean is None else tuple(float(v) for v in mean),
        normalize_std=None if std is None else tuple(float(v) for v in std),
    )
    train_ds = arrays_to_tensor_dataset(x_train, y_train, task=task)
    eval_ds = arrays_to_tensor_dataset(x_eval, y_eval, task=task)
    generator = torch.Generator().manual_seed(int(seed))
    loaders = {
        "train": torch.utils.data.DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True, generator=generator
        ),
        "validation": torch.utils.data.DataLoader(
            eval_ds, batch_size=cfg.batch_size, shuffle=False
        ),
    }
    bundle = TorchLoaderBundle(
        loaders=loaders,
        contract=contract,
        report=LoaderReport(
            batch_size=cfg.batch_size,
            shuffle_train=True,
            normalize=normalize,
            feature_columns=contract.feature_columns,
            target_column=contract.target_column,
            task=task,  # type: ignore[arg-type]
            n_train=len(train_idx),
            n_validation=len(eval_idx),
            n_test=0,
            class_labels=contract.class_labels,
        ),
    )
    result = train_supervised_module(module, bundle, config=cfg)
    metrics = _eval_accuracy_or_mse(
        result.module, loaders["validation"], task=task, device=result.device.resolved
    )
    return metrics, result.n_epochs_ran, result.device.resolved


def _evaluate_trials_on_indices(
    *,
    frame: pd.DataFrame,
    universe: np.ndarray,
    feature_cols: Sequence[str],
    target: str,
    task: str,
    trial_params: Sequence[Mapping[str, Any]],
    n_folds: int,
    normalize: bool,
    device: str,
    seed: int,
    stratify: bool,
    module_factory: ModuleFactory,
    base_epochs: int,
    base_batch_size: int,
    base_lr: float,
    scoring_metric: str,
) -> list[TorchSearchTrial]:
    y_universe = frame.iloc[list(universe)][target].to_numpy()
    fold_pairs = _make_fold_indices(
        len(universe),
        n_folds=n_folds,
        seed=seed,
        y=y_universe if task == "classification" else None,
        stratify=stratify and task == "classification",
    )
    maximize = scoring_metric in {"accuracy", "f1", "roc_auc", "r2"}
    trials: list[TorchSearchTrial] = []
    for params in trial_params:
        fold_metrics: list[dict[str, float]] = []
        for fold_i, (tr_rel, va_rel) in enumerate(fold_pairs):
            train_idx = universe[tr_rel]
            eval_idx = universe[va_rel]
            metrics, _, _ = _fit_score_params(
                frame=frame,
                train_idx=train_idx,
                eval_idx=eval_idx,
                feature_cols=feature_cols,
                target=target,
                task=task,
                params=params,
                normalize=normalize,
                device=device,
                seed=seed + fold_i,
                module_factory=module_factory,
                base_epochs=base_epochs,
                base_batch_size=base_batch_size,
                base_lr=base_lr,
            )
            fold_metrics.append(metrics)
        keys = sorted({k for m in fold_metrics for k in m})
        mean_metrics = {k: float(np.mean([m[k] for m in fold_metrics if k in m])) for k in keys}
        std_metrics = {
            k: float(np.std([m[k] for m in fold_metrics if k in m], ddof=0)) for k in keys
        }
        if scoring_metric not in mean_metrics:
            raise ValidationError(
                f"scoring_metric {scoring_metric!r} is not in fold metrics "
                f"{sorted(mean_metrics)}. Use accuracy/loss for classification or mse/loss."
            )
        trials.append(
            TorchSearchTrial(
                params=dict(params),
                mean_metrics=mean_metrics,
                std_metrics=std_metrics,
                n_inner_folds=n_folds,
                rank_metric=scoring_metric,
                rank_value=float(mean_metrics[scoring_metric]),
            )
        )
    trials.sort(key=lambda t: t.rank_value, reverse=maximize)
    return trials


def _consensus_params(per_fold: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not per_fold:
        return None
    keys = sorted({k for p in per_fold for k in p})
    consensus: dict[str, Any] = {}
    for key in keys:
        values = [p[key] for p in per_fold if key in p]
        if not values:
            continue
        # Prefer modal value; for numeric floats take median.
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            consensus[key] = float(np.median(np.asarray(values, dtype=np.float64)))
        else:
            # stringify for hashing tuples/lists
            serialized = [repr(v) for v in values]
            mode = max(set(serialized), key=serialized.count)
            consensus[key] = values[serialized.index(mode)]
    return consensus


def search_torch(
    dataset: Dataset,
    *,
    split_plan: SplitPlan | None = None,
    param_grid: Mapping[str, Sequence[Any]] | None = None,
    param_distributions: Mapping[str, Any] | None = None,
    inner_search: SearchMethod = "auto",
    n_iter: int = 5,
    n_folds: int = 3,
    epochs: int = 2,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: Literal["cpu", "cuda", "mps", "auto"] = "auto",
    normalize: bool = True,
    seed: int = 0,
    stratify: bool = True,
    task: Literal["classification", "regression", "auto"] = "auto",
    scoring_metric: str | None = None,
    module_factory: ModuleFactory | None = None,
) -> TorchSearchResult:
    """Find good hyperparameters by cross-validating each candidate.

    Expands the search space into configurations, scores each one with
    cross-validation over the training rows, and ranks them. Session validation
    and test partitions are excluded entirely, so they remain available for an
    unbiased estimate afterwards.

    Parameters
    ----------
    dataset:
        The data, with roles and a numeric target.
    split_plan:
        When supplied, only the training partition is searched over. Without
        one, all rows are used.
    param_grid:
        Every combination to try, as name to list of values. Exhaustive.
    param_distributions:
        Distributions to sample from, as name to sequence or scipy-style object
        with ``rvs``. Supply exactly one of this and ``param_grid``.
    inner_search:
        ``'grid'``, ``'randomized'``, or ``'auto'`` to follow whichever space
        you provided.
    n_iter:
        How many samples to draw for randomized search.
    n_folds:
        Folds per configuration.
    epochs / batch_size / learning_rate:
        Defaults for anything not being searched.
    device:
        Where to train.
    normalize:
        Standardise features using each fold's training statistics.
    seed:
        Controls fold assignment and sampling.
    stratify:
        Preserve class balance across folds, for classification.
    task:
        ``'auto'`` to infer, or an explicit choice.
    scoring_metric:
        What to rank by. Defaults to accuracy for classification, MSE for
        regression.
    module_factory:
        Called as ``factory(in_features, task, n_classes, params)``. Defaults to
        a tabular MLP that reads ``hidden`` and ``dropout`` from the parameters.

    Returns
    -------
    TorchSearchResult
        Ranked trials, the winner, and honest limitations.

    Raises
    ------
    MissingExtraError
        If PyTorch is not installed.
    ValidationError
        If neither or both search spaces are given, if a key is outside the
        searchable set, if a space is empty, if the target is non-numeric, or
        if the scoring metric is not among the computed fold metrics.
    LeakageError
        If the training partition overlaps validation or test.

    Notes
    -----
    **The reported best score is optimistic and should not be published.** It
    is the maximum over many trials, and the maximum of a noisy sample sits
    above the mean by construction. Refit the winner and score on held-out data
    for the number you report.

    **Randomized search usually beats grid search per unit of compute.** A grid
    spends its budget varying every parameter equally, including the ones that
    do not matter; sampling explores more distinct values of the ones that do.

    **Only tabular MLP knobs are searchable by default**: learning rate,
    hidden widths, dropout, batch size, epochs, and weight decay. Anything else
    is rejected explicitly rather than silently ignored. Pass a
    ``module_factory`` to search over your own architecture.

    Examples
    --------
    Search, then evaluate honestly::

        result = search_torch(
            dataset,
            split_plan=split_plan,
            param_grid={"learning_rate": [1e-3, 1e-4], "dropout": [0.0, 0.2]},
            n_folds=3,
        )
        result.best_params      # use this
        result.best_metrics     # do not report this

    See Also
    --------
    nested_cv_torch : Selection and an honest estimate in one run.
    """
    # Validate search space before optional Torch import so AI dispatch /
    # missing-extra environments get a clear ValidationError first.
    method = _resolve_method(
        inner_search=inner_search,
        param_grid=param_grid,
        param_distributions=param_distributions,
    )
    require_torch(feature="Torch hyperparameter search")
    feature_cols, target = resolve_feature_target(dataset)
    frame = dataset._ensure_pandas().copy()
    y_all = frame[target]
    if not pd.api.types.is_numeric_dtype(y_all):
        raise ValidationError(
            f"Target '{target}' must be numeric for Torch search "
            "(encode class labels to integers first)."
        )
    resolved_task = infer_task(y_all, task)
    universe, held_out = _train_universe_indices(dataset, split_plan)
    metric = _rank_metric_for_task(resolved_task, scoring_metric)
    if method == "grid":
        assert param_grid is not None
        trial_params = _expand_grid(param_grid)
    else:
        assert param_distributions is not None
        trial_params = _sample_distributions(param_distributions, n_iter=n_iter, seed=seed)

    factory = module_factory or _default_factory
    trials = _evaluate_trials_on_indices(
        frame=frame,
        universe=universe,
        feature_cols=feature_cols,
        target=target,
        task=resolved_task,
        trial_params=trial_params,
        n_folds=n_folds,
        normalize=normalize,
        device=device,
        seed=seed,
        stratify=stratify,
        module_factory=factory,
        base_epochs=epochs,
        base_batch_size=batch_size,
        base_lr=learning_rate,
        scoring_metric=metric,
    )
    best = trials[0]
    disclosures = (
        f"Torch search ({method}) with n_folds={n_folds}, task={resolved_task}.",
        "Normalize statistics (when enabled) are fit per fold on train indices only.",
        "Session held-out partitions are never used for trial ranking."
        if held_out
        else "No Session validation/test split was attached; search used all rows.",
    )
    limitations = (
        "search_torch is inner-style selection, not a nested outer estimate. "
        "Use nested_cv_torch for an outer generalization estimate after search.",
        "Default search space covers tabular MLP knobs only "
        f"({', '.join(sorted(_SEARCHABLE))}); text/multimodal search is not covered here.",
        "Tiny epoch budgets produce noisy ranks: treat as alpha selection, not production AutoML.",
    )
    return TorchSearchResult(
        task=resolved_task,
        search_method=method,
        scoring_metric=metric,
        n_folds=n_folds,
        trials=tuple(trials),
        best_params=dict(best.params),
        best_metrics=dict(best.mean_metrics),
        held_out_partitions=held_out,
        disclosures=disclosures,
        limitations=limitations,
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "normalize": normalize,
            "seed": seed,
            "n_iter": n_iter if method == "randomized" else len(trial_params),
            "n_trials_evaluated": len(trials),
        },
    )


def nested_cv_torch(
    dataset: Dataset,
    *,
    split_plan: SplitPlan | None = None,
    param_grid: Mapping[str, Sequence[Any]] | None = None,
    param_distributions: Mapping[str, Any] | None = None,
    inner_search: SearchMethod = "auto",
    n_iter: int = 5,
    outer_cv: int = 3,
    inner_cv: int = 2,
    epochs: int = 2,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: Literal["cpu", "cuda", "mps", "auto"] = "auto",
    normalize: bool = True,
    seed: int = 0,
    stratify: bool = True,
    task: Literal["classification", "regression", "auto"] = "auto",
    scoring_metric: str | None = None,
    module_factory: ModuleFactory | None = None,
) -> TorchNestedCVResult:
    """Estimate performance honestly when hyperparameters are also being chosen.

    For each outer fold: hold out its evaluation rows, run a complete
    hyperparameter search on the remaining rows only, refit the winner on all of
    them, and score the held-out rows once. The outer scores never influenced
    any selection, which is what makes their average a fair estimate of the
    whole procedure.

    Parameters
    ----------
    dataset:
        The data, with roles and a numeric target.
    split_plan:
        When supplied, only the training partition participates. Validation and
        test stay entirely outside both loops.
    param_grid:
        Every combination to try, as name to list of values.
    param_distributions:
        Distributions to sample from. Supply exactly one of this and
        ``param_grid``.
    inner_search:
        ``'grid'``, ``'randomized'``, or ``'auto'``.
    n_iter:
        Samples per randomized search.
    outer_cv:
        Outer folds. Each produces one honest score.
    inner_cv:
        Folds inside each search.
    epochs / batch_size / learning_rate:
        Defaults for anything not being searched.
    device:
        Where to train.
    normalize:
        Standardise using each fold's own training statistics.
    seed:
        Controls fold assignment and sampling. Inner and outer loops use
        derived seeds so they do not partition identically.
    stratify:
        Preserve class balance across folds, for classification.
    task:
        ``'auto'`` to infer, or an explicit choice.
    scoring_metric:
        What ranks configurations in the inner loop.
    module_factory:
        Called as ``factory(in_features, task, n_classes, params)``.

    Returns
    -------
    TorchNestedCVResult
        Outer-fold scores with their mean and spread, each fold's chosen
        configuration, and a consensus summary.

    Raises
    ------
    MissingExtraError
        If PyTorch is not installed.
    ValidationError
        If the search space is missing, doubled, empty, or uses unsupported
        keys; or if the target is non-numeric.
    LeakageError
        If the training partition overlaps validation or test, or if outer
        training and evaluation indices overlap.

    Notes
    -----
    **This is expensive, and predictably so.** The total number of training runs
    is roughly ``outer_cv * inner_cv * n_configurations``, plus one refit per
    outer fold. Three outer folds, two inner folds, and eight configurations is
    fifty-one runs. Size the epoch budget accordingly, or use
    :func:`search_torch` and a held-out partition instead.

    **What you get is an estimate of the procedure, not of a model.** Nested CV
    answers "how well does searching this space and fitting the winner tend to
    work?" To deploy, refit once on all your training data with a chosen
    configuration.

    **``consensus_params`` is a convenience, not a validated choice.** It takes
    the median of numeric parameters and the mode of the rest across outer
    winners. No fold actually evaluated that exact combination.

    **Small epoch budgets make the inner ranking noisy**, which propagates: a
    fold that picks arbitrarily among indistinguishable configurations produces
    an outer score for an arbitrary choice.

    Examples
    --------
    Honest estimate with selection included::

        result = nested_cv_torch(
            dataset,
            split_plan=split_plan,
            param_grid={"learning_rate": [1e-3, 1e-4]},
            outer_cv=3,
            inner_cv=2,
            epochs=10,
        )
        result.mean_metrics        # honest
        result.best_params_per_fold  # disagreement means the space is flat

    See Also
    --------
    search_torch : Selection alone, much cheaper.
    buildml.dl.cv.cross_validate_torch : Evaluation without selection.
    """
    method = _resolve_method(
        inner_search=inner_search,
        param_grid=param_grid,
        param_distributions=param_distributions,
    )
    require_torch(feature="Torch nested CV")
    feature_cols, target = resolve_feature_target(dataset)
    frame = dataset._ensure_pandas().copy()
    y_all = frame[target]
    if not pd.api.types.is_numeric_dtype(y_all):
        raise ValidationError(
            f"Target '{target}' must be numeric for nested Torch CV "
            "(encode class labels to integers first)."
        )
    resolved_task = infer_task(y_all, task)
    universe, held_out = _train_universe_indices(dataset, split_plan)
    metric = _rank_metric_for_task(resolved_task, scoring_metric)
    maximize = metric in {"accuracy", "f1", "roc_auc", "r2"}
    factory = module_factory or _default_factory

    if method == "grid":
        assert param_grid is not None
        catalog = _expand_grid(param_grid)
    else:
        assert param_distributions is not None
        catalog = _sample_distributions(param_distributions, n_iter=n_iter, seed=seed)

    y_universe = frame.iloc[list(universe)][target].to_numpy()
    outer_pairs = _make_fold_indices(
        len(universe),
        n_folds=outer_cv,
        seed=seed,
        y=y_universe if resolved_task == "classification" else None,
        stratify=stratify and resolved_task == "classification",
    )

    outer_results: list[TorchOuterFoldResult] = []
    warnings: list[str] = []
    for fold_i, (tr_rel, ev_rel) in enumerate(outer_pairs):
        outer_train = universe[tr_rel]
        outer_eval = universe[ev_rel]
        fold_warnings: list[str] = []
        # Inner search must not see outer_eval rows.
        overlap = set(outer_train.tolist()) & set(outer_eval.tolist())
        if overlap:
            raise LeakageError("Outer train/eval indices overlap; refusing nested Torch CV")

        inner_trials = _evaluate_trials_on_indices(
            frame=frame,
            universe=outer_train,
            feature_cols=feature_cols,
            target=target,
            task=resolved_task,
            trial_params=catalog,
            n_folds=inner_cv,
            normalize=normalize,
            device=device,
            seed=seed + 1000 * (fold_i + 1),
            stratify=stratify,
            module_factory=factory,
            base_epochs=epochs,
            base_batch_size=batch_size,
            base_lr=learning_rate,
            scoring_metric=metric,
        )
        best_inner = inner_trials[0]
        outer_metrics, _, _ = _fit_score_params(
            frame=frame,
            train_idx=outer_train,
            eval_idx=outer_eval,
            feature_cols=feature_cols,
            target=target,
            task=resolved_task,
            params=best_inner.params,
            normalize=normalize,
            device=device,
            seed=seed + 10_000 + fold_i,
            module_factory=factory,
            base_epochs=epochs,
            base_batch_size=batch_size,
            base_lr=learning_rate,
        )
        outer_results.append(
            TorchOuterFoldResult(
                fold=fold_i,
                train_size=len(outer_train),
                eval_size=len(outer_eval),
                best_params=dict(best_inner.params),
                inner_best=best_inner,
                outer_metrics=outer_metrics,
                n_inner_trials=len(inner_trials),
                warnings=tuple(fold_warnings),
            )
        )

    metric_keys = sorted({k for fr in outer_results for k in fr.outer_metrics})
    mean_metrics = {
        k: float(np.mean([fr.outer_metrics[k] for fr in outer_results if k in fr.outer_metrics]))
        for k in metric_keys
    }
    std_metrics = {
        k: float(
            np.std([fr.outer_metrics[k] for fr in outer_results if k in fr.outer_metrics], ddof=0)
        )
        for k in metric_keys
    }
    best_params_per_fold = tuple(dict(fr.best_params) for fr in outer_results)
    consensus = _consensus_params(best_params_per_fold)
    # Prefer consensus ranked by how often it matched maximizing metric when present.
    _ = maximize  # documented in disclosures; consensus uses modal/median policy

    disclosures = (
        f"Nested Torch CV with outer_cv={outer_cv}, inner_cv={inner_cv}, "
        f"search={method}, task={resolved_task}.",
        "Inner search never sees outer-eval rows; normalize is fold-local.",
        (
            f"Session held-out partition(s) stay untouched: {', '.join(held_out)}."
            if held_out
            else "No Session validation/test split was attached; nested CV used all rows."
        ),
    )
    limitations = (
        "Outer mean_metrics estimate selection+fit honesty under the searched space; "
        "they are not a production SLA.",
        "consensus_params is a modal/median summary across outer winners: refit on full "
        "train with an explicit chosen config before deployment claims.",
        "Default factory searches tabular MLP knobs only; multimodal/text nested search "
        "requires a custom module_factory.",
        "Alpha epoch budgets make ranks noisy; increase epochs for more stable selection.",
    )
    return TorchNestedCVResult(
        n_outer_folds=outer_cv,
        n_inner_folds=inner_cv,
        task=resolved_task,
        search_method=method,
        scoring_metric=metric,
        outer_folds=tuple(outer_results),
        mean_metrics=mean_metrics,
        std_metrics=std_metrics,
        best_params_per_fold=best_params_per_fold,
        consensus_params=consensus,
        held_out_partitions=held_out,
        disclosures=disclosures,
        limitations=limitations,
        warnings=tuple(warnings),
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "normalize": normalize,
            "seed": seed,
            "n_catalog_trials": len(catalog),
            "scoring_metric": metric,
            "maximize": maximize,
        },
    )


# Re-export for type checkers that expect fold scores nearby.
__all__ = [
    "TorchFoldScore",
    "TorchNestedCVResult",
    "TorchOuterFoldResult",
    "TorchSearchResult",
    "TorchSearchTrial",
    "nested_cv_torch",
    "search_torch",
]
