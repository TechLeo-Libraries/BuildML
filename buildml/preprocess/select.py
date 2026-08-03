"""Keep the columns that carry signal and drop the ones that only add noise.

More features is not better. Each one adds a dimension the model must estimate
in, and with limited rows that means more parameters fitted on the same
evidence — the model starts learning the quirks of your training set instead of
the pattern underneath. Irrelevant columns also give a tree somewhere to make a
spurious split, and they slow everything down.

Three strategies are available, in increasing order of cost and of how much
they know about your target.

**Variance** drops columns that barely change. A column that is the same value
for 99% of rows cannot help distinguish those rows, whatever the target is.
This is the cheapest check and the only one that does not look at the target at
all, so it is a safe first pass.

**Univariate** scores each column against the target independently and keeps
the top *k*. Fast and often effective, but it judges each column alone: it will
discard a feature that is useless by itself and decisive in combination with
another, and it will keep several columns that all say the same thing.

**Model-based** fits an estimator and keeps the features it relied on. This one
sees interactions and redundancy, which is what makes it the most accurate — at
the cost of fitting a model, and of inheriting that model's biases about what
matters.

All three learn from training rows only. Choosing which columns to keep by
looking at test performance is leakage of exactly the kind that produces a
model which scores beautifully and then fails.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    VarianceThreshold,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.linear_model import LogisticRegression

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.core.validation import validate_column_names
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition, frame_for_partition
from buildml.explain.schemas import (
    Action,
    ActionPriority,
    Evidence,
    EvidenceKind,
    Finding,
    FindingSeverity,
    Recommendation,
)
from buildml.preprocess.columns import select_columns
from buildml.preprocess.result import PreprocessResult

SelectStrategy = Literal["variance", "univariate", "model"]


@dataclass(slots=True)
class FeatureSelectPlan:
    """Which columns survived selection, which did not, and why.

    Attributes
    ----------
    strategy:
        Which selection method produced this — ``'variance'``,
        ``'univariate'``, or ``'model'``.
    selected_features_:
        The features kept, and therefore the exact set the model will be
        trained on and will expect at inference time.
    dropped_features_:
        The features removed. Read this before trusting the result: a column
        you know to be important appearing here usually means it needed
        encoding, scaling, or a different strategy rather than that it is
        genuinely useless.
    scores_:
        The score each candidate received. Interpretation depends on the
        strategy — variance for ``'variance'``, the test statistic for
        ``'univariate'``, the estimator's importance for ``'model'``. Look at
        the gap between the last kept and first dropped feature; if it is
        tiny, the cutoff is arbitrary.
    threshold:
        The variance cutoff, when the strategy is ``'variance'``.
    k:
        How many features were requested, when the strategy is
        ``'univariate'``.
    score_func:
        The scoring function used for univariate selection.
    estimator_name:
        The estimator whose importances drove model-based selection.
    """

    strategy: SelectStrategy
    selected_features_: tuple[str, ...]
    dropped_features_: tuple[str, ...]
    scores_: dict[str, float] = field(default_factory=dict)
    threshold: float | None = None
    k: int | None = None
    score_func: str | None = None
    estimator_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the selection outcome as plain JSON-safe values.

        Used by model cards and checkpoints. Worth recording: which features a
        model was allowed to see is part of what makes a result reproducible.

        Returns
        -------
        dict
            Every attribute in plain-data form, including the full score map.
        """
        return {
            "strategy": self.strategy,
            "selected_features_": list(self.selected_features_),
            "dropped_features_": list(self.dropped_features_),
            "scores_": dict(self.scores_),
            "threshold": self.threshold,
            "k": self.k,
            "score_func": self.score_func,
            "estimator_name": self.estimator_name,
        }


def fit_feature_selector(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    strategy: SelectStrategy = "variance",
    columns: list[str] | None = None,
    threshold: float = 0.0,
    k: int = 10,
    score_func: Literal["f_classif", "f_regression", "mutual_info"] = "f_classif",
    estimator: Any | None = None,
) -> FeatureSelectPlan:
    """Decide which feature columns to keep, judging only by the training rows.

    Nothing is removed here — pass the plan to
    :func:`transform_feature_selector` to apply it. The separation matters:
    you should read ``dropped_features_`` before acting on it.

    Parameters
    ----------
    dataset:
        The full dataset. Only rows the split assigns to ``train`` are read.
    split_plan:
        The split defining the training rows. Required, because a feature set
        chosen with any knowledge of the test rows makes the eventual test
        score meaningless.
    strategy:
        ``'variance'``, ``'univariate'``, or ``'model'``. See the module
        docstring for what each one can and cannot see.
    columns:
        Candidate features. Defaults to the ``feature``-role columns. Columns
        holding protected roles are never dropped regardless of score, so your
        target and identifiers survive.
    threshold:
        Minimum variance a column must have to survive, for the ``'variance'``
        strategy. The default of 0.0 removes only genuinely constant columns,
        which is nearly always safe. Raising it starts removing
        low-variability columns, and since variance depends on scale, a raised
        threshold behaves very differently before and after scaling.
    k:
        How many top-scoring features to keep, for the ``'univariate'``
        strategy. Capped at the number of candidates.
    score_func:
        How univariate relevance is measured. ``'f_classif'`` and
        ``'f_regression'`` test for a *linear* relationship and are fast but
        blind to curved ones. ``'mutual_info'`` detects any dependence
        including non-linear, at more computational cost and with more variance
        in its estimates — usually the better choice when you have the rows to
        support it.
    estimator:
        The model whose importances drive ``'model'`` selection. Left as
        ``None``, a logistic regression is used for classification and a random
        forest for regression. Supply your own when you want the selection to
        reflect the model you actually intend to deploy — a linear model and a
        gradient booster disagree substantially about which features matter.

    Returns
    -------
    FeatureSelectPlan
        The kept and dropped sets with their scores.

    Raises
    ------
    ~buildml.core.errors.LeakageError
        No split plan was supplied.
    ~buildml.core.errors.ValidationError
        The strategy is unrecognised, the training features contain missing
        values, or any candidate is non-numeric. Both of the latter are
        actionable: impute first, and encode categoricals first.

    Notes
    -----
    **Order in the pipeline matters.** Selection needs complete numeric data,
    so it comes after imputation and encoding. It generally belongs before
    scaling, since a variance threshold means something different once every
    column has unit variance.

    **Univariate selection misses interactions.** Two columns that predict
    nothing alone but everything together will both be dropped. If you suspect
    that structure, use the model strategy.

    **Selection is a modelling decision, not a cleanup step.** It changes what
    the model can learn. When you tune the feature count, tune it inside cross
    validation via :class:`~buildml.preprocess.fold.PreprocessRecipe` rather
    than choosing it once against a holdout.

    Examples
    --------
    >>> plan = fit_feature_selector(  # doctest: +SKIP
    ...     dataset, split_plan, strategy="univariate", k=20
    ... )
    >>> len(plan.selected_features_)  # doctest: +SKIP
    20

    See Also
    --------
    transform_feature_selector : Applies the plan produced here.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None
    if strategy not in {"variance", "univariate", "model"}:
        raise ValidationError(f"Unsupported feature selection strategy '{strategy}'")

    train = frame_for_partition(dataset, split_plan, "train")
    feature_cols = _resolve_feature_columns(dataset, train, columns)
    protected = _protected_columns(dataset)
    x = train[feature_cols]
    if x.isna().any().any():
        raise ValidationError(
            "Feature selection requires non-null train features. Call session.impute(...) first."
        )
    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(x[c])]
    if non_numeric:
        raise ValidationError(
            "Feature selection currently requires numeric features; "
            f"encode categoricals first. Non-numeric: {non_numeric[:12]}"
        )

    scores: dict[str, float] = {}
    selected: list[str]
    score_name: str | None = None
    estimator_name: str | None = None
    used_threshold: float | None = threshold if strategy == "variance" else None
    used_k: int | None = k if strategy == "univariate" else None

    if strategy == "variance":
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(x)
        support = selector.get_support()
        variances = np.asarray(selector.variances_, dtype=float)
        selected = [col for col, keep in zip(feature_cols, support, strict=True) if keep]
        scores = {col: float(var) for col, var in zip(feature_cols, variances, strict=True)}
    elif strategy == "univariate":
        target = dataset.require_target()
        y = train[target]
        task = _infer_task(y)
        func, score_name = _resolve_score_func(score_func, task)
        k_eff = min(k, len(feature_cols))
        if k_eff < 1:
            raise ValidationError("k must be at least 1")
        used_k = k_eff
        selector = SelectKBest(score_func=func, k=k_eff)
        selector.fit(x, y)
        support = selector.get_support()
        raw_scores = np.asarray(selector.scores_, dtype=float)
        selected = [col for col, keep in zip(feature_cols, support, strict=True) if keep]
        scores = {
            col: float(score) if np.isfinite(score) else 0.0
            for col, score in zip(feature_cols, raw_scores, strict=True)
        }
    else:
        target = dataset.require_target()
        y = train[target]
        task = _infer_task(y)
        model = estimator
        if model is None:
            model = (
                LogisticRegression(max_iter=500)
                if task == "classification"
                else RandomForestRegressor(n_estimators=50, random_state=0)
            )
            if task == "classification" and hasattr(model, "fit"):
                # Prefer a tree fallback when linear separability is unlikely / multiclass heavy.
                n_classes = int(pd.Series(y).nunique())
                if n_classes > 2:
                    model = RandomForestClassifier(n_estimators=50, random_state=0)
        estimator_name = type(model).__name__
        model.fit(x, y)
        selector = SelectFromModel(model, prefit=True)
        support = selector.get_support()
        selected = [col for col, keep in zip(feature_cols, support, strict=True) if keep]
        if hasattr(model, "feature_importances_"):
            importances = np.asarray(model.feature_importances_, dtype=float)
            scores = {
                col: float(value) for col, value in zip(feature_cols, importances, strict=True)
            }
        elif hasattr(model, "coef_"):
            coef = np.asarray(model.coef_, dtype=float)
            if coef.ndim > 1:
                coef = np.mean(np.abs(coef), axis=0)
            else:
                coef = np.abs(coef)
            scores = {col: float(value) for col, value in zip(feature_cols, coef, strict=True)}
        if not selected:
            # Keep at least the top-scoring feature to avoid empty matrices.
            if scores:
                top = max(scores, key=scores.get)
                selected = [top]
            else:
                selected = feature_cols[:1]

    dropped = [c for c in feature_cols if c not in set(selected)]
    # Always retain protected non-feature columns via transform path.
    _ = protected
    return FeatureSelectPlan(
        strategy=strategy,
        selected_features_=tuple(selected),
        dropped_features_=tuple(dropped),
        scores_=scores,
        threshold=used_threshold,
        k=used_k,
        score_func=score_name,
        estimator_name=estimator_name,
    )


def transform_feature_selector(
    dataset: Dataset,
    plan: FeatureSelectPlan,
) -> tuple[Dataset, PreprocessResult]:
    """Narrow the dataset to the features the plan selected.

    Removes the dropped columns from every partition, so training and test rows
    carry the same feature set. Columns holding protected roles — target, id,
    group, time, weight — are retained regardless of the plan, since dropping
    them would break the split and the evaluation.

    Parameters
    ----------
    dataset:
        The dataset to narrow.
    plan:
        A plan from :func:`fit_feature_selector`.

    Returns
    -------
    tuple of (~buildml.data.dataset.Dataset, ~buildml.preprocess.result.PreprocessResult)
        The narrowed dataset, and a narrated record of what was kept, what was
        dropped, and how close to the cutoff the marginal features fell.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        A feature the plan selected is missing from the dataset.

    Notes
    -----
    The dropped columns are gone from the returned dataset, so widen by
    re-running selection from the original data rather than by undoing this.
    The plan itself is preserved, which is what lets the same narrowing be
    replayed at inference time.

    See Also
    --------
    fit_feature_selector : Produces the plan this consumes.
    """
    keep = list(plan.selected_features_)
    protected = _protected_columns(dataset)
    for column in protected:
        if column not in keep and column in dataset.columns:
            keep.append(column)
    missing = [c for c in keep if c not in dataset.columns]
    if missing:
        raise ValidationError(f"Feature selection keep-list missing from dataset: {missing}")
    if not plan.selected_features_:
        raise ValidationError("Feature selection plan retained no features")

    new_dataset = select_columns(dataset, keep)
    return new_dataset, _build_result(plan)


def _protected_columns(dataset: Dataset) -> list[str]:
    from buildml.preprocess.columns import protected_role_columns

    return protected_role_columns(dataset)


def _resolve_feature_columns(
    dataset: Dataset,
    train: pd.DataFrame,
    columns: list[str] | None,
) -> list[str]:
    if columns is not None:
        names = validate_column_names(columns, dataset.columns)
        protected = set(_protected_columns(dataset))
        return [name for name in names if name not in protected]
    feature_roles = dataset.role_columns(ColumnRole.FEATURE)
    if feature_roles:
        return [str(c) for c in feature_roles if c in train.columns]
    protected = set(_protected_columns(dataset))
    return [str(c) for c in train.columns if c not in protected]


def _infer_task(y: pd.Series) -> Literal["classification", "regression"]:
    if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > 20:
        return "regression"
    return "classification"


def _resolve_score_func(
    name: str,
    task: Literal["classification", "regression"],
) -> tuple[Any, str]:
    if name == "mutual_info":
        if task == "classification":
            return mutual_info_classif, "mutual_info_classif"
        return mutual_info_regression, "mutual_info_regression"
    if name == "f_regression" or (name == "f_classif" and task == "regression"):
        return f_regression, "f_regression"
    return f_classif, "f_classif"


def _build_result(plan: FeatureSelectPlan) -> PreprocessResult:
    evidence = [
        Evidence(
            key="select_features.scores",
            kind=EvidenceKind.METRIC,
            summary="Train-only feature selection scores and retained columns.",
            value={
                "strategy": plan.strategy,
                "selected": list(plan.selected_features_),
                "dropped": list(plan.dropped_features_),
                "scores": dict(plan.scores_),
            },
            source="train.feature_selection",
            limitations=(
                "Univariate and model-based scores are heuristic; interactions may be missed.",
            ),
        )
    ]
    findings = [
        Finding(
            key="select_features.applied",
            title="Feature subset selected on train",
            detail=(
                f"Strategy '{plan.strategy}' retained {len(plan.selected_features_)} "
                f"feature(s) and dropped {len(plan.dropped_features_)}."
            ),
            severity=FindingSeverity.INFO if plan.dropped_features_ else FindingSeverity.LOW,
            evidence=tuple(evidence),
            affected_columns=plan.dropped_features_,
        )
    ]
    recommendations = [
        Recommendation(
            key="select_features.recheck-holdout",
            title="Confirm holdout metric after selection",
            rationale=(
                "Selection is fit on train. Re-evaluate on validation/test with the "
                "frozen feature subset before claiming improvement."
            ),
            priority=ActionPriority.NEXT,
            action=Action(
                key="select_features.eval-action",
                label="Session.evaluate(partition='validation')",
                operation="evaluate",
                parameters={"partition": "validation"},
            ),
            based_on=("select_features.applied",),
            caveats=("Selection inside nested CV is safer when the subset itself is tuned.",),
        )
    ]
    return PreprocessResult(
        operation="select_features",
        plan=plan.to_dict(),
        evidence=evidence,
        findings=findings,
        interpretation=[
            f"Kept {len(plan.selected_features_)} feature(s) via '{plan.strategy}'.",
            "Target/id/group/time/weight columns are preserved when present.",
        ],
        limitations=[
            "Scores are computed on train only and can overfit noisy labels.",
            "Variance thresholds ignore predictive relevance.",
            "Model-based selection reflects the chosen estimator's inductive bias.",
        ],
        recommendations=recommendations,
        methods=[
            f"Strategy={plan.strategy}.",
            (
                f"Variance threshold={plan.threshold}."
                if plan.strategy == "variance"
                else (
                    f"Univariate score_func={plan.score_func}, k={plan.k}."
                    if plan.strategy == "univariate"
                    else f"Model-based estimator={plan.estimator_name}."
                )
            ),
        ],
    )
