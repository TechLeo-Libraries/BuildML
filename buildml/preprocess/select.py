"""Train-fitted feature selection utilities."""

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
    """Train-fitted feature selection plan."""

    strategy: SelectStrategy
    selected_features_: tuple[str, ...]
    dropped_features_: tuple[str, ...]
    scores_: dict[str, float] = field(default_factory=dict)
    threshold: float | None = None
    k: int | None = None
    score_func: str | None = None
    estimator_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
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
    """Learn which feature columns to keep using train rows only."""
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
    """Drop features not selected by a train-fitted plan; keep target/id/group/time/weight."""
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
