"""Train-fitted text feature utilities (count / hashing / TF-IDF)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)

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
from buildml.ingest.detect import schema_from_dataframe
from buildml.preprocess.result import PreprocessResult

TextMethod = Literal["count", "tfidf", "hashing"]


@dataclass(slots=True)
class TextFeaturePlan:
    """Train-fitted text vectorization plan.

    Hashing does not store a vocabulary; count/TF-IDF store sklearn vectorizers
    that are joblib-serializable for pipeline/checkpoint replay.
    """

    columns: tuple[str, ...]
    method: TextMethod
    max_features: int | None
    ngram_range: tuple[int, int]
    feature_names_: tuple[str, ...]
    vectorizers_: dict[str, Any] = field(repr=False)
    n_features_per_column_: dict[str, int] = field(default_factory=dict)
    drop_input_columns: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "method": self.method,
            "max_features": self.max_features,
            "ngram_range": list(self.ngram_range),
            "feature_names_": list(self.feature_names_),
            "n_features_per_column_": dict(self.n_features_per_column_),
            "drop_input_columns": self.drop_input_columns,
        }


def fit_text_features(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    columns: list[str] | None = None,
    method: TextMethod = "tfidf",
    max_features: int | None = 128,
    ngram_range: tuple[int, int] = (1, 1),
    drop_input_columns: bool = True,
) -> TextFeaturePlan:
    """Fit text vectorizers on the train partition only."""
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None
    if method not in {"count", "tfidf", "hashing"}:
        raise ValidationError(f"Unsupported text feature method '{method}'")
    if max_features is not None and max_features < 1:
        raise ValidationError("max_features must be >= 1 when provided")
    if len(ngram_range) != 2 or ngram_range[0] < 1 or ngram_range[1] < ngram_range[0]:
        raise ValidationError("ngram_range must be a (min_n, max_n) pair with 1 <= min_n <= max_n")

    train = frame_for_partition(dataset, split_plan, "train")
    cols = _resolve_text_columns(dataset, train, columns)
    vectorizers: dict[str, Any] = {}
    feature_names: list[str] = []
    n_per_col: dict[str, int] = {}

    for column in cols:
        documents = _as_text(train[column])
        vectorizer = _build_vectorizer(method, max_features=max_features, ngram_range=ngram_range)
        matrix = vectorizer.fit_transform(documents)
        n_features = int(matrix.shape[1])
        n_per_col[column] = n_features
        names = _feature_names_for_column(column, vectorizer, n_features, method)
        feature_names.extend(names)
        vectorizers[column] = vectorizer

    return TextFeaturePlan(
        columns=tuple(cols),
        method=method,
        max_features=max_features,
        ngram_range=ngram_range,
        feature_names_=tuple(feature_names),
        vectorizers_=vectorizers,
        n_features_per_column_=n_per_col,
        drop_input_columns=drop_input_columns,
    )


def transform_text_features(
    dataset: Dataset,
    plan: TextFeaturePlan,
) -> tuple[Dataset, PreprocessResult]:
    """Apply a train-fitted text plan to the full dataset."""
    missing = [c for c in plan.columns if c not in dataset.columns]
    if missing:
        raise ValidationError(f"Text plan columns missing from dataset: {missing}")

    frame = dataset._ensure_pandas().copy()
    roles = dict(dataset.roles)
    blocks: list[pd.DataFrame] = []
    for column in plan.columns:
        documents = _as_text(frame[column])
        matrix = plan.vectorizers_[column].transform(documents)
        dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
        n_features = plan.n_features_per_column_[column]
        names = _feature_names_for_column(
            column,
            plan.vectorizers_[column],
            n_features,
            plan.method,
        )
        block = pd.DataFrame(dense, columns=names, index=frame.index)
        blocks.append(block)
        roles.pop(column, None)
        for name in names:
            roles[name] = ColumnRole.FEATURE

    feature_frame = pd.concat(blocks, axis=1)
    if list(feature_frame.columns) != list(plan.feature_names_):
        # Keep a stable contract even if hashing name helpers change.
        feature_frame.columns = list(plan.feature_names_)
        for name in plan.feature_names_:
            roles[name] = ColumnRole.FEATURE

    if plan.drop_input_columns:
        frame = frame.drop(columns=list(plan.columns))
    out = pd.concat([frame, feature_frame], axis=1)
    new_dataset = Dataset.from_transformed(
        dataset,
        out,
        schema=schema_from_dataframe(out),
        roles=roles,
    )
    return new_dataset, _build_result(plan)


def _build_vectorizer(
    method: TextMethod,
    *,
    max_features: int | None,
    ngram_range: tuple[int, int],
) -> Any:
    common = {
        "ngram_range": ngram_range,
        "lowercase": True,
        "dtype": np.float64,
    }
    if method == "count":
        return CountVectorizer(max_features=max_features, **common)
    if method == "tfidf":
        return TfidfVectorizer(max_features=max_features, **common)
    # Hashing is stateless; max_features is the fixed output width.
    n_features = 128 if max_features is None else max_features
    return HashingVectorizer(n_features=n_features, alternate_sign=False, **common)


def _feature_names_for_column(
    column: str,
    vectorizer: Any,
    n_features: int,
    method: TextMethod,
) -> list[str]:
    if method != "hashing" and hasattr(vectorizer, "get_feature_names_out"):
        raw = [str(name) for name in vectorizer.get_feature_names_out()]
        return [f"{column}__{name}" for name in raw]
    return [f"{column}__hash_{i}" for i in range(n_features)]


def _as_text(series: pd.Series) -> list[str]:
    return series.astype("string").fillna("").tolist()


def _resolve_text_columns(
    dataset: Dataset,
    train: pd.DataFrame,
    columns: list[str] | None,
) -> list[str]:
    if columns is not None:
        names = validate_column_names(columns, dataset.columns)
    else:
        protected = {
            ColumnRole.TARGET,
            ColumnRole.ID,
            ColumnRole.GROUP,
            ColumnRole.TIME,
            ColumnRole.WEIGHT,
        }
        feature_roles = dataset.role_columns(ColumnRole.FEATURE)
        candidates = feature_roles or [
            str(c) for c in train.columns if dataset.roles.get(str(c)) not in protected
        ]
        names = [
            str(c)
            for c in candidates
            if c in train.columns
            and (
                pd.api.types.is_string_dtype(train[c])
                or pd.api.types.is_object_dtype(train[c])
            )
        ]
    if not names:
        raise ValidationError(
            "No text/object columns available for text_features. Pass columns=... explicitly."
        )
    for column in names:
        if pd.api.types.is_numeric_dtype(train[column]):
            raise ValidationError(
                f"Column '{column}' is numeric; text_features expects string-like values."
            )
    return names


def _build_result(plan: TextFeaturePlan) -> PreprocessResult:
    evidence = [
        Evidence(
            key="text_features.width",
            kind=EvidenceKind.METRIC,
            summary="Train-fitted text feature width by source column.",
            value={
                "method": plan.method,
                "n_features_per_column": dict(plan.n_features_per_column_),
                "total_features": len(plan.feature_names_),
                "max_features": plan.max_features,
                "ngram_range": list(plan.ngram_range),
            },
            source="train.text_features",
            limitations=(
                "Bag features are bag-of-n-grams style; they ignore word order beyond n-grams.",
            ),
        )
    ]
    findings = [
        Finding(
            key="text_features.applied",
            title="Text features fitted on train",
            detail=(
                f"Method '{plan.method}' expanded {len(plan.columns)} text column(s) "
                f"into {len(plan.feature_names_)} numeric feature(s)."
            ),
            severity=FindingSeverity.INFO,
            evidence=tuple(evidence),
            affected_columns=plan.columns,
        )
    ]
    recommendations = [
        Recommendation(
            key="text_features.review-width",
            title="Review text feature width before scale-sensitive models",
            rationale=(
                "Wide sparse-style expansions can dominate linear models. Confirm "
                "max_features and holdout metrics before claiming improvement."
            ),
            priority=ActionPriority.NEXT,
            action=Action(
                key="text_features.eval-action",
                label="Session.evaluate(partition='validation')",
                operation="evaluate",
                parameters={"partition": "validation"},
            ),
            based_on=("text_features.applied",),
            caveats=(
                "Hashing collisions are irreversible; "
                "prefer TF-IDF when interpretability matters.",
            ),
        )
    ]
    return PreprocessResult(
        operation="text_features",
        plan=plan.to_dict(),
        evidence=evidence,
        findings=findings,
        interpretation=[
            f"Expanded {len(plan.columns)} text column(s) with method '{plan.method}'.",
            "Vectorizers were fitted on train documents only.",
        ],
        limitations=[
            "Missing text becomes empty strings before vectorization.",
            "Hashing has no invertible vocabulary; feature names are positional hashes.",
            "Dense materialization can be wide; keep max_features modest for tabular models.",
        ],
        recommendations=recommendations,
        methods=[
            f"method={plan.method}",
            f"max_features={plan.max_features}",
            f"ngram_range={plan.ngram_range}",
        ],
    )
