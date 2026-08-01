"""Leakage tests for fold-local text features and PCA in PreprocessRecipe."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.preprocess import SAFE_RECIPE_KNOBS, PreprocessRecipe, build_fold_preprocessor


def test_fold_local_text_vocabulary_ignores_eval_tokens() -> None:
    x = pd.DataFrame(
        {
            "review": [
                "good product",
                "good value",
                "good quality",
                "good item",
                "bad product",
                "bad value",
                "unique_eval_only_token xyz",
            ],
            "x": [0.1, 0.2, 0.0, -0.1, 0.3, -0.2, 0.0],
        }
    )
    y = pd.Series([1, 1, 1, 1, 0, 0, 0])
    prep = build_fold_preprocessor(
        x.iloc[:6],
        PreprocessRecipe(
            impute=None,
            text="count",
            text_columns=("review",),
            text_max_features=32,
            text_drop_input=True,
        ),
        y.iloc[:6],
    )
    vocab_names = " ".join(prep._text_feature_names)
    assert "unique_eval_only_token" not in vocab_names
    out = prep.transform(x.iloc[[6]])
    assert all(pd.api.types.is_numeric_dtype(out[c]) for c in out.columns)
    assert "review" not in out.columns


def test_fold_local_pca_fit_ignores_eval_rows() -> None:
    # Fold-train lives on a 2-D plane; eval row has a large orthogonal spike.
    x = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0],
            "b": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 0.0],
            "c": [0.0, 0.1, -0.1, 0.05, -0.05, 0.0, 50.0],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1, 0])
    prep = build_fold_preprocessor(
        x.iloc[:6],
        PreprocessRecipe(
            impute=None,
            scale="standard",
            reduce="pca",
            reduce_n_components=2,
            reduce_prefix="pc",
            reduce_columns=("a", "b", "c"),
        ),
        y.iloc[:6],
    )
    assert prep._reducer is not None
    assert prep._reduce_feature_names == ["pc_1", "pc_2"]
    # Components are shaped by fold-train only (6 rows), not the eval spike.
    assert prep._reducer.n_samples_ == 6
    out = prep.transform(x.iloc[[6]])
    assert list(out.columns) == ["pc_1", "pc_2"]


def test_fold_local_text_and_pca_in_cv_score() -> None:
    rng = np.random.default_rng(11)
    n = 72
    labels = np.array([0, 1] * (n // 2))
    reviews = np.where(labels == 1, "good product quality", "bad product quality")
    # Inject a holdout-looking token on later rows only.
    reviews = reviews.astype(object)
    reviews[60:] = "holdout_only_token " + reviews[60:]
    frame = pd.DataFrame(
        {
            "review": reviews,
            "a": rng.normal(size=n) + labels * 0.8,
            "b": rng.normal(size=n),
            "y": labels,
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"review": "feature", "a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
    )
    test_idx = set(session.split_plan.test_indices)  # type: ignore[union-attr]
    recipe = PreprocessRecipe(
        impute="median",
        text="tfidf",
        text_columns=("review",),
        text_max_features=16,
        scale="standard",
        reduce="pca",
        reduce_n_components=3,
        reduce_prefix="pc",
    )
    assert "text" in recipe.to_dict()["fold_local_order"]
    assert "reduce" in recipe.to_dict()["fold_local_order"]
    assert any("custom_transform" in step for step in recipe.to_dict()["session_global_only"])
    result = session.cv_score(
        LogisticRegression(max_iter=400),
        cv=KFold(n_splits=3, shuffle=True, random_state=0),
        preprocess=recipe,
    )
    assert result.fold_preprocess is not None
    assert result.fold_preprocess["text"] == "tfidf"
    assert result.fold_preprocess["reduce"] == "pca"
    assert any("text vectorizers" in tip for tip in result.limitations)
    assert any("PCA fits the rotation" in tip for tip in result.limitations)
    assert set(session.split_plan.test_indices) == test_idx  # type: ignore[union-attr]


def test_fold_local_text_requires_string_columns() -> None:
    x = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(ValidationError, match="No text/object columns"):
        build_fold_preprocessor(
            x,
            PreprocessRecipe(impute=None, text="tfidf"),
            y_train=pd.Series([0, 1, 0, 1]),
        )


def test_recipe_knobs_include_text_and_reduce() -> None:
    base = PreprocessRecipe(text="tfidf", reduce="pca", reduce_n_components=2)
    updated = base.with_knobs({"text_max_features": 8, "reduce_n_components": 1})
    assert updated.text_max_features == 8
    assert updated.reduce_n_components == 1
    assert "text_max_features" in SAFE_RECIPE_KNOBS
    assert "reduce_n_components" in SAFE_RECIPE_KNOBS
