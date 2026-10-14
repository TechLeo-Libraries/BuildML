"""Fold-local recipes cannot mutate supplied estimators or disguise missing labels."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from buildml.core.errors import ValidationError
from buildml.preprocess import PreprocessRecipe, build_fold_preprocessor


@pytest.mark.parametrize("estimator", [LogisticRegression(), RandomForestClassifier(n_estimators=5, random_state=1)])
def test_each_fold_clones_model_selector_without_mutating_recipe_estimator(estimator):
    rng = np.random.default_rng(32)
    train = pd.DataFrame(rng.normal(size=(60, 3)), columns=["signal", "noise", "other"])
    target = pd.Series(np.where(train.signal > 0, "yes", "no"))
    recipe = PreprocessRecipe(impute=None, select="model", select_estimator=estimator)
    first = build_fold_preprocessor(train, recipe, target)
    before = first.transform(train).copy(deep=True)
    model = first._selector.estimator
    assert model is not estimator
    assert not hasattr(estimator, "classes_")
    second = build_fold_preprocessor(train, recipe, pd.Series(np.where(train.other > 0, "yes", "no")))
    assert second._selector.estimator is not model
    pd.testing.assert_frame_equal(first.transform(train), before)
    assert not hasattr(estimator, "classes_")


@pytest.mark.parametrize("null", [None, np.nan, pd.NA])
@pytest.mark.parametrize("method", ["univariate", "model"])
def test_object_labels_with_nulls_cannot_be_factorized_as_an_extra_class(null, method):
    train = pd.DataFrame({"x": np.arange(12), "z": np.arange(12) % 3})
    labels = pd.Series(["yes", "no"] * 6, dtype=object)
    labels.iloc[3] = null
    recipe = PreprocessRecipe(impute=None, select=method)
    with pytest.raises(ValidationError, match="non-null target"):
        build_fold_preprocessor(train, recipe, labels)
