"""Finite training values define fold-local fences and discretization edges."""

import numpy as np
import pandas as pd
import pytest

from buildml.core.errors import ValidationError
from buildml.preprocess import PreprocessRecipe, build_fold_preprocessor


@pytest.mark.parametrize("method", ["iqr", "zscore"])
def test_infinite_observations_do_not_corrupt_train_derived_outlier_fences(method):
    values = np.arange(1.0, 9.0)
    train = pd.DataFrame({"x": np.concatenate([values, [np.nan, np.inf, -np.inf]])})
    prep = build_fold_preprocessor(train, PreprocessRecipe(impute=None, outliers=method))
    if method == "iqr":
        q1, q3 = np.quantile(values, [0.25, 0.75])
        expected = [q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)]
    else:
        expected = [values.mean() - 3 * values.std(), values.mean() + 3 * values.std()]
    eval_frame = pd.DataFrame({"x": [-np.inf, np.inf]})
    np.testing.assert_allclose(prep.transform(eval_frame).x, expected)


@pytest.mark.parametrize("method", ["uniform", "quantile"])
def test_binning_uses_finite_training_edges_and_applies_frozen_edges(method):
    clean = pd.DataFrame({"x": np.arange(1.0, 9.0)})
    dirty = pd.concat([clean, pd.DataFrame({"x": [np.nan, np.inf, -np.inf]})], ignore_index=True)
    recipe = PreprocessRecipe(impute=None, binning=method, n_bins=3)
    expected = build_fold_preprocessor(clean, recipe)
    actual = build_fold_preprocessor(dirty, recipe)
    eval_frame = pd.DataFrame({"x": [-1e6, 1, 4, 8, 1e6]})
    pd.testing.assert_frame_equal(actual.transform(eval_frame), expected.transform(eval_frame))
    assert np.isfinite(actual._binning_edges["x"][1:-1]).all()


@pytest.mark.parametrize("options", [
    {"outliers": "iqr"}, {"outliers": "zscore"},
    {"binning": "uniform"}, {"binning": "quantile"},
])
def test_no_finite_training_values_are_refused(options):
    train = pd.DataFrame({"x": [np.inf, -np.inf, np.nan]})
    with pytest.raises(ValidationError, match="no finite fold-train values"):
        build_fold_preprocessor(train, PreprocessRecipe(impute=None, **options))
