"""Fold-local sklearn consumers keep the current fit's state and column layout."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from buildml.core.errors import ValidationError
from buildml.preprocess import FoldLocalPreprocessor, PreprocessRecipe


@pytest.mark.parametrize("previous", ("dates", "outliers", "retired-outlier", "both"))
@pytest.mark.parametrize("scaling", ("standard", "minmax"))
def test_refit_pipeline_learns_only_the_replacement_recipe(previous, scaling):
    first = pd.DataFrame(
        {
            "signal": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0],
            "retired": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 80.0],
            "when": pd.date_range("2025-04-01", periods=8),
        }
    )
    dates = previous in {"dates", "both"}
    caps = previous != "dates"
    recipe = PreprocessRecipe(
        impute=None,
        dates=dates,
        date_columns=("when",),
        outliers="iqr" if caps else None,
        outlier_columns=("retired" if previous == "retired-outlier" else "signal",),
    )
    pipe = Pipeline([("prep", FoldLocalPreprocessor(recipe)), ("model", LinearRegression())])
    pipe.fit(first, np.cos(np.arange(len(first))))
    original = pipe.named_steps["prep"].transform(first)
    assert ("when_year" in original) is dates
    if caps:
        column = recipe.outlier_columns[0]
        low, high = np.quantile(first[column], [0.25, 0.75])
        assert original[column].iloc[-1] == pytest.approx(high + 1.5 * (high - low))

    x = pd.DataFrame({"signal": np.arange(20.0, 30.0), "contrast": np.cos(np.arange(10))})
    y = 3 + 2 * x["signal"].to_numpy() - x["contrast"].to_numpy() + 0.1 * np.sin(np.arange(10))
    pipe.set_params(prep__recipe=PreprocessRecipe(impute=None, scale=scaling))
    pipe.fit(x, y)
    matrix = x.to_numpy()
    if scaling == "standard":
        center, width = matrix.mean(axis=0), matrix.std(axis=0)
    else:
        center, width = matrix.min(axis=0), np.ptp(matrix, axis=0)
    scored = pd.DataFrame(
        {"signal": [14.0, 35.0, 17.0], "contrast": [1.1, -0.4, 0.5]}, index=[7, 7, -2]
    )
    fit_values = (matrix - center) / width
    expected_features = (scored.to_numpy() - center) / width
    weights = np.linalg.lstsq(np.column_stack([np.ones(len(x)), fit_values]), y, rcond=None)[0]
    transformed = pipe.named_steps["prep"].transform(scored)
    assert transformed.columns.tolist() == x.columns.tolist() and transformed.index.equals(
        scored.index
    )
    np.testing.assert_allclose(transformed, expected_features, atol=1e-12)
    np.testing.assert_allclose(
        pipe.predict(scored), np.column_stack([np.ones(3), expected_features]) @ weights
    )


@pytest.mark.parametrize("select", (None, "variance"))
@pytest.mark.parametrize("replay", (False, True), ids=("memory", "saved"))
def test_pipeline_score_uses_fitted_order_and_only_fitted_features(select, replay, tmp_path):
    i = np.arange(12, dtype=float)
    train = pd.DataFrame({"left": np.sin(i), "right": i, "constant": 3.0})
    train_y = 2.0 + 0.7 * train["left"] + 3.0 * train["right"]
    recipe = PreprocessRecipe(impute=None, scale="standard", select=select)
    pipe = Pipeline([("prep", FoldLocalPreprocessor(recipe)), ("model", LinearRegression())])
    pipe.fit(train, train_y)
    if replay:
        path = tmp_path / "pipeline.joblib"
        joblib.dump(pipe, path)
        pipe = joblib.load(path)
    scoring = pd.DataFrame(
        {"right": [4.0, -2.0], "extra": [1.0, 999.0], "constant": 3.0, "left": [-0.4, 0.8]},
        index=[5, 5],
    )
    prep = pipe.named_steps["prep"]
    actual = prep.transform(scoring)
    names = ["left", "right"] if select else ["left", "right", "constant"]
    assert actual.columns.tolist() == names and actual.index.equals(scoring.index)
    expected = (scoring[names] - train[names].mean()) / train[names].std(ddof=0).replace(0, 1)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(
        pipe.predict(scoring), 2 + 0.7 * scoring["left"] + 3 * scoring["right"]
    )


@pytest.mark.parametrize("select", (None, "variance"))
def test_missing_passthrough_feature_fails_at_transform_boundary(select):
    train = pd.DataFrame({"signal": [1.0, 3.0, 9.0], "second": [7.0, 2.0, 4.0]})
    fit = FoldLocalPreprocessor(
        PreprocessRecipe(impute=None, select=select, scale="standard", scale_columns=("signal",))
    ).fit(train)
    with pytest.raises(ValidationError, match="fitted features missing.*second"):
        fit.transform(train[["signal"]])


@pytest.mark.parametrize("failure", ("before-fit", "during-selection"))
def test_failed_refit_cannot_score_and_a_later_valid_fit_recovers(failure):
    x = pd.DataFrame({"a": [1.0, 2.0, 7.0], "b": [3.0, 4.0, 2.0]})
    good = PreprocessRecipe(impute=None, scale="standard", select="variance")
    preprocessor = FoldLocalPreprocessor(good).fit(x)
    if failure == "before-fit":
        bad = PreprocessRecipe(impute=None, binning="uniform", n_bins=1)
        detail = "n_bins"
    else:
        bad = PreprocessRecipe(impute=None, select="variance", select_columns=("unseen",))
        detail = "select_columns missing"
    preprocessor.set_params(recipe=bad)
    with pytest.raises(ValidationError, match=detail):
        preprocessor.fit(x)
    with pytest.raises(ValidationError, match="must be fitted"):
        preprocessor.transform(x)
    assert preprocessor.set_params(recipe=good).fit(x) is preprocessor
    np.testing.assert_allclose(preprocessor.transform(x), (x - x.mean()) / x.std(ddof=0))


@pytest.mark.parametrize(
    ("step", "column", "settings"),
    (
        ("date", "when", {"dates": True, "date_columns": ("when",)}),
        ("text", "note", {"text": "tfidf", "text_columns": ("note",)}),
        ("outlier", "signal", {"outliers": "iqr", "outlier_columns": ("signal",)}),
        ("impute", "signal", {"impute": "median", "impute_columns": ("signal",)}),
        ("scale", "signal", {"scale": "standard", "scale_columns": ("signal",)}),
        ("binning", "signal", {"binning": "uniform", "binning_columns": ("signal",)}),
        ("Categorical recipe", "group", {"encode": "onehot", "encode_columns": ("group",)}),
        (
            "reduce",
            "signal",
            {"reduce": "pca", "reduce_columns": ("signal", "second"), "reduce_n_components": 1},
        ),
    ),
)
def test_missing_source_column_names_the_fitted_step(step, column, settings):
    train = pd.DataFrame(
        {
            "signal": [1.0, 3.0, 5.0, 9.0],
            "second": [7.0, 2.0, 4.0, 10.0],
            "group": ["one", "two", "one", "three"],
            "note": ["clear sky", "low clouds", "bright sun", "clear view"],
            "when": pd.date_range("2025-01-30", periods=4),
        }
    )
    prep = FoldLocalPreprocessor(PreprocessRecipe(**{"impute": None, **settings})).fit(train)
    with pytest.raises(ValidationError, match=rf"{step} columns missing: .*{column}"):
        prep.transform(train.drop(columns=[column]))
