"""Real attribution checks against known linear effects and SHAP additivity."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from buildml.core.errors import ValidationError
from buildml.model.shap_explain import explain_with_shap


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, float("inf"), "5", None])
def test_invalid_sample_budget_fails_before_optional_import(budget, monkeypatch):
    def unexpected(**kwargs):
        pytest.fail("invalid input must fail before loading SHAP")
    monkeypatch.setattr("buildml.model.shap_explain.require_shap", unexpected)
    with pytest.raises(ValidationError, match="positive integer"):
        explain_with_shap(None, pd.DataFrame({"x": [1.]}), max_samples=budget)


def test_empty_attribution_population_is_rejected():
    with pytest.raises(ValidationError, match="non-empty"):
        explain_with_shap(None, pd.DataFrame(columns=["x"]))


def test_real_linear_attributions_match_known_effects_and_sampling():
    pytest.importorskip("shap")
    frame = pd.DataFrame(np.random.default_rng(23).normal(size=(60, 2)), columns=["signal", "weak"])
    model = LinearRegression().fit(frame, 4 * frame.signal + .2 * frame.weak + 3)
    result = explain_with_shap(model, frame, max_samples=25, random_state=17)
    sample = frame.sample(n=25, random_state=17)
    expected = (sample - sample.mean()).abs().mean() * pd.Series([4., .2], index=frame.columns)
    assert result.backend == "explainer"
    assert result.n_rows == 25
    for name in frame:
        assert result.mean_abs_shap[name] == pytest.approx(expected[name], rel=1e-6)
    assert result.warnings
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("classification", [False, True])
def test_real_tree_attributions_preserve_feature_order_and_prediction_additivity(classification):
    shap = pytest.importorskip("shap")
    frame = pd.DataFrame(np.random.default_rng(91).normal(size=(60, 2)), columns=["signal", "noise"])
    target = (frame.signal > 0).astype(int) if classification else 3 * frame.signal
    cls = RandomForestClassifier if classification else RandomForestRegressor
    model = cls(n_estimators=8, max_depth=3, random_state=1).fit(frame, target)
    result = explain_with_shap(model, frame, max_samples=60)
    values = shap.TreeExplainer(model)(frame.to_numpy())
    predicted = model.predict_proba(frame) if classification else model.predict(frame)
    np.testing.assert_allclose(values.base_values + values.values.sum(axis=1), predicted, atol=1e-6)
    assert result.backend == "tree_explainer"
    assert result.feature_names == ("signal", "noise")
    assert result.mean_abs_shap["signal"] > result.mean_abs_shap["noise"]
    assert result.n_rows == 60
    json.dumps(result.to_dict(), allow_nan=False)
