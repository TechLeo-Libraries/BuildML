"""Independent weighted CV arithmetic and malformed custom-fold boundaries."""
import json
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor

from buildml import Session
from buildml.core.errors import ValidationError


def session():
    x = np.arange(60, dtype=float)
    frame = pd.DataFrame({"x": x, "y": .2 * x + np.sin(x), "weight": 1 + x % 5})
    return (Session.ingest(frame).set_roles({"x": "feature", "y": "target", "weight": "weight"})
            .split(test_size=.2, validation_size=.1, random_state=11))


@pytest.mark.parametrize("metric", ["mae", "mse", "rmse", "r2"])
def test_weighted_cv_matches_hand_calculated_train_only_folds(metric, capsys):
    s = session()
    held_out = s.partition("test").copy()
    train = s.partition("train")
    expected = []
    splitter = KFold(n_splits=3, shuffle=True, random_state=42)
    for fit_pos, eval_pos in splitter.split(train):
        fit, score = train.iloc[fit_pos], train.iloc[eval_pos]
        prediction = np.average(fit.y, weights=fit.weight)
        errors = score.y.to_numpy() - prediction
        mse = np.average(errors ** 2, weights=score.weight)
        variance = np.average((score.y - np.average(score.y, weights=score.weight)) ** 2,
                              weights=score.weight)
        expected.append({"mae": np.average(abs(errors), weights=score.weight),
                         "mse": mse, "rmse": np.sqrt(mse), "r2": 1 - mse / variance})
    result = s.cv_score(DummyRegressor(strategy="mean"), task="regression", cv=3,
                        cv_strategy="kfold", scoring_metric=metric)
    for fold, oracle in zip(result.folds, expected, strict=True):
        assert fold.metrics == pytest.approx(oracle)
    for key in expected[0]:
        values = [fold[key] for fold in expected]
        assert result.mean_metrics[key] == pytest.approx(np.mean(values))
        assert result.std_metrics[key] == pytest.approx(np.std(values, ddof=1))
    assert result.to_frame()[metric].tolist() == pytest.approx([fold[metric] for fold in expected])
    assert json.loads(json.dumps(result.to_dict(), allow_nan=False))["population"] == "train"
    result.show()
    assert metric in capsys.readouterr().out
    pd.testing.assert_frame_equal(s.partition("test"), held_out)


class MalformedSplitter:
    def __init__(self, kind):
        self.kind = kind

    def split(self, x, y=None, groups=None):
        cases = {
            "negative_alias": ([-1, 0, 1], [len(x) - 1]),
            "duplicate": ([0, 0, 1], [2, 3]),
            "out_of_bounds": ([0, len(x)], [1, 2]),
            "fractional": ([0.5, 1.5], [2, 3]),
            "boolean": ([True, False], [2, 3]),
            "empty_fit": ([], [0, 1]),
            "empty_eval": ([0, 1], []),
            "two_dimensional": ([[0, 1]], [2, 3]),
        }
        yield cases[self.kind]


@pytest.mark.parametrize("method", ["cv_score", "nested_cv_score"])
@pytest.mark.parametrize("kind", ["negative_alias", "duplicate", "out_of_bounds", "fractional",
                                  "boolean", "empty_fit", "empty_eval", "two_dimensional"])
def test_custom_cv_rejects_invalid_positions_before_any_fit(method, kind, monkeypatch):
    fitting = Mock(side_effect=AssertionError("Invalid fold reached fitting"))
    monkeypatch.setattr(DummyRegressor, "fit", fitting)
    options = {"cv": MalformedSplitter(kind)} if method == "cv_score" else {
        "outer_cv": MalformedSplitter(kind), "inner_cv": 2, "param_grid": {"strategy": ["mean"]},
    }
    with pytest.raises(ValidationError, match="fold.*indices"):
        s = session().set_roles({"weight": "ignore"})
        getattr(s, method)(DummyRegressor(), task="regression", **options)
    fitting.assert_not_called()


@pytest.mark.parametrize("strategy", ["group", "stratified_group", "time", "unknown"])
def test_missing_or_invalid_split_strategy_fails_before_fit(strategy, monkeypatch):
    fitting = Mock(side_effect=AssertionError("Invalid strategy reached fitting"))
    monkeypatch.setattr(DummyRegressor, "fit", fitting)
    with pytest.raises(ValidationError):
        session().cv_score(DummyRegressor(), task="regression", cv=3, cv_strategy=strategy)
    fitting.assert_not_called()


def test_temporal_weighted_cv_uses_chronological_order_without_future_rows():
    s = session()
    frame = s.to_pandas().copy()
    frame["clock"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(frame.x, unit="D")
    s = (Session.ingest(frame).set_roles({"x": "feature", "y": "target", "weight": "weight", "clock": "time"})
         .split(test_size=.2, random_state=17))
    train = s.partition("train").sort_values("clock", kind="mergesort")
    expected = []
    for fit_pos, eval_pos in TimeSeriesSplit(n_splits=3).split(train):
        fit, score = train.iloc[fit_pos], train.iloc[eval_pos]
        assert fit.clock.max() < score.clock.min()
        prediction = np.average(fit.y, weights=fit.weight)
        expected.append(np.average(abs(score.y - prediction), weights=score.weight))
    result = s.cv_score(DummyRegressor(), task="regression", cv=3, cv_strategy="auto", scoring_metric="mae")
    assert result.cv_strategy == "time"
    assert [fold.metrics["mae"] for fold in result.folds] == pytest.approx(expected)


@pytest.mark.parametrize("search", ["grid", "randomized"])
def test_nested_selection_reports_outer_scores_and_serializes_separate_inner_results(search, capsys):
    s = session()
    options = {"param_grid": {"max_depth": [1, 2]}} if search == "grid" else {
        "param_distributions": {"max_depth": [1, 2]}, "n_iter": 2,
    }
    result = s.nested_cv_score(DecisionTreeRegressor(random_state=7), task="regression",
                               inner_search=search, outer_cv=3, inner_cv=2, scoring_metric="mae", **options)
    outer = [fold.metrics["mae"] for fold in result.outer_folds]
    assert result.mean_metrics["mae"] == pytest.approx(np.mean(outer))
    assert result.std_metrics["mae"] == pytest.approx(np.std(outer, ddof=1))
    payload = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert payload["population"] == "train"
    assert len(result.to_frame()) == 3
    assert payload["inner_selection_summary"]["n_outer_folds"] == 3
    result.show()
    assert "outer mae" in capsys.readouterr().out
