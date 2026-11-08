"""Exercise FLAML's public search and validation results with real fitted learners."""

import numpy as np
import pandas as pd
import pytest

from buildml import Session


@pytest.fixture(autouse=True)
def _require_provider():
    provider = pytest.importorskip("flaml")
    if not hasattr(provider, "AutoML"):
        pytest.skip("FLAML AutoML dependencies are unavailable")


def _regression_session(*, signal_name="signal", target_name="target", zero_loss=False):
    rng = np.random.default_rng(82)
    n = 120
    signal = np.tile([-1.0, 1.0], n // 2)
    target = np.full(n, 4.0) if zero_loss else 6 * signal + rng.normal(scale=0.08, size=n)
    frame = pd.DataFrame(
        {
            "noise": rng.normal(size=n),
            signal_name: signal,
            target_name: target,
            "row_weight": rng.integers(1, 6, size=n),
        }
    )
    return (
        Session.ingest(frame)
        .set_roles(
            {
                "noise": "feature",
                signal_name: "feature",
                target_name: "target",
                "row_weight": "weight",
            }
        )
        .inject_split(
            train_indices=list(range(80)),
            validation_indices=list(range(80, 100)),
            test_indices=list(range(100, n)),
        )
    )


@pytest.mark.parametrize(("metric", "zero_loss"), [("r2", False), ("mse", True)])
def test_real_flaml_reports_learner_results_on_the_requested_scale(metric, zero_loss):
    session = _regression_session(zero_loss=zero_loss)
    result = session.automl.run(
        backend="flaml", task="regression", ranking_metric=metric, time_budget=4, random_state=17
    )
    provider = session.fit_result.estimator._automl
    best = provider.best_loss
    expected = {
        learner: 1.0 - loss if metric == "r2" else loss
        for learner, loss in provider.best_loss_per_estimator.items()
        if np.isfinite(loss) and provider.best_config_per_estimator[learner] is not None
    }
    assert expected
    if zero_loss:
        assert best == pytest.approx(0.0, abs=1e-10)
    assert result.best_score == pytest.approx(1.0 - best if metric == "r2" else best)
    assert {trial.family: trial.mean_score for trial in result.trials} == pytest.approx(expected)
    assert session.automl_plan.n_trials == result.config["n_trials"] == len(result.trials)
    for trial in result.trials:
        assert trial.params == provider.best_config_per_estimator[trial.family]
        assert trial.mean_metrics[metric] == trial.mean_score


@pytest.mark.parametrize(
    ("signal_name", "target_name"),
    [
        ("_sample_weight", "target"),
        ("signal", "_sample_weight"),
    ],
)
def test_real_weighted_flaml_preserves_reserved_spelling_in_user_columns(signal_name, target_name):
    session = _regression_session(signal_name=signal_name, target_name=target_name)
    result = session.automl.run(
        backend="flaml",
        task="regression",
        selection="validation",
        ranking_metric="mse",
        time_budget=4,
        random_state=17,
    )
    fitted = session.fit_result
    distinct_signal = pd.DataFrame({"noise": [0.0, 0.0], signal_name: [-1.0, 1.0]})
    predictions = np.asarray(fitted.estimator.predict(distinct_signal))
    assert predictions[1] - predictions[0] > 8.0
    validation = session.partition("validation")
    predicted = np.asarray(
        fitted.estimator.predict(validation.loc[:, list(fitted.feature_columns)])
    )
    expected = np.average(
        (validation[target_name].to_numpy() - predicted) ** 2, weights=validation["row_weight"]
    )
    assert result.best_score == pytest.approx(expected)


@pytest.mark.parametrize("ranking_metric", [None, "f1_macro"])
def test_real_flaml_multiclass_f1_uses_the_requested_average(ranking_metric):
    rng = np.random.default_rng(22)
    labels = ("alpha", "beta", "gamma")
    actual = np.tile(np.repeat(labels, [6, 3, 1]), 18)
    centers = pd.Series(actual).map(dict(zip(labels, [-1.0, 0.0, 1.0], strict=True)))
    frame = pd.DataFrame(
        {
            "x": centers + rng.normal(scale=0.2, size=len(actual)),
            "noise": rng.normal(size=len(actual)),
            "label": actual,
            "row_weight": rng.integers(1, 5, size=len(actual)),
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles(
            {
                "x": "feature",
                "noise": "feature",
                "label": "target",
                "row_weight": "weight",
            }
        )
        .inject_split(
            train_indices=list(range(120)),
            validation_indices=list(range(120, 150)),
            test_indices=list(range(150, 180)),
        )
    )
    result = session.automl.run(
        backend="flaml",
        task="classification",
        selection="validation",
        ranking_metric=ranking_metric,
        time_budget=4,
        random_state=17,
    )
    assert result.ranking_metric == (ranking_metric or "f1_weighted")
    fitted = session.fit_result.estimator
    validation = session.partition("validation")
    predicted = np.asarray(fitted.predict(validation[["x", "noise"]]))
    actual = validation["label"].to_numpy()
    weights = validation["row_weight"].to_numpy()
    scores = []
    support = []
    for label in labels:
        true_weight = weights[actual == label].sum()
        predicted_weight = weights[predicted == label].sum()
        true_positives = weights[(actual == label) & (predicted == label)].sum()
        scores.append(2.0 * true_positives / (true_weight + predicted_weight))
        support.append(true_weight)
    expected = np.average(scores, weights=support if ranking_metric is None else None)
    assert result.best_score == pytest.approx(expected)
    assert result.trials[0].mean_score == pytest.approx(1.0 - fitted._automl.best_loss)
