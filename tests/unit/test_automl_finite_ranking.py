"""AutoML only selects candidates with a finite ranking score."""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import UndefinedMetricWarning
from test_automl_budget_input_consumers import assert_preserved, snapshot
from test_automl_refit_consumers import options, predict, score

from buildml import Session
from buildml.automl import AutoMLBudget
from buildml.core.errors import ValidationError


@pytest.mark.parametrize(
    "method, secondary",
    [
        ("grid", None),
        ("randomized", None),
        ("evolutionary", None),
        ("optuna", None),
        ("optuna", "r2"),
    ],
)
def test_undefined_validation_ranking_preserves_finite_winner(method, secondary, tmp_path):
    if method == "optuna":
        pytest.importorskip("optuna")

    x = np.arange(16, dtype=float)
    frame = pd.DataFrame({"x": x, "z": np.cos(x), "y": 2 * x + np.sin(x), "weight": 1 + x % 4})
    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "z": "feature", "y": "target", "weight": "weight"})
        .inject_split(
            train_indices=list(range(12)), validation_indices=[12], test_indices=[13, 14, 15]
        )
    )
    settings = options("validation", "mse")
    settings["method"] = method

    with pytest.warns(UndefinedMetricWarning):
        result = session.automl.run(**settings)

    train = session.partition("train")
    validation = session.partition("validation")
    for trial in result.trials:
        expected = predict(train, validation, trial.params["alpha"], weighted=True)
        assert trial.mean_score == pytest.approx(score(validation, expected, "mse", weighted=True))
        assert np.isnan(trial.mean_metrics["r2"])
    assert result.best_score == min(trial.mean_score for trial in result.trials)
    held_out = session.partition("test")
    expected = predict(train, held_out, result.best_params["alpha"], weighted=True)
    np.testing.assert_allclose(session.predict(partition="test"), expected)

    before = snapshot(session)
    settings["ranking_metric"] = "mse" if secondary else "r2"
    if method == "optuna":
        storage = f"sqlite:///{(tmp_path / 'ranking.db').as_posix()}"
        settings["budget"] = AutoMLBudget(
            study_storage=storage, multi_objective=secondary is not None, secondary_metric=secondary
        )
    with (
        pytest.warns(UndefinedMetricWarning),
        pytest.raises(
            ValidationError, match=f"finite '{settings['ranking_metric']}' ranking score"
        ),
    ):
        session.automl.run(**settings)
    assert_preserved(session, before)

    if method == "optuna":
        import optuna

        study = optuna.load_study(study_name="buildml_automl", storage=storage)
        assert len(study.trials) == settings["n_trials"]
        assert all(trial.state is optuna.trial.TrialState.PRUNED for trial in study.trials)
