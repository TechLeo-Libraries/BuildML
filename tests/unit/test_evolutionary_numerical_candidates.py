"""Real evolutionary candidates are scored with independent prediction arithmetic."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor

from buildml import Session


def session(changed=False):
    rng = np.random.default_rng(597)
    x = rng.normal(size=(72, 2))
    frame = pd.DataFrame(
        {"a": x[:, 0], "b": x[:, 1], "y": 1.7 * x[:, 0] - 0.6 * x[:, 1] + rng.normal(0, 0.5, 72)}
    )
    s = (
        Session.ingest(frame)
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=9)
    )
    if changed:
        positions = list(s.split_plan.test_indices) + list(s.split_plan.validation_indices)
        s.dataset._ensure_pandas().iloc[positions, :] = 1e8
    return s


def oracle(train_x, train_y, test_x, params, kind):
    if kind.startswith("ridge"):
        offset = train_x.mean(axis=0) if params["fit_intercept"] else np.zeros(2)
        center_y = train_y.mean() if params["fit_intercept"] else 0.0
        x = train_x - offset
        coef = np.linalg.solve(x.T @ x + params["alpha"] * np.eye(2), x.T @ (train_y - center_y))
        return (test_x - offset) @ coef + center_y
    # Build a separate reference tree, then traverse its arrays independently
    # rather than calling predict or any BuildML scoring implementation.
    tree = DecisionTreeRegressor(random_state=5, **params).fit(train_x, train_y).tree_
    out = []
    for row in test_x.astype(np.float32):
        node = 0
        while tree.children_left[node] != tree.children_right[node]:
            node = (
                tree.children_left[node]
                if row[tree.feature[node]] <= tree.threshold[node]
                else tree.children_right[node]
            )
        out.append(float(tree.value[node, 0, 0]))
    return np.asarray(out)


@pytest.mark.parametrize("kind", ["ridge_linear", "ridge_log", "tree"])
@pytest.mark.parametrize("seed", [2, 7, 19])
def test_evolutionary_mutation_candidates_bounds_scores_and_holdout_isolation(kind, seed):
    s = session()
    changed = session(True)
    if kind.startswith("ridge"):
        estimator = Ridge(solver="svd")
        space = {
            "alpha": {"type": "float", "low": 0.02, "high": 3.0, "log": kind.endswith("log")},
            "fit_intercept": [False, True],
        }
    else:
        estimator = DecisionTreeRegressor(random_state=5)
        space = {
            "max_depth": {"type": "int", "low": 1, "high": 6},
            "min_weight_fraction_leaf": {"type": "float", "low": 0.0, "high": 0.2},
            "ccp_alpha": {"type": "float", "low": 0.0001, "high": 0.2, "log": True},
            "criterion": ["squared_error", "absolute_error"],
        }
    kwargs = dict(
        param_space=space,
        population_size=6,
        n_generations=3,
        elite_size=1,
        mutation_rate=1.0,
        crossover_rate=0.8,
        max_evaluations=14,
        random_state=seed,
        cv=KFold(3, shuffle=True, random_state=11),
        ranking_metric="rmse",
        task="regression",
        refit=False,
    )
    result = s.evolutionary_search(estimator, **kwargs)
    repeated = changed.evolutionary_search(estimator, **kwargs)
    assert 6 < len(result.trials) <= 14
    assert [t.params for t in result.trials] == [t.params for t in repeated.trials]
    np.testing.assert_allclose(
        [t.mean_score for t in result.trials], [t.mean_score for t in repeated.trials], atol=1e-12
    )
    train = s.partition("train")
    x, y = train[["a", "b"]].to_numpy(), train.y.to_numpy()
    for trial in result.trials:
        for key, spec in space.items():
            if isinstance(spec, list):
                assert trial.params[key] in spec
            else:
                assert spec["low"] <= trial.params[key] <= spec["high"]
                if spec["type"] == "int":
                    assert isinstance(trial.params[key], int) and not isinstance(
                        trial.params[key], bool
                    )
        fold_scores = []
        for fit, evaluate in KFold(3, shuffle=True, random_state=11).split(x):
            predicted = oracle(x[fit], y[fit], x[evaluate], trial.params, kind)
            fold_scores.append(np.sqrt(np.mean((y[evaluate] - predicted) ** 2)))
        assert trial.mean_score == pytest.approx(np.mean(fold_scores), rel=1e-10, abs=1e-12)
        assert trial.std_score == pytest.approx(np.std(fold_scores, ddof=1), rel=1e-10, abs=1e-12)
    assert result.best_score == pytest.approx(min(t.mean_score for t in result.trials))
    assert s.fit_result is None and changed.fit_result is None
    assert "test" in result.best_cv.held_out_partitions
