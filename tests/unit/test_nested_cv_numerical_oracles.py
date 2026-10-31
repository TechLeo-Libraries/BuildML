"""Group and temporal nested evaluation match independent sklearn searches."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from buildml import Session
from buildml.preprocess import PreprocessRecipe


class AuditedRidge(Ridge):
    fitted_targets = []

    def fit(self, X, y, sample_weight=None):
        self.fitted_targets.append(frozenset(np.asarray(y, dtype=float).ravel()))
        return super().fit(X, y, sample_weight=sample_weight)


def _frame():
    rng = np.random.default_rng(640)
    n = 144
    clock = rng.permutation(n)
    x = rng.normal(size=n)
    z = rng.normal(size=n) * 17
    return pd.DataFrame(
        {
            "x": x,
            "z": z,
            "y": 2 * x - 0.13 * z + np.sin(clock / 9),
            "clock": pd.Timestamp("2024-01-01") + pd.to_timedelta(clock, unit="D"),
            "subject": np.arange(n) % 24,
        },
        index=np.arange(2000, 2000 + n),
    )


def _session(frame):
    return (
        Session.ingest(frame)
        .set_roles(
            {"x": "feature", "z": "feature", "y": "target", "clock": "time", "subject": "id"}
        )
        .split(test_size=0.2, validation_size=0.2, random_state=19)
    )


@pytest.mark.parametrize("strategy", ["group", "time"])
@pytest.mark.parametrize("scaled", [False, True])
def test_nested_scores_and_fit_populations_match_sklearn(strategy, scaled):
    frame = _frame()
    session = _session(frame)
    train = session.partition("train")
    # Explicit groups are positional in Session train order, regardless of the
    # Series' own labels. Time CV instead sorts the shuffled time-role column.
    groups = pd.Series(train.subject.to_numpy(), index=np.arange(9000, 9000 + len(train)))
    ordered = train.sort_values("clock", kind="mergesort") if strategy == "time" else train
    x, y = ordered[["x", "z"]], ordered.y
    outer = TimeSeriesSplit(3) if strategy == "time" else GroupKFold(3)
    inner = TimeSeriesSplit(2) if strategy == "time" else GroupKFold(2)
    oracle_groups = None if strategy == "time" else ordered.subject.to_numpy()
    expected = []
    allowed_fit_targets = set()
    for outer_train, outer_eval in outer.split(x, y, oracle_groups):
        inner_groups = None if oracle_groups is None else oracle_groups[outer_train]
        allowed_fit_targets.add(frozenset(y.iloc[outer_train]))
        for inner_train, _ in inner.split(x.iloc[outer_train], y.iloc[outer_train], inner_groups):
            allowed_fit_targets.add(frozenset(y.iloc[outer_train].iloc[inner_train]))
        pipeline = Pipeline(
            [("scale", StandardScaler() if scaled else "passthrough"), ("model", Ridge())]
        )
        reference = GridSearchCV(
            pipeline,
            {"model__alpha": [0.01, 1.0, 20.0]},
            cv=inner,
            scoring="neg_mean_squared_error",
        )
        fit_kwargs = {} if inner_groups is None else {"groups": inner_groups}
        reference.fit(x.iloc[outer_train], y.iloc[outer_train], **fit_kwargs)
        expected.append(
            (
                reference.best_params_["model__alpha"],
                mean_squared_error(y.iloc[outer_eval], reference.predict(x.iloc[outer_eval])),
                len(outer_train),
                len(outer_eval),
            )
        )

    AuditedRidge.fitted_targets = []
    kwargs = dict(
        param_grid={"alpha": [0.01, 1.0, 20.0]},
        task="regression",
        outer_cv=3,
        inner_cv=2,
        cv_strategy=strategy,
        scoring_metric="mse",
        preprocess=PreprocessRecipe(scale="standard") if scaled else None,
    )
    if strategy == "group":
        kwargs["groups"] = groups
    result = session.nested_cv_score(AuditedRidge(), **kwargs)
    for actual, (alpha, mse, n_train, n_eval) in zip(result.outer_folds, expected, strict=True):
        assert actual.best_params == {"alpha": alpha}
        assert actual.metrics["mse"] == pytest.approx(mse, rel=1e-10, abs=1e-10)
        assert (actual.n_train, actual.n_eval) == (n_train, n_eval)
    scores = [row[1] for row in expected]
    assert result.mean_metrics["mse"] == pytest.approx(np.mean(scores))
    assert result.std_metrics["mse"] == pytest.approx(np.std(scores, ddof=1))
    assert set(AuditedRidge.fitted_targets) == allowed_fit_targets

    changed = frame.copy()
    holdout_positions = [*session.split_plan.test_indices, *session.split_plan.validation_indices]
    changed.iloc[holdout_positions, changed.columns.get_loc("y")] += 1_000_000
    repeated = _session(changed).nested_cv_score(AuditedRidge(), **kwargs)
    assert repeated.mean_metrics == pytest.approx(result.mean_metrics)
    assert [fold.best_params for fold in repeated.outer_folds] == [
        fold.best_params for fold in result.outer_folds
    ]
    assert set(AuditedRidge.fitted_targets) == allowed_fit_targets
    assert session.fit_result is None


@pytest.mark.parametrize("strategy", ["group", "time"])
@pytest.mark.parametrize("scaled", [False, True])
def test_evolutionary_nested_search_matches_manual_candidate_scores(monkeypatch, strategy, scaled):
    """Audit actual sampled candidates, without assuming an exhaustive GA search."""
    import buildml.model.selection as selection

    frame = _frame()
    session = _session(frame)
    train = session.partition("train")
    ordered = train.sort_values("clock", kind="mergesort") if strategy == "time" else train
    x, y = ordered[["x", "z"]], ordered.y
    groups = None if strategy == "time" else ordered.subject.to_numpy()
    outer = TimeSeriesSplit(3) if strategy == "time" else GroupKFold(3)
    inner = TimeSeriesSplit(2) if strategy == "time" else GroupKFold(2)
    searches = []
    real_search = selection.evolutionary_search

    def record_search(*args, **kwargs):
        result = real_search(*args, **kwargs)
        searches.append(result)
        return result

    monkeypatch.setattr(selection, "evolutionary_search", record_search)
    kwargs = dict(
        inner_search="evolutionary",
        param_space={"alpha": [0.01, 1.0, 20.0]},
        population_size=4,
        n_generations=2,
        n_trials=8,
        random_state=37,
        task="regression",
        outer_cv=3,
        inner_cv=2,
        cv_strategy=strategy,
        scoring_metric="mse",
        preprocess=PreprocessRecipe(scale="standard") if scaled else None,
    )
    if groups is not None:
        kwargs["groups"] = pd.Series(groups, index=np.arange(9000, 9000 + len(train)))
    AuditedRidge.fitted_targets = []
    result = session.nested_cv_score(AuditedRidge(), **kwargs)
    allowed_fit_targets = set()
    outer_scores = []
    for actual, search, (fit_rows, eval_rows) in zip(
        result.outer_folds, searches, outer.split(x, y, groups), strict=True
    ):
        inner_groups = None if groups is None else groups[fit_rows]
        splits = list(inner.split(x.iloc[fit_rows], y.iloc[fit_rows], inner_groups))
        allowed_fit_targets.add(frozenset(y.iloc[fit_rows]))
        candidate_scores = []
        for trial in search.trials:
            scores = []
            for inner_fit, inner_eval in splits:
                allowed_fit_targets.add(frozenset(y.iloc[fit_rows].iloc[inner_fit]))
                reference = Pipeline(
                    [
                        ("scale", StandardScaler() if scaled else "passthrough"),
                        ("model", Ridge(**trial.params)),
                    ]
                )
                reference.fit(x.iloc[fit_rows].iloc[inner_fit], y.iloc[fit_rows].iloc[inner_fit])
                scores.append(
                    mean_squared_error(
                        y.iloc[fit_rows].iloc[inner_eval],
                        reference.predict(x.iloc[fit_rows].iloc[inner_eval]),
                    )
                )
            assert trial.mean_score == pytest.approx(np.mean(scores), rel=1e-10, abs=1e-10)
            assert trial.std_score == pytest.approx(np.std(scores, ddof=1), rel=1e-10, abs=1e-10)
            candidate_scores.append((np.mean(scores), trial.params))
        best_score, best_params = min(candidate_scores, key=lambda row: row[0])
        assert actual.best_params == best_params
        assert actual.inner_best_score == pytest.approx(best_score, rel=1e-10, abs=1e-10)
        assert actual.inner_n_trials == len(candidate_scores)
        assert 2 <= actual.inner_n_trials <= 3  # Three possible genomes, cached on repetition.
        assert (actual.n_train, actual.n_eval) == (len(fit_rows), len(eval_rows))
        reference = Pipeline(
            [
                ("scale", StandardScaler() if scaled else "passthrough"),
                ("model", Ridge(**best_params)),
            ]
        ).fit(x.iloc[fit_rows], y.iloc[fit_rows])
        score = mean_squared_error(y.iloc[eval_rows], reference.predict(x.iloc[eval_rows]))
        assert actual.metrics["mse"] == pytest.approx(score, rel=1e-10, abs=1e-10)
        outer_scores.append(score)
    assert result.mean_metrics["mse"] == pytest.approx(np.mean(outer_scores))
    assert result.std_metrics["mse"] == pytest.approx(np.std(outer_scores, ddof=1))
    assert set(AuditedRidge.fitted_targets) == allowed_fit_targets

    poisoned = frame.copy()
    held_out = [*session.split_plan.test_indices, *session.split_plan.validation_indices]
    poisoned.iloc[held_out, poisoned.columns.get_loc("y")] += 1_000_000
    repeated = _session(poisoned).nested_cv_score(AuditedRidge(), **kwargs)
    assert repeated.mean_metrics == pytest.approx(result.mean_metrics)
    assert [fold.best_params for fold in repeated.outer_folds] == [
        fold.best_params for fold in result.outer_folds
    ]
    assert set(AuditedRidge.fitted_targets) == allowed_fit_targets
    assert session.fit_result is None
