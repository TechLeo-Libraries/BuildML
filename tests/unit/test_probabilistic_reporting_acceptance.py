"""Regression checks for the independently reproduced reporting defects."""

from dataclasses import replace
from importlib import import_module

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError


def fitted(task="regression", conformal=True):
    rng = np.random.default_rng(9)
    x = rng.normal(size=180)
    y = 2 * x + rng.normal(size=180)
    if task == "classification":
        y = (y > 0).astype(int)
    session = Session.ingest(pd.DataFrame({"x": x, "y": y})).set_roles(
        {"x": "feature", "y": "target"}
    ).split(test_size=.25, validation_size=.2, random_state=0)
    session.probabilistic.fit(
        estimator="gaussian_nb" if task == "classification" else "bayesian_ridge",
        conformal=conformal, alpha=.1,
    )
    return session


@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("method", ["split_conformal", "both"])
def test_conformal_alpha_override_refused(task, method):
    session = fitted(task)
    with pytest.raises(ValidationError, match="differs from calibrated alpha"):
        session.probabilistic.predict_interval(alpha=.01, method=method)
    with pytest.raises(ValidationError, match="differs from calibrated alpha"):
        session.probabilistic.evaluate(alpha=.01)
    result = session.probabilistic.predict_interval(alpha=.1, method=method)
    assert result.alpha == result.to_dict()["alpha"] == .1


def test_posterior_intervals_recompute_for_requested_alpha():
    session = fitted()
    a = session.probabilistic.predict_interval(alpha=.1, method="posterior_std")
    b = session.probabilistic.predict_interval(alpha=.01, method="posterior_std")
    assert b.alpha == .01
    assert np.all(np.subtract(b.upper, b.lower) > np.subtract(a.upper, a.lower))
    uncalibrated = fitted(conformal=False)
    assert uncalibrated.probabilistic.evaluate(alpha=.01).alpha == .01


@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("partition", ["train", "all", "validation", "test"])
def test_population_disclosures_do_not_certify_holdout(task, partition):
    result = fitted(task).probabilistic.evaluate(partition=partition)
    assert any(f"population: {partition} rows" in s for s in result.disclosures)
    assert any("do not establish independence" in s for s in result.disclosures)
    assert not any("rows were never" in s for s in result.disclosures)
    if partition in {"train", "all"}:
        assert any("not as independent holdout evidence" in s for s in result.warnings)
    assert result.to_dict()["disclosures"] == list(result.disclosures)


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_evaluation_preserves_interval_warnings(task, monkeypatch):
    module = import_module("buildml.probabilistic.evaluate")
    original = module.predict_interval

    def with_warning(*args, **kwargs):
        return replace(original(*args, **kwargs), warnings=("backend interval limitation",))

    monkeypatch.setattr(module, "predict_interval", with_warning)
    result = fitted(task).probabilistic.evaluate()
    assert "backend interval limitation" in result.warnings


def test_loaded_bundle_on_new_data_does_not_claim_holdout(tmp_path):
    session = fitted()
    session.probabilistic.save_bundle(tmp_path / "model")
    other = Session.ingest(pd.DataFrame({"x": [0., 1., 2.], "y": [0., 2., 4.]}))
    other.probabilistic.load_bundle(tmp_path / "model", trusted=True)
    result = other.probabilistic.evaluate(partition="all")
    assert result.n_rows == 3
    assert any("verify that provenance" in s for s in result.disclosures)
    assert any("diagnostic" in s for s in result.warnings)


@pytest.mark.parametrize("alpha", [0, 1, float("nan"), -0.1])
def test_invalid_evaluation_alpha_is_not_silently_skipped(alpha):
    with pytest.raises(ValidationError, match="alpha must be"):
        fitted(conformal=False).probabilistic.evaluate(alpha=alpha)
