"""Industry backend tests for probabilistic ML (MAPIE / NGBoost)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import MissingExtraError
from buildml.probabilistic.catalog import probabilistic_capability_matrix
from buildml.probabilistic.extras import mapie_available, ngboost_available


def _reg_session() -> Session:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(180, 2))
    y = 1.0 * x[:, 0] + rng.normal(scale=0.3, size=180)
    frame = pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "y": y})
    return (
        Session.ingest(frame)
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=1)
        .scale(method="standard")
    )


def test_capability_matrix_native_always_available() -> None:
    matrix = probabilistic_capability_matrix()
    assert matrix["backends"]["native"]["available"] is True
    assert "bayesian_ridge" in matrix["backends"]["native"]["estimators"]


@pytest.mark.skipif(not mapie_available(), reason="mapie not installed")
def test_mapie_split_regression_intervals() -> None:
    session = _reg_session()
    fit = session.fit_probabilistic(
        backend="mapie",
        estimator="split",
        task="regression",
        alpha=0.1,
    )
    assert fit.backend == "mapie"
    interval = session.predict_interval(partition="test")
    assert interval.lower is not None
    assert len(interval.lower) == len(interval.upper)
    ev = session.evaluate_probabilistic(partition="validation")
    assert ev.metrics.get("interval_coverage") is not None


@pytest.mark.skipif(not mapie_available(), reason="mapie not installed")
def test_mapie_cv_plus_regression() -> None:
    session = _reg_session()
    session.fit_probabilistic(
        backend="mapie",
        estimator="cv_plus",
        task="regression",
        alpha=0.1,
    )
    ev = session.evaluate_probabilistic(partition="test")
    assert "rmse" in ev.metrics
    assert ev.metrics.get("interval_coverage") is not None


@pytest.mark.skipif(not ngboost_available(), reason="ngboost not installed")
def test_ngboost_regressor_nll_crps() -> None:
    session = _reg_session()
    fit = session.fit_probabilistic(
        backend="ngboost",
        estimator="ngboost_regressor",
        conformal=True,
        alpha=0.1,
        n_estimators=40,
        learning_rate=0.1,
    )
    assert fit.backend == "ngboost"
    pred = session.predict_probabilistic(partition="test", return_std=True)
    assert pred.std is not None
    ev = session.evaluate_probabilistic(partition="validation")
    assert "nll" in ev.metrics
    assert "crps" in ev.metrics


def test_mapie_backend_raises_without_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    if mapie_available():
        pytest.skip("mapie is installed in this environment")
    monkeypatch.setattr(
        "buildml.probabilistic.catalog.mapie_available",
        lambda: False,
    )
    monkeypatch.setattr(
        "buildml.probabilistic.catalog.backend_available",
        lambda name: name == "native",
    )
    session = _reg_session()
    with pytest.raises(MissingExtraError, match="probabilistic-industry"):
        session.fit_probabilistic(backend="mapie", estimator="split", task="regression")
