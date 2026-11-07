"""Decomposition is descriptive, reconstructs observations, and isolates holdout rows."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.timeseries.decompose import decompose_series


@pytest.mark.parametrize("method", ["moving_average", "stl", "classical"])
def test_train_components_are_invariant_to_future_observations(method):
    if method != "moving_average":
        pytest.importorskip("statsmodels")
    time = np.arange(120)
    values = 20 + .07 * time + 3 * np.sin(2 * np.pi * time / 6)
    frame = pd.DataFrame({"clock": pd.date_range("2024-01-01", periods=120), "value": values})

    def analyze(data):
        session = (Session.ingest(data).set_roles({"clock": "time", "value": "target"})
                   .time_split(test_size=.2, validation_size=.2))
        result = session.timeseries.decompose(scope="train", seasonal_period=6, decompose_method=method)
        return session, result.decompose

    session, original = analyze(frame)
    changed = frame.copy()
    future = [*session.split_plan.validation_indices, *session.split_plan.test_indices]
    changed.loc[future, "value"] += 100_000
    _, perturbed = analyze(changed)
    assert original.method == method
    assert original.n_points == len(session.split_plan.train_indices)
    for component in ("trend", "seasonal", "residual", "observed"):
        np.testing.assert_allclose(getattr(original, component), getattr(perturbed, component))
    reconstructed = np.asarray(original.trend) + original.seasonal + original.residual
    np.testing.assert_allclose(reconstructed, original.observed, atol=1e-10)
    assert all(np.isfinite(getattr(original, component)).all()
               for component in ("trend", "seasonal", "residual"))
    # An interior full cycle retains the known seasonal phase and amplitude.
    np.testing.assert_allclose(np.asarray(original.seasonal)[24:30],
                               3 * np.sin(2 * np.pi * np.arange(6) / 6), atol=.6)


@pytest.mark.parametrize("method", ["stl", "classical"])
def test_missing_optional_backend_is_disclosed_and_reconstructs_data(method, monkeypatch):
    monkeypatch.setattr("buildml.timeseries.decompose.statsmodels_available", lambda: False)
    values = 4 + np.sin(np.arange(30) * np.pi / 3)
    result = decompose_series(values, method=method, seasonal_period=6)
    assert result.method == "moving_average"
    assert any("falling back" in warning and method in warning for warning in result.warnings)
    np.testing.assert_allclose(np.asarray(result.trend) + result.seasonal + result.residual, values)
