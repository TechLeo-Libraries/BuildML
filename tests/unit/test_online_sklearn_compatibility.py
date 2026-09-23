"""Version compatibility for scikit-learn's passive-aggressive schedules."""

import numpy as np
import pytest

from buildml.online.adapters import sklearn as adapter


@pytest.mark.parametrize(
    ("version", "expected"),
    [("1.7.2", False), ("1.8rc1", True), ("1.9.dev0", True), ("2.0.0", True)],
)
def test_pa_schedule_version_handles_prereleases(monkeypatch, version, expected):
    monkeypatch.setattr(adapter.sklearn, "__version__", version)
    assert adapter._has_pa_learning_rate() is expected


@pytest.mark.parametrize("task", ["classifier", "regressor"])
def test_passive_aggressive_incremental_updates_are_finite(task):
    model = adapter.build_sklearn_estimator(f"passive_aggressive_{task}", 0)
    x = np.array([[0.0, 1.0], [1.0, 0.0], [0.2, 0.8], [0.8, 0.2]])
    y = np.array([0, 1, 0, 1])
    kwargs = {"classes": np.array([0, 1])} if task == "classifier" else {}
    model.partial_fit(x, y, **kwargs)
    before = model.coef_.copy()
    model.partial_fit(x, 1 - y, **kwargs)
    assert np.isfinite(model.predict(x)).all()
    assert not np.array_equal(before, model.coef_)
