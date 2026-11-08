"""Federated round budgets and proximal strength preserve the training contract."""

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError


@pytest.fixture(params=["native", "flower"])
def backend(request):
    if request.param == "flower":
        pytest.importorskip("flwr")
    return request.param


@pytest.fixture
def session():
    frame = pd.DataFrame(
        {
            "x": [0, 1, 2, 3, 4, 5, -2, -1, 0, 1, 0, 1, -3, 2, 8],
            "y": [1, 3, 5, 7, 9, 11, 7, 6, 5, 4, 900, -900, 100, 101, 102],
            "client": ["a"] * 6 + ["b"] * 4 + ["tiny"] * 2 + ["held_out"] * 3,
        }
    )
    return (
        Session.ingest(frame)
        .set_roles({"x": "feature", "y": "target", "client": "group"})
        .inject_split(train_indices=list(range(12)), test_indices=[12, 13, 14])
    )


def _fit(session, backend, **overrides):
    settings = {
        "backend": backend,
        "method": "fedprox",
        "estimator": "linear_regression",
        "n_rounds": np.int64(3),
        "local_epochs": np.int32(2),
        "min_client_rows": np.int64(3),
        "client_fraction": np.float64(1),
        "mu": np.float64(0.4),
        **overrides,
    }
    return session.federated.fit(**settings)


@pytest.mark.parametrize("method", ["fedavg", "fedprox"])
def test_federated_rounds_match_weighted_client_lines_on_unseen_rows(session, backend, method):
    fit = _fit(session, backend, method=method)
    plan = session.federated_plan
    # The two eligible client lines are exact. The larger client seeds the global line.
    client_a = np.array([2.0, 1.0])
    client_b = np.array([-1.0, 5.0])
    average = (6 * client_a + 4 * client_b) / 10
    mu = 0.4 if method == "fedprox" else 0.0
    expected = average + mu**3 * (client_a - average)
    assert fit.backend == backend and plan.mu == mu
    assert len(fit.round_history) == 3
    for index, entry in enumerate(fit.round_history, start=1):
        assert entry["round"] == index
        assert entry["n_samples"] == 10
        assert entry["client_weights"] == {"a": 6.0, "b": 4.0}
    assert any("Skipped 1 client" in warning for warning in fit.warnings)
    predicted = session.federated.predict(partition="test").predictions
    expected_predictions = np.array([-3, 2, 8]) * expected[0] + expected[1]
    np.testing.assert_allclose(predicted, expected_predictions, atol=1e-12, rtol=0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_rounds", 2.5),
        ("local_epochs", True),
        ("min_client_rows", 2.5),
        ("client_fraction", True),
        ("client_fraction", "0.5"),
        ("mu", True),
        ("mu", float("nan")),
        ("mu", float("inf")),
    ],
)
def test_invalid_federated_request_preserves_the_prior_plan(session, backend, field, value):
    _fit(session, backend)
    original = session.federated_plan
    history = list(session.history)
    with pytest.raises(ValidationError, match=field):
        _fit(session, backend, **{field: value})
    assert session.federated_plan is original
    assert session.history == history
