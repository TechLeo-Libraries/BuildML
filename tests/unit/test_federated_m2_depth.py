"""M2 depth coverage for federated low-level APIs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.federated.evaluate import evaluate_federated
from buildml.federated.fit import fit_federated
from buildml.federated.predict import predict_federated


def _frame(n_clients: int = 6, n_per: int = 40, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for client in range(n_clients):
        shift = rng.normal(0, 0.8, size=2)
        for i in range(n_per):
            label = i % 2
            center = shift + (1.1 if label else -1.1)
            x = rng.normal(center, 0.3, size=2)
            rows.append(
                {
                    "x": float(x[0]),
                    "y": float(x[1]),
                    "label": int(label),
                    "client_id": f"c{client}",
                }
            )
    return pd.DataFrame(rows)


def _session() -> Session:
    return (
        Session.ingest(_frame())
        .set_roles(
            {
                "x": "feature",
                "y": "feature",
                "label": "target",
                "client_id": "group",
            }
        )
        .split(test_size=0.2, validation_size=0.2, random_state=2)
        .scale(method="standard")
    )


def test_low_level_logistic_fedavg() -> None:
    session = _session()
    plan, fit = fit_federated(
        session.dataset,
        session.split_plan,
        backend="native",
        method="fedavg",
        estimator="logistic_regression",
        n_rounds=3,
        local_epochs=1,
        reduce_plan=session._reduce_plan,
    )
    assert fit.method == "fedavg"
    assert plan.estimator_ is not None
    ev = evaluate_federated(
        session.dataset,
        plan,
        session.split_plan,
        partition="test",
        per_client=True,
    )
    assert ev.n_rows > 0
    assert "accuracy" in ev.metrics
    pred = predict_federated(
        session.dataset,
        plan,
        session.split_plan,
        partition="test",
    )
    assert len(pred.predictions) == ev.n_rows


def test_fedprox_requires_mu() -> None:
    session = _session()
    with pytest.raises(ValidationError, match="mu > 0"):
        fit_federated(
            session.dataset,
            session.split_plan,
            backend="native",
            method="fedprox",
            mu=0.0,
            reduce_plan=session._reduce_plan,
        )


def test_fedprox_runs() -> None:
    session = _session()
    plan, fit = fit_federated(
        session.dataset,
        session.split_plan,
        backend="native",
        method="fedprox",
        estimator="sgd_classifier",
        mu=0.05,
        n_rounds=2,
        local_epochs=2,
        reduce_plan=session._reduce_plan,
    )
    assert plan.method == "fedprox"
    assert plan.mu == 0.05
    assert fit.n_clients >= 2
    assert any("FedProx" in d or "proximal" in d.lower() for d in plan.disclosures)


def test_explicit_client_column() -> None:
    frame = _frame()
    session = (
        Session.ingest(frame)
        .set_roles(
            {
                "x": "feature",
                "y": "feature",
                "label": "target",
                "client_id": "feature",
            }
        )
        .split(test_size=0.2, validation_size=0.2, random_state=0)
        .scale(method="standard")
    )
    plan, fit = fit_federated(
        session.dataset,
        session.split_plan,
        backend="native",
        method="fedavg",
        client_column="client_id",
        n_rounds=2,
        reduce_plan=session._reduce_plan,
    )
    assert plan.client_column == "client_id"
    assert "client_id" not in plan.columns
    assert fit.n_clients >= 2


def test_refuse_unknown_method() -> None:
    session = _session()
    with pytest.raises(ValidationError, match="Unknown federated method"):
        fit_federated(
            session.dataset,
            session.split_plan,
            method="scaffold",  # type: ignore[arg-type]
        )


def test_regression_ridge_path() -> None:
    rng = np.random.default_rng(9)
    rows = []
    for client in range(5):
        bias = float(rng.normal(0, 1.0))
        for _ in range(30):
            x = float(rng.normal())
            y = 1.5 * x + bias + float(rng.normal(0, 0.2))
            rows.append({"x": x, "y": y, "client_id": client})
    frame = pd.DataFrame(rows)
    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "y": "target", "client_id": "group"})
        .split(test_size=0.25, random_state=0)
        .scale(method="standard")
    )
    plan, fit = fit_federated(
        session.dataset,
        session.split_plan,
        backend="native",
        method="fedavg",
        estimator="ridge",
        n_rounds=3,
        reduce_plan=session._reduce_plan,
    )
    assert plan.task == "regression"
    ev = evaluate_federated(
        session.dataset,
        plan,
        session.split_plan,
        partition="test",
        per_client=False,
    )
    assert "r2" in ev.metrics or "mae" in ev.metrics
    assert fit.n_clients >= 2


def test_explain_before_prereq() -> None:
    session = _session()
    before = session.explain("evaluate_federated", moment="before")
    assert before.prerequisite_status.get("federated-plan") is False
    session.fit_federated(backend="native", method="fedavg", n_rounds=2)
    after = session.explain("evaluate_federated", moment="before")
    assert after.prerequisite_status.get("federated-plan") is True
