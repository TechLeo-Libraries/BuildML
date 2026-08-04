"""Industry-depth tests for federated backends (Flower skip when absent)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import MissingExtraError, ValidationError
from buildml.federated.catalog import (
    federated_capability_matrix,
    list_federated_methods,
    resolve_backend,
)
from buildml.federated.extras import flwr_available
from buildml.federated.fit import fit_federated
from buildml.federated.results import export_round_history


def _frame(n_clients: int = 6, n_per: int = 40, seed: int = 3) -> pd.DataFrame:
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
        .split(test_size=0.2, validation_size=0.2, random_state=1)
        .scale(method="standard")
    )


def test_capability_matrix_native_always_available() -> None:
    matrix = federated_capability_matrix()
    assert matrix["backends"]["native"]["available"] is True
    assert "fedavg" in list_federated_methods(backend="native")
    assert matrix["backends"]["native"]["secure_aggregation"] is False
    assert matrix["backends"]["flower"]["network_runtime"] is False
    assert "local simulation" in matrix["honesty"].lower() or "local sim" in (
        matrix["honesty"].lower()
    )
    assert "industry_runtime_present" in matrix
    assert "accuracy" in matrix["evaluation_metrics"]["classification"]
    assert "roc_auc" in matrix["evaluation_metrics"]["classification"]


def test_resolve_backend_native_explicit() -> None:
    assert resolve_backend("native", method="fedavg") == "native"


def test_missing_extra_when_flower_requested_without_install() -> None:
    if flwr_available():
        pytest.skip("flwr installed — MissingExtraError path not testable here.")
    with pytest.raises(MissingExtraError, match="federated-industry"):
        resolve_backend("flower", method="fedavg")


@pytest.mark.skipif(not flwr_available(), reason="buildml[federated-industry] flwr not installed")
def test_flower_backend_runs_and_exports_history(tmp_path: Path) -> None:
    session = _session()
    plan, fit = fit_federated(
        session.dataset,
        session.split_plan,
        backend="flower",
        method="fedavg",
        n_rounds=3,
        local_epochs=1,
        reduce_plan=session._reduce_plan,
    )
    assert fit.backend == "flower"
    assert plan.backend == "flower"
    assert len(fit.round_history) >= 1
    assert fit.round_history[0].get("aggregation") == "flwr.server.strategy.aggregate"
    out = export_round_history(plan, tmp_path / "rounds.json")
    assert out.is_file()
    payload = out.read_text(encoding="utf-8")
    assert "round_history" in payload
    assert '"backend": "flower"' in payload


def test_native_round_history_has_client_weights() -> None:
    session = _session()
    plan, fit = fit_federated(
        session.dataset,
        session.split_plan,
        backend="native",
        method="fedavg",
        n_rounds=2,
        reduce_plan=session._reduce_plan,
    )
    assert fit.backend == "native"
    assert "client_weights" in fit.round_history[0]
    assert fit.round_history[0]["weighting"] == "sample_size"


def test_session_backend_routing_native() -> None:
    session = _session()
    fit = session.fit_federated(backend="native", method="fedavg", n_rounds=2)
    assert fit.backend == "native"


def test_evaluate_backend_mismatch_refused() -> None:
    session = _session()
    session.fit_federated(backend="native", method="fedavg", n_rounds=2)
    with pytest.raises(ValidationError, match="does not match"):
        session.evaluate_federated(backend="flower")
