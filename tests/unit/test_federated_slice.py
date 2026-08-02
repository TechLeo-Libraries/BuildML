"""Slice tests for federated Session API + catalog + bundle boundary."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import LeakageError, ValidationError
from buildml.explain.catalog import OPERATION_CATALOG
from buildml.federated.checkpoint import BUNDLE_FORMAT, load_federated_bundle
from buildml.ai.tools import registered_tool_names


def _frame(n_clients: int = 6, n_per: int = 36, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for client in range(n_clients):
        shift = rng.normal(0, 0.7, size=2)
        for i in range(n_per):
            label = i % 2
            center = shift + (1.0 if label else -1.0)
            x = rng.normal(center, 0.35, size=2)
            rows.append(
                {
                    "x": float(x[0]),
                    "y": float(x[1]),
                    "label": int(label),
                    "client_id": client,
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


def test_public_surface_and_catalog() -> None:
    import buildml.federated as federated

    assert hasattr(federated, "fit_federated")
    assert hasattr(Session, "fit_federated")
    for name in (
        "fit_federated",
        "evaluate_federated",
        "predict_federated",
        "save_federated_bundle",
        "load_federated_bundle",
    ):
        assert name in OPERATION_CATALOG
    assert "federated-simulation" in OPERATION_CATALOG["fit_federated"].concept_links
    assert (
        "federated-bundle-boundary"
        in OPERATION_CATALOG["save_federated_bundle"].concept_links
    )
    tools = set(registered_tool_names())
    for name in (
        "fit_federated",
        "evaluate_federated",
        "predict_federated",
        "save_federated_bundle",
        "load_federated_bundle",
    ):
        assert name in tools
    assert BUNDLE_FORMAT == "buildml.federated_bundle.v1"


def test_session_fedavg_loop_and_bundle(tmp_path: Path) -> None:
    session = _session()
    fit = session.fit_federated(
        method="fedavg",
        estimator="sgd_classifier",
        n_rounds=3,
        local_epochs=2,
    )
    assert session.federated_plan is not None
    assert session.federated_plan.client_column == "client_id"
    assert fit.n_clients >= 2
    assert len(fit.round_history) >= 1
    assert "client_id" not in session.federated_plan.columns

    ev = session.evaluate_federated(partition="test", per_client=True)
    assert "accuracy" in ev.metrics
    assert session.federated_eval_result is not None

    preds = session.predict_federated(partition="test")
    assert len(preds.predictions) == ev.n_rows

    before = session.explain("evaluate_federated", moment="before")
    assert before.prerequisite_status.get("federated-plan") is True

    bundle = session.save_federated_bundle(tmp_path / "federated_bundle")
    plan = load_federated_bundle(bundle)
    assert plan.method == "fedavg"

    restored = Session.ingest(_frame()).set_roles(
        {
            "x": "feature",
            "y": "feature",
            "label": "target",
            "client_id": "group",
        }
    )
    restored._split_plan = session.split_plan
    restored._dataset = session.dataset
    restored.load_federated_bundle(bundle)
    assert restored.federated_plan is not None
    assert restored.federated_plan.estimator_name == "sgd_classifier"


def test_refuse_without_split() -> None:
    session = Session.ingest(_frame()).set_roles(
        {
            "x": "feature",
            "y": "feature",
            "label": "target",
            "client_id": "group",
        }
    )
    with pytest.raises(LeakageError):
        session.fit_federated()


def test_refuse_without_client_column() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles(
            {
                "x": "feature",
                "y": "feature",
                "label": "target",
                "client_id": "feature",
            }
        )
        .split(test_size=0.2, random_state=0)
    )
    with pytest.raises(ValidationError, match="client/group"):
        session.fit_federated()
