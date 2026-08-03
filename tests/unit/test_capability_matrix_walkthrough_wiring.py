"""Second-pass capability-matrix walkthrough / audit / resolver wiring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from buildml import Session
from buildml.explain.capability_status import (
    CAPABILITY_MATRIX_OPERATIONS,
    attach_capability_matrix,
    capability_introspection_status,
    load_capability_matrix,
    suggest_capability_introspection,
)
from buildml.explain.resolver import resolve_workflow
from buildml.explain.schemas import WorkflowStepStatus
from buildml.session.audit import suggest_next_operations


def _sales_frame(n: int = 90) -> pd.DataFrame:
    t = pd.date_range("2023-06-01", periods=n, freq="D")
    y = 5 + 0.03 * np.arange(n) + np.cos(np.arange(n) / 5.0)
    promo = (np.arange(n) % 14 == 0).astype(float)
    return pd.DataFrame({"clock": t, "promo": promo, "sales": y})


def test_load_capability_matrix_forecast_shape() -> None:
    matrix = load_capability_matrix("forecast_capability_matrix")
    assert "backends" in matrix or "methods" in matrix


def test_attach_capability_matrix_is_idempotent() -> None:
    base = {"enabled": False, "present": False}
    once = attach_capability_matrix(base, "forecast_capability_matrix")
    twice = attach_capability_matrix(once, "forecast_capability_matrix")
    assert once is twice or twice["capability_matrix"] == once["capability_matrix"]
    assert once["capability_operation"] == "forecast_capability_matrix"
    assert "Session.forecast_capability_matrix()" in once["capability_introspection"]


def test_domain_status_blocks_include_capability_matrix() -> None:
    session = (
        Session.ingest(_sales_frame())
        .set_roles({"clock": "time", "promo": "feature", "sales": "target"})
        .time_split(test_size=0.2)
    )
    lazy = session.walkthrough(capability_probe="lazy").to_dict()
    assert lazy["audit_summary"]["capability_probe"] == "lazy"
    assert lazy["rag_status"].get("status") == "idle"
    assert lazy["capability_introspection_status"]["capability_probe"] == "lazy"

    report = session.walkthrough(capability_probe="eager").to_dict()
    for field in (
        "forecasting_status",
        "timeseries_status",
        "rag_status",
        "cbr_status",
        "unsupervised_status",
        "nlp_status",
        "ranking_status",
    ):
        status = report[field]
        assert "capability_matrix" in status, field
        intro = status.get("capability_introspection")
        if intro:
            assert str(intro).startswith("Session."), field
    cap = report["capability_introspection_status"]
    assert cap["n_domains"] >= 20
    assert any(row["operation"] == "forecast_capability_matrix" for row in cap["domains"])


def test_resolver_keeps_capability_matrices_always_available() -> None:
    session = Session.ingest(pd.DataFrame({"a": [1.0], "y": [0]}))
    steps = {step.operation: step for step in resolve_workflow(session)}
    for operation in (
        "forecast_capability_matrix",
        "rl_capability_matrix",
        "nlp_capability_matrix",
    ):
        assert steps[operation].status == WorkflowStepStatus.AVAILABLE


def test_audit_suggests_capability_matrix_before_domain_fit() -> None:
    session = (
        Session.ingest(_sales_frame())
        .set_roles({"clock": "time", "promo": "feature", "sales": "target"})
        .time_split(test_size=0.2)
    )
    suggestions = suggest_next_operations(session, limit=48)
    ops = [item["operation"] for item in suggestions]
    assert "forecast_capability_matrix" in ops
    assert "fit_forecast" in ops
    assert ops.index("forecast_capability_matrix") < ops.index("fit_forecast")


def test_suggest_capability_introspection_respects_history() -> None:
    raw = suggest_capability_introspection([], available_fit_ops={"fit_forecast"})
    assert raw and raw[0]["operation"] == "forecast_capability_matrix"
    after = suggest_capability_introspection(
        [{"operation_id": "forecast_capability_matrix", "sequence": 1}],
        available_fit_ops={"fit_forecast"},
    )
    assert after == []


def test_capability_introspection_lists_all_operations() -> None:
    payload = capability_introspection_status(capability_probe="eager")
    assert CAPABILITY_MATRIX_OPERATIONS <= set(payload["operations"])
    assert "fairness_capability_matrix" in payload["operations"]
