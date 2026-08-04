"""Namespaced Session facades, discovery tiers, and flat deprecation warnings."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.session.facade_registry import (
    DEPRECATED_FLAT_ACTIONS,
    DOMAIN_FACADES,
    resolve_operation_name,
)
from buildml.session.facades import list_facades, preferred_path


def test_all_domains_have_facades() -> None:
    catalog = list_facades()
    assert catalog["n_facades"] == len(DOMAIN_FACADES) == 35
    session = object.__new__(Session)
    for attr in DOMAIN_FACADES:
        facade = getattr(session, attr)
        assert facade.domain == DOMAIN_FACADES[attr]["mixin_key"]
        assert set(facade.__dir__()) == set(DOMAIN_FACADES[attr]["bindings"])


def test_fairness_facade_parity_and_no_warning_on_facade() -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=120),
            "group": rng.integers(0, 2, size=120),
            "y": rng.integers(0, 2, size=120),
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "group": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .fit(LogisticRegression(max_iter=200), task="classification")
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        report = session.fairness.evaluate(sensitive_column="group", positive_label=1)
    assert report is not None
    assert not any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert session.fairness.last_report is report


def test_flat_domain_action_emits_deprecation_warning() -> None:
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=120),
            "group": rng.integers(0, 2, size=120),
            "y": rng.integers(0, 2, size=120),
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "group": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=1)
        .fit(LogisticRegression(max_iter=200), task="classification")
    )
    assert "evaluate_fairness" in DEPRECATED_FLAT_ACTIONS
    with pytest.warns(DeprecationWarning, match="session.fairness.evaluate"):
        session.evaluate_fairness(sensitive_column="group", positive_label=1)


def test_classical_flat_fit_does_not_warn() -> None:
    rng = np.random.default_rng(2)
    frame = pd.DataFrame(
        {"x": rng.normal(size=80), "y": rng.integers(0, 2, size=80)}
    )
    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=2)
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        session.fit(LogisticRegression(max_iter=200), task="classification")
        session.classical.fit(
            LogisticRegression(max_iter=200), task="classification"
        )
    assert not any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_discovery_preferred_path_and_tiers() -> None:
    caps = Session.list_capabilities(domain="fairness")
    assert caps["n_domains"] == 1
    row = caps["domains"][0]
    assert row["preferred_facade"] == "session.fairness.capability_matrix"
    assert row["stability_tier"] == "domain"
    assert caps["facades"]["n_facades"] == 35

    desc = Session.describe_method("evaluate_fairness")
    assert desc["preferred_path"] == "session.fairness.evaluate"
    assert desc["stability_tier"] == "domain"
    assert desc["flat_deprecated"] is True

    desc_facade = Session.describe_method("fairness.evaluate")
    assert desc_facade["name"] == "evaluate_fairness"
    assert desc_facade["preferred_path"] == "session.fairness.evaluate"

    fit_desc = Session.describe_method("fit")
    assert fit_desc["preferred_path"] == "session.classical.fit"
    assert fit_desc["flat_deprecated"] is False
    assert fit_desc["stability_tier"] == "core"


def test_explore_and_audit_collision_renames() -> None:
    assert "explore" in DOMAIN_FACADES
    assert "audit" in DOMAIN_FACADES
    assert "eda" not in DOMAIN_FACADES
    assert "workflow" not in DOMAIN_FACADES
    assert preferred_path("eda") == "session.explore.run"
    assert preferred_path("walkthrough") == "session.audit.walkthrough"
    # Flat methods remain public callables for teaching sync.
    assert callable(Session.eda)
    assert callable(Session.workflow)


def test_anomaly_facade_delegates_capability_matrix() -> None:
    session = object.__new__(Session)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        via_facade = session.anomaly.capability_matrix()
    assert isinstance(via_facade, dict)
    assert "default_backend" in via_facade or "backends" in via_facade
    assert not any(issubclass(w.category, DeprecationWarning) for w in caught)
    with pytest.warns(DeprecationWarning, match="session.anomaly.capability_matrix"):
        Session.anomaly_capability_matrix()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("evaluate_fairness", "evaluate_fairness"),
        ("fairness.evaluate", "evaluate_fairness"),
        ("session.fairness.evaluate", "evaluate_fairness"),
        ("eda", "eda"),
        ("explore.run", "eda"),
        ("session.audit.walkthrough", "walkthrough"),
        ("not_a_real_op_zz", "not_a_real_op_zz"),
    ],
)
def test_resolve_operation_name_dual_forms(raw: str, expected: str) -> None:
    assert resolve_operation_name(raw) == expected


def test_domain_variable_shadow_warns_once() -> None:
    rag = Session.ingest(
        pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [0, 1, 0]})
    ).set_roles({"x": "feature", "y": "target"})
    with pytest.warns(UserWarning, match="collides with the namespaced facade"):
        _ = rag.rag
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        _ = rag.rag
        _ = rag.anomaly
    assert not any("collides with the namespaced facade" in str(w.message) for w in caught)


def test_list_facades_discloses_surface_policy() -> None:
    disclosures = " ".join(list_facades()["disclosures"])
    assert "3.0" in disclosures
    assert "organized" in disclosures.lower() or "not reduced" in disclosures.lower()
