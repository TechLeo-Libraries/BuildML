"""Additive Session discoverability API tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.core.errors import ValidationError
import pytest


def test_list_capabilities_groups_domains() -> None:
    caps = Session.list_capabilities()
    assert caps["n_domains"] >= 10
    domains = {row["domain"] for row in caps["domains"]}
    assert "fairness" in domains
    assert "online" in domains
    assert "activelearning" in domains
    fairness = Session.list_capabilities(domain="fairness", include_matrices=True)
    assert fairness["n_domains"] == 1
    assert fairness["domains"][0]["matrix"]["default_backend"] == "native"


def test_describe_method_fairness_and_classical() -> None:
    fair = Session.describe_method("evaluate_fairness")
    assert fair["name"] == "evaluate_fairness"
    assert fair["capability_matrix_operation"] == "fairness_capability_matrix"
    assert fair["preferred_path"] == "session.fairness.evaluate"
    assert fair["stability_tier"] == "domain"
    assert fair["flat_deprecated"] is True
    assert "summary" in fair
    fit = Session.describe_method("fit")
    assert fit["name"] == "fit"
    assert fit["summary"]
    assert fit["preferred_path"] == "session.classical.fit"
    assert fit["flat_deprecated"] is False


def test_describe_method_unknown_raises() -> None:
    with pytest.raises(ValidationError, match="Unknown Session method"):
        Session.describe_method("definitely_not_a_real_method_zz")


def test_list_active_domains_after_fit() -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=80),
            "y": rng.integers(0, 2, size=80),
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .fit(LogisticRegression(max_iter=200), task="classification")
    )
    active = session.list_active_domains()
    assert "classical" in active["active_domains"]
    assert "fairness" in active["idle_probed_domains"]
