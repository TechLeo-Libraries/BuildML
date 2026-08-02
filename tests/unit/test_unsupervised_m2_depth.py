"""Unit coverage for unsupervised M2 depth (PCA, leakage, metrics, explain)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import LeakageError, ValidationError
from buildml.explain.catalog import OPERATION_CATALOG
from buildml.unsupervised.explain_hooks import unsupervised_status_for_session


def _frame(n_per: int = 50, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal([0.0, 0.0, 0.0], 0.4, size=(n_per, 3))
    b = rng.normal([2.8, 2.8, 2.8], 0.4, size=(n_per, 3))
    frame = pd.DataFrame(np.vstack([a, b]), columns=["a", "b", "c"])
    frame["segment"] = [0] * n_per + [1] * n_per
    return frame


def test_catalog_parameter_surface() -> None:
    params = {p.name for p in OPERATION_CATALOG["fit_clusters"].parameters}
    assert "method" in params
    assert "prefer_reduce_components" in params
    assert "eps" in params
    eval_params = {p.name for p in OPERATION_CATALOG["evaluate_clusters"].parameters}
    assert "external_label_column" in eval_params


def test_refuse_null_features() -> None:
    frame = _frame()
    frame.loc[0, "a"] = np.nan
    session = (
        Session.ingest(frame)
        .set_roles({"a": "feature", "b": "feature", "c": "feature", "segment": "ignore"})
        .split(test_size=0.25, random_state=0)
    )
    with pytest.raises(ValidationError, match="non-null"):
        session.fit_clusters(method="kmeans", n_clusters=2)


def test_target_column_excluded_from_auto_features() -> None:
    frame = _frame()
    session = (
        Session.ingest(frame)
        .set_roles({"a": "feature", "b": "feature", "c": "feature", "segment": "target"})
        .split(test_size=0.25, random_state=0, stratify=True)
        .scale(method="standard")
    )
    fit = session.fit_clusters(method="kmeans", n_clusters=2)
    assert "segment" not in fit.columns


def test_validation_fallback_to_test() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"a": "feature", "b": "feature", "c": "feature", "segment": "ignore"})
        .split(test_size=0.25, random_state=0)  # no validation carve
        .scale(method="standard")
    )
    session.fit_clusters(method="kmeans", n_clusters=2)
    result = session.evaluate_clusters(partition="validation")
    assert result.partition == "test"


def test_train_partition_eval_recommends_holdout() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"a": "feature", "b": "feature", "c": "feature", "segment": "ignore"})
        .split(test_size=0.25, validation_size=0.2, random_state=0)
        .scale(method="standard")
    )
    session.fit_clusters(method="kmeans", n_clusters=2)
    result = session.evaluate_clusters(partition="train")
    assert any("optimistic" in r.lower() or "train" in r.lower() for r in result.recommendations)


def test_explicit_columns_override_reduce_preference() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"a": "feature", "b": "feature", "c": "feature", "segment": "ignore"})
        .split(test_size=0.25, random_state=0)
        .scale(method="standard")
        .reduce_dimensions(method="pca", n_components=2, prefix="pc", drop_input_columns=False)
    )
    fit = session.fit_clusters(
        method="kmeans",
        n_clusters=2,
        columns=["a", "b"],
        prefer_reduce_components=True,
    )
    assert fit.used_reduce_components is False
    assert list(fit.columns) == ["a", "b"]


def test_status_and_explain_prerequisites() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"a": "feature", "b": "feature", "c": "feature", "segment": "ignore"})
        .split(test_size=0.25, random_state=0)
        .scale(method="standard")
    )
    before = session.explain("assign_clusters", moment="before")
    assert before.prerequisite_status.get("cluster-plan") is False
    session.fit_clusters(method="kmeans", n_clusters=2)
    after = session.explain("assign_clusters", moment="before")
    assert after.prerequisite_status.get("cluster-plan") is True
    status = unsupervised_status_for_session(session)
    assert status["has_cluster_plan"] is True
    assert status["method"] == "kmeans"


def test_n_clusters_exceeds_train_rows() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0]})
    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "y": "feature"})
        .split(test_size=0.5, random_state=0)
        .scale(method="standard")
    )
    n_train = len(session.partition("train"))
    with pytest.raises(ValidationError, match="n_train"):
        session.fit_clusters(method="kmeans", n_clusters=n_train + 1)


def test_fit_without_split_raises_leakage() -> None:
    session = Session.ingest(_frame()).set_roles(
        {"a": "feature", "b": "feature", "c": "feature", "segment": "ignore"}
    )
    with pytest.raises(LeakageError):
        session.assert_can_fit("train")
