"""Unit coverage for the unsupervised clustering thin slice."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import LeakageError, ValidationError
from buildml.explain.catalog import OPERATION_CATALOG
from buildml.unsupervised.checkpoint import (
    BUNDLE_FORMAT,
    load_unsupervised_bundle,
    save_unsupervised_bundle,
)
from buildml.unsupervised.cluster import assign_clusters, fit_clusterer
from buildml.unsupervised.evaluate import evaluate_clustering


def _blob_frame(n_per: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal([0.0, 0.0], 0.35, size=(n_per, 2))
    b = rng.normal([3.0, 3.0], 0.35, size=(n_per, 2))
    frame = pd.DataFrame(np.vstack([a, b]), columns=["x", "y"])
    frame["segment"] = [0] * n_per + [1] * n_per
    return frame


def _ready_session(*, validation: bool = False) -> Session:
    kwargs = {"test_size": 0.25, "random_state": 0}
    if validation:
        kwargs["validation_size"] = 0.2
    return (
        Session.ingest(_blob_frame())
        .set_roles({"x": "feature", "y": "feature", "segment": "ignore"})
        .split(**kwargs)
        .scale(method="standard")
    )


def test_core_import_does_not_require_extra() -> None:
    import buildml
    import buildml.unsupervised as unsup

    assert hasattr(buildml, "Session")
    assert hasattr(unsup, "fit_clusterer")
    assert hasattr(Session, "fit_clusters")


def test_catalog_covers_unsupervised_operations() -> None:
    for name in (
        "fit_clusters",
        "assign_clusters",
        "evaluate_clusters",
        "save_unsupervised_bundle",
        "load_unsupervised_bundle",
    ):
        assert name in OPERATION_CATALOG
    assert (
        "unsupervised-train-fit-holdout-assign"
        in OPERATION_CATALOG["fit_clusters"].concept_links
    )
    assert "cluster-validity-not-truth" in OPERATION_CATALOG["evaluate_clusters"].concept_links
    assert (
        "unsupervised-bundle-boundary"
        in OPERATION_CATALOG["save_unsupervised_bundle"].concept_links
    )
    assert "pca-cluster-integration" in OPERATION_CATALOG["fit_clusters"].concept_links


def test_fit_requires_split() -> None:
    session = Session.ingest(_blob_frame()).set_roles(
        {"x": "feature", "y": "feature", "segment": "ignore"}
    )
    with pytest.raises((LeakageError, ValidationError)):
        session.fit_clusters(method="kmeans", n_clusters=2)


def test_kmeans_fit_assign_evaluate_and_bundle(tmp_path: Path) -> None:
    session = _ready_session()
    fit = session.fit_clusters(method="kmeans", n_clusters=2, random_state=0)
    assert fit.n_clusters == 2
    assert fit.assign_strategy == "native"
    assert session.cluster_plan is not None
    assert sum(fit.cluster_sizes.values()) == fit.n_train_rows

    assigned = session.assign_clusters(partition="test")
    assert assigned.n_rows > 0
    assert assigned.assign_strategy == "native"
    assert set(assigned.labels).issubset({0, 1})

    metrics = session.evaluate_clusters(
        partition="test", external_label_column="segment"
    )
    assert "silhouette" in metrics.metrics
    assert "adjusted_rand_index" in metrics.external_metrics
    assert 0.0 <= metrics.external_metrics["adjusted_rand_index"] <= 1.0

    path = session.save_unsupervised_bundle(tmp_path / "unsup")
    assert (path / "meta.json").is_file()
    assert (path / "cluster_plan.joblib").is_file()
    plan = load_unsupervised_bundle(path)
    assert plan.method == "kmeans"
    assert plan.columns == session.cluster_plan.columns

    restored = Session.ingest(session.to_pandas()).set_roles(
        {"x": "feature", "y": "feature", "segment": "ignore"}
    )
    restored.split(test_size=0.25, random_state=0).scale(method="standard")
    restored.load_unsupervised_bundle(path)
    again = restored.assign_clusters(partition="test")
    assert again.labels == assigned.labels

    with pytest.raises(ValidationError, match=BUNDLE_FORMAT):
        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "meta.json").write_text('{"format": "buildml.rag_bundle.v1"}', encoding="utf-8")
        (bad / "cluster_plan.joblib").write_bytes(b"not-a-real-joblib")
        # force format check before joblib load when meta is wrong
        load_unsupervised_bundle(bad)


def test_prefer_reduce_components() -> None:
    session = (
        Session.ingest(_blob_frame())
        .set_roles({"x": "feature", "y": "feature", "segment": "ignore"})
        .split(test_size=0.25, random_state=0)
        .scale(method="standard")
        .reduce_dimensions(method="pca", n_components=2, prefix="pc")
    )
    fit = session.fit_clusters(method="kmeans", n_clusters=2, prefer_reduce_components=True)
    assert fit.used_reduce_components is True
    assert all(c.startswith("pc_") for c in fit.columns)


def test_agglomerative_and_dbscan_disclosures() -> None:
    session = _ready_session()
    agg = session.fit_clusters(method="agglomerative", n_clusters=2)
    assert agg.assign_strategy == "nearest_centroid"
    assert any("centroid" in d.lower() for d in agg.disclosures)
    labels = session.assign_clusters(partition="test")
    assert labels.n_rows > 0

    db = session.fit_clusters(method="dbscan", eps=1.0, min_samples=3, n_clusters=None)
    assert db.assign_strategy == "nearest_core"
    assert any("core" in d.lower() for d in db.disclosures)
    assigned = session.assign_clusters(partition="test")
    assert assigned.n_rows > 0


def test_attach_requires_all_partition() -> None:
    session = _ready_session()
    session.fit_clusters(method="kmeans", n_clusters=2)
    with pytest.raises(ValidationError, match="partition='all'"):
        session.assign_clusters(partition="test", attach=True)
    attached = session.assign_clusters(partition="all", attach=True)
    assert attached.attached is True
    assert "cluster_id" in session.dataset.columns


def test_low_level_fit_clusterer_matches_session(tmp_path: Path) -> None:
    session = _ready_session()
    plan, fit = fit_clusterer(
        session.dataset,
        session.split_plan,
        method="kmeans",
        n_clusters=2,
        random_state=0,
        prefer_reduce_components=False,
    )
    _, assign = assign_clusters(
        session.dataset, plan, session.split_plan, partition="test"
    )
    eval_result = evaluate_clustering(
        session.dataset, plan, session.split_plan, partition="test"
    )
    assert fit.n_clusters == 2
    assert assign.n_rows == eval_result.n_rows
    path = save_unsupervised_bundle(tmp_path / "direct", plan, fit_result=fit)
    restored = load_unsupervised_bundle(path)
    assert restored.method == plan.method
    assert restored.columns == plan.columns
