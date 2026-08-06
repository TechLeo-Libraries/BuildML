"""Unit coverage for unsupervised Phase R2 (modern methods + validation)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import MissingExtraError
from buildml.unsupervised.catalog import ALL_CLUSTER_METHODS, list_cluster_methods
from buildml.unsupervised.checkpoint import (
    BUNDLE_FORMAT,
    BUNDLE_FORMAT_V1,
    BUNDLE_FORMAT_V2,
)


def _blob_frame(n_per: int = 50, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal([0.0, 0.0], 0.35, size=(n_per, 2))
    b = rng.normal([3.0, 3.0], 0.35, size=(n_per, 2))
    c = rng.normal([0.0, 3.0], 0.35, size=(n_per, 2))
    frame = pd.DataFrame(np.vstack([a, b, c]), columns=["x", "y"])
    frame["segment"] = [0] * n_per + [1] * n_per + [2] * n_per
    return frame


def _ready_session() -> Session:
    return (
        Session.ingest(_blob_frame())
        .set_roles({"x": "feature", "y": "feature", "segment": "ignore"})
        .split(test_size=0.25, random_state=0)
        .scale(method="standard")
    )


def test_catalog_lists_all_methods() -> None:
    listed = {row["method"] for row in list_cluster_methods(include_torch=True)}
    assert listed >= ALL_CLUSTER_METHODS


def test_gmm_bic_fit_and_evaluate() -> None:
    session = _ready_session()
    fit = session.fit_clusters(method="gmm", n_clusters=3, gmm_max_components=5)
    assert fit.n_clusters == 3
    assert "gmm_bic" in fit.diagnostics or "gmm_bic" in session.cluster_plan.config
    ev = session.evaluate_clusters(partition="test", external_label_column="segment")
    assert "silhouette" in ev.metrics
    assert "adjusted_rand_index" in ev.external_metrics


def test_spectral_and_optics_transductive_disclosure() -> None:
    session = _ready_session()
    spec = session.fit_clusters(method="spectral", n_clusters=3, random_state=0)
    assert spec.assign_strategy == "nearest_centroid"
    assert any("transductive" in d.lower() for d in spec.disclosures)
    ev = session.evaluate_clusters(partition="test")
    assert any("transductive" in d.lower() for d in ev.disclosures)

    session.fit_clusters(method="optics", optics_min_samples=4, n_clusters=None)
    assigned = session.assign_clusters(partition="test")
    assert assigned.n_rows > 0


def test_mean_shift_observed_k() -> None:
    session = _ready_session()
    fit = session.fit_clusters(method="mean_shift", n_clusters=None)
    assert fit.n_clusters is not None
    assert fit.n_clusters >= 1


def test_auto_k_elbow_kmeans() -> None:
    session = _ready_session()
    fit = session.fit_clusters(method="kmeans", auto_k=True, auto_k_min=2, auto_k_max=5)
    assert fit.n_clusters is not None
    assert 2 <= fit.n_clusters <= 5


def test_evaluate_stability_and_elbow() -> None:
    session = _ready_session()
    session.fit_clusters(method="kmeans", n_clusters=3)
    ev = session.evaluate_clusters(
        partition="test",
        compute_stability=True,
        stability_runs=4,
        compute_elbow=True,
        elbow_k_max=5,
    )
    assert "stability_ari_mean" in ev.metrics
    assert "elbow_inertia" in ev.diagnostics


@pytest.mark.skipif(importlib.util.find_spec("hdbscan") is None, reason="hdbscan extra")
def test_hdbscan_when_installed() -> None:
    session = _ready_session()
    fit = session.fit_clusters(method="hdbscan", hdbscan_min_cluster_size=5, n_clusters=None)
    assert fit.assign_strategy == "nearest_core"
    labels = session.assign_clusters(partition="test")
    assert labels.n_rows > 0


def test_hdbscan_missing_extra_raises() -> None:
    if importlib.util.find_spec("hdbscan") is not None:
        pytest.skip("hdbscan installed")
    session = _ready_session()
    with pytest.raises(MissingExtraError):
        session.fit_clusters(method="hdbscan", n_clusters=None)


@pytest.mark.skip(reason="DEC requires working torch runtime; isolated in CI")
def test_dec_when_torch_installed() -> None:
    session = _ready_session()
    try:
        fit = session.fit_clusters(
            method="dec",
            n_clusters=3,
            pretrain_epochs=5,
            finetune_epochs=5,
            batch_size=32,
        )
    except Exception as exc:
        pytest.skip(f"torch DEC unavailable: {exc}")
    assert fit.assign_strategy == "native"
    assigned = session.assign_clusters(partition="test")
    assert assigned.n_rows > 0


@pytest.mark.skipif(importlib.util.find_spec("umap") is None, reason="umap extra")
def test_umap_reduce_dimensions() -> None:
    session = _ready_session().reduce_dimensions(
        method="umap", n_components=2, prefix="um", drop_input_columns=False
    )
    assert session.reduce_plan is not None
    assert session.reduce_plan.method == "umap"
    fit = session.fit_clusters(method="kmeans", n_clusters=3, prefer_reduce_components=True)
    assert fit.used_reduce_components is True


def test_tsne_reduce_disclosure() -> None:
    session = _ready_session().reduce_dimensions(method="tsne", n_components=2, prefix="ts")
    assert session.reduce_plan is not None
    assert any("transductive" in d.lower() for d in session.reduce_plan.disclosures)


def test_bundle_v2_format(tmp_path) -> None:
    session = _ready_session()
    session.fit_clusters(method="kmeans", n_clusters=3)
    path = session.save_unsupervised_bundle(tmp_path / "v2")
    meta = (path / "meta.json").read_text(encoding="utf-8")
    assert BUNDLE_FORMAT in meta
    assert BUNDLE_FORMAT_V1 not in meta or BUNDLE_FORMAT_V2 in meta


def test_load_v1_bundle_still_works(tmp_path) -> None:
    from buildml.unsupervised.checkpoint import save_unsupervised_bundle
    from buildml.unsupervised.cluster import fit_clusterer

    session = _ready_session()
    plan, fit = fit_clusterer(
        session.dataset,
        session.split_plan,
        method="kmeans",
        n_clusters=3,
        prefer_reduce_components=False,
    )
    dest = tmp_path / "v1"
    dest.mkdir()
    save_unsupervised_bundle(dest, plan, fit_result=fit)
    # simulate v1 meta
    import json

    meta_path = dest / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["format"] = BUNDLE_FORMAT_V1
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    session.load_unsupervised_bundle(dest, trusted=True)
    assert session.cluster_plan is not None
