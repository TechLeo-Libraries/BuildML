"""Cluster geometry and external agreement keep sampling/noise disclosures explicit."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import silhouette_score

from buildml import Session
from buildml.core.errors import ValidationError


def _session(method):
    rng = np.random.default_rng(109)
    points = np.vstack([rng.normal(-3, .06, (40, 2)), rng.normal(3, .06, (40, 2)), [[20, 20]]])
    frame = pd.DataFrame(points, columns=["x", "z"])
    frame["reference"] = [*["left"] * 40, *["right"] * 40, "isolated"]
    session = (Session.ingest(frame).set_roles({"x": "feature", "z": "feature", "reference": "target"})
               .split(test_size=.2, validation_size=.2, random_state=12))
    kwargs = {"n_clusters": 2} if method == "kmeans" else {"n_clusters": None, "eps": .3, "min_samples": 3}
    session.unsupervised.fit(method=method, columns=["x", "z"], **kwargs)
    return session


def test_dbscan_external_agreement_excludes_noise_explicitly():
    session = _session("dbscan")
    report = session.unsupervised.evaluate(partition="all", external_label_column="reference",
                                           compute_stability=False, compute_elbow=False)
    assert report.metrics["noise_rate"] == pytest.approx(1 / 81)
    assert report.external_metrics["adjusted_rand_index"] == 1.
    assert report.external_metrics["normalized_mutual_info"] == 1.
    assert any("non-noise assigned rows only" in disclosure for disclosure in report.disclosures)
    assert any("do not imply causal" in disclosure for disclosure in report.disclosures)


def test_sampled_silhouette_uses_requested_rows_without_refitting():
    session = _session("kmeans")
    before = joblib.hash(session.cluster_plan)
    assigned = session.unsupervised.assign(partition="test", attach=False)
    frame = session.partition("test")
    take = np.random.default_rng(81).choice(len(frame), size=10, replace=False)
    expected = silhouette_score(frame[["x", "z"]].to_numpy()[take], np.asarray(assigned.labels)[take])
    report = session.unsupervised.evaluate(partition="test", sample_size=10, random_state=81,
                                           compute_stability=False, compute_elbow=False)
    assert report.metrics["silhouette"] == pytest.approx(expected)
    assert any("sampled to 10 rows" in disclosure for disclosure in report.disclosures)
    assert joblib.hash(session.cluster_plan) == before


def test_external_agreement_refuses_a_missing_reference_column():
    session = _session("kmeans")
    with pytest.raises(ValidationError, match="not found"):
        session.unsupervised.evaluate(partition="test", external_label_column="absent",
                                       compute_stability=False, compute_elbow=False)
