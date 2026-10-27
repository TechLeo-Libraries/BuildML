"""Public clustering diagnostics agree on row identities and density settings."""

from collections import Counter

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import OPTICS, KMeans

from buildml import Session
from buildml.core.errors import ValidationError


def _population(shape="separated", reverse=False):
    rng = np.random.default_rng(372)
    scale = .08 if shape == "separated" else 1.8
    points = np.vstack([rng.normal(center, scale, (24, 2)) for center in [-4, 0, 4]])
    if reverse:
        points = points[::-1]
    frame = pd.DataFrame(points, columns=["x", "z"], index=np.arange(72) % 9)
    # Deliberately heterogeneous scalar labels: ordering them is unnecessary.
    frame["reference"] = pd.Series([17] * 24 + ["middle"] * 24 + [(-2, "right")] * 24,
                                    dtype=object).to_numpy()
    session = Session.ingest(frame).set_roles({"x": "feature", "z": "feature", "reference": "ignore"})
    train = np.r_[0:18, 24:42, 48:66]
    other = [i for i in range(72) if i not in train]
    session.inject_split(train_indices=train.tolist(), validation_indices=other[:9], test_indices=other[9:])
    return session


def _replay(session, tmp_path):
    path = session.unsupervised.save_bundle(tmp_path / "cluster")
    restored = Session(dataset=session.dataset, split_plan=session.split_plan)
    restored.unsupervised.load_bundle(path, trusted=True)
    return restored


def _ari(left, right):
    """Contingency-table ARI, independent of sklearn's metric implementation."""
    def pairs(counts):
        return sum(n * (n - 1) / 2 for n in counts)
    joint = pairs(Counter(zip(left, right)).values())
    a, b = pairs(Counter(left).values()), pairs(Counter(right).values())
    total = len(left) * (len(left) - 1) / 2
    expected = a * b / total
    denominator = (a + b) / 2 - expected
    return 1. if denominator == 0 else (joint - expected) / denominator


def _stability_oracle(x, fraction, runs, seed):
    rng = np.random.default_rng(seed)
    reference = None
    scores = []
    for _ in range(runs):
        positions = rng.choice(len(x), size=max(4, int(len(x) * fraction)), replace=False)
        learner = KMeans(n_clusters=3, random_state=int(rng.integers(0, 2**31)), n_init=3)
        labels = learner.fit_predict(x[positions])
        keyed = dict(zip(positions.tolist(), labels.tolist()))
        if reference is None:
            reference = keyed
        else:
            shared = sorted(reference.keys() & keyed.keys())
            if len(shared) >= 2:
                scores.append(_ari([reference[i] for i in shared], [keyed[i] for i in shared]))
    return np.mean(scores), np.std(scores), len(scores)


@pytest.mark.parametrize("shape", ["separated", "overlapping"])
@pytest.mark.parametrize("fraction", [.45, 1.])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("method", ["kmeans", "agglomerative"])
@pytest.mark.parametrize("replay", [False, True])
def test_stability_compares_the_same_training_rows(shape, fraction, reverse, method, replay, tmp_path):
    session = _population(shape, reverse)
    session.unsupervised.fit(method=method, n_clusters=3, random_state=73)
    if replay:
        session = _replay(session, tmp_path)
    before = joblib.hash(session.cluster_plan)
    expected = _stability_oracle(session.partition("train")[["x", "z"]].to_numpy(), fraction, 5, 89)
    result = session.unsupervised.evaluate(partition="test", compute_stability=True,
                                          stability_sample_fraction=fraction, stability_runs=5,
                                          random_state=89, compute_elbow=True, elbow_k_max=4)
    assert result.metrics["stability_ari_mean"] == pytest.approx(expected[0], abs=1e-12)
    assert result.metrics["stability_ari_std"] == pytest.approx(expected[1], abs=1e-12)
    assert result.diagnostics["stability_comparisons"] == expected[2]
    assert any("shared training rows" in item for item in result.disclosures)
    assert joblib.hash(session.cluster_plan) == before
    # Both elbow and stability are train diagnostics, invariant to holdout changes.
    changed = session.to_pandas().copy()
    changed.iloc[list(session.split_plan.test_indices), :2] += 100
    other = Session.ingest(changed).set_roles(session.dataset.roles)
    other._split_plan = session.split_plan
    other._cluster_plan = session.cluster_plan
    second = other.unsupervised.evaluate(partition="test", compute_stability=True,
                                         stability_sample_fraction=fraction, stability_runs=5,
                                         random_state=89, compute_elbow=True, elbow_k_max=4)
    for key in ["stability_ari_mean", "stability_ari_std"]:
        assert second.metrics[key] == result.metrics[key]
    assert second.diagnostics == result.diagnostics


@pytest.mark.parametrize("minimum", [.15, .4, 9, None])
@pytest.mark.parametrize("shape", ["separated", "overlapping"])
@pytest.mark.parametrize("replay", [False, True])
def test_optics_cluster_size_matches_the_real_backend(minimum, shape, replay, tmp_path):
    session = _population(shape)
    x = session.partition("train")[["x", "z"]].to_numpy()
    direct = OPTICS(min_samples=4, xi=.05, min_cluster_size=minimum).fit(x)
    session.unsupervised.fit(method="optics", n_clusters=None, optics_min_samples=4,
                             optics_min_cluster_size=minimum)
    np.testing.assert_array_equal(session.cluster_plan.train_labels_, direct.labels_)
    if replay:
        session = _replay(session, tmp_path)
    before = joblib.hash(session.cluster_plan)
    ids = sorted(set(direct.labels_) - {-1})
    centers = np.asarray([x[direct.labels_ == label].mean(axis=0) for label in ids])
    holdout = session.partition("test")[["x", "z"]].to_numpy()
    expected = [ids[np.argmin(np.sum((centers - row) ** 2, axis=1))] for row in holdout]
    assigned = session.unsupervised.assign(partition="test", attach=False)
    np.testing.assert_array_equal(assigned.labels, expected)
    assert any("nearest train centroid" in text for text in assigned.disclosures)
    assert joblib.hash(session.cluster_plan) == before


@pytest.mark.parametrize("option,value", [
    ("stability_runs", 0), ("stability_runs", 1), ("stability_runs", 2.5),
    ("stability_runs", True), ("stability_runs", "5"),
    ("stability_sample_fraction", 0), ("stability_sample_fraction", -.1),
    ("stability_sample_fraction", 1.1), ("stability_sample_fraction", np.nan),
    ("stability_sample_fraction", np.inf), ("stability_sample_fraction", True),
    ("stability_sample_fraction", "0.8"),
])
def test_invalid_stability_settings_leave_previous_report_intact(option, value):
    session = _population()
    session.unsupervised.fit(method="kmeans", n_clusters=3)
    previous = session.unsupervised.evaluate(partition="test")
    history = joblib.hash(session.history)
    before = joblib.hash(session.cluster_plan)
    with pytest.raises(ValidationError, match=option):
        session.unsupervised.evaluate(partition="test", compute_stability=True, **{option: value})
    assert session._cluster_eval_result is previous
    assert joblib.hash(session.cluster_plan) == before
    assert joblib.hash(session.history) == history


@pytest.mark.parametrize("method", ["kmeans", "agglomerative", "dbscan", "optics"])
@pytest.mark.parametrize("replay", [False, True])
def test_external_labels_need_no_scalar_ordering(method, replay, tmp_path):
    session = _population()
    options = dict(n_clusters=None, eps=.5, min_samples=3) if method == "dbscan" else dict(n_clusters=3)
    session.unsupervised.fit(method=method, **options)
    if replay:
        session = _replay(session, tmp_path)
    result = session.unsupervised.evaluate(partition="test", external_label_column="reference")
    assert result.external_metrics["adjusted_rand_index"] == 1.
    assert result.external_metrics["normalized_mutual_info"] == 1.


def test_all_noise_external_agreement_is_unavailable():
    session = _population()
    session.unsupervised.fit(method="dbscan", n_clusters=None, eps=1e-8, min_samples=3)
    result = session.unsupervised.evaluate(partition="test", external_label_column="reference")
    assert result.n_clusters_observed == 0
    assert result.metrics["noise_rate"] == 1.
    assert result.external_metrics == {}
    assert any("no non-noise rows" in warning for warning in result.warnings)


def test_stability_with_no_comparable_rows_reports_unavailable():
    session = Session.ingest(pd.DataFrame({"x": [-5, -4, -3, 3, 4, 5, -2, 2]}))
    session.set_roles({"x": "feature"})
    session.inject_split(train_indices=list(range(6)), validation_indices=[6], test_indices=[7])
    session.unsupervised.fit(method="kmeans", n_clusters=2)
    # The seeded samples [4, 5, 3] and [0, 4, 1] have only one shared row.
    result = session.unsupervised.evaluate(partition="test", compute_stability=True,
                                          stability_runs=2, stability_sample_fraction=.1,
                                          random_state=0)
    assert result.diagnostics["stability_comparisons"] == 0
    assert "stability_ari_mean" not in result.metrics
    assert "stability_ari_std" not in result.metrics
    assert any("fewer than 2 training rows" in warning for warning in result.warnings)


def test_stability_with_insufficient_training_rows_is_disclosed():
    session = Session.ingest(pd.DataFrame({"x": [-5, 0, 5, -4, 4]})).set_roles({"x": "feature"})
    session.inject_split(train_indices=[0, 1, 2], validation_indices=[3], test_indices=[4])
    session.unsupervised.fit(method="kmeans", n_clusters=2)
    result = session.unsupervised.evaluate(partition="test", compute_stability=True)
    assert "stability_ari_mean" not in result.metrics
    assert any("insufficient train rows" in warning for warning in result.warnings)


def test_null_external_labels_refuse_without_replacing_report():
    session = _population()
    session.unsupervised.fit(method="kmeans", n_clusters=3)
    previous = session.unsupervised.evaluate(partition="test")
    frame = session.to_pandas().copy()
    frame.iloc[session.split_plan.test_indices[0], 2] = None
    replacement = Session.ingest(frame).set_roles(session.dataset.roles)
    replacement._split_plan = session.split_plan
    replacement._cluster_plan = session.cluster_plan
    replacement._cluster_eval_result = previous
    before = joblib.hash(replacement.history)
    with pytest.raises(ValidationError, match="contains nulls"):
        replacement.unsupervised.evaluate(partition="test", external_label_column="reference")
    assert replacement._cluster_eval_result is previous
    assert joblib.hash(replacement.history) == before
