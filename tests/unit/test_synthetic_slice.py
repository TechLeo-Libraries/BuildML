"""Session-facing slice tests for synthetic-data systems."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.datasets import make_classification

from buildml import Session
from buildml.ai.tools import build_default_registry
from buildml.core.errors import ValidationError
from buildml.explain.catalog import OPERATION_CATALOG


def _mixed_session() -> Session:
    x, y = make_classification(
        n_samples=220,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        weights=[0.6, 0.4],
        random_state=0,
    )
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    frame["y"] = y
    frame["grp"] = pd.Series(y).map({0: "A", 1: "B"})
    return (
        Session.ingest(frame)
        .set_roles(
            {
                **{c: "feature" for c in frame.columns if c.startswith("f")},
                "grp": "feature",
                "y": "target",
            }
        )
        .split(test_size=0.25, validation_size=0.25, random_state=0)
    )


def test_core_import_and_catalog() -> None:
    import buildml.synthetic as synthetic

    assert hasattr(synthetic, "fit_synthesizer")
    assert hasattr(Session, "fit_synthesizer")
    for op in (
        "fit_synthesizer",
        "sample_synthetic",
        "evaluate_synthetic",
        "save_synthetic_bundle",
        "load_synthetic_bundle",
    ):
        assert op in OPERATION_CATALOG
    assert (
        "synthetic-train-only-generator"
        in OPERATION_CATALOG["fit_synthesizer"].concept_links
    )
    assert "synthetic-vs-resample" in OPERATION_CATALOG["fit_synthesizer"].concept_links
    assert (
        "synthetic-bundle-boundary"
        in OPERATION_CATALOG["save_synthetic_bundle"].concept_links
    )

    registry = build_default_registry()
    for name in (
        "fit_synthesizer",
        "sample_synthetic",
        "evaluate_synthetic",
        "save_synthetic_bundle",
        "load_synthetic_bundle",
    ):
        assert name in registry


def test_requires_split() -> None:
    x, y = make_classification(n_samples=80, n_features=4, random_state=1)
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(4)])
    frame["y"] = y
    session = Session.ingest(frame).set_roles(
        {**{c: "feature" for c in frame.columns if c.startswith("f")}, "y": "target"}
    )
    with pytest.raises(ValidationError, match="split"):
        session.fit_synthesizer(method="bootstrap")


def test_bootstrap_and_copula_e2e() -> None:
    session = _mixed_session()
    boot = session.fit_synthesizer(method="bootstrap", smooth_sigma=0.1, random_state=0)
    assert boot.method == "bootstrap"
    assert session.synthesizer_plan is not None
    sample = session.sample_synthetic(n=40, random_state=1)
    assert sample.n_rows == 40
    assert sample.frame is not None
    assert sample.merged is False

    session.fit_synthesizer(method="gaussian_copula", random_state=0)
    sample2 = session.sample_synthetic(n=50, random_state=2)
    assert "grp" in sample2.frame.columns
    assert set(sample2.frame["grp"].astype(str).unique()).issubset({"A", "B"})

    fid = session.evaluate_synthetic(mode="fidelity", partition="test")
    assert "mean_ks" in fid.metrics
    assert "mean_tv" in fid.metrics

    tstr = session.evaluate_synthetic(mode="tstr", partition="test")
    assert "score" in tstr.metrics
    assert "differential" in " ".join(tstr.disclosures).lower() or any(
        "privacy" in d.lower() for d in tstr.disclosures
    )


def test_merge_extend_train_provenance() -> None:
    session = _mixed_session()
    n_train_before = len(session._split_plan.train_indices)
    n_test_before = len(session._split_plan.test_indices)
    test_before = session.dataset.frame.iloc[
        list(session._split_plan.test_indices)
    ].reset_index(drop=True)

    session.fit_synthesizer(method="bootstrap", random_state=0)
    result = session.sample_synthetic(
        n=15, merge_mode="extend_train", provenance_column="_synthetic"
    )
    assert result.merged is True
    assert "_synthetic" in session.dataset.frame.columns
    assert session.dataset.roles["_synthetic"].value == "ignore"
    assert len(session._split_plan.train_indices) == n_train_before + 15
    assert len(session._split_plan.test_indices) == n_test_before
    test_after = session.dataset.frame.iloc[
        list(session._split_plan.test_indices)
    ].drop(columns=["_synthetic"]).reset_index(drop=True)
    assert test_after.equals(test_before)


def test_bundle_roundtrip(tmp_path) -> None:
    session = _mixed_session()
    session.fit_synthesizer(method="gaussian_copula", random_state=0)
    path = tmp_path / "syn_bundle"
    session.save_synthetic_bundle(path)
    other = _mixed_session()
    other.load_synthetic_bundle(path)
    assert other.synthesizer_plan is not None
    assert other.synthesizer_plan.method == "gaussian_copula"
    sample = other.sample_synthetic(n=20, random_state=9)
    assert sample.n_rows == 20


def test_walkthrough_synthetic_status() -> None:
    session = _mixed_session()
    session.fit_synthesizer(method="bootstrap", random_state=0)
    walk = session.walkthrough()
    assert walk.synthetic_status.get("enabled") is True
    assert walk.synthetic_status.get("method") == "bootstrap"
