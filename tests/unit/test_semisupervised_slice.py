"""Unit coverage for the semi-supervised thin slice."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.tools import build_default_registry
from buildml.core.errors import LeakageError, ValidationError
from buildml.data.dataset import Dataset
from buildml.explain.catalog import OPERATION_CATALOG
from buildml.ingest.detect import schema_from_dataframe
from buildml.semisupervised.checkpoint import BUNDLE_FORMAT, load_semisupervised_bundle


def _frame(n: int = 160, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.normal([-1.0, -1.0], 0.55, size=(n // 2, 2))
    x1 = rng.normal([1.2, 1.0], 0.55, size=(n - n // 2, 2))
    frame = pd.DataFrame(np.vstack([x0, x1]), columns=["x", "y"])
    frame["label"] = [0] * (n // 2) + [1] * (n - n // 2)
    return frame


def _mask_train(session: Session, fraction: float = 0.65, seed: int = 1) -> Session:
    rng = np.random.default_rng(seed)
    full = session.to_pandas().copy()
    train_idx = list(session.split_plan.train_indices)
    blank = rng.choice(train_idx, size=max(1, int(fraction * len(train_idx))), replace=False)
    full.loc[blank, "label"] = np.nan
    session._dataset = Dataset.from_transformed(
        session.dataset,
        full,
        schema=schema_from_dataframe(full),
        roles=dict(session.dataset.roles),
    )
    return session


def _ready_session(*, mask: bool = True) -> Session:
    session = (
        Session.ingest(_frame())
        .set_roles({"x": "feature", "y": "feature", "label": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    if mask:
        session = _mask_train(session)
    return session


def test_core_import_and_catalog() -> None:
    import buildml.semisupervised as semi

    assert hasattr(semi, "fit_semisupervised")
    assert hasattr(Session, "fit_semisupervised")
    for name in (
        "fit_semisupervised",
        "predict_semisupervised",
        "evaluate_semisupervised",
        "save_semisupervised_bundle",
        "load_semisupervised_bundle",
    ):
        assert name in OPERATION_CATALOG
    assert (
        "semisupervised-label-missingness"
        in OPERATION_CATALOG["fit_semisupervised"].concept_links
    )
    assert (
        "semisupervised-bundle-boundary"
        in OPERATION_CATALOG["save_semisupervised_bundle"].concept_links
    )


def test_fit_requires_split() -> None:
    session = Session.ingest(_frame()).set_roles(
        {"x": "feature", "y": "feature", "label": "target"}
    )
    with pytest.raises((LeakageError, ValidationError)):
        session.fit_semisupervised()


def test_label_propagation_fit_predict_evaluate_bundle(tmp_path: Path) -> None:
    session = _ready_session()
    fit = session.fit_semisupervised(method="label_propagation", n_neighbors=5)
    assert fit.method == "label_propagation"
    assert fit.n_unlabeled_train > 0
    assert fit.n_labeled_train >= 2
    assert session.semisupervised_plan is not None

    preds = session.predict_semisupervised(partition="test")
    assert preds.n_rows > 0
    assert len(preds.predictions) == preds.n_rows

    ev = session.evaluate_semisupervised(partition="test")
    assert ev.n_labeled_eval == ev.n_rows  # holdout fully labeled
    assert "accuracy" in ev.metrics
    assert ev.metrics["accuracy"] >= 0.5

    bundle = session.save_semisupervised_bundle(tmp_path / "semi")
    assert (bundle / "meta.json").is_file()
    plan = load_semisupervised_bundle(bundle, trusted=True)
    assert plan.method == "label_propagation"
    import json

    meta = json.loads((bundle / "meta.json").read_text(encoding="utf-8"))
    assert meta["format"] == BUNDLE_FORMAT


def test_self_training_and_ai_allowlist() -> None:
    session = _ready_session()
    fit = session.fit_semisupervised(
        method="self_training",
        base_estimator="logistic_regression",
        threshold=0.7,
    )
    assert fit.method == "self_training"
    ev = session.evaluate_semisupervised(partition="test")
    assert "f1_macro" in ev.metrics

    registry = build_default_registry()
    for name in (
        "fit_semisupervised",
        "evaluate_semisupervised",
        "save_semisupervised_bundle",
        "load_semisupervised_bundle",
    ):
        assert registry.get(name) is not None


def test_walkthrough_status() -> None:
    session = _ready_session()
    session.fit_semisupervised(method="label_spreading", n_neighbors=5)
    report = session.walkthrough()
    status = report.semisupervised_status
    assert status["enabled"] is True
    assert status["method"] == "label_spreading"
