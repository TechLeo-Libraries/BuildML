"""Unit coverage for the self-supervised thin slice."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.tools import build_default_registry
from buildml.core.errors import LeakageError, ValidationError
from buildml.explain.catalog import OPERATION_CATALOG
from buildml.selfsupervised.checkpoint import BUNDLE_FORMAT, load_ssl_bundle


def _frame(n: int = 160, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.normal([-1.0, -1.0], 0.7, size=(n // 2, 2))
    x1 = rng.normal([1.4, 1.1], 0.7, size=(n - n // 2, 2))
    frame = pd.DataFrame(np.vstack([x0, x1]), columns=["x", "y"])
    frame["label"] = [0] * (n // 2) + [1] * (n - n // 2)
    return frame


def _ready() -> Session:
    return (
        Session.ingest(_frame())
        .set_roles({"x": "feature", "y": "feature", "label": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )


def test_core_import_and_catalog() -> None:
    import buildml.selfsupervised as ssl

    assert hasattr(ssl, "fit_ssl_pretext")
    assert hasattr(Session, "fit_ssl_pretext")
    for name in (
        "fit_ssl_pretext",
        "transform_ssl",
        "finetune_ssl_head",
        "evaluate_ssl",
        "save_ssl_bundle",
        "load_ssl_bundle",
    ):
        assert name in OPERATION_CATALOG
    assert "ssl-pretext-then-head" in OPERATION_CATALOG["fit_ssl_pretext"].concept_links
    assert "ssl-bundle-boundary" in OPERATION_CATALOG["save_ssl_bundle"].concept_links


def test_fit_requires_split() -> None:
    session = Session.ingest(_frame()).set_roles(
        {"x": "feature", "y": "feature", "label": "target"}
    )
    with pytest.raises((LeakageError, ValidationError)):
        session.fit_ssl_pretext()


def test_masked_tabular_pretext_head_eval_bundle(tmp_path: Path) -> None:
    session = _ready()
    pre = session.fit_ssl_pretext(
        method="masked_tabular",
        latent_dim=6,
        mask_ratio=0.2,
        max_iter=80,
        random_state=0,
    )
    assert pre.method == "masked_tabular"
    assert pre.latent_dim == 6
    assert session.ssl_plan is not None
    assert pre.reconstruction_mae is not None

    tr = session.transform_ssl(partition="test")
    assert tr.n_rows > 0
    assert len(tr.representation_columns) == 6

    head = session.finetune_ssl_head(estimator="logistic_regression", random_state=0)
    assert head.n_labeled_train >= 2
    assert session.ssl_head_plan is not None

    ev = session.evaluate_ssl(partition="test")
    assert "accuracy" in ev.metrics
    assert ev.n_labeled_eval == ev.n_rows

    bundle = session.save_ssl_bundle(tmp_path / "ssl")
    assert (bundle / "meta.json").is_file()
    plan, loaded_head = load_ssl_bundle(bundle, trusted=True)
    assert plan.latent_dim == 6
    assert loaded_head is not None
    import json

    meta = json.loads((bundle / "meta.json").read_text(encoding="utf-8"))
    assert meta["format"] == BUNDLE_FORMAT


def test_attach_embeddings_and_ai_allowlist() -> None:
    session = _ready()
    session.fit_ssl_pretext(latent_dim=4, max_iter=60, random_state=0)
    attached = session.transform_ssl(partition="all", attach=True)
    assert attached.attached is True
    for col in session.ssl_plan.representation_columns:
        assert col in session.to_pandas().columns

    registry = build_default_registry()
    for name in (
        "fit_ssl_pretext",
        "finetune_ssl_head",
        "evaluate_ssl",
        "save_ssl_bundle",
        "load_ssl_bundle",
    ):
        assert registry.get(name) is not None


def test_walkthrough_status() -> None:
    session = _ready()
    session.fit_ssl_pretext(latent_dim=4, max_iter=50, random_state=0)
    session.finetune_ssl_head()
    report = session.walkthrough()
    status = report.selfsupervised_status
    assert status["enabled"] is True
    assert status["has_ssl_head"] is True
