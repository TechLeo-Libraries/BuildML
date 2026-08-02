"""Unit coverage for the online / continual thin slice."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.tools import build_default_registry
from buildml.core.errors import ValidationError
from buildml.explain.catalog import OPERATION_CATALOG
from buildml.online.checkpoint import BUNDLE_FORMAT, load_online_bundle


def _frame(n: int = 220, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.normal([-1.0, -1.0], 0.55, size=(n // 2, 2))
    x1 = rng.normal([1.2, 1.0], 0.55, size=(n - n // 2, 2))
    frame = pd.DataFrame(np.vstack([x0, x1]), columns=["x", "y"])
    frame["label"] = [0] * (n // 2) + [1] * (n - n // 2)
    return frame


def _ready_session() -> Session:
    return (
        Session.ingest(_frame())
        .set_roles({"x": "feature", "y": "feature", "label": "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0)
        .scale(method="standard")
    )


def test_core_import_and_catalog() -> None:
    import buildml.online as online

    assert hasattr(online, "fit_online")
    assert hasattr(online, "partial_fit")
    assert hasattr(Session, "fit_online")
    for name in (
        "fit_online",
        "partial_fit_online",
        "evaluate_online",
        "predict_online",
        "save_online_bundle",
        "load_online_bundle",
    ):
        assert name in OPERATION_CATALOG
    assert "online-partial-fit" in OPERATION_CATALOG["fit_online"].concept_links
    assert (
        "online-bundle-boundary"
        in OPERATION_CATALOG["save_online_bundle"].concept_links
    )


def test_fit_partial_evaluate_bundle(tmp_path: Path) -> None:
    session = _ready_session()
    fit = session.fit_online(estimator="sgd_classifier", chunk_size=40, n_init=40)
    assert fit.n_init_rows == 40
    assert session.online_plan is not None
    assert session.online_plan.classes_ is not None

    update = session.partial_fit_online(n_rows=40)
    assert update.n_updates == 1
    assert update.update_mode == "partial_fit"
    assert update.n_seen_rows >= 80

    ev = session.evaluate_online(partition="validation")
    assert "accuracy" in ev.metrics
    assert session.online_eval_result is not None

    preds = session.predict_online(partition="test")
    assert preds.n_rows == len(preds.predictions)

    before = session.explain("partial_fit_online", moment="before")
    assert before.prerequisite_status.get("online-plan") is True

    bundle = session.save_online_bundle(tmp_path / "online_bundle")
    assert (bundle / "meta.json").is_file()
    plan = load_online_bundle(bundle)
    assert plan.n_updates == 1

    restored = Session.ingest(session.to_pandas()).set_roles(
        {"x": "feature", "y": "feature", "label": "target"}
    )
    restored._split_plan = session.split_plan
    restored._dataset = session.dataset
    restored.load_online_bundle(bundle)
    assert restored.online_plan is not None
    assert restored.online_plan.n_updates == 1


def test_refuse_holdout_indices() -> None:
    session = _ready_session()
    session.fit_online(chunk_size=30, n_init=30)
    test_idx = list(session.split_plan.test_indices)[:3]
    with pytest.raises(ValidationError, match="non-train indices"):
        session.partial_fit_online(indices=test_idx)


def test_unknown_estimator() -> None:
    session = _ready_session()
    with pytest.raises(ValidationError, match="Unknown online estimator|not valid for backend"):
        session.fit_online(estimator="hist_gradient_boosting")  # type: ignore[arg-type]


def test_ai_allowlist() -> None:
    registry = build_default_registry()
    names = {t.name for t in registry.tools}
    for name in (
        "fit_online",
        "partial_fit_online",
        "evaluate_online",
        "save_online_bundle",
        "load_online_bundle",
    ):
        assert name in names


def test_bundle_format_constant() -> None:
    assert BUNDLE_FORMAT == "buildml.online_bundle.v1"
