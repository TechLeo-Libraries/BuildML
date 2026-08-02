"""Unit coverage for native ensemble voting / stacking / blending."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from buildml import Session
from buildml.core.errors import LeakageError, ValidationError
from buildml.ensemble.checkpoint import BUNDLE_FORMAT
from buildml.explain.catalog import OPERATION_CATALOG


def _clf_frame(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    logits = 0.8 * x1 - 0.5 * x2
    y = (logits + rng.normal(scale=0.4, size=n) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def _reg_frame(n: int = 120, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.5 * x1 - 0.7 * x2 + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def _ready_clf(**split_kwargs: object) -> Session:
    kwargs = {"test_size": 0.25, "random_state": 0, "stratify": True}
    kwargs.update(split_kwargs)
    return (
        Session.ingest(_clf_frame())
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(**kwargs)
        .scale(method="standard")
    )


def test_core_import_does_not_require_extra() -> None:
    import buildml.ensemble as ens

    assert hasattr(Session, "fit_voting")
    assert hasattr(Session, "fit_stacking")
    assert hasattr(Session, "fit_blending")
    assert hasattr(ens, "fit_voting_ensemble")


def test_catalog_covers_ensemble_operations() -> None:
    for name in (
        "fit_voting",
        "fit_stacking",
        "fit_blending",
        "evaluate_ensemble",
        "save_ensemble_bundle",
        "load_ensemble_bundle",
    ):
        assert name in OPERATION_CATALOG
    assert (
        "ensemble-voting-vs-single-tree"
        in OPERATION_CATALOG["fit_voting"].concept_links
    )
    assert "ensemble-stacking-oof" in OPERATION_CATALOG["fit_stacking"].concept_links
    assert "ensemble-blending-holdout" in OPERATION_CATALOG["fit_blending"].concept_links
    assert (
        "ensemble-bundle-boundary"
        in OPERATION_CATALOG["save_ensemble_bundle"].concept_links
    )


def test_fit_requires_split() -> None:
    session = Session.ingest(_clf_frame()).set_roles(
        {"x1": "feature", "x2": "feature", "y": "target"}
    )
    with pytest.raises((LeakageError, ValidationError)):
        session.fit_voting(
            {
                "lr": LogisticRegression(max_iter=500),
                "rf": RandomForestClassifier(n_estimators=20, random_state=0),
            }
        )


def test_voting_fit_evaluate_and_bundle(tmp_path: Path) -> None:
    session = _ready_clf()
    fit = session.fit_voting(
        {
            "lr": LogisticRegression(max_iter=500),
            "rf": RandomForestClassifier(n_estimators=30, random_state=0),
        },
        voting="soft",
        task="classification",
    )
    assert fit.strategy == "voting"
    assert session.ensemble_plan is not None
    assert session.fit_result is not None
    assert set(fit.estimator_names) == {"lr", "rf"}

    metrics = session.evaluate_ensemble(partition="test")
    assert "accuracy" in metrics.metrics or "f1_weighted" in metrics.metrics
    assert metrics.diagnostics.get("ensemble", {}).get("strategy") == "voting"

    bundle = session.save_ensemble_bundle(tmp_path / "ens_bundle")
    assert (bundle / "meta.json").is_file()
    assert (bundle / "ensemble_plan.joblib").is_file()
    meta = (bundle / "meta.json").read_text(encoding="utf-8")
    assert BUNDLE_FORMAT in meta

    restored = (
        Session.ingest(session.to_pandas())
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, random_state=0, stratify=True)
    )
    restored.load_ensemble_bundle(bundle)
    assert restored.ensemble_plan is not None
    assert restored.fit_result is not None
    again = restored.evaluate_ensemble(partition="test")
    assert again.metrics


def test_stacking_and_blending_paths() -> None:
    session = _ready_clf(validation_size=0.2)
    bases = {
        "lr": LogisticRegression(max_iter=500),
        "rf": RandomForestClassifier(n_estimators=25, random_state=0),
    }
    stack = session.fit_stacking(bases, cv=3, task="classification")
    assert stack.strategy == "stacking"
    assert stack.cv == 3
    eval_s = session.evaluate(partition="test")
    assert eval_s.n_rows > 0

    blend = session.fit_blending(
        bases,
        holdout_fraction=0.25,
        task="classification",
        random_state=0,
    )
    assert blend.strategy == "blending"
    assert blend.holdout_fraction == 0.25
    assert any("train only" in d.lower() or "holdout_fraction" in d for d in blend.disclosures)
    eval_b = session.evaluate_ensemble(partition="validation")
    assert eval_b.partition == "validation"


def test_regression_voting() -> None:
    session = (
        Session.ingest(_reg_frame())
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, random_state=0)
        .scale(method="standard")
    )
    fit = session.fit_voting(
        {
            "ridge": Ridge(),
            "rf": RandomForestRegressor(n_estimators=20, random_state=0),
        },
        task="regression",
    )
    assert fit.task == "regression"
    metrics = session.evaluate_ensemble(partition="test")
    assert "r2" in metrics.metrics or "rmse" in metrics.metrics


def test_rejects_single_estimator() -> None:
    session = _ready_clf()
    with pytest.raises(ValidationError):
        session.fit_voting({"lr": LogisticRegression(max_iter=200)})
