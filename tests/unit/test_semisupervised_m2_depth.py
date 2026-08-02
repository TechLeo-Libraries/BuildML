"""Deeper semi-supervised coverage: leakage, low-level API, explain."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.ingest.detect import schema_from_dataframe
from buildml.semisupervised.evaluate import evaluate_semisupervised
from buildml.semisupervised.fit import fit_semisupervised
from buildml.semisupervised.predict import predict_semisupervised


def _frame(n: int = 140, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.normal(0.0, 0.8, size=(n // 2, 2))
    x1 = rng.normal(2.5, 0.8, size=(n - n // 2, 2))
    frame = pd.DataFrame(np.vstack([x0, x1]), columns=["a", "b"])
    frame["y"] = [0] * (n // 2) + [1] * (n - n // 2)
    return frame


def _mask(session: Session, fraction: float = 0.6) -> Session:
    rng = np.random.default_rng(4)
    full = session.to_pandas().copy()
    idx = list(session.split_plan.train_indices)
    blank = rng.choice(idx, size=max(1, int(fraction * len(idx))), replace=False)
    full.loc[blank, "y"] = np.nan
    session._dataset = Dataset.from_transformed(
        session.dataset,
        full,
        schema=schema_from_dataframe(full),
        roles=dict(session.dataset.roles),
    )
    return session


def test_low_level_fit_predict_evaluate(tmp_path: Path) -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    session = _mask(session)
    plan, fit = fit_semisupervised(
        session.dataset,
        session.split_plan,
        method="label_propagation",
        prefer_reduce_components=False,
    )
    assert fit.n_unlabeled_train > 0
    _, preds = predict_semisupervised(
        session.dataset, plan, session.split_plan, partition="test"
    )
    ev = evaluate_semisupervised(
        session.dataset, plan, session.split_plan, partition="test"
    )
    assert preds.n_rows == ev.n_rows
    assert ev.metrics["accuracy"] >= 0.0

    from buildml.semisupervised.checkpoint import save_semisupervised_bundle

    out = save_semisupervised_bundle(tmp_path / "direct", plan, fit_result=fit, eval_result=ev)
    assert (out / "semisupervised_plan.joblib").is_file()


def test_needs_labeled_rows() -> None:
    frame = _frame()
    frame["y"] = np.nan
    session = (
        Session.ingest(frame)
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.25, random_state=0)
        .scale(method="standard")
    )
    with pytest.raises(ValidationError, match="at least 2 labeled"):
        session.fit_semisupervised()


def test_explain_before() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    before = session.explain("fit_semisupervised", moment="before")
    assert before.operation == "fit_semisupervised"
    assert before.prerequisite_status.get("split") is True
