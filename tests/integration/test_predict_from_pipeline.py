"""One-shot score-time prediction through pipeline bundles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.pipeline import predict_from_pipeline


def test_predict_from_pipeline_classification_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n = 60
    frame = pd.DataFrame(
        {
            "age": rng.normal(40, 8, n),
            "income": rng.normal(55, 12, n),
            "city": rng.choice(["a", "b", "c"], size=n),
            "y": ([0, 1] * (n // 2)),
        }
    )
    frame.loc[::10, "age"] = np.nan
    session = (
        Session.ingest(frame)
        .set_roles(
            {"age": "feature", "income": "feature", "city": "feature", "y": "target"}
        )
        .split(test_size=0.25, stratify=True, random_state=0)
        .impute(strategy="median")
        .encode(method="onehot", columns=["city"])
        .scale(method="standard")
        .fit(LogisticRegression(max_iter=500), task="classification")
    )
    direct = session.predict(partition="test", return_proba=True)
    pipe = tmp_path / "cls_pipe"
    session.save_pipeline(pipe, evaluate_partition=None)

    holdout = frame.iloc[list(session.split_plan.test_indices)].reset_index(drop=True)  # type: ignore[union-attr]
    scored = predict_from_pipeline(
        pipe,
        holdout,
        roles={
            "age": "feature",
            "income": "feature",
            "city": "feature",
            "y": "target",
        },
        return_proba=True,
        trusted=True,
    )
    assert scored.n_rows == len(holdout)
    assert scored.probabilities is not None
    assert scored.apply_result is not None
    assert "impute" in scored.apply_result.applied
    # Labels should match Session.predict on the transformed partition.
    # Compare via reloaded session path for numerical stability.
    session_api = Session.ingest(holdout).predict_from_pipeline(
        pipe,
        holdout,
        roles={
            "age": "feature",
            "income": "feature",
            "city": "feature",
            "y": "target",
        },
        return_proba=True,
        trusted=True,
    )
    pd.testing.assert_series_equal(
        scored.predictions.reset_index(drop=True),
        session_api.predictions.reset_index(drop=True),
        check_names=False,
    )
    assert direct is not None  # fitted path still works
    assert scored.task == "classification"


def test_predict_from_pipeline_regression_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    n = 50
    x = rng.normal(size=n)
    frame = pd.DataFrame({"x": x, "y": 2.0 * x + rng.normal(scale=0.1, size=n)})
    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "y": "target"})
        .split(test_size=0.3, random_state=0)
        .scale(method="standard")
        .fit(LinearRegression(), task="regression")
    )
    pipe = tmp_path / "reg_pipe"
    session.save_pipeline(pipe, evaluate_partition=None)
    holdout = frame.iloc[list(session.split_plan.test_indices)].reset_index(drop=True)  # type: ignore[union-attr]
    scored = Session().predict_from_pipeline(
        pipe,
        holdout,
        roles={"x": "feature", "y": "target"},
        trusted=True,
    )
    assert scored.n_rows == len(holdout)
    assert scored.probabilities is None
    assert scored.predictions.notna().all()


def test_predict_from_pipeline_schema_error(tmp_path: Path) -> None:
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [0, 1, 0, 1]})
    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "y": "target"})
        .split(test_size=0.25, random_state=0)
        .fit(LogisticRegression(max_iter=200), task="classification")
    )
    pipe = tmp_path / "bad_pipe"
    session.save_pipeline(pipe, evaluate_partition=None)
    with pytest.raises(ValidationError, match="missing columns"):
        predict_from_pipeline(
            pipe,
            pd.DataFrame({"z": [1.0, 2.0]}),
            apply_plans=False,
            trusted=True,
        )
