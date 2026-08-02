"""Default scale/encode/impute skip ignore/id (and related) roles."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.preprocess import DEFAULT_SKIP_ROLES


def _knapsack_style_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [21.0, 35.0, 40.0, 29.0, 33.0, 52.0, 47.0, 38.0],
            "income": [40.0, 60.0, 80.0, 50.0, 70.0, 90.0, 65.0, 55.0],
            "segment": ["a", "b", "a", "b", "a", "b", "a", "b"],
            "app_id": [f"id-{i}" for i in range(8)],
            "review_cost": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "approved": [0, 1, 0, 1, 0, 1, 1, 0],
        }
    )


def test_default_skip_roles_include_ignore_and_id() -> None:
    names = {role.value for role in DEFAULT_SKIP_ROLES}
    assert {"ignore", "id", "target", "group", "time", "weight"} <= names


def test_scale_skips_ignore_and_id_numeric_by_default() -> None:
    frame = _knapsack_style_frame()
    cost_before = frame["review_cost"].tolist()
    session = (
        Session.ingest(frame)
        .set_roles(
            {
                "age": "feature",
                "income": "feature",
                "segment": "feature",
                "app_id": "id",
                "review_cost": "ignore",
                "approved": "target",
            }
        )
        .split(test_size=0.25, random_state=0)
    )
    session.scale(method="standard")
    assert session.scale_plan is not None
    assert set(session.scale_plan.columns) == {"age", "income"}
    out = session.to_pandas()
    assert out["review_cost"].tolist() == cost_before
    # Features should have been mutated (not identical to raw).
    assert not np.allclose(out["age"].to_numpy(), frame["age"].to_numpy())


def test_encode_skips_id_categorical_by_default() -> None:
    frame = _knapsack_style_frame()
    ids_before = frame["app_id"].tolist()
    session = (
        Session.ingest(frame)
        .set_roles(
            {
                "age": "feature",
                "income": "feature",
                "segment": "feature",
                "app_id": "id",
                "review_cost": "ignore",
                "approved": "target",
            }
        )
        .split(test_size=0.25, random_state=0)
    )
    session.encode(method="onehot")
    assert session.encode_plan is not None
    assert "app_id" not in session.encode_plan.columns
    assert set(session.encode_plan.columns) == {"segment"}
    out = session.to_pandas()
    assert "app_id" in out.columns
    assert out["app_id"].tolist() == ids_before
    # One-hot expanded the feature categorical.
    assert any(c.startswith("segment") for c in out.columns)


def test_impute_skips_ignore_numeric_by_default() -> None:
    frame = _knapsack_style_frame()
    frame.loc[0, "age"] = np.nan
    frame.loc[1, "review_cost"] = np.nan
    session = (
        Session.ingest(frame)
        .set_roles(
            {
                "age": "feature",
                "income": "feature",
                "segment": "feature",
                "app_id": "id",
                "review_cost": "ignore",
                "approved": "target",
            }
        )
        .split(test_size=0.25, random_state=0)
    )
    session.impute(strategy="median")
    assert session.impute_plan is not None
    assert "review_cost" not in session.impute_plan.columns
    assert "age" in session.impute_plan.columns
    out = session.to_pandas()
    assert out["age"].isna().sum() == 0
    assert out["review_cost"].isna().sum() == 1


def test_scale_columns_opt_in_includes_ignore() -> None:
    frame = _knapsack_style_frame()
    cost_before = frame["review_cost"].to_numpy(dtype=float)
    session = (
        Session.ingest(frame)
        .set_roles(
            {
                "age": "feature",
                "income": "feature",
                "review_cost": "ignore",
                "approved": "target",
            }
        )
        .split(test_size=0.25, random_state=0)
    )
    session.scale(method="standard", columns=["review_cost"])
    assert session.scale_plan is not None
    assert session.scale_plan.columns == ("review_cost",)
    out = session.to_pandas()["review_cost"].to_numpy(dtype=float)
    assert not np.allclose(out, cost_before)


def test_default_scale_then_knapsack_costs_stay_non_negative() -> None:
    """Regression: Ledger/Aegis-style ignore costs must remain usable."""
    frame = _knapsack_style_frame()
    session = (
        Session.ingest(frame)
        .set_roles(
            {
                "age": "feature",
                "income": "feature",
                "segment": "feature",
                "app_id": "id",
                "review_cost": "ignore",
                "approved": "target",
            }
        )
        .split(test_size=0.25, validation_size=0.25, stratify=True, random_state=0)
    )
    session.encode(method="onehot")
    session.scale(method="standard")
    costs = session.to_pandas()["review_cost"]
    assert (costs >= 0).all()
    assert costs.tolist() == frame["review_cost"].tolist()
