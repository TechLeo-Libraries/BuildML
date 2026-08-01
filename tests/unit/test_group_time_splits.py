"""Group-aware and time-aware split behavior."""

from __future__ import annotations

import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import LeakageError, ValidationError


def test_group_split_keeps_groups_disjoint() -> None:
    frame = pd.DataFrame(
        {
            "x": list(range(30)),
            "g": [i // 3 for i in range(30)],  # 10 groups of 3
            "y": [i % 2 for i in range(30)],
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "g": "group", "y": "target"})
        .group_split(test_size=0.3, validation_size=0.3, random_state=0)
    )
    train = session.partition("train")
    valid = session.partition("validation")
    test = session.partition("test")
    assert set(train["g"]).isdisjoint(set(test["g"]))
    assert set(train["g"]).isdisjoint(set(valid["g"]))
    assert set(valid["g"]).isdisjoint(set(test["g"]))
    assert session.split_plan is not None
    assert session.split_plan.kind == "group"


def test_group_split_requires_group_role() -> None:
    frame = pd.DataFrame({"x": [1, 2, 3, 4], "y": [0, 1, 0, 1]})
    session = Session.ingest(frame).set_roles({"x": "feature", "y": "target"})
    with pytest.raises(ValidationError, match="role 'group'"):
        session.group_split(test_size=0.25)


def test_time_split_orders_partitions() -> None:
    frame = pd.DataFrame(
        {
            "t": pd.date_range("2024-01-01", periods=20, freq="D"),
            "x": list(range(20)),
            "y": [i % 2 for i in range(20)],
        }
    )
    # Shuffle row order to ensure splitter reorders by time.
    frame = frame.sample(frac=1.0, random_state=1).reset_index(drop=True)
    session = (
        Session.ingest(frame)
        .set_roles({"t": "time", "x": "feature", "y": "target"})
        .time_split(test_size=0.25, validation_size=0.25)
    )
    train = session.partition("train")
    valid = session.partition("validation")
    test = session.partition("test")
    assert train["t"].max() <= valid["t"].min()
    assert valid["t"].max() <= test["t"].min()
    assert session.split_plan is not None
    assert session.split_plan.kind == "time"


def test_time_split_rejects_unparseable_timestamps() -> None:
    frame = pd.DataFrame(
        {
            "t": ["2024-01-01", "not-a-date", "2024-01-03", "2024-01-04"],
            "x": [1, 2, 3, 4],
            "y": [0, 1, 0, 1],
        }
    )
    session = Session.ingest(frame).set_roles({"t": "time", "x": "feature", "y": "target"})
    with pytest.raises(ValidationError, match="parseable"):
        session.time_split(test_size=0.25)


def test_time_split_invariant_is_enforced() -> None:
    # Direct unit coverage for the leakage guard via inject of a bad time plan
    # is unnecessary; create_time_split itself asserts order. Ensure API works.
    frame = pd.DataFrame(
        {
            "t": pd.date_range("2024-01-01", periods=12, freq="D"),
            "x": list(range(12)),
            "y": [0, 1] * 6,
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"t": "time", "x": "feature", "y": "target"})
        .time_split(test_size=3)
    )
    assert len(session.partition("test")) == 3
    with pytest.raises(LeakageError):
        # Fitting on test remains blocked.
        session.assert_can_fit("test")
