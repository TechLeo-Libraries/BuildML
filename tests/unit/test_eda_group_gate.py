"""Group readiness evidence follows the declared population boundary."""

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.dashboard.gates import build_gate_context, build_gates_payload


def _gate_rows(report):
    return {row["id"]: row for row in build_gates_payload(report)["rows"]}


def test_eda_group_gates_count_train_labels_before_sampling():
    train_groups = ["a"] * 5 + ["b"] * 4 + ["c"] * 3 + [None]
    groups = train_groups + ["future-a", "future-b", "future-b"]
    frame = pd.DataFrame({"account": groups, "value": np.arange(len(groups), dtype=float)})
    session = (
        Session.ingest(frame)
        .set_roles({"account": "group", "value": "feature"})
        .inject_split(
            train_indices=list(range(len(train_groups))),
            test_indices=list(range(len(train_groups), len(groups))),
        )
    )
    report = session.eda(partition="train", sample_rows=4).to_dict()
    assert report["overview"]["analysis_rows"] == 4
    assert build_gate_context(report)["groupCol"] == {
        "name": "account",
        "groups": len(set(train_groups) - {None}),
        "observedRows": sum(group is not None for group in train_groups),
    }
    gates = _gate_rows(report)
    assert "account" in gates["03.1"]["evidence"]
    assert "3 observed groups across 12 labeled rows" in gates["03.6"]["evidence"]

    observed = frame.dropna(subset=["account"])
    split = (
        Session.ingest(observed)
        .set_roles({"account": "group", "value": "feature"})
        .group_split(test_size=0.3, random_state=7)
    )
    assert set(split.partition("train")["account"]).isdisjoint(split.partition("test")["account"])


def test_eda_group_gates_preserve_known_role_with_missing_or_legacy_counts():
    report = (
        Session.ingest(pd.DataFrame({"account": [None] * 6, "value": np.arange(6, dtype=float)}))
        .set_roles({"account": "group", "value": "feature"})
        .eda()
        .to_dict()
    )
    assert build_gate_context(report)["groupCol"] == {
        "name": "account",
        "groups": 0,
        "observedRows": 0,
    }
    assert "no observed group labels" in _gate_rows(report)["03.6"]["evidence"]

    report["overview"].pop("group_summary")
    assert build_gate_context(report)["groupCol"] == {
        "name": "account",
        "groups": None,
        "observedRows": None,
    }
    assert "group counts are unavailable" in _gate_rows(report)["03.6"]["evidence"]


def test_eda_group_gates_require_a_choice_when_several_group_roles_are_declared():
    frame = pd.DataFrame(
        {
            "account": ["a", "a", "b", "b", "c", "c"],
            "region": ["north", "south"] * 3,
            "value": np.arange(6, dtype=float),
        }
    )
    session = Session.ingest(frame).set_roles(
        {"account": "group", "region": "group", "value": "feature"}
    )
    report = session.eda().to_dict()
    assert report["overview"].get("group_summary") is None
    assert build_gate_context(report)["groupCol"] == {"names": ["account", "region"]}
    gates = _gate_rows(report)
    assert "Choose a grouping column from account, region" in gates["03.1"]["evidence"]
    assert "which declared grouping column is selected" in gates["03.6"]["evidence"]

    with pytest.raises(ValidationError, match="group_column"):
        session.group_split(test_size=1)
    split = session.group_split(group_column="region", test_size=1)
    assert set(split.partition("train")["region"]).isdisjoint(split.partition("test")["region"])
