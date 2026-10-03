"""Causal bundle metadata preserves numeric and categorical treatment levels."""

import json

import numpy as np
import pandas as pd
import pytest

from buildml import Session


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.float32, np.bool_, str])
def test_treatment_levels_survive_bundle_roundtrip(tmp_path, dtype):
    rng = np.random.default_rng(42)
    treatment = np.tile([0, 1], 60).astype(dtype)
    x = rng.normal(size=len(treatment))
    frame = pd.DataFrame({"x": x, "treatment": treatment, "outcome": 2 * np.tile([0, 1], 60) + x})
    session = Session.ingest(frame).set_roles(
        {"x": "feature", "treatment": "ignore", "outcome": "target"}
    )
    session.split(test_size=0.2, random_state=42)
    session.causal.declare_assumptions(
        treatment="treatment",
        outcome="outcome",
        confounders=["x"],
        acknowledge_unconfoundedness=True,
        acknowledge_positivity=True,
    )
    session.causal.fit(method="t_learner", bootstrap_samples=0)
    before = session.causal.estimate(partition="test", bootstrap_samples=0)
    path = session.causal.save_bundle(tmp_path / "bundle")
    metadata = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    expected = [
        value.item() if isinstance(value, np.generic) else value
        for value in session.causal.plan.treatment_levels
    ]
    assert metadata["plan"]["treatment_levels"] == expected
    restored = Session.ingest(frame).set_roles(dict(session.dataset.roles))
    split = session.split_plan
    restored.inject_split(train_indices=split.train_indices, test_indices=split.test_indices)
    restored.causal.load_bundle(path, trusted=True)
    after = restored.causal.estimate(partition="test", bootstrap_samples=0)
    assert after.ate == pytest.approx(before.ate)
    assert restored.causal.plan.treatment_levels == session.causal.plan.treatment_levels
