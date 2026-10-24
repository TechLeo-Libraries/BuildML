"""Ordinary dataframe names must not change the declared backdoor graph."""

import numpy as np
import pandas as pd
import pytest

from buildml import Session


@pytest.mark.parametrize("renames", [
    {}, {"x": "blood pressure"},
    {"x": 'severity "score"', "z": "heart-rate"},
    {"t": "therapy group", "y": "clinical outcome", "x": "风险评分"},
])
def test_declared_named_confounders_recover_known_linear_effect(renames, tmp_path):
    pytest.importorskip("dowhy")
    rng = np.random.default_rng(907)
    x, z = rng.normal(size=(2, 1000))
    probability = 1 / (1 + np.exp(-(2 * x - z)))
    treatment = (rng.uniform(size=len(x)) < probability).astype(int)
    outcome = 1.7 * treatment + 4 * x - 3 * z + rng.normal(scale=0.1, size=len(x))
    frame = pd.DataFrame({"x": x, "z": z, "t": treatment, "y": outcome}).rename(columns=renames)
    def name(column):
        return renames.get(column, column)
    session = (Session.ingest(frame)
               .set_roles({name("x"): "feature", name("z"): "feature", name("t"): "feature",
                           name("y"): "target"})
               .split(test_size=0.2, validation_size=0.2, random_state=8))
    session.causal.declare_assumptions(treatment=name("t"), outcome=name("y"),
                                      confounders=[name("x"), name("z")],
                                      acknowledge_unconfoundedness=True, acknowledge_positivity=True)
    before = session.dataset.frame.copy(deep=True)
    result = session.causal.fit(backend="dowhy", method="backdoor_linear", random_state=8)
    assert result.ate == pytest.approx(1.7, abs=0.05)
    assert result.n_train_rows == len(session.split_plan.train_indices)
    pd.testing.assert_frame_equal(session.dataset.frame, before)
    destination = session.causal.save_bundle(tmp_path / "causal")
    restored = Session.ingest(frame).set_roles(dict(session.dataset.roles))
    restored.split(test_size=0.2, validation_size=0.2, random_state=8)
    restored.causal.load_bundle(destination, trusted=True)
    assert restored.causal_plan.ate == pytest.approx(result.ate)
    assert set(restored.causal_plan.backend_artifact_["graph"].nodes) == set(frame.columns)
