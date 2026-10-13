"""A real policy learner must learn treatment benefit rather than observed assignment."""

import numpy as np
import pandas as pd
import pytest

from buildml import Session


@pytest.mark.parametrize("string_treatment", [False, True])
@pytest.mark.parametrize("binary_outcome", [False, True])
def test_policy_tree_learns_opposite_treatment_choices_for_known_benefit_groups(string_treatment, binary_outcome):
    pytest.importorskip("econml")
    rng = np.random.default_rng(613)
    x = rng.uniform(-2, 2, 1000)
    z = rng.normal(size=len(x))
    treatment = rng.integers(0, 2, len(x))
    outcome = treatment * (3 * x) + 0.5 * z + rng.normal(scale=0.1, size=len(x))
    if binary_outcome:
        probability = np.where(treatment == 0, 0.5, np.where(x > 0, 0.9, 0.1))
        outcome = (rng.uniform(size=len(x)) < probability).astype(int)
    labels = np.where(treatment == 0, "control", "treated") if string_treatment else treatment
    frame = pd.DataFrame({"x": x, "z": z, "t": labels, "y": outcome})
    session = (Session.ingest(frame)
               .set_roles({"x": "feature", "z": "feature", "t": "feature", "y": "target"})
               .split(test_size=0.2, validation_size=0.2, random_state=4))
    session.causal.declare_assumptions(treatment="t", outcome="y", confounders=["x", "z"],
                                      acknowledge_unconfoundedness=True, acknowledge_positivity=True)
    before = session.dataset.frame.copy(deep=True)
    result = session.causal.fit(backend="econml", method="policy_tree", bootstrap_samples=0, random_state=4)
    assert result.backend == "econml"
    assert result.n_train_rows == len(session.split_plan.train_indices)
    model = session.causal_plan.backend_artifact_["estimator"]
    held_out = session.partition("test")
    choices = np.asarray(model.predict(held_out[["x", "z"]].to_numpy()))
    clear_benefit = np.abs(held_out.x.to_numpy()) > 0.5
    expected = (held_out.x.to_numpy() > 0).astype(int)
    assert (choices[clear_benefit] == expected[clear_benefit]).mean() > (0.85 if binary_outcome else 0.9)
    assert set(choices) == {0, 1}
    pd.testing.assert_frame_equal(session.dataset.frame, before)
