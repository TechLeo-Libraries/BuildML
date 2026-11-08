"""Serving replays frozen learned transforms without batch-dependent refitting."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.preprocess.apply import apply_preprocess_plans
from buildml.preprocess.binning import fit_binning
from buildml.preprocess.custom import fit_custom_transform, register_transform, unregister_transform
from buildml.preprocess.reduce import fit_reducer
from buildml.preprocess.select import fit_feature_selector
from buildml.preprocess.text import fit_text_features


def _fitted(step):
    x = np.linspace(-3, 3, 80)
    frame = pd.DataFrame({"x": x, "z": np.sin(x), "constant": 1.,
                          "text": np.where(x > 0, "orchard apple", "engine valve"),
                          "target": (x > 0).astype(int)})
    session = (Session.ingest(frame).set_roles({"x": "feature", "z": "feature",
                "constant": "feature", "text": "feature", "target": "target"})
               .split(test_size=.25, validation_size=.25, random_state=32))
    arguments = session.dataset, session.split_plan
    if step == "text":
        plan = fit_text_features(*arguments, columns=["text"], max_features=8)
    elif step == "binning":
        plan = fit_binning(*arguments, columns=["x", "z"], n_bins=4)
    elif step == "reduce":
        plan = fit_reducer(*arguments, columns=["x", "z"], n_components=1)
    else:
        plan = fit_feature_selector(*arguments, columns=["x", "z", "constant"])
    return plan, frame.drop(columns="target")


@pytest.mark.parametrize("step", ["text", "binning", "reduce", "feature_select"])
def test_scoring_is_batch_invariant_and_preserves_fitted_artifacts(step):
    plan, frame = _fitted(step)
    scored = frame.iloc[:3].copy()
    scored.index = [1001, 1002, 1003]
    scored.loc[1003, "x"] = 100_000
    scored.loc[1003, "text"] = "previously unseen vocabulary"
    original = scored.copy()
    before = joblib.hash(plan)
    together = apply_preprocess_plans(scored, {step: plan})
    separately = pd.concat([
        apply_preprocess_plans(scored.loc[[index]], {step: plan}).dataset._ensure_pandas()
        for index in scored.index
    ])
    pd.testing.assert_frame_equal(together.dataset._ensure_pandas(), separately)
    assert together.applied == (step,)
    assert together.dataset.n_rows == len(scored)
    assert joblib.hash(plan) == before
    pd.testing.assert_frame_equal(scored, original)


def test_selected_feature_missing_at_serving_is_an_explicit_schema_error():
    plan, frame = _fitted("feature_select")
    missing = plan.selected_features_[0]
    with pytest.raises(ValidationError, match="missing after earlier score-time steps"):
        apply_preprocess_plans(frame.drop(columns=missing), {"feature_select": plan})


def test_custom_replay_uses_only_frozen_training_statistics():
    fitted_rows = []

    def fit(frame, params):
        fitted_rows.extend(frame.index)
        return float(frame.x.mean())

    def transform(frame, mean):
        return frame.assign(x=frame.x - mean)

    name = "test_frozen_custom_replay_center"
    register_transform(name, fit=fit, transform=transform)
    try:
        frame = pd.DataFrame({"x": np.arange(40.)})
        session = Session.ingest(frame).split(test_size=.25, validation_size=.25, random_state=13)
        plan = fit_custom_transform(session.dataset, session.split_plan, name=name, columns=["x"])
        assert set(fitted_rows) == set(session.split_plan.train_indices)
        expected_mean = frame.iloc[list(session.split_plan.train_indices)].x.mean()
        score = pd.DataFrame({"x": [10_000., -3.]}, index=[500, 501])
        result = apply_preprocess_plans(score, custom_plan=plan)
        np.testing.assert_allclose(result.dataset._ensure_pandas().x, score.x - expected_mean)
        assert result.applied == ("custom",)
        assert len(fitted_rows) == len(session.split_plan.train_indices)
        assert plan.artifact_ == expected_mean
    finally:
        unregister_transform(name)
