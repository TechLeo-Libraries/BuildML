"""Binary nuisance metrics and causal scores agree with independent arithmetic."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.causal.evaluate import evaluate_causal
from buildml.data.dataset import Dataset


@pytest.mark.parametrize("method", ["t_learner", "ipw", "aipw"])
@pytest.mark.parametrize("shifted", [False, True])
def test_binary_outcome_and_overlap_diagnostics_match_frozen_nuisances(method, shifted):
    rng = np.random.default_rng(417)
    x = rng.normal(size=(360, 2))
    treatment = (rng.random(len(x)) < 1 / (1 + np.exp(-x[:, 0]))).astype(int)
    probability = 1 / (1 + np.exp(-(0.8 * treatment + 0.5 * x[:, 0] - 0.2 * x[:, 1])))
    outcome = (rng.random(len(x)) < probability).astype(int)
    frame = pd.DataFrame(
        {"x": x[:, 0], "z": x[:, 1], "t": treatment, "y": outcome}, index=np.arange(4000, 4360)
    )
    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "z": "feature", "t": "ignore", "y": "target"})
        .split(test_size=0.25, validation_size=0.2, random_state=12)
    )
    session.causal.declare_assumptions(
        treatment="t",
        outcome="y",
        confounders=["x", "z"],
        acknowledge_unconfoundedness=True,
        acknowledge_positivity=True,
    )
    session.causal.fit(method=method, bootstrap_samples=0, clip_propensity=(0.3, 0.7))
    plan = session.causal.plan
    evaluation = session.partition("test").copy()
    if shifted:
        # Deliberate extrapolation exercises clipping disclosures, not a claim
        # about identification or effect recovery under population shift.
        evaluation["x"] = np.where(np.arange(len(evaluation)) % 2, 1000.0, -1000.0)
    original = evaluation.copy()
    dataset = Dataset.from_pandas(evaluation)
    fingerprint = joblib.hash(plan)
    result = evaluate_causal(dataset, plan, None, partition="all", bootstrap_samples=0)
    design = evaluation[["x", "z"]].to_numpy()
    t, y = evaluation.t.to_numpy(), evaluation.y.to_numpy()
    assert plan.outcome_kind == "binary" and result.n_rows == len(evaluation)
    assert result.ate_ci_low is None and result.ate_ci_high is None
    assert any("not proof" in text for text in result.disclosures)
    mu0 = mu1 = propensity = None
    if plan.mu0_ is not None:
        mu0, mu1 = plan.mu0_.predict_proba(design)[:, 1], plan.mu1_.predict_proba(design)[:, 1]
        observed_outcome = np.where(t == 1, mu1, mu0)
        assert result.metrics["outcome_accuracy"] == pytest.approx(
            np.mean((observed_outcome >= 0.5) == y)
        )
        assert result.metrics["outcome_brier"] == pytest.approx(
            np.mean((y - observed_outcome) ** 2)
        )
    if plan.propensity_ is not None:
        propensity = np.clip(plan.propensity_.predict_proba(design)[:, 1], 0.3, 0.7)
        assert result.metrics["propensity_mean"] == pytest.approx(propensity.mean())
        assert result.metrics["propensity_min"] == pytest.approx(propensity.min())
        assert result.metrics["propensity_max"] == pytest.approx(propensity.max())
        assert result.metrics["propensity_brier"] == pytest.approx(np.mean((t - propensity) ** 2))
        if shifted:
            assert any("clip floor" in warning for warning in result.warnings)
            assert any("clip ceiling" in warning for warning in result.warnings)
    if method == "t_learner":
        expected_effect = np.mean(mu1 - mu0)
    elif method == "ipw":
        expected_effect = np.mean(t * y / propensity - (1 - t) * y / (1 - propensity))
    else:
        expected_effect = np.mean(
            mu1 - mu0 + t * (y - mu1) / propensity - (1 - t) * (y - mu0) / (1 - propensity)
        )
    assert result.ate == pytest.approx(expected_effect)
    assert joblib.hash(plan) == fingerprint
    pd.testing.assert_frame_equal(dataset._ensure_pandas(), original)
