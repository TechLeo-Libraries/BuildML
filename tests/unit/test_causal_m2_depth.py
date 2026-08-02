"""M2 depth coverage for causal low-level + Session APIs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.causal.estimate import estimate_causal
from buildml.causal.evaluate import evaluate_causal
from buildml.causal.fit import fit_causal
from buildml.causal.refute import refute_causal
from buildml.causal.types import CausalAssumptions
from buildml.core.errors import ValidationError
from buildml.explain.sync import REQUIRED_AI_TOOL_SESSION_METHODS


def _causal_frame(n: int = 360, seed: int = 5, effect: float = 1.6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n, 2))
    logit = 0.85 * w[:, 0] - 0.55 * w[:, 1]
    t = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    y = effect * t + 0.65 * w[:, 0] - 0.45 * w[:, 1] + rng.normal(scale=0.4, size=n)
    return pd.DataFrame({"x1": w[:, 0], "x2": w[:, 1], "t": t, "y": y})


def _session() -> Session:
    return (
        Session.ingest(_causal_frame())
        .set_roles(
            {"x1": "feature", "x2": "feature", "t": "feature", "y": "target"}
        )
        .split(test_size=0.2, validation_size=0.2, random_state=1)
        .scale(method="standard")
    )


def _assumptions(**overrides) -> CausalAssumptions:
    base = dict(
        treatment="t",
        outcome="y",
        confounders=("x1", "x2"),
        acknowledge_unconfoundedness=True,
        acknowledge_positivity=True,
    )
    base.update(overrides)
    assumptions = CausalAssumptions(**base)
    assumptions.validate()
    return assumptions


def test_refuse_without_assumptions_acknowledgements() -> None:
    with pytest.raises(ValidationError, match="acknowledge_unconfoundedness"):
        CausalAssumptions(
            treatment="t",
            outcome="y",
            confounders=("x1",),
            acknowledge_unconfoundedness=False,
            acknowledge_positivity=True,
        ).validate()
    with pytest.raises(ValidationError, match="acknowledge_positivity"):
        CausalAssumptions(
            treatment="t",
            outcome="y",
            confounders=("x1",),
            acknowledge_unconfoundedness=True,
            acknowledge_positivity=False,
        ).validate()
    with pytest.raises(ValidationError, match="confounders is empty"):
        CausalAssumptions(
            treatment="t",
            outcome="y",
            confounders=(),
            acknowledge_unconfoundedness=True,
            acknowledge_positivity=True,
        ).validate()


def test_session_refuses_fit_without_declare() -> None:
    session = _session()
    with pytest.raises(ValidationError, match="No CausalAssumptions"):
        session.fit_causal(method="aipw", bootstrap_samples=0)


def test_session_refuse_incomplete_declare() -> None:
    session = _session()
    with pytest.raises(ValidationError, match="confounders"):
        session.declare_causal_assumptions(
            treatment="t",
            outcome="y",
            confounders=None,
            acknowledge_unconfoundedness=True,
            acknowledge_positivity=True,
        )


def test_aipw_recovers_ate_roughly() -> None:
    session = _session()
    assumptions = _assumptions()
    plan, fit = fit_causal(
        session.dataset,
        session.split_plan,
        assumptions,
        method="aipw",
        bootstrap_samples=30,
        random_state=0,
    )
    assert fit.method == "aipw"
    assert fit.ate_ci_low is not None and fit.ate_ci_high is not None
    # Synthetic true effect ≈ 1.6; allow estimation noise.
    assert 0.8 < fit.ate < 2.4
    assert plan.mu0_ is not None and plan.mu1_ is not None and plan.propensity_ is not None
    est = estimate_causal(
        session.dataset, plan, session.split_plan, partition="validation", bootstrap_samples=15
    )
    assert est.n_rows > 0
    ev = evaluate_causal(
        session.dataset, plan, session.split_plan, partition="validation", bootstrap_samples=10
    )
    assert "propensity_auc" in ev.metrics or "outcome_rmse" in ev.metrics


def test_t_learner_and_ipw_paths() -> None:
    session = _session()
    assumptions = _assumptions()
    for method in ("t_learner", "ipw"):
        plan, fit = fit_causal(
            session.dataset,
            session.split_plan,
            assumptions,
            method=method,  # type: ignore[arg-type]
            bootstrap_samples=0,
            random_state=2,
        )
        assert fit.method == method
        assert np.isfinite(fit.ate)


def test_refute_placebo_and_bundle(tmp_path) -> None:
    session = _session()
    session.declare_causal_assumptions(
        treatment="t",
        outcome="y",
        confounders=["x1", "x2"],
        acknowledge_unconfoundedness=True,
        acknowledge_positivity=True,
    )
    fit = session.fit_causal(method="aipw", bootstrap_samples=20)
    refute = session.refute_causal(kind="placebo_treatment", random_state=0)
    assert refute.kind == "placebo_treatment"
    assert abs(refute.refute_ate) < abs(fit.ate) + 1.0
    out = tmp_path / "causal_bundle"
    session.save_causal_bundle(out)
    other = (
        Session.ingest(_causal_frame())
        .set_roles(
            {"x1": "feature", "x2": "feature", "t": "feature", "y": "target"}
        )
        .split(test_size=0.2, validation_size=0.2, random_state=1)
        .scale(method="standard")
    )
    other.load_causal_bundle(out)
    assert other.causal_plan is not None
    assert other.causal_assumptions is not None
    est = other.estimate_causal(partition="test", bootstrap_samples=10)
    assert np.isfinite(est.ate)


def test_instruments_refused() -> None:
    with pytest.raises(ValidationError, match="Instruments"):
        CausalAssumptions(
            treatment="t",
            outcome="y",
            confounders=("x1",),
            instruments=("z",),
            acknowledge_unconfoundedness=True,
            acknowledge_positivity=True,
        ).validate()


def test_ai_allowlist_includes_causal() -> None:
    for name in (
        "declare_causal_assumptions",
        "fit_causal",
        "evaluate_causal",
        "estimate_causal",
    ):
        assert name in REQUIRED_AI_TOOL_SESSION_METHODS
