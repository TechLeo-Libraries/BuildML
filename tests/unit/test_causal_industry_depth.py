"""Industry-depth tests for causal backends (DoWhy / EconML skip when absent)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.causal.catalog import (
    causal_capability_matrix,
    list_causal_methods,
    resolve_backend_method,
)
from buildml.causal.extras import dowhy_available, econml_available
from buildml.causal.fit import fit_causal
from buildml.causal.refute import refute_causal
from buildml.causal.types import CausalAssumptions
from buildml.core.errors import MissingExtraError, ValidationError


def _causal_frame(n: int = 400, seed: int = 7, effect: float = 1.5) -> pd.DataFrame:
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


def test_capability_matrix_native_always_available() -> None:
    matrix = causal_capability_matrix()
    assert matrix["backends"]["native"]["available"] is True
    assert "aipw" in list_causal_methods(backend="native")


def test_resolve_backend_method_defaults_native() -> None:
    backend, method = resolve_backend_method(backend=None, method="aipw")
    assert backend == "native"
    assert method == "aipw"


def test_missing_extra_when_dowhy_requested_without_install() -> None:
    if dowhy_available():
        pytest.skip("DoWhy installed — MissingExtraError path not testable here.")
    with pytest.raises(MissingExtraError, match="causal-industry"):
        resolve_backend_method(backend="dowhy", method="backdoor_linear")


@pytest.mark.skipif(not dowhy_available(), reason="buildml[causal-industry] dowhy not installed")
def test_dowhy_backend_recovers_ate() -> None:
    session = _session()
    assumptions = _assumptions()
    plan, fit = fit_causal(
        session.dataset,
        session.split_plan,
        assumptions,
        backend="dowhy",
        method="backdoor_linear",
        random_state=0,
    )
    assert fit.backend == "dowhy"
    assert 0.7 < fit.ate < 2.3
    refute = refute_causal(
        session.dataset, plan, session.split_plan, kind="random_common_cause"
    )
    assert refute.backend == "dowhy"
    assert refute.refute_ate is not None


@pytest.mark.skipif(not econml_available(), reason="buildml[causal-industry] econml not installed")
def test_econml_dml_recovers_ate() -> None:
    session = _session()
    assumptions = _assumptions()
    plan, fit = fit_causal(
        session.dataset,
        session.split_plan,
        assumptions,
        backend="econml",
        method="dml",
        bootstrap_samples=20,
        random_state=0,
    )
    assert fit.backend == "econml"
    assert 0.7 < fit.ate < 2.3
    assert fit.ate_ci_low is not None


@pytest.mark.skipif(not econml_available(), reason="buildml[causal-industry] econml not installed")
def test_econml_causal_forest_cate_std() -> None:
    session = _session()
    assumptions = _assumptions()
    plan, fit = fit_causal(
        session.dataset,
        session.split_plan,
        assumptions,
        backend="econml",
        method="causal_forest",
        bootstrap_samples=0,
        random_state=0,
    )
    assert plan.cate_std is not None or fit.cate_std is not None


def test_session_backend_routing_native() -> None:
    session = _session()
    session.declare_causal_assumptions(
        treatment="t",
        outcome="y",
        confounders=["x1", "x2"],
        acknowledge_unconfoundedness=True,
        acknowledge_positivity=True,
    )
    fit = session.fit_causal(backend="native", method="aipw", bootstrap_samples=0)
    assert fit.backend == "native"


def test_dowhy_refute_kind_rejected_on_native_plan() -> None:
    session = _session()
    assumptions = _assumptions()
    plan, _ = fit_causal(
        session.dataset,
        session.split_plan,
        assumptions,
        backend="native",
        method="aipw",
        bootstrap_samples=0,
    )
    with pytest.raises(ValidationError, match="Unknown refute kind"):
        refute_causal(
            session.dataset,
            plan,
            session.split_plan,
            kind="add_unobserved_common_cause",  # type: ignore[arg-type]
        )
