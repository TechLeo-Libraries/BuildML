"""Industry-depth tests for synthetic-data backends (R6.10)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import MissingExtraError, ValidationError
from buildml.synthetic.catalog import (
    list_synthetic_methods,
    resolve_backend_method,
    synthetic_capability_matrix,
)
from buildml.synthetic.extras import sdv_available, synthetic_industry_available
from buildml.synthetic.validation import validate_synthetic


def _session_tabular() -> Session:
    rng = np.random.default_rng(1)
    n = 220
    frame = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n) * 1.5,
            "cat": rng.choice(["a", "b", "c"], size=n),
            "y": rng.integers(0, 2, size=n),
        }
    )
    return (
        Session.ingest(frame)
        .set_roles(
            {
                "x1": "feature",
                "x2": "feature",
                "cat": "feature",
                "y": "target",
            }
        )
        .split(test_size=0.25, validation_size=0.15, random_state=0)
    )


def test_capability_matrix_native_always_available() -> None:
    matrix = synthetic_capability_matrix()
    assert matrix["backends"]["native"]["available"] is True
    assert "gaussian_copula" in matrix["backends"]["native"]["methods"]
    assert "synthetic_vs_resample" in matrix


def test_list_synthetic_methods_includes_native() -> None:
    assert "bootstrap" in list_synthetic_methods(backend="native")


def test_resolve_backend_method_native_copula() -> None:
    backend, method = resolve_backend_method(backend="native", method="gaussian_copula")
    assert backend == "native"
    assert method == "gaussian_copula"


def test_resolve_sdv_requires_extra_when_missing() -> None:
    if synthetic_industry_available():
        backend, method = resolve_backend_method(backend="sdv", method="ctgan")
        assert backend == "sdv"
        assert method == "ctgan"
    else:
        with pytest.raises(MissingExtraError):
            resolve_backend_method(backend="sdv", method="ctgan")


def test_native_copula_session_path() -> None:
    session = _session_tabular()
    fit = session.fit_synthesizer(backend="native", method="gaussian_copula", random_state=0)
    assert fit.backend == "native"
    sample = session.sample_synthetic(n=50, random_state=1, validate=True)
    assert sample.n_rows == 50
    ev = session.evaluate_synthetic(mode="tstr", partition="test")
    assert "score" in ev.metrics


def test_validate_synthetic_builtin_checks() -> None:
    session = _session_tabular()
    session.fit_synthesizer(method="bootstrap", random_state=0)
    sample = session.sample_synthetic(n=30, random_state=2)
    assert session.synthesizer_plan is not None
    assert sample.frame is not None
    result = validate_synthetic(session.synthesizer_plan, sample.frame)
    assert result.n_checks >= 1
    assert result.checks.get("columns_present") is True


def test_unknown_method_raises() -> None:
    session = _session_tabular()
    with pytest.raises(ValidationError, match="Unknown synthesizer"):
        session.fit_synthesizer(method="not_a_real_method")  # type: ignore[arg-type]


@pytest.mark.skipif(not sdv_available(), reason="SDV not installed")
def test_sdv_ctgan_session_path() -> None:
    session = _session_tabular()
    fit = session.fit_synthesizer(
        backend="sdv",
        method="ctgan",
        epochs=5,
        batch_size=64,
        random_state=0,
    )
    assert fit.backend == "sdv"
    assert fit.method == "ctgan"
    sample = session.sample_synthetic(n=40, random_state=1)
    assert sample.n_rows == 40
    ev = session.evaluate_synthetic(mode="fidelity", eval_backend="auto", partition="test")
    assert "mean_ks" in ev.metrics


def test_session_synthetic_capability_matrix() -> None:
    matrix = Session.synthetic_capability_matrix()
    assert matrix["backends"]["native"]["available"] is True


def test_ai_registry_has_synthetic_capability_matrix() -> None:
    from buildml.ai.tools import build_default_registry

    registry = {spec.name for spec in build_default_registry().tools}
    assert "synthetic_capability_matrix" in registry
