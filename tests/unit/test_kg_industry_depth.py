"""Industry-depth tests for KG backends (PyKEEN skip when absent)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import MissingExtraError, ValidationError
from buildml.kg.catalog import (
    kg_capability_matrix,
    list_kg_methods,
    resolve_backend_method,
)
from buildml.kg.extras import pykeen_available
from buildml.kg.fit import fit_kg
from buildml.kg.models import score_complex, score_rotate


def _kg_frame() -> pd.DataFrame:
    rows = [
        ("a", "r1", "x"),
        ("b", "r1", "x"),
        ("a", "r2", "y"),
        ("c", "r1", "x"),
        ("b", "r2", "y"),
        ("c", "r2", "z"),
        ("a", "r1", "z"),
        ("d", "r1", "x"),
        ("a", "r2", "x"),
        ("b", "r1", "y"),
        ("c", "r1", "z"),
        ("d", "r2", "y"),
        ("x", "r1", "a"),
        ("y", "r2", "b"),
        ("z", "r1", "c"),
        ("d", "r2", "z"),
    ]
    return pd.DataFrame(rows, columns=["head", "relation", "tail"])


def _session() -> Session:
    return (
        Session.ingest(_kg_frame())
        .set_roles({"head": "id", "relation": "id", "tail": "id"})
        .split(test_size=0.25, validation_size=0.15, random_state=0)
    )


def test_capability_matrix_native_always_available() -> None:
    matrix = kg_capability_matrix()
    assert matrix["backends"]["native"]["available"] is True
    assert "transe" in list_kg_methods(backend="native")
    assert "rotate" in list_kg_methods(backend="pykeen")


def test_resolve_backend_method_defaults_native_for_transe() -> None:
    backend, method = resolve_backend_method(backend=None, method="transe")
    assert backend == "native"
    assert method == "transe"


def test_rotate_requires_pykeen_backend() -> None:
    if not pykeen_available():
        with pytest.raises(MissingExtraError, match="kg-industry"):
            resolve_backend_method(backend=None, method="rotate")
        return
    backend, method = resolve_backend_method(backend=None, method="rotate")
    assert backend == "pykeen"
    assert method == "rotate"


def test_missing_extra_when_pykeen_requested_without_install() -> None:
    if pykeen_available():
        pytest.skip("PyKEEN installed — MissingExtraError path not testable here.")
    with pytest.raises(MissingExtraError, match="kg-industry"):
        resolve_backend_method(backend="pykeen", method="rotate")


def test_native_rotate_rejected() -> None:
    session = _session()
    with pytest.raises(ValidationError, match="not valid for backend='native'"):
        fit_kg(
            session.dataset,
            session.split_plan,
            backend="native",
            method="rotate",  # type: ignore[arg-type]
            head_column="head",
            relation_column="relation",
            tail_column="tail",
        )


def test_session_backend_routing_native() -> None:
    session = _session()
    fit = session.fit_kg(
        backend="native",
        method="transe",
        head_column="head",
        relation_column="relation",
        tail_column="tail",
        embedding_dim=8,
        epochs=10,
        batch_size=8,
        random_state=0,
    )
    assert fit.backend == "native"
    assert session.kg_plan is not None
    assert session.kg_plan.backend == "native"


@pytest.mark.skipif(
    not pykeen_available(),
    reason="buildml[kg-industry] pykeen not installed",
)
def test_pykeen_rotate_fit_and_eval() -> None:
    session = _session()
    try:
        fit = session.fit_kg(
            backend="pykeen",
            method="rotate",
            head_column="head",
            relation_column="relation",
            tail_column="tail",
            embedding_dim=16,
            epochs=5,
            batch_size=16,
            random_state=0,
        )
    except MissingExtraError as exc:
        pytest.skip(str(exc))
    assert fit.backend == "pykeen"
    assert fit.method == "rotate"
    assert session.kg_plan is not None
    assert session.kg_plan.embedding_kind == "rotate"
    ev = session.evaluate_kg(partition="test", k=5)
    assert "mrr" in ev.metrics


@pytest.mark.skipif(
    not pykeen_available(),
    reason="buildml[kg-industry] pykeen not installed",
)
def test_pykeen_complex_fit_smoke() -> None:
    session = _session()
    try:
        fit = session.fit_kg(
            backend="pykeen",
            method="complex",
            head_column="head",
            relation_column="relation",
            tail_column="tail",
            embedding_dim=16,
            epochs=5,
            batch_size=16,
            random_state=0,
        )
    except MissingExtraError as exc:
        pytest.skip(str(exc))
    assert fit.backend == "pykeen"
    assert session.kg_plan is not None
    assert session.kg_plan.embedding_kind == "complex"


def test_rotate_scoring_prefers_aligned_triple() -> None:
    ent = np.array([1 + 0j, 0 + 1j, 2 + 0j], dtype=np.complex128)
    rel = np.array([[0.0, 0.0]], dtype=float)
    s_true = float(
        score_rotate(np.array([0]), np.array([0]), np.array([2]), ent, rel)[0]
    )
    s_false = float(
        score_rotate(np.array([0]), np.array([0]), np.array([1]), ent, rel)[0]
    )
    assert s_true > s_false


def test_complex_scoring_prefers_aligned_triple() -> None:
    ent = np.array([1 + 0j, 0 + 1j, 2 + 0j], dtype=np.complex128)
    rel = np.array([[1 + 0j, 0 + 0j]], dtype=np.complex128)
    s_true = float(
        score_complex(np.array([0]), np.array([0]), np.array([2]), ent, rel)[0]
    )
    s_false = float(
        score_complex(np.array([0]), np.array([0]), np.array([1]), ent, rel)[0]
    )
    assert s_true > s_false
