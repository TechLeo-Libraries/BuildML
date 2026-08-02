"""Session-facing slice tests for knowledge graphs."""

from __future__ import annotations

import pandas as pd
import pytest

from buildml import Session
from buildml.ai.tools import build_default_registry
from buildml.core.errors import LeakageError, ValidationError
from buildml.explain.catalog import OPERATION_CATALOG
from buildml.kg.features import resolve_triple_columns
from buildml.kg.models import score_transe


def _tiny_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "head": ["a", "b", "a", "c", "b", "c", "a", "d"],
            "relation": ["r1", "r1", "r2", "r1", "r2", "r2", "r1", "r1"],
            "tail": ["x", "x", "y", "x", "y", "z", "z", "x"],
        }
    )


def test_core_import_and_catalog() -> None:
    import buildml.kg as kg

    assert hasattr(kg, "fit_kg")
    assert hasattr(kg, "kg_capability_matrix")
    assert hasattr(Session, "fit_kg")
    for op in (
        "fit_kg",
        "score_triples",
        "predict_links",
        "query_kg",
        "evaluate_kg",
        "save_kg_bundle",
        "load_kg_bundle",
    ):
        assert op in OPERATION_CATALOG
    assert "kg-triples" in OPERATION_CATALOG["fit_kg"].concept_links
    assert "kg-link-prediction" in OPERATION_CATALOG["evaluate_kg"].concept_links
    assert "kg-symbolic-query" in OPERATION_CATALOG["query_kg"].concept_links
    assert "kg-bundle-boundary" in OPERATION_CATALOG["save_kg_bundle"].concept_links

    registry = build_default_registry()
    for name in (
        "fit_kg",
        "score_triples",
        "predict_links",
        "query_kg",
        "evaluate_kg",
        "save_kg_bundle",
        "load_kg_bundle",
    ):
        assert name in registry


def test_fit_requires_split() -> None:
    session = Session.ingest(_tiny_frame()).set_roles(
        {"head": "id", "relation": "id", "tail": "id"}
    )
    with pytest.raises((LeakageError, ValidationError)):
        session.fit_kg(
            head_column="head",
            relation_column="relation",
            tail_column="tail",
        )


def test_resolve_requires_explicit_columns() -> None:
    session = Session.ingest(_tiny_frame()).set_roles(
        {"head": "id", "relation": "id", "tail": "id"}
    )
    with pytest.raises(ValidationError, match="head_column"):
        resolve_triple_columns(
            session.dataset,
            head_column=None,
            relation_column="relation",
            tail_column="tail",
        )


def test_transe_prefers_true_triple() -> None:
    # Hand-craft embeddings where a+r ≈ t for the true triple
    import numpy as np

    ent = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]], dtype=float)
    rel = np.array([[1.0, 0.0]], dtype=float)
    # (0, 0, 2): 1+1 = 2 → distance 0; (0,0,1): 1+1-0 = (2,0) worse
    s_true = float(score_transe(np.array([0]), np.array([0]), np.array([2]), ent, rel)[0])
    s_false = float(score_transe(np.array([0]), np.array([0]), np.array([1]), ent, rel)[0])
    assert s_true > s_false


def test_fit_predict_query_smoke() -> None:
    frame = _tiny_frame()
    # Expand slightly so split keeps enough train entities
    extra = pd.DataFrame(
        {
            "head": ["a", "b", "c", "d", "x", "y"],
            "relation": ["r2", "r1", "r1", "r2", "r1", "r2"],
            "tail": ["x", "y", "z", "y", "a", "b"],
        }
    )
    frame = pd.concat([frame, extra], ignore_index=True)
    session = (
        Session.ingest(frame)
        .set_roles({"head": "id", "relation": "id", "tail": "id"})
        .split(test_size=0.25, validation_size=0.15, random_state=0)
    )
    fit = session.fit_kg(
        method="transe",
        head_column="head",
        relation_column="relation",
        tail_column="tail",
        embedding_dim=8,
        epochs=15,
        batch_size=8,
        random_state=0,
    )
    assert fit.n_train_triples > 0
    assert fit.n_entities >= 2
    assert session.kg_plan is not None
    assert any("Negative sampling" in d for d in fit.disclosures)

    scored = session.score_triples(partition="test")
    assert scored.n_triples >= 0

    preds = session.predict_links(
        mode="tail",
        heads=[frame["head"].iloc[0]],
        relations=[frame["relation"].iloc[0]],
        k=3,
    )
    assert preds.n_queries == 1

    nbrs = session.query_kg(mode="neighbors", entity=frame["head"].iloc[0])
    assert nbrs.mode == "neighbors"

    ev = session.evaluate_kg(partition="test", k=3)
    assert "mrr" in ev.metrics
    assert "hits_at_1" in ev.metrics
