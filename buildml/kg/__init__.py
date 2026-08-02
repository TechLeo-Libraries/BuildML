"""Knowledge graphs (Session-shaped triples → embeddings + symbolic query).

Phase coverage (internal tracker — depth-first; do not spray stubs)
------------------------------------------------------------------
Phase 1–2 complete. Phase 3 — Application systems:
  Recommendation systems (**PASS**).
  Search / LTR (**PASS**).
  **Knowledge graphs (this module)** — **PASS** (Phase-1 bar).
  Optimisation / decision helpers (**PASS** — see ``buildml.optimize``).
  Synthetic-data systems (**PASS** — see ``buildml.synthetic``).

Honesty (this package):
  - Session rows are (head, relation, tail) triples.
  - Train-only materialization; never trains on holdout triples.
  - Algorithms: pure-numpy TransE and DistMult with disclosed uniform
    negative sampling (core — no Neo4j, no torch required).
  - Link prediction: score_triples / predict_links (tail|head|relation);
    evaluate_kg reports filtered MRR and Hits@K.
  - Symbolic query_kg: neighbors / path / typed over the **train** adjacency
    (not an LLM, not Cypher/Neo4j).
  - **Not** Graph ML node classification (``set_graph`` / ``fit_graph``),
    **not** a graph-database product, **not** RAG retrieve/generate.

Dependency policy: core stays numpy/pandas/sklearn. TransE/DistMult-lite
are justified in core (small dense embeddings + SGD; Session-scale graphs).
Optional ``buildml[graph]`` (NetworkX) is unused here — adjacency BFS is
pure-Python on the train triple store.

Lazy imports — keep the core import graph light.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT",
    "CHECKPOINT_BOUNDARY",
    "KgConfig",
    "KgEvalResult",
    "KgFitResult",
    "KgMethod",
    "KgNorm",
    "KgPlan",
    "KgQueryMode",
    "KgQueryResult",
    "LinkPredictionMode",
    "PredictLinksResult",
    "ScoreTriplesResult",
    "evaluate_kg",
    "fit_kg",
    "kg_status",
    "kg_status_for_session",
    "load_kg_bundle",
    "predict_links",
    "query_kg",
    "save_kg_bundle",
    "score_triples",
]


def __getattr__(name: str) -> Any:
    if name in {
        "KgConfig",
        "KgMethod",
        "KgNorm",
        "LinkPredictionMode",
        "KgQueryMode",
    }:
        from buildml.kg import types as types_mod

        return getattr(types_mod, name)
    if name in {
        "KgPlan",
        "KgFitResult",
        "ScoreTriplesResult",
        "PredictLinksResult",
        "KgQueryResult",
        "KgEvalResult",
    }:
        from buildml.kg import results as results_mod

        return getattr(results_mod, name)
    if name == "fit_kg":
        from buildml.kg.fit import fit_kg

        return fit_kg
    if name == "score_triples":
        from buildml.kg.predict import score_triples

        return score_triples
    if name == "predict_links":
        from buildml.kg.predict import predict_links

        return predict_links
    if name == "query_kg":
        from buildml.kg.query import query_kg

        return query_kg
    if name == "evaluate_kg":
        from buildml.kg.evaluate import evaluate_kg

        return evaluate_kg
    if name in {
        "BUNDLE_FORMAT",
        "CHECKPOINT_BOUNDARY",
        "save_kg_bundle",
        "load_kg_bundle",
    }:
        from buildml.kg import checkpoint as checkpoint_mod

        return getattr(checkpoint_mod, name)
    if name in {"kg_status", "kg_status_for_session"}:
        from buildml.kg import explain_hooks as hooks

        return getattr(hooks, name)
    raise AttributeError(f"module 'buildml.kg' has no attribute {name!r}")
