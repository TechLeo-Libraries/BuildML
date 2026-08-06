"""Knowledge-graph bundle persistence (distinct from Session / Graph / RAG)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from buildml._version import __version__
from buildml.core.errors import ValidationError
from buildml.core.serialization import joblib_load_trusted
from buildml.kg.results import (
    KgEvalResult,
    KgFitResult,
    KgPlan,
    KgQueryResult,
    PredictLinksResult,
    ScoreTriplesResult,
)

BUNDLE_FORMAT = "buildml.kg_bundle.v1"
CHECKPOINT_BOUNDARY = (
    "KG bundles, Graph ML bundles, recommender bundles, RAG bundles, and "
    "Session checkpoints are complementary, not interchangeable. A KG bundle "
    "(buildml.kg_bundle.v1) stores a train-fitted KgPlan (entity/relation "
    "vocabularies, embedding weights, train adjacency). A Session "
    "checkpoint stores data, roles, splits, and history; it does not embed "
    "the KG plan. Reload tabular workflow via checkpoint_load; reload the KG "
    "via load_kg_bundle. Honesty: Session KG learning/query: not Neo4j, not "
    "Graph ML node classification, not RAG."
)


def save_kg_bundle(
    path: str | Path,
    plan: KgPlan,
    *,
    fit_result: KgFitResult | None = None,
    eval_result: KgEvalResult | None = None,
    score_result: ScoreTriplesResult | None = None,
    predict_result: PredictLinksResult | None = None,
    query_result: KgQueryResult | None = None,
) -> Path:
    """Write a knowledge-graph bundle directory (``buildml.kg_bundle.v1``).

    Persists the fitted :class:`~buildml.kg.results.KgPlan` separately from
    Session checkpoints so tabular workflow and KG state reload independently.

    Parameters
    ----------
    path:
        Destination directory for ``meta.json`` and ``kg_plan.joblib``.
    plan:
        Train-fitted knowledge-graph plan to persist.
    fit_result, eval_result, score_result, predict_result, query_result:
        Optional last operation reports for bundle metadata.

    Returns
    -------
    pathlib.Path
        The bundle directory that was written.

    Raises
    ------
    ValidationError
        When ``plan`` is ``None``.
    """
    if plan is None:
        raise ValidationError("No KgPlan to save.")
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "plan": plan,
        "entity_embeddings": np.asarray(plan.entity_embeddings_),
        "relation_embeddings": np.asarray(plan.relation_embeddings_),
        "train_heads": np.asarray(plan.train_heads_),
        "train_relations": np.asarray(plan.train_relations_),
        "train_tails": np.asarray(plan.train_tails_),
        "entity_ids": list(plan.entity_ids),
        "relation_ids": list(plan.relation_ids),
        "true_triple_set": list(plan.true_triple_set_),
        "out_edges": {str(k): v for k, v in plan.out_edges_.items()},
        "in_edges": {str(k): v for k, v in plan.in_edges_.items()},
    }
    joblib.dump(payload, destination / "kg_plan.joblib")
    meta: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "buildml_version": __version__,
        "compatibility": CHECKPOINT_BOUNDARY,
        "plan": plan.to_dict(),
        "fit": None if fit_result is None else fit_result.to_dict(),
        "eval": None if eval_result is None else eval_result.to_dict(),
        "score": None if score_result is None else score_result.to_dict(),
        "predict": None if predict_result is None else predict_result.to_dict(),
        "query": None if query_result is None else query_result.to_dict(),
    }
    (destination / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return destination


def load_kg_bundle(path: str | Path, *, trusted: bool = False) -> KgPlan:
    """Load a knowledge-graph bundle into a :class:`~buildml.kg.results.KgPlan`.

    Validates bundle format and restores embeddings, vocabularies, and train
    adjacency for score, predict, query, and evaluate without reloading Session.

    Parameters
    ----------
    path:
        Bundle directory containing ``meta.json`` and ``kg_plan.joblib``.
    trusted:
        Must be ``True`` to deserialize pickle/joblib/torch payloads. Pass
        only for artifacts you created or fully trust. Defaults to ``False``.

    Returns
    -------
    KgPlan
        Deserialised plan ready for link prediction and symbolic query.

    Raises
    ------
    ValidationError
        When files are missing, format is unsupported, or payload is malformed.
    """
    root = Path(path)
    meta_path = root / "meta.json"
    plan_path = root / "kg_plan.joblib"
    if not meta_path.is_file() or not plan_path.is_file():
        raise ValidationError(
            f"Incomplete KG bundle at {root}. "
            f"Expected meta.json and kg_plan.joblib ({BUNDLE_FORMAT})."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fmt = meta.get("format")
    if fmt != BUNDLE_FORMAT:
        raise ValidationError(
            f"Unsupported KG bundle format {fmt!r}; expected {BUNDLE_FORMAT}."
        )
    loaded = joblib_load_trusted(plan_path, trusted=trusted, artifact="joblib plan")
    if isinstance(loaded, KgPlan):
        plan = loaded
    elif isinstance(loaded, dict) and "plan" in loaded:
        plan = loaded["plan"]
        if not isinstance(plan, KgPlan):
            raise ValidationError("Loaded plan object is not a KgPlan")
        if loaded.get("entity_embeddings") is not None and (
            plan.entity_embeddings_ is None or plan.entity_embeddings_.size == 0
        ):
            plan.entity_embeddings_ = np.asarray(
                loaded["entity_embeddings"], dtype=float
            )
        if loaded.get("relation_embeddings") is not None and (
            plan.relation_embeddings_ is None or plan.relation_embeddings_.size == 0
        ):
            plan.relation_embeddings_ = np.asarray(
                loaded["relation_embeddings"], dtype=float
            )
        if loaded.get("train_heads") is not None and plan.train_heads_.size == 0:
            plan.train_heads_ = np.asarray(loaded["train_heads"], dtype=np.int64)
            plan.train_relations_ = np.asarray(
                loaded["train_relations"], dtype=np.int64
            )
            plan.train_tails_ = np.asarray(loaded["train_tails"], dtype=np.int64)
        if loaded.get("entity_ids") and not plan.entity_ids:
            plan.entity_ids = tuple(loaded["entity_ids"])
        if loaded.get("relation_ids") and not plan.relation_ids:
            plan.relation_ids = tuple(loaded["relation_ids"])
        if loaded.get("true_triple_set") is not None and not plan.true_triple_set_:
            plan.true_triple_set_ = frozenset(
                tuple(t) for t in loaded["true_triple_set"]
            )
        if loaded.get("out_edges") is not None and not plan.out_edges_:
            plan.out_edges_ = {
                int(k): [(int(a), int(b)) for a, b in v]
                for k, v in loaded["out_edges"].items()
            }
        if loaded.get("in_edges") is not None and not plan.in_edges_:
            plan.in_edges_ = {
                int(k): [(int(a), int(b)) for a, b in v]
                for k, v in loaded["in_edges"].items()
            }
    else:
        raise ValidationError(
            "kg_plan.joblib must contain a KgPlan or a payload with key 'plan'."
        )

    if not plan.entity_index_ and plan.entity_ids:
        plan.entity_index_ = {e: i for i, e in enumerate(plan.entity_ids)}
    if not plan.relation_index_ and plan.relation_ids:
        plan.relation_index_ = {r: i for i, r in enumerate(plan.relation_ids)}
    if not plan.true_triple_set_ and plan.train_heads_.size:
        plan.true_triple_set_ = frozenset(
            zip(
                plan.train_heads_.tolist(),
                plan.train_relations_.tolist(),
                plan.train_tails_.tolist(),
                strict=True,
            )
        )
    if not plan.out_edges_ and plan.train_heads_.size:
        from buildml.kg.features import build_adjacency

        plan.out_edges_, plan.in_edges_ = build_adjacency(
            plan.train_heads_, plan.train_relations_, plan.train_tails_
        )
    return plan
