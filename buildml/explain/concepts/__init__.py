# ruff: noqa: E501
"""Shared concept notes referenced by the operation catalog and Concept Academy.

Notes are split by domain modules (classical / dl / rag / ai) and merged here.
"""

from __future__ import annotations

from buildml.explain.concepts.activelearning import ACTIVELEARNING_NOTES
from buildml.explain.concepts.ai import AI_NOTES
from buildml.explain.concepts.anomaly import ANOMALY_NOTES
from buildml.explain.concepts.automl import AUTOML_NOTES
from buildml.explain.concepts.classical import CLASSICAL_NOTES
from buildml.explain.concepts.dl import DL_NOTES
from buildml.explain.concepts.ensemble import ENSEMBLE_NOTES
from buildml.explain.concepts.forecasting import FORECASTING_NOTES
from buildml.explain.concepts.federated import FEDERATED_NOTES
from buildml.explain.concepts.metalearning import METALEARNING_NOTES
from buildml.explain.concepts.multitask import MULTITASK_NOTES
from buildml.explain.concepts.online import ONLINE_NOTES
from buildml.explain.concepts.probabilistic import PROBABILISTIC_NOTES
from buildml.explain.concepts.causal import CAUSAL_NOTES
from buildml.explain.concepts.graph import GRAPH_NOTES
from buildml.explain.concepts.symbolic import SYMBOLIC_NOTES
from buildml.explain.concepts.cbr import CBR_NOTES
from buildml.explain.concepts.rl import RL_NOTES
from buildml.explain.concepts.tda import TDA_NOTES
from buildml.explain.concepts.recommenders import RECOMMENDER_NOTES
from buildml.explain.concepts.ranking import RANKING_NOTES
from buildml.explain.concepts.kg import KG_NOTES
from buildml.explain.concepts.optimize import OPTIMIZE_NOTES
from buildml.explain.concepts.synthetic import SYNTHETIC_NOTES
from buildml.explain.concepts.rag import RAG_NOTES
from buildml.explain.concepts.selfsupervised import SELFSUPERVISED_NOTES
from buildml.explain.concepts.semisupervised import SEMISUPERVISED_NOTES
from buildml.explain.concepts.unsupervised import UNSUPERVISED_NOTES
from buildml.explain.schemas import ConceptNote

CONCEPT_NOTES: dict[str, ConceptNote] = {
    **CLASSICAL_NOTES,
    **DL_NOTES,
    **RAG_NOTES,
    **AI_NOTES,
    **UNSUPERVISED_NOTES,
    **ENSEMBLE_NOTES,
    **AUTOML_NOTES,
    **FORECASTING_NOTES,
    **ANOMALY_NOTES,
    **SEMISUPERVISED_NOTES,
    **SELFSUPERVISED_NOTES,
    **ACTIVELEARNING_NOTES,
    **ONLINE_NOTES,
    **MULTITASK_NOTES,
    **METALEARNING_NOTES,
    **FEDERATED_NOTES,
    **PROBABILISTIC_NOTES,
    **CAUSAL_NOTES,
    **GRAPH_NOTES,
    **SYMBOLIC_NOTES,
    **CBR_NOTES,
    **RL_NOTES,
    **TDA_NOTES,
    **RECOMMENDER_NOTES,
    **RANKING_NOTES,
    **KG_NOTES,
    **OPTIMIZE_NOTES,
    **SYNTHETIC_NOTES,
}


def get_concept(key: str) -> ConceptNote:
    """Return a concept note or raise a precise catalog error."""
    try:
        return CONCEPT_NOTES[key]
    except KeyError as exc:
        raise KeyError(f"Unknown BuildML concept: {key}") from exc


def list_concepts() -> tuple[ConceptNote, ...]:
    """Return concept notes in stable key order."""
    return tuple(CONCEPT_NOTES[key] for key in sorted(CONCEPT_NOTES))
