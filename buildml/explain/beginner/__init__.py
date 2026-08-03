"""Beginner teaching layers for every BuildML concept note.

One module per domain, mirroring :mod:`buildml.explain.concepts`. Each module
exports a ``{DOMAIN}_BEGINNER`` mapping from concept key to
:class:`~buildml.explain.beginner._builder.BeginnerLayer`. This package merges
them into a single registry that :mod:`buildml.explain.concepts` folds onto the
technical notes at import time.

The split exists so the expert prose and the beginner prose can be reviewed
independently while remaining a single artifact for readers.
"""

from __future__ import annotations

from buildml.explain.beginner._builder import BeginnerLayer
from buildml.explain.beginner.activelearning import ACTIVELEARNING_BEGINNER
from buildml.explain.beginner.ai import AI_BEGINNER
from buildml.explain.beginner.anomaly import ANOMALY_BEGINNER
from buildml.explain.beginner.automl import AUTOML_BEGINNER
from buildml.explain.beginner.causal import CAUSAL_BEGINNER
from buildml.explain.beginner.cbr import CBR_BEGINNER
from buildml.explain.beginner.classical import CLASSICAL_BEGINNER
from buildml.explain.beginner.dl import DL_BEGINNER
from buildml.explain.beginner.ensemble import ENSEMBLE_BEGINNER
from buildml.explain.beginner.federated import FEDERATED_BEGINNER
from buildml.explain.beginner.forecasting import FORECASTING_BEGINNER
from buildml.explain.beginner.graph import GRAPH_BEGINNER
from buildml.explain.beginner.kg import KG_BEGINNER
from buildml.explain.beginner.metalearning import METALEARNING_BEGINNER
from buildml.explain.beginner.multitask import MULTITASK_BEGINNER
from buildml.explain.beginner.nlp import NLP_BEGINNER
from buildml.explain.beginner.online import ONLINE_BEGINNER
from buildml.explain.beginner.optimize import OPTIMIZE_BEGINNER
from buildml.explain.beginner.probabilistic import PROBABILISTIC_BEGINNER
from buildml.explain.beginner.rag import RAG_BEGINNER
from buildml.explain.beginner.ranking import RANKING_BEGINNER
from buildml.explain.beginner.recommenders import RECOMMENDER_BEGINNER
from buildml.explain.beginner.rl import RL_BEGINNER
from buildml.explain.beginner.selfsupervised import SELFSUPERVISED_BEGINNER
from buildml.explain.beginner.semisupervised import SEMISUPERVISED_BEGINNER
from buildml.explain.beginner.symbolic import SYMBOLIC_BEGINNER
from buildml.explain.beginner.synthetic import SYNTHETIC_BEGINNER
from buildml.explain.beginner.tda import TDA_BEGINNER
from buildml.explain.beginner.teaching import TEACHING_BEGINNER
from buildml.explain.beginner.timeseries import TIMESERIES_BEGINNER
from buildml.explain.beginner.unsupervised import UNSUPERVISED_BEGINNER

_DOMAIN_LAYERS: tuple[tuple[str, dict[str, BeginnerLayer]], ...] = (
    ("activelearning", ACTIVELEARNING_BEGINNER),
    ("ai", AI_BEGINNER),
    ("anomaly", ANOMALY_BEGINNER),
    ("automl", AUTOML_BEGINNER),
    ("causal", CAUSAL_BEGINNER),
    ("cbr", CBR_BEGINNER),
    ("classical", CLASSICAL_BEGINNER),
    ("dl", DL_BEGINNER),
    ("ensemble", ENSEMBLE_BEGINNER),
    ("federated", FEDERATED_BEGINNER),
    ("forecasting", FORECASTING_BEGINNER),
    ("graph", GRAPH_BEGINNER),
    ("kg", KG_BEGINNER),
    ("metalearning", METALEARNING_BEGINNER),
    ("multitask", MULTITASK_BEGINNER),
    ("nlp", NLP_BEGINNER),
    ("online", ONLINE_BEGINNER),
    ("optimize", OPTIMIZE_BEGINNER),
    ("probabilistic", PROBABILISTIC_BEGINNER),
    ("rag", RAG_BEGINNER),
    ("ranking", RANKING_BEGINNER),
    ("recommenders", RECOMMENDER_BEGINNER),
    ("rl", RL_BEGINNER),
    ("selfsupervised", SELFSUPERVISED_BEGINNER),
    ("semisupervised", SEMISUPERVISED_BEGINNER),
    ("symbolic", SYMBOLIC_BEGINNER),
    ("synthetic", SYNTHETIC_BEGINNER),
    ("tda", TDA_BEGINNER),
    ("teaching", TEACHING_BEGINNER),
    ("timeseries", TIMESERIES_BEGINNER),
    ("unsupervised", UNSUPERVISED_BEGINNER),
)


def _merge() -> dict[str, BeginnerLayer]:
    merged: dict[str, BeginnerLayer] = {}
    origin: dict[str, str] = {}
    for domain, layers in _DOMAIN_LAYERS:
        for key, layer in layers.items():
            if key in merged:
                raise ValueError(
                    f"Beginner layer {key!r} defined in both {origin[key]!r} and {domain!r}"
                )
            merged[key] = layer
            origin[key] = domain
    return merged


BEGINNER_LAYERS: dict[str, BeginnerLayer] = _merge()
"""Concept key → beginner layer, across every BuildML domain."""

DOMAIN_OF_CONCEPT: dict[str, str] = {
    key: domain for domain, layers in _DOMAIN_LAYERS for key in layers
}
"""Concept key → the beginner domain module that authored its layer."""


def layer_for(key: str) -> BeginnerLayer | None:
    """Fetch the beginner material authored for one concept.

    Callers normally read the merged :data:`~buildml.explain.CONCEPT_NOTES`
    instead, where this material is already folded into the note. Use this when
    you need the beginner layer on its own — checking coverage, for instance, or
    building tooling over the authored source.

    Parameters
    ----------
    key:
        A concept key, such as ``'leakage-boundary'``.

    Returns
    -------
    BeginnerLayer or None
        The layer, or ``None`` when no beginner material exists for that key.
    """
    return BEGINNER_LAYERS.get(key)


__all__ = [
    "BEGINNER_LAYERS",
    "DOMAIN_OF_CONCEPT",
    "BeginnerLayer",
    "layer_for",
]
