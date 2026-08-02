"""Human teaching overlays for Session operations, split by domain."""

from __future__ import annotations

from buildml.explain.overlays.activelearning import _OPERATIONS as _ACTIVELEARNING
from buildml.explain.overlays.ai import _OPERATIONS as _AI
from buildml.explain.overlays.anomaly import _OPERATIONS as _ANOMALY
from buildml.explain.overlays.automl import _OPERATIONS as _AUTOML
from buildml.explain.overlays.classical import _OPERATIONS as _CLASSICAL
from buildml.explain.overlays.dl import _OPERATIONS as _DL
from buildml.explain.overlays.ensemble import _OPERATIONS as _ENSEMBLE
from buildml.explain.overlays.forecasting import _OPERATIONS as _FORECASTING
from buildml.explain.overlays.timeseries import _OPERATIONS as _TIMESERIES
from buildml.explain.overlays.federated import _OPERATIONS as _FEDERATED
from buildml.explain.overlays.metalearning import _OPERATIONS as _METALEARNING
from buildml.explain.overlays.multitask import _OPERATIONS as _MULTITASK
from buildml.explain.overlays.online import _OPERATIONS as _ONLINE
from buildml.explain.overlays.probabilistic import _OPERATIONS as _PROBABILISTIC
from buildml.explain.overlays.causal import _OPERATIONS as _CAUSAL
from buildml.explain.overlays.graph import _OPERATIONS as _GRAPH
from buildml.explain.overlays.symbolic import _OPERATIONS as _SYMBOLIC
from buildml.explain.overlays.cbr import _OPERATIONS as _CBR
from buildml.explain.overlays.rl import _OPERATIONS as _RL
from buildml.explain.overlays.tda import _OPERATIONS as _TDA
from buildml.explain.overlays.recommenders import _OPERATIONS as _RECOMMENDERS
from buildml.explain.overlays.ranking import _OPERATIONS as _RANKING
from buildml.explain.overlays.kg import _OPERATIONS as _KG
from buildml.explain.overlays.optimize import _OPERATIONS as _OPTIMIZE
from buildml.explain.overlays.synthetic import _OPERATIONS as _SYNTHETIC
from buildml.explain.overlays.rag import _OPERATIONS as _RAG
from buildml.explain.overlays.selfsupervised import _OPERATIONS as _SELFSUPERVISED
from buildml.explain.overlays.semisupervised import _OPERATIONS as _SEMISUPERVISED
from buildml.explain.overlays.unsupervised import _OPERATIONS as _UNSUPERVISED
from buildml.explain.overlays.workflow import _OPERATIONS as _WORKFLOW
from buildml.explain.schemas import OperationSpec

_OPERATIONS: tuple[OperationSpec, ...] = (
    *_CLASSICAL,
    *_DL,
    *_RAG,
    *_UNSUPERVISED,
    *_ENSEMBLE,
    *_AUTOML,
    *_FORECASTING,
    *_TIMESERIES,
    *_ANOMALY,
    *_SEMISUPERVISED,
    *_SELFSUPERVISED,
    *_ACTIVELEARNING,
    *_ONLINE,
    *_MULTITASK,
    *_METALEARNING,
    *_FEDERATED,
    *_PROBABILISTIC,
    *_CAUSAL,
    *_GRAPH,
    *_SYMBOLIC,
    *_CBR,
    *_RL,
    *_TDA,
    *_RECOMMENDERS,
    *_RANKING,
    *_KG,
    *_OPTIMIZE,
    *_SYNTHETIC,
    *_AI,
    *_WORKFLOW,
)

__all__ = ["_OPERATIONS"]
