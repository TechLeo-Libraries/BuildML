"""Session domain mixins composing the public Session class."""

from buildml.session.mixins.activelearning import ActivelearningSessionMixin
from buildml.session.mixins.ai import AiSessionMixin
from buildml.session.mixins.anomaly import AnomalySessionMixin
from buildml.session.mixins.automl import AutomlSessionMixin
from buildml.session.mixins.causal import CausalSessionMixin
from buildml.session.mixins.cbr import CbrSessionMixin
from buildml.session.mixins.classical import ClassicalSessionMixin
from buildml.session.mixins.data import DataSessionMixin
from buildml.session.mixins.decision import DecisionSessionMixin
from buildml.session.mixins.dl import DlSessionMixin
from buildml.session.mixins.eda import EdaSessionMixin
from buildml.session.mixins.ensemble import EnsembleSessionMixin
from buildml.session.mixins.federated import FederatedSessionMixin
from buildml.session.mixins.forecast import ForecastSessionMixin
from buildml.session.mixins.graph import GraphSessionMixin
from buildml.session.mixins.kg import KgSessionMixin
from buildml.session.mixins.metalearning import MetalearningSessionMixin
from buildml.session.mixins.multitask import MultitaskSessionMixin
from buildml.session.mixins.nlp import NlpSessionMixin
from buildml.session.mixins.online import OnlineSessionMixin
from buildml.session.mixins.preprocess import PreprocessSessionMixin
from buildml.session.mixins.probabilistic import ProbabilisticSessionMixin
from buildml.session.mixins.rag import RagSessionMixin
from buildml.session.mixins.ranking import RankingSessionMixin
from buildml.session.mixins.recommender import RecommenderSessionMixin
from buildml.session.mixins.rl import RlSessionMixin
from buildml.session.mixins.selfsupervised import SelfsupervisedSessionMixin
from buildml.session.mixins.semisupervised import SemisupervisedSessionMixin
from buildml.session.mixins.symbolic import SymbolicSessionMixin
from buildml.session.mixins.synthetic import SyntheticSessionMixin
from buildml.session.mixins.tda import TdaSessionMixin
from buildml.session.mixins.timeseries import TimeseriesSessionMixin
from buildml.session.mixins.unsupervised import UnsupervisedSessionMixin
from buildml.session.mixins.workflow import WorkflowSessionMixin

__all__ = [
    "ActivelearningSessionMixin",
    "AiSessionMixin",
    "AnomalySessionMixin",
    "AutomlSessionMixin",
    "CausalSessionMixin",
    "CbrSessionMixin",
    "ClassicalSessionMixin",
    "DataSessionMixin",
    "DecisionSessionMixin",
    "DlSessionMixin",
    "EdaSessionMixin",
    "EnsembleSessionMixin",
    "FederatedSessionMixin",
    "ForecastSessionMixin",
    "GraphSessionMixin",
    "KgSessionMixin",
    "MetalearningSessionMixin",
    "MultitaskSessionMixin",
    "NlpSessionMixin",
    "OnlineSessionMixin",
    "PreprocessSessionMixin",
    "ProbabilisticSessionMixin",
    "RagSessionMixin",
    "RankingSessionMixin",
    "RecommenderSessionMixin",
    "RlSessionMixin",
    "SelfsupervisedSessionMixin",
    "SemisupervisedSessionMixin",
    "SymbolicSessionMixin",
    "SyntheticSessionMixin",
    "TdaSessionMixin",
    "TimeseriesSessionMixin",
    "UnsupervisedSessionMixin",
    "WorkflowSessionMixin",
]
