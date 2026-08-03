"""Shared imports for Session domain mixins."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from buildml.checkpoint.validate import ReattachResult
from buildml.core.errors import ValidationError
from buildml.core.results import IngestReport
from buildml.core.types import ColumnRole, DataMode, EngineName
from buildml.data.dataset import Dataset
from buildml.data.engines.prep import MaterializePrepResult
from buildml.data.splits import PartitionName, SplitPlan
from buildml.eda.report import EDAReport
from buildml.explain.history import normalize_history
from buildml.explain.schemas import WorkflowStep
from buildml.model.compare import ModelComparison
from buildml.model.diagnostics import DiagnosticReport
from buildml.model.plot_boards import PlotBoardReport
from buildml.model.selection import CVScoreResult, NestedCVResult, SearchResult
from buildml.model.supervised import EvaluateResult, FitResult
from buildml.pipeline.card import ModelCard
from buildml.pipeline.score import PipelinePredictResult
from buildml.preprocess.apply import ApplyPlansResult
from buildml.preprocess.binning import BinningPlan
from buildml.preprocess.custom import CustomTransformPlan, CustomTransformSpec
from buildml.preprocess.dates import DateFeaturePlan
from buildml.preprocess.encode import EncodePlan
from buildml.preprocess.fold import PreprocessRecipe
from buildml.preprocess.imbalance import ResamplePlan
from buildml.preprocess.impute import SimpleImputePlan
from buildml.preprocess.outliers import OutlierPlan
from buildml.preprocess.reduce import ReducePlan
from buildml.preprocess.result import PreprocessResult
from buildml.preprocess.scale import ScalePlan
from buildml.preprocess.select import FeatureSelectPlan
from buildml.preprocess.text import TextFeaturePlan
from buildml.session.audit import DryRunReport, HistorySummary
from buildml.session.walkthrough import WorkflowWalkthroughReport

if TYPE_CHECKING:
    from buildml.activelearning.results import (
        ActiveLearningEvalResult,
        ActiveLearningFitResult,
        ActiveLearningLabelResult,
        ActiveLearningPlan,
        ActiveLearningQueryResult,
    )
    from buildml.activelearning.types import (
        ActiveLearningBackend,
        ActiveLearningEstimator,
        ActiveLearningStrategy,
    )
    from buildml.ai.advisor import AdvisorResult
    from buildml.ai.executor import ExecutorProposal, ExecutorResult
    from buildml.ai.planner import BudgetTracker, PlanExecutionResult
    from buildml.ai.privacy import EgressConfig, EgressManifest
    from buildml.ai.provider import ProviderConfig, ProviderProtocol
    from buildml.ai.results import PlanResult
    from buildml.ai.tools import ToolRegistry
    from buildml.ai.transcript import TranscriptStore
    from buildml.anomaly.results import (
        AnomalyEvalResult,
        AnomalyFitResult,
        AnomalyPlan,
        AnomalyScoreResult,
        AnomalyThresholdTuneResult,
    )
    from buildml.anomaly.types import (
        AnomalyBackend,
        AnomalyMethod,
        AnomalyMode,
        ThresholdPolicy,
        ThresholdTuningMetric,
    )
    from buildml.automl.results import AutoMLPlan, AutoMLResult
    from buildml.automl.types import (
        AutoMLBackend,
        AutoMLBudget,
        AutoMLMethod,
        AutoMLSelection,
        EnsembleMode,
    )
    from buildml.causal.results import (
        CausalEstimateResult,
        CausalEvalResult,
        CausalFitResult,
        CausalPlan,
        CausalRefuteResult,
    )
    from buildml.causal.types import (
        CausalAssumptions,
        CausalBackend,
        CausalMethod,
        CausalRefuteKind,
    )
    from buildml.cbr.results import (
        CbrEvalResult,
        CbrFitResult,
        CbrPlan,
        CbrPredictResult,
        CbrRetainResult,
        CbrRetrieveResult,
    )
    from buildml.cbr.types import (
        CbrAdaptMode,
        CbrMetric,
        CbrReuseMode,
        CbrTask,
    )
    from buildml.dashboard.launch import EDAAppHandle
    from buildml.dl.cv import TorchCVResult
    from buildml.dl.results import (
        DLEvaluateResult,
        TorchLoaderBundle,
        TrainingCurveReport,
        TrainResult,
    )
    from buildml.dl.types import TrainConfig
    from buildml.ensemble.results import EnsembleFitResult, EnsemblePlan
    from buildml.ensemble.types import BlendMethod, VotingMethod
    from buildml.federated.results import (
        FederatedEvalResult,
        FederatedFitResult,
        FederatedPlan,
        FederatedPredictResult,
    )
    from buildml.federated.types import (
        FederatedBackend,
        FederatedEstimator,
        FederatedMethod,
        FederatedTask,
    )
    from buildml.forecasting.results import (
        ForecastEvalResult,
        ForecastFitResult,
        ForecastGenerateResult,
        ForecastPlan,
    )
    from buildml.forecasting.types import ForecastEvalStrategy, ForecastMethod
    from buildml.graph.results import (
        GraphEvalResult,
        GraphFitResult,
        GraphPlan,
        GraphPredictResult,
    )
    from buildml.graph.types import (
        ClassicalEstimator,
        GraphMethod,
        GraphMode,
        GraphSpec,
        GraphTask,
        PyGModel,
    )
    from buildml.kg.results import (
        KgEvalResult,
        KgFitResult,
        KgPlan,
        KgQueryResult,
        PredictLinksResult,
        ScoreTriplesResult,
    )
    from buildml.kg.types import (
        KgBackend,
        KgMethod,
        KgNorm,
        KgQueryMode,
        LinkPredictionMode,
    )
    from buildml.metalearning.results import (
        MetaAdaptResult,
        MetaLearningEvalResult,
        MetaLearningFitResult,
        MetaLearningPlan,
    )
    from buildml.metalearning.types import (
        MetaLearningBaseEstimator,
        MetaLearningMethod,
    )
    from buildml.multitask.results import (
        MultiTaskEvalResult,
        MultiTaskFitResult,
        MultiTaskPlan,
        MultiTaskPredictResult,
    )
    from buildml.multitask.types import (
        MultiTaskBackend,
        MultiTaskBaseEstimator,
        MultiTaskMethod,
        MultiTaskTask,
    )
    from buildml.nlp.results import (
        NlpCorpusProfile,
        NlpEntityResult,
        NlpEvalResult,
        NlpFitResult,
        NlpInterpretResult,
        NlpKeyphraseResult,
        NlpLanguageResult,
        NlpPredictResult,
        NlpSentimentResult,
        NlpSummaryResult,
        NlpTextPlan,
        NlpTopicAssignResult,
        NlpTopicPlan,
        NlpTopicResult,
    )
    from buildml.online.results import (
        OnlineEvalResult,
        OnlineFitResult,
        OnlinePlan,
        OnlinePredictResult,
        OnlineUpdateResult,
    )
    from buildml.online.types import (
        OnlineBackend,
        OnlineDriftDetector,
        OnlineEstimator,
        OnlineTask,
    )
    from buildml.optimize.results import (
        ApplyDecisionsResult,
        DecisionEvalResult,
        DecisionFitResult,
        DecisionPlan,
    )
    from buildml.optimize.types import (
        AllocationObjective,
        DecisionMethod,
        KnapsackSolver,
        ScoreSource,
        TuningPartition,
    )
    from buildml.probabilistic.results import (
        ProbabilisticEvalResult,
        ProbabilisticFitResult,
        ProbabilisticIntervalResult,
        ProbabilisticPlan,
        ProbabilisticPredictResult,
    )
    from buildml.probabilistic.types import (
        IntervalMethod,
        ProbabilisticEstimator,
        ProbabilisticTask,
    )
    from buildml.rag.generate import ChatProvider as RagChatProvider
    from buildml.rag.results import GenerateResult, IndexResult, RagEvalResult, RetrieveResult
    from buildml.rag.types import GenerateConfig, RetrieveConfig
    from buildml.ranking.results import (
        RankerEvalResult,
        RankerFitResult,
        RankerPlan,
        RankResult,
    )
    from buildml.ranking.types import (
        PairwiseEstimator,
        PointwiseEstimator,
        RankerBackend,
        RankerMethod,
    )
    from buildml.recommenders.results import (
        RecommenderEvalResult,
        RecommenderFitResult,
        RecommenderPlan,
        RecommendResult,
    )
    from buildml.recommenders.types import (
        ColdStartPolicy,
        FeedbackMode,
        RecommenderBackend,
        RecommenderMethod,
    )
    from buildml.rl.results import (
        ImitationEvalResult,
        ImitationFitResult,
        ImitationPlan,
        ImitationPredictResult,
        RlActResult,
        RlEvalResult,
        RlFitResult,
        RlPlan,
    )
    from buildml.rl.types import (
        BanditAlgorithm,
        ImitationEstimator,
        ImitationTask,
        RlMode,
    )
    from buildml.selfsupervised.results import (
        SelfSupervisedEvalResult,
        SelfSupervisedFitResult,
        SelfSupervisedPlan,
        SelfSupervisedTransformResult,
        SSLHeadFitResult,
        SSLHeadPlan,
    )
    from buildml.selfsupervised.types import SelfSupervisedMethod, SSLHeadEstimator
    from buildml.semisupervised.results import (
        SemiSupervisedEvalResult,
        SemiSupervisedFitResult,
        SemiSupervisedPlan,
        SemiSupervisedPredictResult,
    )
    from buildml.semisupervised.types import SemiSupervisedBackend, SemiSupervisedMethod
    from buildml.session.session import Session  # noqa: F401
    from buildml.symbolic.results import (
        NeuroSymbolicFitResult,
        NeuroSymbolicPlan,
        SymbolicEvalResult,
        SymbolicFitResult,
        SymbolicPlan,
        SymbolicPredictResult,
    )
    from buildml.symbolic.rules import Rule
    from buildml.symbolic.types import (
        BaseEstimatorName,
        IndustrySymbolicMethod,
        NeuroSymbolicBackend,
        NeuroSymbolicMode,
        SymbolicBackend,
        SymbolicSource,
        SymbolicTask,
    )
    from buildml.synthetic.results import (
        SynthesizerFitResult,
        SynthesizerPlan,
        SyntheticEvalResult,
        SyntheticSampleResult,
    )
    from buildml.synthetic.types import (
        EvalBackend,
        EvalMode,
        MergeMode,
        SynthesizerMethod,
        SyntheticBackend,
    )
    from buildml.tda.results import (
        TdaEvalResult,
        TdaFitResult,
        TdaPlan,
        TdaPredictResult,
        TdaTransformResult,
    )
    from buildml.tda.types import (
        DiagramDistanceMetric,
        SubsampleStrategy,
        TdaBackend,
        TdaHead,
        TdaTask,
        Vectorization,
    )
    from buildml.unsupervised.results import (
        ClusterAssignResult,
        ClusterEvalResult,
        ClusterFitResult,
        ClusterPlan,
    )
    from buildml.unsupervised.types import ClusterMethod

# Runtime placeholder so ``from ._shared import *`` always binds Session.
Session = Any  # type: ignore[misc,assignment]
