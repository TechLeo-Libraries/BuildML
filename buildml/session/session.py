"""BuildML Session — thin OOP facade that delegates to domain ops."""

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

from . import (
    ai_ops,
    anomaly_ops,
    automl_ops,
    classical_ops,
    data_ops,
    dl_ops,
    eda_ops,
    ensemble_ops,
    forecast_ops,
    preprocess_ops,
    rag_ops,
    activelearning_ops,
    federated_ops,
    metalearning_ops,
    multitask_ops,
    online_ops,
    probabilistic_ops,
    causal_ops,
    graph_ops,
    symbolic_ops,
    cbr_ops,
    rl_ops,
    tda_ops,
    recommender_ops,
    ranking_ops,
    kg_ops,
    decision_ops,
    synthetic_ops,
    selfsupervised_ops,
    semisupervised_ops,
    timeseries_ops,
    state,
    unsupervised_ops,
    workflow_ops,
)
from .audit import DryRunReport, HistorySummary
from .walkthrough import WorkflowWalkthroughReport

if TYPE_CHECKING:
    from buildml.ai.advisor import AdvisorResult
    from buildml.ai.executor import ExecutorProposal, ExecutorResult
    from buildml.ai.planner import BudgetTracker, PlanExecutionResult
    from buildml.ai.privacy import EgressConfig, EgressManifest
    from buildml.ai.provider import ProviderConfig, ProviderProtocol
    from buildml.ai.results import PlanResult
    from buildml.ai.tools import ToolRegistry
    from buildml.ai.transcript import TranscriptStore
    from buildml.dashboard.launch import EDAAppHandle
    from buildml.dl.cv import TorchCVResult
    from buildml.dl.results import (
        DLEvaluateResult,
        TorchLoaderBundle,
        TrainingCurveReport,
        TrainResult,
    )
    from buildml.dl.types import TrainConfig
    from buildml.rag.generate import ChatProvider as RagChatProvider
    from buildml.rag.results import GenerateResult, IndexResult, RagEvalResult, RetrieveResult
    from buildml.rag.types import GenerateConfig, RetrieveConfig
    from buildml.automl.results import AutoMLPlan, AutoMLResult
    from buildml.automl.types import (
        AutoMLBackend,
        AutoMLBudget,
        AutoMLMethod,
        AutoMLSelection,
        EnsembleMode,
    )
    from buildml.ensemble.results import EnsembleFitResult, EnsemblePlan
    from buildml.ensemble.types import BlendMethod, VotingMethod
    from buildml.forecasting.results import (
        ForecastEvalResult,
        ForecastFitResult,
        ForecastGenerateResult,
        ForecastPlan,
    )
    from buildml.forecasting.types import ForecastEvalStrategy, ForecastMethod
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
    from buildml.semisupervised.results import (
        SemiSupervisedEvalResult,
        SemiSupervisedFitResult,
        SemiSupervisedPlan,
        SemiSupervisedPredictResult,
    )
    from buildml.semisupervised.types import SemiSupervisedBackend, SemiSupervisedMethod
    from buildml.selfsupervised.results import (
        SSLHeadFitResult,
        SSLHeadPlan,
        SelfSupervisedEvalResult,
        SelfSupervisedFitResult,
        SelfSupervisedPlan,
        SelfSupervisedTransformResult,
    )
    from buildml.selfsupervised.types import SelfSupervisedMethod, SSLHeadEstimator
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
    from buildml.recommenders.results import (
        RecommendResult,
        RecommenderEvalResult,
        RecommenderFitResult,
        RecommenderPlan,
    )
    from buildml.recommenders.types import (
        ColdStartPolicy,
        FeedbackMode,
        RecommenderBackend,
        RecommenderMethod,
    )
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
    from buildml.synthetic.results import (
        SyntheticEvalResult,
        SyntheticSampleResult,
        SynthesizerFitResult,
        SynthesizerPlan,
    )
    from buildml.synthetic.types import (
        EvalBackend,
        EvalMode,
        MergeMode,
        SyntheticBackend,
        SynthesizerMethod,
    )
    from buildml.unsupervised.results import (
        ClusterAssignResult,
        ClusterEvalResult,
        ClusterFitResult,
        ClusterPlan,
    )
    from buildml.unsupervised.types import ClusterMethod


class Session:
    """Primary user-facing object for BuildML 2.x workflows.

    A session owns ingested data, roles, splits, history, and checkpoint
    reattach state. Methods delegate to domain packages / session ops and do
    not reimplement transform or trainer logic.

    Examples
    --------
    >>> import pandas as pd
    >>> from buildml import Session
    >>> frame = pd.DataFrame({"a": [1, 2, 3, 4], "y": [0, 1, 0, 1]})
    >>> session = Session.ingest(frame)
    >>> session.set_roles({"a": "feature", "y": "target"})
    >>> session.split(test_size=0.25, stratify=True)
    >>> session.partition("train").shape[0] > 0
    True

    Notes
    -----
    ``with session:`` calls :meth:`close_native` on exit so owned DuckDB
    connections on the Session dataset are released safely.
    """

    def __init__(
        self,
        dataset: Dataset | None = None,
        ingest_report: IngestReport | None = None,
        split_plan: SplitPlan | None = None,
        history: list[dict[str, Any]] | None = None,
        reattach_result: ReattachResult | None = None,
    ) -> None:
        self._dataset = dataset
        self._ingest_report = ingest_report
        self._split_plan = split_plan
        self._history: list[dict[str, Any]] = normalize_history(history)
        self._reattach_result = reattach_result
        self._impute_plan: SimpleImputePlan | None = None
        self._encode_plan: EncodePlan | None = None
        self._scale_plan: ScalePlan | None = None
        self._outlier_plan: OutlierPlan | None = None
        self._binning_plan: BinningPlan | None = None
        self._feature_select_plan: FeatureSelectPlan | None = None
        self._text_plan: TextFeaturePlan | None = None
        self._reduce_plan: ReducePlan | None = None
        self._custom_plan: CustomTransformPlan | None = None
        self._last_preprocess: PreprocessResult | None = None
        self._fit_result: FitResult | None = None
        self._date_plan: DateFeaturePlan | None = None
        self._last_comparison: ModelComparison | None = None
        self._resample_plan: ResamplePlan | None = None
        self._last_diagnostic: DiagnosticReport | None = None
        self._last_plot_board: PlotBoardReport | None = None
        self._last_walkthrough: WorkflowWalkthroughReport | None = None
        self._last_dry_run: DryRunReport | None = None
        self._last_history_summary: HistorySummary | None = None
        self._last_eda: EDAReport | None = None
        self._eda_app_handle: EDAAppHandle | None = None
        self._last_cv: CVScoreResult | None = None
        self._last_nested_cv: NestedCVResult | None = None
        self._last_search: SearchResult | None = None
        self._model_card: ModelCard | None = None
        self._torch_loaders: TorchLoaderBundle | None = None
        self._dl_train_result: TrainResult | None = None
        self._dl_cv_result: TorchCVResult | None = None
        self._dl_search_result: Any | None = None
        self._dl_nested_cv_result: Any | None = None
        self._dl_export_result: Any | None = None
        self._dl_ddp_result: Any | None = None
        self._dl_speech_result: Any | None = None
        self._dl_backbone: Any | None = None
        self._dl_backbone_head: Any | None = None
        self._dl_asr_eval: Any | None = None
        self._dl_packaging_result: Any | None = None
        self._dl_k8s_result: Any | None = None
        self._serve_handle: Any | None = None
        self._last_pipeline_path: Path | None = None
        self._ai_autonomy_result: Any | None = None
        self._rag_corpus: Any | None = None
        self._rag_chunks: Any | None = None
        self._rag_index: Any | None = None
        self._rag_index_result: IndexResult | None = None
        self._rag_retrieve_result: RetrieveResult | None = None
        self._rag_eval_result: RagEvalResult | None = None
        self._rag_generate_result: GenerateResult | None = None
        self._cluster_plan: ClusterPlan | None = None
        self._cluster_fit_result: ClusterFitResult | None = None
        self._cluster_assign_result: ClusterAssignResult | None = None
        self._cluster_eval_result: ClusterEvalResult | None = None
        self._ensemble_plan: EnsemblePlan | None = None
        self._ensemble_fit_result: EnsembleFitResult | None = None
        self._automl_plan: AutoMLPlan | None = None
        self._automl_result: AutoMLResult | None = None
        self._forecast_plan: ForecastPlan | None = None
        self._forecast_fit_result: ForecastFitResult | None = None
        self._forecast_generate_result: ForecastGenerateResult | None = None
        self._forecast_eval_result: ForecastEvalResult | None = None
        self._ts_analysis_result: Any | None = None
        self._anomaly_plan: AnomalyPlan | None = None
        self._anomaly_fit_result: AnomalyFitResult | None = None
        self._anomaly_score_result: AnomalyScoreResult | None = None
        self._anomaly_eval_result: AnomalyEvalResult | None = None
        self._semisupervised_plan: SemiSupervisedPlan | None = None
        self._semisupervised_fit_result: SemiSupervisedFitResult | None = None
        self._semisupervised_predict_result: SemiSupervisedPredictResult | None = None
        self._semisupervised_eval_result: SemiSupervisedEvalResult | None = None
        self._ssl_plan: SelfSupervisedPlan | None = None
        self._ssl_fit_result: SelfSupervisedFitResult | None = None
        self._ssl_transform_result: SelfSupervisedTransformResult | None = None
        self._ssl_head_plan: SSLHeadPlan | None = None
        self._ssl_head_fit_result: SSLHeadFitResult | None = None
        self._ssl_eval_result: SelfSupervisedEvalResult | None = None
        self._activelearning_plan: ActiveLearningPlan | None = None
        self._activelearning_fit_result: ActiveLearningFitResult | None = None
        self._activelearning_query_result: ActiveLearningQueryResult | None = None
        self._activelearning_label_result: ActiveLearningLabelResult | None = None
        self._activelearning_eval_result: ActiveLearningEvalResult | None = None
        self._online_plan: OnlinePlan | None = None
        self._online_fit_result: OnlineFitResult | None = None
        self._online_update_result: OnlineUpdateResult | None = None
        self._online_eval_result: OnlineEvalResult | None = None
        self._online_predict_result: OnlinePredictResult | None = None
        self._multitask_plan: MultiTaskPlan | None = None
        self._multitask_fit_result: MultiTaskFitResult | None = None
        self._multitask_predict_result: MultiTaskPredictResult | None = None
        self._multitask_eval_result: MultiTaskEvalResult | None = None
        self._metalearning_plan: MetaLearningPlan | None = None
        self._metalearning_fit_result: MetaLearningFitResult | None = None
        self._metalearning_adapt_result: MetaAdaptResult | None = None
        self._metalearning_eval_result: MetaLearningEvalResult | None = None
        self._federated_plan: FederatedPlan | None = None
        self._federated_fit_result: FederatedFitResult | None = None
        self._federated_eval_result: FederatedEvalResult | None = None
        self._federated_predict_result: FederatedPredictResult | None = None
        self._probabilistic_plan: ProbabilisticPlan | None = None
        self._probabilistic_fit_result: ProbabilisticFitResult | None = None
        self._probabilistic_eval_result: ProbabilisticEvalResult | None = None
        self._probabilistic_predict_result: ProbabilisticPredictResult | None = None
        self._probabilistic_interval_result: ProbabilisticIntervalResult | None = None
        self._causal_assumptions: CausalAssumptions | None = None
        self._causal_plan: CausalPlan | None = None
        self._causal_fit_result: CausalFitResult | None = None
        self._causal_estimate_result: CausalEstimateResult | None = None
        self._causal_eval_result: CausalEvalResult | None = None
        self._causal_refute_result: CausalRefuteResult | None = None
        self._graph_spec: GraphSpec | None = None
        self._graph_plan: GraphPlan | None = None
        self._graph_fit_result: GraphFitResult | None = None
        self._graph_predict_result: GraphPredictResult | None = None
        self._graph_eval_result: GraphEvalResult | None = None
        self._symbolic_plan: SymbolicPlan | None = None
        self._symbolic_fit_result: SymbolicFitResult | None = None
        self._symbolic_eval_result: SymbolicEvalResult | None = None
        self._symbolic_predict_result: SymbolicPredictResult | None = None
        self._neuro_symbolic_plan: NeuroSymbolicPlan | None = None
        self._neuro_symbolic_fit_result: NeuroSymbolicFitResult | None = None
        self._neuro_symbolic_predict_result: SymbolicPredictResult | None = None
        self._cbr_plan: CbrPlan | None = None
        self._cbr_fit_result: CbrFitResult | None = None
        self._cbr_eval_result: CbrEvalResult | None = None
        self._cbr_predict_result: CbrPredictResult | None = None
        self._cbr_retrieve_result: CbrRetrieveResult | None = None
        self._cbr_retain_result: CbrRetainResult | None = None
        self._imitation_plan: ImitationPlan | None = None
        self._imitation_fit_result: ImitationFitResult | None = None
        self._imitation_eval_result: ImitationEvalResult | None = None
        self._imitation_predict_result: ImitationPredictResult | None = None
        self._rl_plan: RlPlan | None = None
        self._rl_fit_result: RlFitResult | None = None
        self._rl_eval_result: RlEvalResult | None = None
        self._rl_act_result: RlActResult | None = None
        self._tda_plan: TdaPlan | None = None
        self._tda_fit_result: TdaFitResult | None = None
        self._tda_eval_result: TdaEvalResult | None = None
        self._tda_transform_result: TdaTransformResult | None = None
        self._tda_predict_result: TdaPredictResult | None = None
        self._recommender_plan: RecommenderPlan | None = None
        self._recommender_fit_result: RecommenderFitResult | None = None
        self._recommender_eval_result: RecommenderEvalResult | None = None
        self._recommender_recommend_result: RecommendResult | None = None
        self._ranker_plan: RankerPlan | None = None
        self._ranker_fit_result: RankerFitResult | None = None
        self._ranker_eval_result: RankerEvalResult | None = None
        self._ranker_rank_result: RankResult | None = None
        self._kg_plan: KgPlan | None = None
        self._kg_fit_result: KgFitResult | None = None
        self._kg_eval_result: KgEvalResult | None = None
        self._kg_score_result: ScoreTriplesResult | None = None
        self._kg_predict_result: PredictLinksResult | None = None
        self._kg_query_result: KgQueryResult | None = None
        self._decision_plan: DecisionPlan | None = None
        self._decision_fit_result: DecisionFitResult | None = None
        self._decision_eval_result: DecisionEvalResult | None = None
        self._decision_apply_result: ApplyDecisionsResult | None = None
        self._synthesizer_plan: SynthesizerPlan | None = None
        self._synthetic_fit_result: SynthesizerFitResult | None = None
        self._synthetic_eval_result: SyntheticEvalResult | None = None
        self._synthetic_sample_result: SyntheticSampleResult | None = None
        self._ai_provider: ProviderProtocol | ProviderConfig | None = None
        self._ai_egress_config: EgressConfig | None = None
        self._ai_transcript: TranscriptStore | None = None
        self._ai_result: Any | None = None
        self._ai_advisor_result: AdvisorResult | None = None
        self._ai_executor_result: ExecutorProposal | ExecutorResult | None = None
        self._ai_registry: ToolRegistry | None = None
        self._ai_max_iterations: int = 10
        self._ai_budget_tracker: BudgetTracker | None = None
        self._ai_plan_result: PlanResult | None = None

    def __enter__(self) -> Session:
        """Return ``self`` for ``with session:`` ownership scopes."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        """Release owned native resources via :meth:`close_native`."""
        self.close_native()

    def close_native(self) -> None:
        """Close an owned DuckDB connection on the session dataset, if any.

        Safe to call when no dataset is attached or the engine is not DuckDB.
        Derived Datasets that share a connection are not owners; only the root
        handle closes the connection."""
        return data_ops.close_native(self)

    @classmethod
    def ingest(
        cls,
        source: pd.DataFrame | str | Path,
        *,
        mode: DataMode | str | None = None,
        engine: EngineName | str | None = None,
        dry_run: bool = False,
        mock_byte_estimate: int | None = None,
        read_nrows: int | None = None,
    ) -> Session:
        """Create a session by ingesting a tabular source.

        Parameters
        ----------
        source:
            DataFrame or path to CSV/Parquet/Arrow.
        mode:
            Optional data-mode override.
        engine:
            Optional engine override.
        dry_run:
            If True, build a session with report only (no dataset) when the
            pipeline does not materialize data.
        mock_byte_estimate:
            Optional scale override for tests/heuristics.
        read_nrows:
            Optional CSV row cap.

        Returns
        -------
        Session
            Session containing dataset and/or ingest report.

        Notes
        -----
        **Scale:** Large paths are not silently loaded into Pandas. Use
        ``dry_run=True``, ``read_nrows``, ``mode='memory'`` (force), or engine
        extras.

        **Leakage:** Call :meth:`split` before fit-capable operations. Use
        :meth:`assert_can_fit` to enforce train-only fit scope."""
        return data_ops.ingest_session(
            cls,
            source=source,
            mode=mode,
            engine=engine,
            dry_run=dry_run,
            mock_byte_estimate=mock_byte_estimate,
            read_nrows=read_nrows,
        )

    @property
    def dataset(self) -> Dataset:
        """Return the current dataset handle.

        Raises
        ------
        ValidationError
            If no dataset is loaded.
        """
        if self._dataset is None:
            raise ValidationError("Session has no dataset. Call Session.ingest(...) first.")
        return self._dataset

    @property
    def ingest_report(self) -> IngestReport | None:
        """Most recent automated ingest report, if any."""
        return self._ingest_report

    @property
    def split_plan(self) -> SplitPlan | None:
        """Current split membership plan, if any."""
        return self._split_plan

    @property
    def history(self) -> list[dict[str, Any]]:
        """Shallow copy of the operation history."""
        return list(self._history)

    @property
    def reattach_result(self) -> ReattachResult | None:
        """Validation outcome from the last checkpoint load, if any."""
        return self._reattach_result

    def set_roles(self, mapping: dict[str, str | ColumnRole]) -> Session:
        """Assign column roles on the current dataset.

        Parameters
        ----------
        mapping:
            Column → role mapping.

        Returns
        -------
        Session
            ``self`` for fluent chaining."""
        return data_ops.set_roles(self, mapping=mapping)

    def split(
        self,
        *,
        test_size: float | int = 0.2,
        validation_size: float | int | None = None,
        random_state: int | None = 42,
        stratify: bool = False,
    ) -> Session:
        """Create a train/test (optional validation) split.

        Parameters
        ----------
        test_size:
            Test fraction or count.
        validation_size:
            Optional validation fraction/count from the train pool.
        random_state:
            RNG seed.
        stratify:
            If True, stratify on the target role column.

        Notes
        -----
        **Leakage:** After splitting, fit-capable operations must use the train
        partition only (:meth:`assert_can_fit`)."""
        return data_ops.split(
            self,
            test_size=test_size,
            validation_size=validation_size,
            random_state=random_state,
            stratify=stratify,
        )

    def inject_split(
        self,
        *,
        train_indices: list[int] | tuple[int, ...],
        test_indices: list[int] | tuple[int, ...],
        validation_indices: list[int] | tuple[int, ...] | None = None,
    ) -> Session:
        """Inject externally defined partition indices.

        Parameters
        ----------
        train_indices / test_indices / validation_indices:
            Positional indices into the current dataset."""
        return data_ops.inject_split(
            self,
            train_indices=train_indices,
            test_indices=test_indices,
            validation_indices=validation_indices,
        )

    def group_split(
        self,
        *,
        test_size: float | int = 0.2,
        validation_size: float | int | None = None,
        random_state: int | None = 42,
        group_column: str | None = None,
    ) -> Session:
        """Create a group-aware train/test(/validation) split.

        No group identifier appears in more than one partition. Sizes are
        interpreted over groups, not rows.

        Parameters
        ----------
        test_size / validation_size:
            Fraction or count of groups.
        random_state:
            RNG seed.
        group_column:
            Optional override; defaults to the sole ``group`` role column.

        Notes
        -----
        **Leakage:** Prefer this over :meth:`split` when rows share entities
        (customers, sites, documents). Random row splits leak across groups."""
        return data_ops.group_split(
            self,
            test_size=test_size,
            validation_size=validation_size,
            random_state=random_state,
            group_column=group_column,
        )

    def time_split(
        self,
        *,
        test_size: float | int = 0.2,
        validation_size: float | int | None = None,
        time_column: str | None = None,
    ) -> Session:
        """Create a chronological train/test(/validation) split.

        Rows are ordered by the time-role column. The latest rows form test;
        optional validation is carved from the end of the remaining pool.

        Parameters
        ----------
        test_size / validation_size:
            Fraction or absolute row count after time ordering.
        time_column:
            Optional override; defaults to the sole ``time`` role column.

        Notes
        -----
        **Leakage:** Prefer this over shuffled splits for temporal processes.
        The splitter does not add a calendar embargo beyond strict ordering."""
        return data_ops.time_split(
            self, test_size=test_size, validation_size=validation_size, time_column=time_column
        )

    def partition(
        self,
        name: PartitionName | Literal["train", "validation", "test"],
    ) -> pd.DataFrame:
        """Return a copy of rows for a named partition.

        Raises
        ------
        ValidationError
            If no split exists."""
        return data_ops.partition(self, name=name)

    def assert_can_fit(self, partition: PartitionName = "train") -> Session:
        """Enforce leakage-safe fit scope.

        Parameters
        ----------
        partition:
            Partition the caller intends to fit on (must be ``train``).

        Raises
        ------
        LeakageError
            If no split exists or partition is not train."""
        return data_ops.assert_can_fit(self, partition=partition)

    def drop_columns(self, columns: list[str] | tuple[str, ...]) -> Session:
        """Drop columns from the current dataset.

        Parameters
        ----------
        columns:
            Column names to remove.

        Returns
        -------
        Session
            ``self`` for fluent chaining.

        Notes
        -----
        Split membership is preserved (row order unchanged). Roles for dropped
        columns are removed."""
        return preprocess_ops.drop_columns(self, columns=columns)

    def impute(
        self,
        *,
        columns: list[str] | None = None,
        strategy: Literal["mean", "median", "most_frequent", "constant"] = "median",
        fill_value: Any | None = None,
    ) -> Session:
        """Fit imputation on train and transform the full dataset.

        Parameters
        ----------
        columns:
            Columns to impute. Defaults to numeric ``feature``-role columns
            (skips ``ignore`` / ``id`` / ``target`` / ``group`` / ``time`` /
            ``weight``). Pass ``columns=[...]`` to force-include any column.
        strategy:
            Imputation strategy.
        fill_value:
            Constant fill when ``strategy='constant'``.

        Notes
        -----
        **Leakage:** Requires an existing split. Statistics are learned from
        the train partition only, then applied to all rows."""
        return preprocess_ops.impute(
            self, columns=columns, strategy=strategy, fill_value=fill_value
        )

    @property
    def impute_plan(self) -> SimpleImputePlan | None:
        """Last fitted impute plan, if any."""
        return self._impute_plan

    def encode(
        self,
        *,
        columns: list[str] | None = None,
        method: Literal["onehot", "ordinal", "infrequent", "target"] = "onehot",
        min_frequency: float | int = 0.05,
        n_folds: int = 5,
        random_state: int = 0,
        smoothing: float = 10.0,
    ) -> Session:
        """Fit categorical encoding on train and transform the full dataset.

        Parameters
        ----------
        columns:
            Columns to encode. Defaults to categorical ``feature``-role columns
            (skips ``ignore`` / ``id`` / ``target`` / ``group`` / ``time`` /
            ``weight``). Pass ``columns=[...]`` to force-include any column.
        method:
            ``onehot`` / ``ordinal`` for standard encodings; ``infrequent`` to
            pool rare train levels before one-hot; ``target`` for smoothed mean
            target encoding with out-of-fold values on train rows.
        min_frequency:
            For ``infrequent``: float in (0, 1) as a train fraction, or an
            absolute integer count threshold.
        n_folds / random_state / smoothing:
            Target-encoding controls (ignored for other methods).

        Notes
        -----
        **Leakage:** Requires a split. Vocabularies and target means are learned
        on train only. Target encoding writes out-of-fold values on train and
        full-train means on holdouts."""
        return preprocess_ops.encode(
            self,
            columns=columns,
            method=method,
            min_frequency=min_frequency,
            n_folds=n_folds,
            random_state=random_state,
            smoothing=smoothing,
        )

    @property
    def encode_plan(self) -> EncodePlan | None:
        """Last fitted encode plan, if any."""
        return self._encode_plan

    def handle_outliers(
        self,
        *,
        columns: list[str] | None = None,
        method: Literal["iqr", "zscore"] = "iqr",
        action: Literal["detect", "cap", "drop"] = "cap",
        iqr_multiplier: float = 1.5,
        zscore_threshold: float = 3.0,
    ) -> Session:
        """Screen or treat numeric outliers using train-fitted fences.

        Parameters
        ----------
        method:
            ``iqr`` (Tukey fences) or ``zscore``.
        action:
            ``detect`` records the screen without mutating values; ``cap``
            winsorizes to the fences; ``drop`` removes flagged rows and rebuilds
            split membership.

        Notes
        -----
        **Leakage:** Fence statistics are learned on train only, then applied
        with the frozen bounds. Heuristic screens are not proof of error."""
        return preprocess_ops.handle_outliers(
            self,
            columns=columns,
            method=method,
            action=action,
            iqr_multiplier=iqr_multiplier,
            zscore_threshold=zscore_threshold,
        )

    @property
    def outlier_plan(self) -> OutlierPlan | None:
        """Last outlier plan, if any."""
        return self._outlier_plan

    def bin(
        self,
        *,
        columns: list[str] | None = None,
        strategy: Literal["quantile", "uniform"] = "quantile",
        n_bins: int = 5,
        encode_as: Literal["ordinal", "onehot"] = "ordinal",
    ) -> Session:
        """Discretize numeric columns with train-fitted bin edges.

        Notes
        -----
        **Leakage:** Edges are learned on train only. End bins use open
        ``±inf`` edges so score-time extremes remain defined."""
        return preprocess_ops.bin(
            self, columns=columns, strategy=strategy, n_bins=n_bins, encode_as=encode_as
        )

    @property
    def binning_plan(self) -> BinningPlan | None:
        """Last binning plan, if any."""
        return self._binning_plan

    def select_features(
        self,
        *,
        strategy: Literal["variance", "univariate", "model"] = "variance",
        columns: list[str] | None = None,
        threshold: float = 0.0,
        k: int = 10,
        score_func: Literal["f_classif", "f_regression", "mutual_info"] = "f_classif",
        estimator: Any | None = None,
    ) -> Session:
        """Select a feature subset using train-only scores or model reliance.

        Parameters
        ----------
        strategy:
            ``variance`` (VarianceThreshold), ``univariate`` (SelectKBest), or
            ``model`` (SelectFromModel).
        threshold / k / score_func / estimator:
            Strategy-specific controls. Non-feature roles (target, id, group,
            time, weight) are preserved.

        Notes
        -----
        **Leakage:** Selection fits on train only. Encode categoricals and
        impute before calling when features are non-numeric or contain nulls."""
        return preprocess_ops.select_features(
            self,
            strategy=strategy,
            columns=columns,
            threshold=threshold,
            k=k,
            score_func=score_func,
            estimator=estimator,
        )

    @property
    def feature_select_plan(self) -> FeatureSelectPlan | None:
        """Last feature-selection plan, if any."""
        return self._feature_select_plan

    @property
    def last_preprocess(self) -> PreprocessResult | None:
        """Most recent structured preprocess result, if any."""
        return self._last_preprocess

    def scale(
        self,
        *,
        columns: list[str] | None = None,
        method: Literal["standard", "minmax"] = "standard",
    ) -> Session:
        """Fit scaling on train and transform the full dataset.

        Parameters
        ----------
        columns:
            Columns to scale. Defaults to numeric ``feature``-role columns
            (skips ``ignore`` / ``id`` / ``target`` / ``group`` / ``time`` /
            ``weight`` — so costs and identifiers stay unmutated). Pass
            ``columns=[...]`` to force-include any column.
        method:
            ``standard`` or ``minmax``.

        Notes
        -----
        **Leakage:** Requires a split. Scaler is fit on train only."""
        return preprocess_ops.scale(self, columns=columns, method=method)

    @property
    def scale_plan(self) -> ScalePlan | None:
        """Last fitted scale plan, if any."""
        return self._scale_plan

    def text_features(
        self,
        *,
        columns: list[str] | None = None,
        method: Literal["count", "tfidf", "hashing"] = "tfidf",
        max_features: int | None = 128,
        ngram_range: tuple[int, int] = (1, 1),
        drop_input_columns: bool = True,
    ) -> Session:
        """Fit text vectorizers on train and expand columns into numeric features.

        Parameters
        ----------
        method:
            ``tfidf`` (default), ``count``, or ``hashing``.
        max_features:
            Vocabulary width for count/TF-IDF, or hashing output width.
        ngram_range:
            Inclusive n-gram bounds passed to the sklearn vectorizer.

        Notes
        -----
        **Leakage:** Requires a split. Vocabularies and IDF weights are learned
        from train documents only. Missing text becomes empty strings."""
        return preprocess_ops.text_features(
            self,
            columns=columns,
            method=method,
            max_features=max_features,
            ngram_range=ngram_range,
            drop_input_columns=drop_input_columns,
        )

    @property
    def text_plan(self) -> TextFeaturePlan | None:
        """Last fitted text-feature plan, if any."""
        return self._text_plan

    def reduce_dimensions(
        self,
        *,
        columns: list[str] | None = None,
        method: Literal["pca", "umap", "tsne"] = "pca",
        n_components: int | float | None = None,
        drop_input_columns: bool = True,
        prefix: str = "pc",
        random_state: int | None = 0,
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        tsne_perplexity: float = 30.0,
        tsne_learning_rate: str | float = "auto",
    ) -> Session:
        """Fit dimensionality reduction on train and replace numeric columns.

        Parameters
        ----------
        method:
            ``pca`` (core sklearn), ``umap`` (umap-learn when
            ``buildml[unsupervised]`` installed), or ``tsne`` (sklearn; transductive
            train embed with disclosed holdout NN transfer).
        n_components:
            Integer count, float variance target in (0, 1] for PCA, or ``None``.
        prefix:
            Output column prefix (``pc_1``, ``umap_1``, …).

        Notes
        -----
        **Leakage:** Requires a split. The transform is learned on train only.
        Explained variance / embedding quality is unsupervised — not predictive utility.
        Scale numeric inputs first when magnitudes differ."""
        return preprocess_ops.reduce_dimensions(
            self,
            columns=columns,
            method=method,
            n_components=n_components,
            drop_input_columns=drop_input_columns,
            prefix=prefix,
            random_state=random_state,
            umap_n_neighbors=umap_n_neighbors,
            umap_min_dist=umap_min_dist,
            tsne_perplexity=tsne_perplexity,
            tsne_learning_rate=tsne_learning_rate,
        )

    @property
    def reduce_plan(self) -> ReducePlan | None:
        """Last fitted dimensionality-reduction plan, if any."""
        return self._reduce_plan

    def fit_clusters(
        self,
        *,
        method: ClusterMethod = "kmeans",
        n_clusters: int | None = 8,
        columns: list[str] | None = None,
        random_state: int | None = 0,
        n_init: int | str = "auto",
        max_iter: int = 300,
        linkage: str = "ward",
        eps: float = 0.5,
        min_samples: int = 5,
        gmm_covariance_type: str = "full",
        gmm_max_components: int = 10,
        gmm_select_by: str = "bic",
        hdbscan_min_cluster_size: int = 5,
        hdbscan_min_samples: int | None = None,
        spectral_affinity: str = "nearest_neighbors",
        spectral_n_neighbors: int = 10,
        optics_min_samples: int = 5,
        optics_xi: float = 0.05,
        optics_min_cluster_size: float | None = None,
        bandwidth: float | None = None,
        latent_dim: int = 10,
        pretrain_epochs: int = 50,
        finetune_epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        prefer_reduce_components: bool = True,
        label_column: str = "cluster_id",
        auto_k: bool = False,
        auto_k_min: int = 2,
        auto_k_max: int = 10,
    ) -> ClusterFitResult:
        """Fit a clusterer on the train partition only.

        Parameters
        ----------
        method:
            Core: ``kmeans``, ``agglomerative``, ``dbscan``, ``gmm`` (BIC k),
            ``spectral``, ``optics``, ``mean_shift``. Industry extras:
            ``hdbscan`` (``buildml[unsupervised]``). Deep: ``dec`` / ``idec``
            (``buildml[torch]``).
        n_clusters:
            Required for partition-based methods; density methods observe k.
        prefer_reduce_components:
            When True and :meth:`reduce_dimensions` components are on the frame,
            cluster those components instead of raw features.
        auto_k:
            Elbow (k-means) or BIC range (GMM) selection on train.

        Notes
        -----
        **Leakage:** Requires a split. Geometry is learned on train only.
        Scale numeric inputs first for distance-based methods."""
        return unsupervised_ops.fit_clusters(
            self,
            method=method,
            n_clusters=n_clusters,
            columns=columns,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
            linkage=linkage,
            eps=eps,
            min_samples=min_samples,
            gmm_covariance_type=gmm_covariance_type,
            gmm_max_components=gmm_max_components,
            gmm_select_by=gmm_select_by,
            hdbscan_min_cluster_size=hdbscan_min_cluster_size,
            hdbscan_min_samples=hdbscan_min_samples,
            spectral_affinity=spectral_affinity,
            spectral_n_neighbors=spectral_n_neighbors,
            optics_min_samples=optics_min_samples,
            optics_xi=optics_xi,
            optics_min_cluster_size=optics_min_cluster_size,
            bandwidth=bandwidth,
            latent_dim=latent_dim,
            pretrain_epochs=pretrain_epochs,
            finetune_epochs=finetune_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            prefer_reduce_components=prefer_reduce_components,
            label_column=label_column,
            auto_k=auto_k,
            auto_k_min=auto_k_min,
            auto_k_max=auto_k_max,
        )

    def assign_clusters(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        attach: bool = False,
    ) -> ClusterAssignResult:
        """Assign cluster labels with the train-fitted plan (no refit).

        Parameters
        ----------
        partition:
            ``train``, ``validation``, ``test``, or ``all``.
        attach:
            When True, requires ``partition='all'`` and writes ``label_column``
            onto the Session frame as a feature role column."""
        return unsupervised_ops.assign_clusters_op(
            self, partition=partition, attach=attach
        )

    def evaluate_clusters(
        self,
        *,
        partition: PartitionName | Literal["all"] = "validation",
        external_label_column: str | None = None,
        sample_size: int | None = 2000,
        random_state: int | None = 0,
        compute_stability: bool = False,
        stability_runs: int = 10,
        stability_sample_fraction: float = 0.8,
        compute_elbow: bool = False,
        elbow_k_min: int = 2,
        elbow_k_max: int = 10,
    ) -> ClusterEvalResult:
        """Evaluate train-fitted clusters on a partition without refitting.

        Internal metrics (silhouette, Calinski–Harabasz, Davies–Bouldin) describe
        geometry — not supervised accuracy. Optional bootstrap stability and elbow
        diagnostics available. Optional ``external_label_column`` adds ARI/NMI."""
        return unsupervised_ops.evaluate_clusters(
            self,
            partition=partition,
            external_label_column=external_label_column,
            sample_size=sample_size,
            random_state=random_state,
            compute_stability=compute_stability,
            stability_runs=stability_runs,
            stability_sample_fraction=stability_sample_fraction,
            compute_elbow=compute_elbow,
            elbow_k_min=elbow_k_min,
            elbow_k_max=elbow_k_max,
        )

    @property
    def cluster_plan(self) -> ClusterPlan | None:
        """Last fitted unsupervised :class:`~buildml.unsupervised.results.ClusterPlan`."""
        return self._cluster_plan

    @property
    def cluster_fit_result(self) -> ClusterFitResult | None:
        """Last :class:`~buildml.unsupervised.results.ClusterFitResult`, if any."""
        return self._cluster_fit_result

    @property
    def cluster_assign_result(self) -> ClusterAssignResult | None:
        """Last :class:`~buildml.unsupervised.results.ClusterAssignResult`, if any."""
        return self._cluster_assign_result

    @property
    def cluster_eval_result(self) -> ClusterEvalResult | None:
        """Last :class:`~buildml.unsupervised.results.ClusterEvalResult`, if any."""
        return self._cluster_eval_result

    def save_unsupervised_bundle(self, path: str | Path) -> Path:
        """Persist the active cluster plan as ``buildml.unsupervised_bundle.v2``.

        Distinct from Session checkpoints, classical pipelines, Torch trainer
        bundles, and RAG bundles. See
        :data:`buildml.unsupervised.checkpoint.CHECKPOINT_BOUNDARY`."""
        return unsupervised_ops.save_unsupervised_bundle_op(self, path=path)

    def load_unsupervised_bundle(self, path: str | Path) -> Session:
        """Load an unsupervised bundle into this Session."""
        return unsupervised_ops.load_unsupervised_bundle_op(self, path=path)

    def fit_voting(
        self,
        estimators: Mapping[str, Any] | Sequence[tuple[str, Any]],
        *,
        voting: VotingMethod = "hard",
        weights: Sequence[float] | None = None,
        task: Literal["classification", "regression", "auto"] = "auto",
    ) -> EnsembleFitResult:
        """Fit a VotingClassifier / VotingRegressor on the train partition only.

        Parameters
        ----------
        estimators:
            Mapping or sequence of ``(name, estimator)`` base learners.
        voting:
            ``hard`` or ``soft`` (classification; soft needs ``predict_proba``).
        weights:
            Optional per-estimator weights.

        Notes
        -----
        **Leakage:** Requires a split. Fits on train only. Sets
        :attr:`fit_result` so :meth:`evaluate` / :meth:`predict` /
        :meth:`save_pipeline` work. Distinct from passing a single RandomForest
        to :meth:`fit`."""
        return ensemble_ops.fit_voting(
            self, estimators, voting=voting, weights=weights, task=task
        )

    def fit_stacking(
        self,
        estimators: Mapping[str, Any] | Sequence[tuple[str, Any]],
        *,
        final_estimator: Any | None = None,
        cv: int = 5,
        passthrough: bool = False,
        stack_method: str = "auto",
        task: Literal["classification", "regression", "auto"] = "auto",
    ) -> EnsembleFitResult:
        """Fit a StackingClassifier / StackingRegressor on the train partition only.

        Notes
        -----
        **Leakage:** Stacking CV folds stay inside train. Session test is never
        used for out-of-fold meta features."""
        return ensemble_ops.fit_stacking(
            self,
            estimators,
            final_estimator=final_estimator,
            cv=cv,
            passthrough=passthrough,
            stack_method=stack_method,
            task=task,
        )

    def fit_blending(
        self,
        estimators: Mapping[str, Any] | Sequence[tuple[str, Any]],
        *,
        final_estimator: Any | None = None,
        holdout_fraction: float = 0.2,
        blend_method: BlendMethod = "predict_proba",
        random_state: int | None = 0,
        refit_bases_on_full_train: bool = True,
        passthrough: bool = False,
        task: Literal["classification", "regression", "auto"] = "auto",
    ) -> EnsembleFitResult:
        """Fit a holdout-blend ensemble on the train partition only.

        The blend holdout is carved from **train** (not Session validation/test).
        Prefer :meth:`fit_stacking` when you want CV out-of-fold meta features.

        Notes
        -----
        **Leakage:** Meta-learner fits on an inner train holdout only. Bases are
        optionally refit on full train for deployment (disclosed)."""
        return ensemble_ops.fit_blending(
            self,
            estimators,
            final_estimator=final_estimator,
            holdout_fraction=holdout_fraction,
            blend_method=blend_method,
            random_state=random_state,
            refit_bases_on_full_train=refit_bases_on_full_train,
            passthrough=passthrough,
            task=task,
        )

    def evaluate_ensemble(
        self,
        *,
        partition: Literal["train", "validation", "test"] = "test",
    ) -> EvaluateResult:
        """Evaluate the last native ensemble with classical supervised metrics.

        Same metric path as :meth:`evaluate`, with ensemble strategy disclosures
        attached to recommendations / diagnostics."""
        return ensemble_ops.evaluate_ensemble(self, partition=partition)

    @property
    def ensemble_plan(self) -> EnsemblePlan | None:
        """Last fitted native :class:`~buildml.ensemble.results.EnsemblePlan`."""
        return self._ensemble_plan

    @property
    def ensemble_fit_result(self) -> EnsembleFitResult | None:
        """Last :class:`~buildml.ensemble.results.EnsembleFitResult`, if any."""
        return self._ensemble_fit_result

    def save_ensemble_bundle(self, path: str | Path) -> Path:
        """Persist the active ensemble plan as ``buildml.ensemble_bundle.v1``.

        Distinct from Session checkpoints and classical pipeline bundles. See
        :data:`buildml.ensemble.checkpoint.CHECKPOINT_BOUNDARY`. Prefer
        :meth:`save_pipeline` when preprocess plans must travel with the
        estimator."""
        return ensemble_ops.save_ensemble_bundle_op(self, path=path)

    def load_ensemble_bundle(self, path: str | Path) -> Session:
        """Load an ensemble bundle into this Session."""
        return ensemble_ops.load_ensemble_bundle_op(self, path=path)

    def run_automl(
        self,
        *,
        backend: AutoMLBackend = "native",
        task: Literal["classification", "regression", "auto"] = "auto",
        method: AutoMLMethod = "randomized",
        selection: AutoMLSelection = "cv",
        n_trials: int = 20,
        cv: int | Any = 3,
        outer_cv: int | Any = 3,
        cv_strategy: Literal[
            "auto", "kfold", "stratified", "group", "stratified_group", "time"
        ] = "auto",
        ranking_metric: str | None = None,
        families: Sequence[str] | None = None,
        include_recipe_search: bool = True,
        include_industry_families: bool = True,
        include_ensembles: bool = False,
        ensemble_mode: EnsembleMode = "voting",
        max_ensemble_bases: int = 3,
        preprocess: PreprocessRecipe | None = None,
        allow_session_global_preprocess: bool = False,
        refit: bool = True,
        random_state: int | None = 0,
        groups: pd.Series | None = None,
        budget: AutoMLBudget | None = None,
        time_budget: float | None = None,
    ) -> AutoMLResult:
        """Search model families and fold-local preprocess strategies on train.

        Goes beyond single-estimator :meth:`grid_search` / :meth:`optuna_search`
        by jointly ranking estimator **families**, discrete **recipe
        strategies** (impute/scale/encode/select), modest hyperparameter
        catalogs, and optionally voting ensembles of diverse top families.

        Parameters
        ----------
        backend:
            ``native`` (default), ``optuna`` (deepened Optuna),
            ``flaml`` or ``autogluon`` (``buildml[automl-industry]``).
        method:
            ``randomized`` (default), ``grid``, ``optuna`` (``buildml[automl]``),
            or ``evolutionary`` (in-tree GA).
        selection:
            ``cv`` (train-fold CV), ``nested`` (outer train estimate after
            inner selection), or ``validation`` (rank on Session validation;
            never test).
        include_recipe_search:
            Search discrete fold-local :class:`PreprocessRecipe` strategies.
        include_industry_families:
            When ``buildml[automl-industry]`` GBDT libs are installed, extend
            the native catalog with LightGBM / XGBoost / CatBoost.
        include_ensembles:
            Optionally score voting/stacking ensembles of diverse top families.
        ensemble_mode:
            ``voting``, ``stacking``, or ``both`` when ``include_ensembles=True``.
        time_budget:
            Optional wall-clock cap in seconds (disclosed in results).
        allow_session_global_preprocess:
            Same hard refusal contract as classical CV/search.

        Notes
        -----
        **Leakage:** Session test never enters selection. Fold-local recipes
        refit on fold-train only. **Not** NAS, not causal discovery, not a
        fully automated AI scientist — finite disclosed catalogs under a
        trial budget. Sets :attr:`fit_result` so :meth:`evaluate` /
        :meth:`predict` / :meth:`save_pipeline` work.
        """
        return automl_ops.run_automl_op(
            self,
            backend=backend,
            task=task,
            method=method,
            selection=selection,
            n_trials=n_trials,
            cv=cv,
            outer_cv=outer_cv,
            cv_strategy=cv_strategy,
            ranking_metric=ranking_metric,
            families=families,
            include_recipe_search=include_recipe_search,
            include_industry_families=include_industry_families,
            include_ensembles=include_ensembles,
            ensemble_mode=ensemble_mode,
            max_ensemble_bases=max_ensemble_bases,
            preprocess=preprocess,
            allow_session_global_preprocess=allow_session_global_preprocess,
            refit=refit,
            random_state=random_state,
            groups=groups,
            budget=budget,
            time_budget=time_budget,
        )

    def evaluate_automl(
        self,
        *,
        partition: Literal["train", "validation", "test"] = "test",
    ) -> EvaluateResult:
        """Evaluate the last AutoML winner with classical supervised metrics.

        Same metric path as :meth:`evaluate`, with AutoML disclosures attached
        to recommendations / diagnostics."""
        return automl_ops.evaluate_automl(self, partition=partition)

    @property
    def automl_plan(self) -> AutoMLPlan | None:
        """Last selected :class:`~buildml.automl.results.AutoMLPlan`."""
        return self._automl_plan

    @property
    def automl_result(self) -> AutoMLResult | None:
        """Last :class:`~buildml.automl.results.AutoMLResult`, if any."""
        return self._automl_result

    def save_automl_bundle(self, path: str | Path) -> Path:
        """Persist the active AutoML plan as ``buildml.automl_bundle.v1``.

        Distinct from Session checkpoints and classical pipeline bundles. See
        :data:`buildml.automl.checkpoint.CHECKPOINT_BOUNDARY`. Prefer
        :meth:`save_pipeline` when Session-global preprocess plans must travel
        with the estimator."""
        return automl_ops.save_automl_bundle_op(self, path=path)

    def load_automl_bundle(self, path: str | Path) -> Session:
        """Load an AutoML bundle into this Session."""
        return automl_ops.load_automl_bundle_op(self, path=path)

    def fit_forecast(
        self,
        *,
        method: ForecastMethod = "auto",
        horizon: int = 1,
        lags: list[int] | tuple[int, ...] | None = None,
        seasonal_period: int | None = None,
        exog_columns: list[str] | None = None,
        target_column: str | None = None,
        time_column: str | None = None,
        random_state: int | None = 0,
        alpha: float = 1.0,
        max_iter: int = 100,
        max_depth: int | None = 3,
        learning_rate: float = 0.1,
        order: tuple[int, int, int] | None = None,
        seasonal_order: tuple[int, int, int, int] | None = None,
        nbeats_input_size: int = 24,
        nbeats_horizon: int | None = None,
    ) -> ForecastFitResult:
        """Fit a forecaster on the train partition only.

        Parameters
        ----------
        method:
            ``auto`` (ETS when statsmodels installed, else ``lag_ridge``),
            baselines, ``lag_ridge``/``lag_hgb``, or with ``buildml[timeseries]``:
            ``arima``, ``auto_arima``, ``ets``, ``sarimax``. Prophet / N-BEATS
            behind ``timeseries-prophet`` / ``timeseries-ml``.
        horizon:
            Default generate horizon stored on the plan.
        lags:
            Positive lag orders for lag models (defaults to ``(1, 2, 3, 7)``).
        seasonal_period:
            Required semantics for ``seasonal_naive`` (defaults to ``max(lags)``).
        exog_columns:
            Optional numeric exogenous columns. Empty ⇒ univariate.

        Notes
        -----
        **Leakage:** Requires :meth:`time_split` (or chronologically ordered
        :meth:`inject_split`). Random/stratified/group splits are refused.
        Not a digital twin. With ``buildml[timeseries]``, statsmodels ETS/ARIMA
        are industry defaults; core lag/baseline fallback when extras absent.
        """
        return forecast_ops.fit_forecast(
            self,
            method=method,
            horizon=horizon,
            lags=lags,
            seasonal_period=seasonal_period,
            exog_columns=exog_columns,
            target_column=target_column,
            time_column=time_column,
            random_state=random_state,
            alpha=alpha,
            max_iter=max_iter,
            max_depth=max_depth,
            learning_rate=learning_rate,
            order=order,
            seasonal_order=seasonal_order,
            nbeats_input_size=nbeats_input_size,
            nbeats_horizon=nbeats_horizon,
        )

    def generate_forecast(
        self,
        *,
        horizon: int | None = None,
        origin: str = "train_end",
        future_exog: Any | None = None,
    ) -> ForecastGenerateResult:
        """Generate an H-step forecast from the train-fitted plan (no refit).

        Parameters
        ----------
        horizon:
            Steps ahead; defaults to the plan horizon.
        origin:
            ``train_end``, ``validation_end``, or ``test_end``.
        future_exog:
            Required when the plan uses exogenous columns.
        """
        return forecast_ops.generate_forecast_op(
            self, horizon=horizon, origin=origin, future_exog=future_exog
        )

    def evaluate_forecast(
        self,
        *,
        partition: PartitionName = "test",
        strategy: ForecastEvalStrategy = "rolling_one_step",
    ) -> ForecastEvalResult:
        """Evaluate the train-fitted forecaster on a holdout partition.

        Metrics: MAE, RMSE, MAPE (MAPE unstable near zero — disclosed).
        Defaults to validation when requested but missing, falling back to test
        via the ops layer for empty validation only when partition='validation'.
        """
        return forecast_ops.evaluate_forecast_op(
            self, partition=partition, strategy=strategy
        )

    @property
    def forecast_plan(self) -> ForecastPlan | None:
        """Last fitted :class:`~buildml.forecasting.results.ForecastPlan`."""
        return self._forecast_plan

    @property
    def forecast_fit_result(self) -> ForecastFitResult | None:
        """Last :class:`~buildml.forecasting.results.ForecastFitResult`, if any."""
        return self._forecast_fit_result

    @property
    def forecast_generate_result(self) -> ForecastGenerateResult | None:
        """Last :class:`~buildml.forecasting.results.ForecastGenerateResult`, if any."""
        return self._forecast_generate_result

    @property
    def forecast_eval_result(self) -> ForecastEvalResult | None:
        """Last :class:`~buildml.forecasting.results.ForecastEvalResult`, if any."""
        return self._forecast_eval_result

    def save_forecast_bundle(self, path: str | Path) -> Path:
        """Persist the active forecast plan as ``buildml.forecast_bundle.v1``.

        Distinct from Session checkpoints, classical pipelines, Torch trainer
        bundles, and RAG bundles. See
        :data:`buildml.forecasting.checkpoint.CHECKPOINT_BOUNDARY`."""
        return forecast_ops.save_forecast_bundle_op(self, path=path)

    def load_forecast_bundle(self, path: str | Path) -> Session:
        """Load a forecast bundle into this Session."""
        return forecast_ops.load_forecast_bundle_op(self, path=path)

    def analyze_timeseries(
        self,
        *,
        target_column: str | None = None,
        time_column: str | None = None,
        scope: str = "train",
        seasonal_period: int | None = None,
        decompose_method: str | None = None,
        include_decompose: bool = True,
        include_diagnostics: bool = True,
        include_changepoints: bool = True,
        include_features: bool = True,
        acf_lags: int = 40,
        pacf_lags: int = 40,
        changepoint_penalty: float = 10.0,
        rolling_window: int = 7,
    ) -> Any:
        """Run time-series analysis (decompose, diagnostics, changepoints, features).

        Notes
        -----
        **Leakage:** Requires :meth:`time_split`. Default ``scope='train'``.
        Industry defaults (STL, ADF/KPSS) when ``buildml[timeseries]`` installed.
        """
        return timeseries_ops.analyze_timeseries_op(
            self,
            target_column=target_column,
            time_column=time_column,
            scope=scope,  # type: ignore[arg-type]
            seasonal_period=seasonal_period,
            decompose_method=decompose_method,  # type: ignore[arg-type]
            include_decompose=include_decompose,
            include_diagnostics=include_diagnostics,
            include_changepoints=include_changepoints,
            include_features=include_features,
            acf_lags=acf_lags,
            pacf_lags=pacf_lags,
            changepoint_penalty=changepoint_penalty,
            rolling_window=rolling_window,
        )

    def ts_decompose(self, **kwargs: Any) -> Any:
        """STL/classical decomposition on train-only scope (default)."""
        return timeseries_ops.ts_decompose_op(self, **kwargs)

    def ts_diagnostics(self, **kwargs: Any) -> Any:
        """ACF/PACF and ADF/KPSS stationarity tests."""
        return timeseries_ops.ts_diagnostics_op(self, **kwargs)

    @property
    def ts_analysis_result(self) -> Any | None:
        """Last time-series analysis result, if any."""
        return self._ts_analysis_result

    def fit_anomaly(
        self,
        *,
        backend: AnomalyBackend | None = None,
        method: AnomalyMethod = "isolation_forest",
        mode: AnomalyMode = "unsupervised",
        columns: list[str] | None = None,
        random_state: int | None = 0,
        contamination: float = 0.05,
        threshold_policy: ThresholdPolicy = "contamination",
        score_threshold: float | None = None,
        quantile: float | None = None,
        n_estimators: int = 100,
        max_samples: str | int | float = "auto",
        n_neighbors: int = 20,
        nu: float = 0.05,
        kernel: str = "rbf",
        gamma: str | float = "scale",
        latent_dim: int = 8,
        ae_epochs: int = 40,
        ae_batch_size: int = 64,
        normal_label_column: str | None = None,
        normal_label_value: Any = 0,
        positive_label: Any = 1,
        prefer_reduce_components: bool = True,
        flag_column: str = "is_anomaly",
        score_column: str = "anomaly_score",
    ) -> AnomalyFitResult:
        """Fit an anomaly / fraud detector on the train partition only.

        Parameters
        ----------
        backend:
            ``sklearn`` (core), ``pyod`` (``buildml[anomaly-industry]``), or
            ``torch`` (``buildml[torch]`` autoencoder path).
        method:
            Catalog method for the backend — see
            :func:`buildml.anomaly.anomaly_capability_matrix`.
        mode:
            ``unsupervised`` (fit all train rows), ``novelty`` (normal-only
            train subset via ``normal_label_column``), or ``supervised``.
        contamination:
            Prior alert fraction used when ``threshold_policy='contamination'``
            (and IsolationForest/LOF contamination knobs).
        threshold_policy:
            ``contamination``, ``quantile``, ``score_threshold``, or
            ``decision_zero`` (One-Class SVM). Use
            :meth:`tune_anomaly_threshold` for validation-tuned cutoffs.

        Notes
        -----
        **Leakage:** Requires a split. Detector fit on train only. Higher
        ``anomaly_score`` means more anomalous; thresholds and alert rates are
        always disclosed. Distinct from EDA IsolationForest screens and
        :meth:`handle_outliers`. Not a graph-fraud or streaming platform; no
        causal fraud claims. Clustering (:meth:`fit_clusters`) remains a
        separate structure API.
        """
        return anomaly_ops.fit_anomaly(
            self,
            backend=backend,
            method=method,
            mode=mode,
            columns=columns,
            random_state=random_state,
            contamination=contamination,
            threshold_policy=threshold_policy,
            score_threshold=score_threshold,
            quantile=quantile,
            n_estimators=n_estimators,
            max_samples=max_samples,
            n_neighbors=n_neighbors,
            nu=nu,
            kernel=kernel,
            gamma=gamma,
            latent_dim=latent_dim,
            ae_epochs=ae_epochs,
            ae_batch_size=ae_batch_size,
            normal_label_column=normal_label_column,
            normal_label_value=normal_label_value,
            positive_label=positive_label,
            prefer_reduce_components=prefer_reduce_components,
            flag_column=flag_column,
            score_column=score_column,
        )

    def tune_anomaly_threshold(
        self,
        *,
        partition: PartitionName = "validation",
        label_column: str | None = None,
        positive_label: Any | None = None,
        metric: ThresholdTuningMetric = "f1",
        fbeta: float = 2.0,
        allow_test_tuning: bool = False,
        update_plan: bool = True,
    ) -> AnomalyThresholdTuneResult:
        """Tune anomaly threshold on validation labels (never test by default).

        Same leakage discipline as :meth:`tune_threshold` — tune on validation,
        evaluate final claims on untouched test.
        """
        return anomaly_ops.tune_anomaly_threshold_op(
            self,
            partition=partition,
            label_column=label_column,
            positive_label=positive_label,
            metric=metric,
            fbeta=fbeta,
            allow_test_tuning=allow_test_tuning,
            update_plan=update_plan,
        )

    @staticmethod
    def anomaly_capability_matrix() -> dict[str, Any]:
        """Honest capability matrix for anomaly backends and extras."""
        return anomaly_ops.anomaly_capability_matrix_op()

    def score_anomalies(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        attach: bool = False,
        override_threshold: float | None = None,
    ) -> AnomalyScoreResult:
        """Score and flag rows with the train-fitted anomaly plan (no refit).

        Parameters
        ----------
        partition:
            ``train``, ``validation``, ``test``, or ``all``.
        attach:
            When True, requires ``partition='all'`` and writes score/flag
            columns onto the Session frame.
        override_threshold:
            Optional absolute threshold for this call only (does not mutate
            the stored plan threshold).
        """
        return anomaly_ops.score_anomalies_op(
            self,
            partition=partition,
            attach=attach,
            override_threshold=override_threshold,
        )

    def evaluate_anomaly(
        self,
        *,
        partition: PartitionName | Literal["all"] = "validation",
        label_column: str | None = None,
        positive_label: Any | None = None,
        k: int | None = None,
        override_threshold: float | None = None,
    ) -> AnomalyEvalResult:
        """Evaluate train-fitted anomaly scores on a partition without refitting.

        Always reports threshold and alert rate. When labels are available
        (``label_column`` or a target role), also reports precision/recall/F1,
        PR-AUC, ROC-AUC, and precision/recall@k with imbalance disclosures.
        Defaults to validation, falling back to test when no validation
        partition exists.
        """
        return anomaly_ops.evaluate_anomaly_op(
            self,
            partition=partition,
            label_column=label_column,
            positive_label=positive_label,
            k=k,
            override_threshold=override_threshold,
        )

    @property
    def anomaly_plan(self) -> AnomalyPlan | None:
        """Last fitted :class:`~buildml.anomaly.results.AnomalyPlan`."""
        return self._anomaly_plan

    @property
    def anomaly_fit_result(self) -> AnomalyFitResult | None:
        """Last :class:`~buildml.anomaly.results.AnomalyFitResult`, if any."""
        return self._anomaly_fit_result

    @property
    def anomaly_score_result(self) -> AnomalyScoreResult | None:
        """Last :class:`~buildml.anomaly.results.AnomalyScoreResult`, if any."""
        return self._anomaly_score_result

    @property
    def anomaly_eval_result(self) -> AnomalyEvalResult | None:
        """Last :class:`~buildml.anomaly.results.AnomalyEvalResult`, if any."""
        return self._anomaly_eval_result

    def save_anomaly_bundle(self, path: str | Path) -> Path:
        """Persist the active anomaly plan as ``buildml.anomaly_bundle.v1``.

        Distinct from Session checkpoints, unsupervised bundles, classical
        pipelines, Torch trainer bundles, and RAG bundles. See
        :data:`buildml.anomaly.checkpoint.CHECKPOINT_BOUNDARY`."""
        return anomaly_ops.save_anomaly_bundle_op(self, path=path)

    def load_anomaly_bundle(self, path: str | Path) -> Session:
        """Load an anomaly bundle into this Session."""
        return anomaly_ops.load_anomaly_bundle_op(self, path=path)

    def fit_semisupervised(
        self,
        *,
        backend: SemiSupervisedBackend | None = None,
        method: SemiSupervisedMethod = "label_propagation",
        columns: list[str] | None = None,
        random_state: int | None = 0,
        kernel: str = "knn",
        n_neighbors: int = 7,
        max_iter: int = 1000,
        alpha: float = 0.2,
        base_estimator: str = "logistic_regression",
        threshold: float = 0.75,
        criterion: str = "threshold",
        k_best: int = 10,
        max_self_train_iter: int = 10,
        epochs: int = 40,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        consistency_weight: float = 1.0,
        mixup_alpha: float = 0.75,
        device: str = "cpu",
        text_column: str | None = None,
        text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        unlabeled_marker: Any = None,
        prefer_reduce_components: bool = True,
    ) -> SemiSupervisedFitResult:
        """Fit a semi-supervised classifier on the train partition only.

        Parameters
        ----------
        backend:
            ``sklearn`` (default), ``industry`` (XGB/LGBM pseudo-label),
            ``torch`` (FixMatch/MixMatch tabular), or ``hf`` (text pseudo-label).
        method:
            Algorithm within the backend — see
            :func:`buildml.semisupervised.semisupervised_capability_matrix`.
        unlabeled_marker:
            Extra sentinel treated as unlabeled. Default ``None`` means pandas
            missing values (NaN) in the target role mark unlabeled rows.

        Notes
        -----
        **Leakage:** Requires a split. Fit uses train only. Validation/test
        never invent labels for model selection. Distinct from anomaly novelty
        and from self-supervised pretext (:meth:`fit_ssl_pretext`).

        **SSL integration:** ``fit_ssl_pretext`` → ``transform_ssl`` (or reduce
        on embeddings) → ``fit_semisupervised`` uses partial labels on SSL features.
        """
        return semisupervised_ops.fit_semisupervised_op(
            self,
            backend=backend,
            method=method,
            columns=columns,
            random_state=random_state,
            kernel=kernel,
            n_neighbors=n_neighbors,
            max_iter=max_iter,
            alpha=alpha,
            base_estimator=base_estimator,
            threshold=threshold,
            criterion=criterion,
            k_best=k_best,
            max_self_train_iter=max_self_train_iter,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            consistency_weight=consistency_weight,
            mixup_alpha=mixup_alpha,
            device=device,
            text_column=text_column,
            text_model_name=text_model_name,
            unlabeled_marker=unlabeled_marker,
            prefer_reduce_components=prefer_reduce_components,
        )

    def predict_semisupervised(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        attach: bool = False,
        prediction_column: str = "semisupervised_prediction",
    ) -> SemiSupervisedPredictResult:
        """Predict with the train-fitted semi-supervised plan (no refit)."""
        return semisupervised_ops.predict_semisupervised_op(
            self,
            partition=partition,
            attach=attach,
            prediction_column=prediction_column,
        )

    def evaluate_semisupervised(
        self,
        *,
        partition: PartitionName | Literal["all"] = "validation",
        unlabeled_marker: Any = None,
    ) -> SemiSupervisedEvalResult:
        """Evaluate semi-supervised predictions on labeled partition rows only.

        Unlabeled holdout rows are disclosed and excluded from metrics. Defaults
        to validation, falling back to test when no validation partition exists.
        """
        return semisupervised_ops.evaluate_semisupervised_op(
            self,
            partition=partition,
            unlabeled_marker=unlabeled_marker,
        )

    @property
    def semisupervised_plan(self) -> SemiSupervisedPlan | None:
        """Last fitted :class:`~buildml.semisupervised.results.SemiSupervisedPlan`."""
        return self._semisupervised_plan

    @property
    def semisupervised_fit_result(self) -> SemiSupervisedFitResult | None:
        """Last :class:`~buildml.semisupervised.results.SemiSupervisedFitResult`."""
        return self._semisupervised_fit_result

    @property
    def semisupervised_predict_result(self) -> SemiSupervisedPredictResult | None:
        """Last :class:`~buildml.semisupervised.results.SemiSupervisedPredictResult`."""
        return self._semisupervised_predict_result

    @property
    def semisupervised_eval_result(self) -> SemiSupervisedEvalResult | None:
        """Last :class:`~buildml.semisupervised.results.SemiSupervisedEvalResult`."""
        return self._semisupervised_eval_result

    def save_semisupervised_bundle(self, path: str | Path) -> Path:
        """Persist the active plan as ``buildml.semisupervised_bundle.v1``.

        See :data:`buildml.semisupervised.checkpoint.CHECKPOINT_BOUNDARY`."""
        return semisupervised_ops.save_semisupervised_bundle_op(self, path=path)

    def load_semisupervised_bundle(self, path: str | Path) -> Session:
        """Load a semi-supervised bundle into this Session."""
        return semisupervised_ops.load_semisupervised_bundle_op(self, path=path)

    def fit_ssl_pretext(
        self,
        *,
        method: SelfSupervisedMethod | None = None,
        columns: list[str] | None = None,
        text_column: str | None = None,
        image_column: str | None = None,
        random_state: int | None = 0,
        latent_dim: int = 16,
        hidden: tuple[int, ...] | list[int] = (64,),
        mask_ratio: float = 0.15,
        n_mask_views: int = 3,
        max_iter: int = 200,
        epochs: int = 40,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        temperature: float = 0.5,
        projector_dim: int = 32,
        projector_hidden: tuple[int, ...] | list[int] = (64,),
        prefer_reduce_components: bool = True,
        representation_prefix: str = "ssl_emb",
        backbone: str = "resnet18",
        weight_mode: str = "mock",
        hf_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> SelfSupervisedFitResult:
        """Fit a self-supervised pretext on the train partition only.

        Parameters
        ----------
        method:
            Torch tabular defaults: ``simclr_tabular``, ``byol_tabular``,
            ``vicreg_tabular``, ``mae_tabular``, ``vae_tabular``.
            Legacy ``masked_tabular`` (sklearn) is deprecated.
            Text: ``hf_text_ssl``. Vision: ``vision_ssl``.
        latent_dim:
            Bottleneck width exported as representation columns.

        Notes
        -----
        **Leakage:** Requires a split. Pretext fits on train features only
        (labels ignored). Install ``buildml[torch]`` for industry defaults.
        """
        return selfsupervised_ops.fit_ssl_pretext_op(
            self,
            method=method,
            columns=columns,
            text_column=text_column,
            image_column=image_column,
            random_state=random_state,
            latent_dim=latent_dim,
            hidden=hidden,
            mask_ratio=mask_ratio,
            n_mask_views=n_mask_views,
            max_iter=max_iter,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            temperature=temperature,
            projector_dim=projector_dim,
            projector_hidden=projector_hidden,
            prefer_reduce_components=prefer_reduce_components,
            representation_prefix=representation_prefix,
            backbone=backbone,
            weight_mode=weight_mode,
            hf_model_name=hf_model_name,
            device=device,
        )

    def transform_ssl(
        self,
        *,
        partition: PartitionName | Literal["all"] = "train",
        attach: bool = False,
    ) -> SelfSupervisedTransformResult:
        """Export SSL representations with the train-fitted pretext (no refit)."""
        return selfsupervised_ops.transform_ssl_op(
            self,
            partition=partition,
            attach=attach,
        )

    def finetune_ssl_head(
        self,
        *,
        estimator: SSLHeadEstimator = "logistic_regression",
        random_state: int | None = 0,
        unlabeled_marker: Any = None,
    ) -> SSLHeadFitResult:
        """Fit a supervised head on frozen SSL embeddings (labeled train only).

        Unlabeled train targets (NaN by default) are skipped. Holdout partitions
        are evaluation-only; the pretext encoder is not updated.
        """
        return selfsupervised_ops.finetune_ssl_head_op(
            self,
            estimator=estimator,
            random_state=random_state,
            unlabeled_marker=unlabeled_marker,
        )

    def evaluate_ssl(
        self,
        *,
        partition: PartitionName | Literal["all"] = "validation",
        unlabeled_marker: Any = None,
    ) -> SelfSupervisedEvalResult:
        """Evaluate frozen SSL pretext + head on labeled partition rows."""
        return selfsupervised_ops.evaluate_ssl_op(
            self,
            partition=partition,
            unlabeled_marker=unlabeled_marker,
        )

    @property
    def ssl_plan(self) -> SelfSupervisedPlan | None:
        """Last fitted :class:`~buildml.selfsupervised.results.SelfSupervisedPlan`."""
        return self._ssl_plan

    @property
    def ssl_fit_result(self) -> SelfSupervisedFitResult | None:
        """Last :class:`~buildml.selfsupervised.results.SelfSupervisedFitResult`."""
        return self._ssl_fit_result

    @property
    def ssl_transform_result(self) -> SelfSupervisedTransformResult | None:
        """Last :class:`~buildml.selfsupervised.results.SelfSupervisedTransformResult`."""
        return self._ssl_transform_result

    @property
    def ssl_head_plan(self) -> SSLHeadPlan | None:
        """Last :class:`~buildml.selfsupervised.results.SSLHeadPlan`, if any."""
        return self._ssl_head_plan

    @property
    def ssl_head_fit_result(self) -> SSLHeadFitResult | None:
        """Last :class:`~buildml.selfsupervised.results.SSLHeadFitResult`, if any."""
        return self._ssl_head_fit_result

    @property
    def ssl_eval_result(self) -> SelfSupervisedEvalResult | None:
        """Last :class:`~buildml.selfsupervised.results.SelfSupervisedEvalResult`."""
        return self._ssl_eval_result

    def save_ssl_bundle(self, path: str | Path) -> Path:
        """Persist the active SSL plan as ``buildml.ssl_bundle.v2``.

        See :data:`buildml.selfsupervised.checkpoint.CHECKPOINT_BOUNDARY`."""
        return selfsupervised_ops.save_ssl_bundle_op(self, path=path)

    def load_ssl_bundle(self, path: str | Path) -> Session:
        """Load a self-supervised bundle into this Session."""
        return selfsupervised_ops.load_ssl_bundle_op(self, path=path)

    def fit_active_learner(
        self,
        *,
        backend: ActiveLearningBackend | None = None,
        strategy: ActiveLearningStrategy = "margin",
        base_estimator: ActiveLearningEstimator = "logistic_regression",
        columns: list[str] | None = None,
        random_state: int | None = 0,
        batch_size: int = 5,
        label_budget: int | None = 50,
        unlabeled_marker: Any = None,
        prefer_reduce_components: bool = True,
        committee_size: int = 5,
        auto_refit: bool = True,
        epochs: int = 60,
        learning_rate: float = 1e-3,
        mc_samples: int = 20,
        device: str = "cpu",
    ) -> ActiveLearningFitResult:
        """Fit / initialize an active learner on labeled train rows only.

        Parameters
        ----------
        backend:
            ``sklearn`` (default), ``industry`` (scikit-activeml), or ``torch``.
        strategy:
            Uncertainty / committee / CoreSet / BALD strategies — see
            :func:`activelearning_capability_matrix`.
        label_budget:
            Cap on labels acquired via :meth:`label_rows` (``None`` = unlimited).
        unlabeled_marker:
            Extra sentinel treated as unlabeled pool. Default ``None`` means
            pandas missing values (NaN) in the target role mark the pool.

        Notes
        -----
        **Leakage:** Requires a split. Fit uses labeled train rows only. The
        query pool is train missingness — never validation/test. Labels come
        from the user (no oracle in core). Distinct from
        :meth:`fit_semisupervised` propagation and :meth:`fit_ssl_pretext`.
        """
        return activelearning_ops.fit_active_learner_op(
            self,
            backend=backend,
            strategy=strategy,
            base_estimator=base_estimator,
            columns=columns,
            random_state=random_state,
            batch_size=batch_size,
            label_budget=label_budget,
            unlabeled_marker=unlabeled_marker,
            prefer_reduce_components=prefer_reduce_components,
            committee_size=committee_size,
            auto_refit=auto_refit,
            epochs=epochs,
            learning_rate=learning_rate,
            mc_samples=mc_samples,
            device=device,
        )

    def suggest_query(
        self,
        *,
        batch_size: int | None = None,
        strategy: ActiveLearningStrategy | None = None,
    ) -> ActiveLearningQueryResult:
        """Suggest unlabeled *train* indices for human labeling (no oracle).

        Never queries validation/test. Honors the remaining label budget.
        Low-level package alias: ``buildml.activelearning.query_indices``.
        """
        return activelearning_ops.suggest_query_op(
            self,
            batch_size=batch_size,
            strategy=strategy,
        )

    def label_rows(
        self,
        *,
        indices: list[Any] | tuple[Any, ...],
        labels: list[Any] | tuple[Any, ...],
        refit: bool | None = None,
    ) -> ActiveLearningLabelResult:
        """Incorporate user-provided labels on train-pool rows; optionally refit.

        Labels must come from the user (or a test harness oracle). Core never
        invents labels. Refuses validation/test indices and enforces
        ``label_budget``.
        """
        return activelearning_ops.label_rows_op(
            self,
            indices=indices,
            labels=labels,
            refit=refit,
        )

    def evaluate_active_learning(
        self,
        *,
        partition: PartitionName | Literal["all"] = "validation",
        unlabeled_marker: Any = None,
    ) -> ActiveLearningEvalResult:
        """Evaluate the active learner on labeled partition rows only.

        Unlabeled holdout rows are disclosed and excluded from metrics. Defaults
        to validation, falling back to test when no validation partition exists.
        """
        return activelearning_ops.evaluate_active_learning_op(
            self,
            partition=partition,
            unlabeled_marker=unlabeled_marker,
        )

    @property
    def activelearning_plan(self) -> ActiveLearningPlan | None:
        """Last fitted :class:`~buildml.activelearning.results.ActiveLearningPlan`."""
        return self._activelearning_plan

    @property
    def activelearning_fit_result(self) -> ActiveLearningFitResult | None:
        """Last :class:`~buildml.activelearning.results.ActiveLearningFitResult`."""
        return self._activelearning_fit_result

    @property
    def activelearning_query_result(self) -> ActiveLearningQueryResult | None:
        """Last :class:`~buildml.activelearning.results.ActiveLearningQueryResult`."""
        return self._activelearning_query_result

    @property
    def activelearning_label_result(self) -> ActiveLearningLabelResult | None:
        """Last :class:`~buildml.activelearning.results.ActiveLearningLabelResult`."""
        return self._activelearning_label_result

    @property
    def activelearning_eval_result(self) -> ActiveLearningEvalResult | None:
        """Last :class:`~buildml.activelearning.results.ActiveLearningEvalResult`."""
        return self._activelearning_eval_result

    def save_active_learning_bundle(self, path: str | Path) -> Path:
        """Persist the active plan as ``buildml.activelearning_bundle.v1``.

        See :data:`buildml.activelearning.checkpoint.CHECKPOINT_BOUNDARY`."""
        return activelearning_ops.save_active_learning_bundle_op(self, path=path)

    def load_active_learning_bundle(self, path: str | Path) -> Session:
        """Load an active-learning bundle into this Session."""
        return activelearning_ops.load_active_learning_bundle_op(self, path=path)

    def fit_online(
        self,
        *,
        backend: OnlineBackend | None = None,
        estimator: OnlineEstimator | str = "sgd_classifier",
        task: OnlineTask | None = None,
        columns: list[str] | None = None,
        random_state: int | None = 0,
        chunk_size: int = 50,
        n_init: int | None = None,
        indices: list[Any] | tuple[Any, ...] | None = None,
        classes: list[Any] | tuple[Any, ...] | None = None,
        prefer_reduce_components: bool = True,
        allow_refit_fallback: bool = False,
        drift_disclose: bool = True,
        drift_detector: OnlineDriftDetector | None = None,
        buffer_size: int = 512,
        epochs_per_update: int = 5,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        ewc_lambda: float = 100.0,
        hidden_dim: int = 64,
        device: str = "cpu",
    ) -> OnlineFitResult:
        """Warm-start an incremental ``partial_fit`` estimator on a train chunk.

        Parameters
        ----------
        backend:
            ``sklearn`` (default partial_fit family), ``industry`` (River +
            ``buildml[online-industry]``), or ``torch`` (replay/EWC continual MLP +
            ``buildml[torch]``).
        estimator:
            Backend-specific estimator name (see ``online_capability_matrix()``).
        classes:
            Full class vocabulary for classifiers. When omitted, discovered from
            the full train target column (labels only).
        allow_refit_fallback:
            If ``True``, estimators without ``partial_fit`` may full-refit on
            cumulative seen rows with an explicit disclosure. Default ``False``
            refuses silent full refits.

        Notes
        -----
        **Leakage:** Requires a split. Init uses a train chunk only.
        Validation/test are never used for updates. Honesty: batch/stream-chunk
        Session updates — not a distributed streaming platform.
        """
        return online_ops.fit_online_op(
            self,
            backend=backend,
            estimator=estimator,
            task=task,
            columns=columns,
            random_state=random_state,
            chunk_size=chunk_size,
            n_init=n_init,
            indices=indices,
            classes=classes,
            prefer_reduce_components=prefer_reduce_components,
            allow_refit_fallback=allow_refit_fallback,
            drift_disclose=drift_disclose,
            drift_detector=drift_detector,
            buffer_size=buffer_size,
            epochs_per_update=epochs_per_update,
            batch_size=batch_size,
            learning_rate=learning_rate,
            ewc_lambda=ewc_lambda,
            hidden_dim=hidden_dim,
            device=device,
        )

    def partial_fit_online(
        self,
        *,
        n_rows: int | None = None,
        indices: list[Any] | tuple[Any, ...] | None = None,
        frame: pd.DataFrame | None = None,
    ) -> OnlineUpdateResult:
        """Incremental ``partial_fit`` update on the next train chunk or frame.

        Provide at most one of ``indices`` or ``frame``. Default advances the
        train cursor by ``n_rows`` (or the plan ``chunk_size``). Refuses
        validation/test indices.
        """
        return online_ops.partial_fit_online_op(
            self,
            n_rows=n_rows,
            indices=indices,
            frame=frame,
        )

    def evaluate_online(
        self,
        *,
        partition: PartitionName | Literal["all"] = "validation",
        drift_check: bool = True,
    ) -> OnlineEvalResult:
        """Evaluate the online learner on a holdout partition (never for updates).

        Defaults to validation, falling back to test when no validation
        partition exists. ``drift_check`` surfaces River ADWIN/Page-Hinkley or
        mean-shift disclosure without updating the model.
        """
        return online_ops.evaluate_online_op(
            self, partition=partition, drift_check=drift_check
        )

    def predict_online(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
    ) -> OnlinePredictResult:
        """Predict with the incremental online estimator (no update)."""
        return online_ops.predict_online_op(self, partition=partition)

    @property
    def online_plan(self) -> OnlinePlan | None:
        """Last fitted :class:`~buildml.online.results.OnlinePlan`."""
        return self._online_plan

    @property
    def online_fit_result(self) -> OnlineFitResult | None:
        """Last :class:`~buildml.online.results.OnlineFitResult`."""
        return self._online_fit_result

    @property
    def online_update_result(self) -> OnlineUpdateResult | None:
        """Last :class:`~buildml.online.results.OnlineUpdateResult`."""
        return self._online_update_result

    @property
    def online_eval_result(self) -> OnlineEvalResult | None:
        """Last :class:`~buildml.online.results.OnlineEvalResult`."""
        return self._online_eval_result

    @property
    def online_predict_result(self) -> OnlinePredictResult | None:
        """Last :class:`~buildml.online.results.OnlinePredictResult`."""
        return self._online_predict_result

    def save_online_bundle(self, path: str | Path) -> Path:
        """Persist the active plan as ``buildml.online_bundle.v1``.

        See :data:`buildml.online.checkpoint.CHECKPOINT_BOUNDARY`."""
        return online_ops.save_online_bundle_op(self, path=path)

    def load_online_bundle(self, path: str | Path) -> Session:
        """Load an online-learning bundle into this Session."""
        return online_ops.load_online_bundle_op(self, path=path)

    def fit_multitask(
        self,
        *,
        backend: MultiTaskBackend | None = None,
        method: MultiTaskMethod = "multi_output",
        task: MultiTaskTask = "auto",
        targets: list[str] | tuple[str, ...] | None = None,
        columns: list[str] | None = None,
        base_estimator: MultiTaskBaseEstimator | str = "logistic_regression",
        random_state: int | None = 0,
        order: list[str] | tuple[str, ...] | None = None,
        prefer_reduce_components: bool = True,
        prediction_prefix: str = "multitask_pred",
        epochs: int = 60,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        device: str = "cpu",
    ) -> MultiTaskFitResult:
        """Fit a multi-target estimator on train only.

        Parameters
        ----------
        backend:
            ``sklearn`` (default when no extras), ``industry`` (XGB/LGBM/CatBoost
            multi-target), or ``torch`` (shared-trunk multi-head). See
            :func:`buildml.multitask.multitask_capability_matrix`.
        method:
            Algorithm within the backend — e.g. ``multi_output``,
            ``multi_output_xgb``, ``shared_trunk_multihead``.
        task:
            ``classification``, ``regression``, ``auto`` (infers; mixed kinds
            allowed only on torch), or ``mixed`` (torch only).
        targets:
            Optional explicit target columns. When omitted, all
            ``role='target'`` columns are used (requires ``>= 2``).
        base_estimator:
            Sklearn backend only — classification: ``logistic_regression``,
            ``hist_gradient_boosting``; regression: ``ridge``,
            ``hist_gradient_boosting_regressor``.

        Notes
        -----
        **Leakage:** Requires a split. Fit uses train only. Validation/test are
        never used for fitting. Classical :meth:`fit` remains single-target via
        ``require_target()``.
        """
        return multitask_ops.fit_multitask_op(
            self,
            backend=backend,
            method=method,
            task=task,
            targets=targets,
            columns=columns,
            base_estimator=base_estimator,
            random_state=random_state,
            order=order,
            prefer_reduce_components=prefer_reduce_components,
            prediction_prefix=prediction_prefix,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
        )

    def predict_multitask(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        attach: bool = False,
        prediction_prefix: str | None = None,
    ) -> MultiTaskPredictResult:
        """Predict per-task outputs with the frozen multi-task plan (no refit).

        ``attach=True`` requires ``partition='all'`` and writes
        ``{prediction_prefix}_{target}`` feature columns.
        """
        return multitask_ops.predict_multitask_op(
            self,
            partition=partition,
            attach=attach,
            prediction_prefix=prediction_prefix,
        )

    def evaluate_multitask(
        self,
        *,
        partition: PartitionName | Literal["all"] = "validation",
    ) -> MultiTaskEvalResult:
        """Evaluate multi-task predictions with per-task and aggregate metrics.

        Defaults to validation, falling back to test when no validation
        partition exists. Holdout rows are never used for fitting.
        """
        return multitask_ops.evaluate_multitask_op(self, partition=partition)

    @property
    def multitask_plan(self) -> MultiTaskPlan | None:
        """Last fitted :class:`~buildml.multitask.results.MultiTaskPlan`."""
        return self._multitask_plan

    @property
    def multitask_fit_result(self) -> MultiTaskFitResult | None:
        """Last :class:`~buildml.multitask.results.MultiTaskFitResult`."""
        return self._multitask_fit_result

    @property
    def multitask_predict_result(self) -> MultiTaskPredictResult | None:
        """Last :class:`~buildml.multitask.results.MultiTaskPredictResult`."""
        return self._multitask_predict_result

    @property
    def multitask_eval_result(self) -> MultiTaskEvalResult | None:
        """Last :class:`~buildml.multitask.results.MultiTaskEvalResult`."""
        return self._multitask_eval_result

    def save_multitask_bundle(self, path: str | Path) -> Path:
        """Persist the active plan as ``buildml.multitask_bundle.v1``.

        See :data:`buildml.multitask.checkpoint.CHECKPOINT_BOUNDARY`."""
        return multitask_ops.save_multitask_bundle_op(self, path=path)

    def load_multitask_bundle(self, path: str | Path) -> Session:
        """Load a multi-task bundle into this Session."""
        return multitask_ops.load_multitask_bundle_op(self, path=path)

    def fit_metalearning(
        self,
        *,
        backend: str | None = None,
        method: MetaLearningMethod = "prototypical",
        task_column: str | None = None,
        columns: list[str] | None = None,
        n_way: int | None = None,
        k_shot: int = 5,
        n_query: int = 10,
        n_episodes: int = 20,
        base_estimator: MetaLearningBaseEstimator | str = "logistic_regression",
        random_state: int | None = 0,
        prefer_reduce_components: bool = True,
        task_holdout_fraction: float = 0.25,
        meta_epochs: int = 40,
        inner_lr: float = 0.05,
        inner_steps: int = 5,
        meta_lr: float = 1e-3,
        embed_dim: int = 32,
        hidden_dim: int = 64,
        device: str = "cpu",
    ) -> MetaLearningFitResult:
        """Meta-train a tabular few-shot / episodic learner on train tasks only.

        Parameters
        ----------
        backend:
            ``sklearn`` (default), ``torch`` (``buildml[torch]``), or
            ``industry`` (``buildml[metalearning-industry,torch]``).
        method:
            ``prototypical``, ``warm_start``, ``prototypical_torch``,
            ``maml``, or ``reptile`` depending on backend.
        task_column:
            Episodic task id column. When omitted, the single
            ``role='group'`` column is used.
        k_shot / n_query / n_episodes:
            Episodic protocol knobs for meta-train disclosure metrics.
        task_holdout_fraction:
            Fraction of train task ids held out internally when enough
            tasks exist (``>= 3``).

        Notes
        -----
        **Leakage:** Requires a split. Meta-train uses train only.
        Validation/test are never used for meta-training. Needs exactly one
        ``role='target'`` and a task/group column. Honesty: practical tabular
        few-shot / episodic Session protocol — not foundation-model
        meta-learning or MAML-at-scale.
        """
        return metalearning_ops.fit_metalearning_op(
            self,
            backend=backend,
            method=method,
            task_column=task_column,
            columns=columns,
            n_way=n_way,
            k_shot=k_shot,
            n_query=n_query,
            n_episodes=n_episodes,
            base_estimator=base_estimator,
            random_state=random_state,
            prefer_reduce_components=prefer_reduce_components,
            task_holdout_fraction=task_holdout_fraction,
            meta_epochs=meta_epochs,
            inner_lr=inner_lr,
            inner_steps=inner_steps,
            meta_lr=meta_lr,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            device=device,
        )

    def adapt_to_task(
        self,
        *,
        task_id: Any | None = None,
        partition: PartitionName = "train",
        support_frame: Any | None = None,
        max_support_per_class: int | None = None,
        random_state: int | None = 0,
    ) -> MetaAdaptResult:
        """Fast-adapt the meta-learner to one task's labeled support set.

        Provide ``task_id`` (rows pulled from ``partition``) or an explicit
        ``support_frame``. Does not refit the global meta-train plan.
        """
        return metalearning_ops.adapt_to_task_op(
            self,
            task_id=task_id,
            partition=partition,
            support_frame=support_frame,
            max_support_per_class=max_support_per_class,
            random_state=random_state,
        )

    def evaluate_metalearning(
        self,
        *,
        partition: PartitionName | Literal["all"] = "validation",
        k_shot: int | None = None,
        n_query: int | None = None,
        n_way: int | None = None,
        prefer_novel_tasks: bool = True,
        random_state: int | None = 0,
    ) -> MetaLearningEvalResult:
        """Evaluate episodic few-shot performance on a holdout partition.

        Prefers novel task ids absent from meta-train. Defaults to
        validation, falling back to test when no validation partition
        exists. Holdout rows are never used for meta-training.
        """
        return metalearning_ops.evaluate_metalearning_op(
            self,
            partition=partition,
            k_shot=k_shot,
            n_query=n_query,
            n_way=n_way,
            prefer_novel_tasks=prefer_novel_tasks,
            random_state=random_state,
        )

    @property
    def metalearning_plan(self) -> MetaLearningPlan | None:
        """Last fitted :class:`~buildml.metalearning.results.MetaLearningPlan`."""
        return self._metalearning_plan

    @property
    def metalearning_fit_result(self) -> MetaLearningFitResult | None:
        """Last :class:`~buildml.metalearning.results.MetaLearningFitResult`."""
        return self._metalearning_fit_result

    @property
    def metalearning_adapt_result(self) -> MetaAdaptResult | None:
        """Last :class:`~buildml.metalearning.results.MetaAdaptResult`."""
        return self._metalearning_adapt_result

    @property
    def metalearning_eval_result(self) -> MetaLearningEvalResult | None:
        """Last :class:`~buildml.metalearning.results.MetaLearningEvalResult`."""
        return self._metalearning_eval_result

    def save_metalearning_bundle(self, path: str | Path) -> Path:
        """Persist the active plan as ``buildml.metalearning_bundle.v1``.

        See :data:`buildml.metalearning.checkpoint.CHECKPOINT_BOUNDARY`."""
        return metalearning_ops.save_metalearning_bundle_op(self, path=path)

    def load_metalearning_bundle(self, path: str | Path) -> Session:
        """Load a meta-learning bundle into this Session."""
        return metalearning_ops.load_metalearning_bundle_op(self, path=path)

    def fit_federated(
        self,
        *,
        backend: FederatedBackend | None = None,
        method: FederatedMethod = "fedavg",
        estimator: FederatedEstimator = "sgd_classifier",
        task: FederatedTask | None = None,
        client_column: str | None = None,
        columns: list[str] | None = None,
        n_rounds: int = 5,
        local_epochs: int = 1,
        client_fraction: float = 1.0,
        mu: float = 0.0,
        random_state: int | None = 0,
        prefer_reduce_components: bool = True,
        min_client_rows: int = 2,
    ) -> FederatedFitResult:
        """Simulate federated averaging on Session train clients.

        Parameters
        ----------
        backend:
            ``native`` (default in-process FedAvg/FedProx) or ``flower``
            (``buildml[federated-industry]`` NumPyClient + flwr aggregation).
            When omitted and ``flwr`` is installed, defaults to ``flower``.
        method:
            ``fedavg`` (weighted coefficient averaging) or ``fedprox``
            (FedAvg + proximal pull; requires ``mu > 0``).
        estimator:
            Linear / SGD family supporting ``coef_`` / ``intercept_``
            aggregation (``sgd_classifier``, ``sgd_regressor``,
            ``logistic_regression``, ``ridge``, ``linear_regression``).
        client_column:
            Client id column. When omitted, the single ``role='group'``
            column is used.
        n_rounds / local_epochs / client_fraction:
            Federation schedule knobs.
        mu:
            FedProx proximal strength (required ``> 0`` when
            ``method='fedprox'``).

        Notes
        -----
        **Leakage:** Requires a split. Local client updates use train only.
        Validation/test are never used for training. Needs exactly one
        ``role='target'`` and a client/group column. Honesty: local
        FedAvg-style simulation on partitioned Session data — not a
        networked FL deployment unless you operate one separately; not
        cryptographic secure aggregation.
        """
        return federated_ops.fit_federated_op(
            self,
            backend=backend,
            method=method,
            estimator=estimator,
            task=task,
            client_column=client_column,
            columns=columns,
            n_rounds=n_rounds,
            local_epochs=local_epochs,
            client_fraction=client_fraction,
            mu=mu,
            random_state=random_state,
            prefer_reduce_components=prefer_reduce_components,
            min_client_rows=min_client_rows,
        )

    def evaluate_federated(
        self,
        *,
        backend: FederatedBackend | None = None,
        partition: PartitionName | Literal["all"] = "validation",
        per_client: bool = True,
    ) -> FederatedEvalResult:
        """Evaluate the global federated model on a holdout partition.

        Defaults to validation, falling back to test when no validation
        partition exists. Holdout rows are never used for local updates.
        Optional ``backend=`` validates consistency with the fitted plan.
        """
        return federated_ops.evaluate_federated_op(
            self,
            backend=backend,
            partition=partition,
            per_client=per_client,
        )

    def predict_federated(
        self,
        *,
        backend: FederatedBackend | None = None,
        partition: PartitionName | Literal["all"] = "test",
    ) -> FederatedPredictResult:
        """Predict with the global federated model (no update).

        Optional ``backend=`` validates consistency with the fitted plan.
        """
        return federated_ops.predict_federated_op(
            self,
            backend=backend,
            partition=partition,
        )

    @property
    def federated_plan(self) -> FederatedPlan | None:
        """Last fitted :class:`~buildml.federated.results.FederatedPlan`."""
        return self._federated_plan

    @property
    def federated_fit_result(self) -> FederatedFitResult | None:
        """Last :class:`~buildml.federated.results.FederatedFitResult`."""
        return self._federated_fit_result

    @property
    def federated_eval_result(self) -> FederatedEvalResult | None:
        """Last :class:`~buildml.federated.results.FederatedEvalResult`."""
        return self._federated_eval_result

    @property
    def federated_predict_result(self) -> FederatedPredictResult | None:
        """Last :class:`~buildml.federated.results.FederatedPredictResult`."""
        return self._federated_predict_result

    def save_federated_bundle(self, path: str | Path) -> Path:
        """Persist the active plan as ``buildml.federated_bundle.v1``.

        See :data:`buildml.federated.checkpoint.CHECKPOINT_BOUNDARY`."""
        return federated_ops.save_federated_bundle_op(self, path=path)

    def load_federated_bundle(self, path: str | Path) -> Session:
        """Load a federated-learning bundle into this Session."""
        return federated_ops.load_federated_bundle_op(self, path=path)

    def fit_probabilistic(
        self,
        *,
        backend: str | None = None,
        estimator: ProbabilisticEstimator = "bayesian_ridge",
        task: ProbabilisticTask | None = None,
        columns: list[str] | None = None,
        random_state: int | None = 0,
        alpha: float = 0.1,
        conformal: bool = True,
        conformal_calibration_fraction: float = 0.2,
        interval_method: IntervalMethod | None = None,
        prefer_reduce_components: bool = True,
        n_restarts_optimizer: int = 0,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
    ) -> ProbabilisticFitResult:
        """Fit a Bayesian / probabilistic estimator with uncertainty.

        Parameters
        ----------
        backend:
            ``native`` (sklearn + in-tree conformal), ``mapie``, or ``ngboost``
            when ``buildml[probabilistic-industry]`` is installed.
        estimator:
            Native: ``bayesian_ridge``, ``gaussian_process_*``, ``gaussian_nb``.
            MAPIE: ``split``, ``cv_plus``, ``jackknife_plus``.
            NGBoost: ``ngboost_regressor``, ``ngboost_classifier``.
        alpha:
            Miscoverage level for intervals / prediction sets (default 0.1 →
            nominal 90% coverage).
        conformal:
            When True, carve a split-conformal calibration subset from the
            Session **train** partition only (native/ngboost; MAPIE owns conformal).
        interval_method:
            ``posterior_std``, ``split_conformal``, ``both``, or ``none``.
            Inferred when omitted.

        Notes
        -----
        **Leakage:** Requires a split. Fit and conformal calibration use train
        only. Holdout is for ``evaluate_probabilistic`` / ``predict_interval``.
        Honesty: tabular uncertainty quantification — **not** PyMC/Stan MCMC or
        Bayesian deep nets. Classical :meth:`calibration` remains unchanged.
        """
        return probabilistic_ops.fit_probabilistic_op(
            self,
            backend=backend,
            estimator=estimator,
            task=task,
            columns=columns,
            random_state=random_state,
            alpha=alpha,
            conformal=conformal,
            conformal_calibration_fraction=conformal_calibration_fraction,
            interval_method=interval_method,
            prefer_reduce_components=prefer_reduce_components,
            n_restarts_optimizer=n_restarts_optimizer,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
        )

    def evaluate_probabilistic(
        self,
        *,
        partition: PartitionName | Literal["all"] = "validation",
        alpha: float | None = None,
    ) -> ProbabilisticEvalResult:
        """Evaluate probabilistic predictions with proper scoring rules.

        Defaults to validation, falling back to test when no validation
        partition exists. Reports NLL / coverage / Brier (as applicable).
        Holdout rows are never used for fit or conformal calibration.
        """
        return probabilistic_ops.evaluate_probabilistic_op(
            self,
            partition=partition,
            alpha=alpha,
        )

    def predict_probabilistic(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        return_std: bool = True,
        return_proba: bool = True,
    ) -> ProbabilisticPredictResult:
        """Point predictions (optional posterior std / class probabilities)."""
        return probabilistic_ops.predict_probabilistic_op(
            self,
            partition=partition,
            return_std=return_std,
            return_proba=return_proba,
        )

    def predict_interval(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        alpha: float | None = None,
        method: str | None = None,
    ) -> ProbabilisticIntervalResult:
        """Predictive intervals (regression) or conformal prediction sets."""
        return probabilistic_ops.predict_interval_op(
            self,
            partition=partition,
            alpha=alpha,
            method=method,
        )

    @property
    def probabilistic_plan(self) -> ProbabilisticPlan | None:
        """Last fitted :class:`~buildml.probabilistic.results.ProbabilisticPlan`."""
        return self._probabilistic_plan

    @property
    def probabilistic_fit_result(self) -> ProbabilisticFitResult | None:
        """Last :class:`~buildml.probabilistic.results.ProbabilisticFitResult`."""
        return self._probabilistic_fit_result

    @property
    def probabilistic_eval_result(self) -> ProbabilisticEvalResult | None:
        """Last :class:`~buildml.probabilistic.results.ProbabilisticEvalResult`."""
        return self._probabilistic_eval_result

    @property
    def probabilistic_predict_result(self) -> ProbabilisticPredictResult | None:
        """Last :class:`~buildml.probabilistic.results.ProbabilisticPredictResult`."""
        return self._probabilistic_predict_result

    @property
    def probabilistic_interval_result(self) -> ProbabilisticIntervalResult | None:
        """Last :class:`~buildml.probabilistic.results.ProbabilisticIntervalResult`."""
        return self._probabilistic_interval_result

    def save_probabilistic_bundle(self, path: str | Path) -> Path:
        """Persist the active plan as ``buildml.probabilistic_bundle.v1``.

        See :data:`buildml.probabilistic.checkpoint.CHECKPOINT_BOUNDARY`."""
        return probabilistic_ops.save_probabilistic_bundle_op(self, path=path)

    def load_probabilistic_bundle(self, path: str | Path) -> Session:
        """Load a probabilistic bundle into this Session."""
        return probabilistic_ops.load_probabilistic_bundle_op(self, path=path)

    def declare_causal_assumptions(
        self,
        *,
        treatment: str,
        outcome: str,
        confounders: Sequence[str] | None,
        estimand: str = "ATE",
        identification: str = "backdoor",
        instruments: Sequence[str] | None = None,
        acknowledge_unconfoundedness: bool = False,
        acknowledge_positivity: bool = False,
        allow_empty_confounders: bool = False,
    ) -> CausalAssumptions:
        """Declare identification assumptions required for causal estimation.

        Causal APIs refuse estimation until treatment, outcome, confounders
        (or an explicit empty-confounder waiver), estimand, and the
        unconfoundedness / positivity acknowledgements are set. EDA /
        association / feature-importance paths never satisfy these fields.
        """
        return causal_ops.declare_causal_assumptions_op(
            self,
            treatment=treatment,
            outcome=outcome,
            confounders=confounders,
            estimand=estimand,
            identification=identification,
            instruments=instruments,
            acknowledge_unconfoundedness=acknowledge_unconfoundedness,
            acknowledge_positivity=acknowledge_positivity,
            allow_empty_confounders=allow_empty_confounders,
        )

    def fit_causal(
        self,
        *,
        backend: CausalBackend | None = None,
        method: CausalMethod = "aipw",
        assumptions: CausalAssumptions | dict[str, Any] | None = None,
        bootstrap_samples: int = 200,
        random_state: int | None = 0,
        clip_propensity: tuple[float, float] = (0.01, 0.99),
        outcome_model: str = "ridge",
        propensity_model: str = "logistic_regression",
    ) -> CausalFitResult:
        """Fit causal models (train-only) and estimate backdoor ATE.

        Parameters
        ----------
        backend:
            ``native`` (default), ``dowhy``, or ``econml``. Industry backends
            require ``buildml[causal-industry]``.
        method:
            Native: ``t_learner``, ``ipw``, ``aipw``. DoWhy: ``backdoor_linear``,
            ``backdoor_propensity_score``, ``backdoor_propensity_weighting``.
            EconML: ``dml``, ``causal_forest``, ``policy_tree``.
        assumptions:
            Optional explicit :class:`CausalAssumptions` / mapping. When
            omitted, uses the object from :meth:`declare_causal_assumptions`.
        bootstrap_samples:
            Full retrain bootstrap on train for uncertainty (0 disables;
            native and econml; DoWhy uses estimator CIs).

        Notes
        -----
        **Leakage:** Requires a split. Nuisances fit on train only.
        **Assumptions:** Refuses without validated CausalAssumptions.
        Not causal discovery. EDA remains associational.
        """
        return causal_ops.fit_causal_op(
            self,
            backend=backend,
            method=method,
            assumptions=assumptions,
            bootstrap_samples=bootstrap_samples,
            random_state=random_state,
            clip_propensity=clip_propensity,
            outcome_model=outcome_model,
            propensity_model=propensity_model,
        )

    def estimate_causal(
        self,
        *,
        partition: PartitionName | Literal["all"] = "train",
        bootstrap_samples: int | None = None,
        random_state: int | None = None,
    ) -> CausalEstimateResult:
        """Estimate ATE on a partition with fitted train nuisances."""
        return causal_ops.estimate_causal_op(
            self,
            partition=partition,
            bootstrap_samples=bootstrap_samples,
            random_state=random_state,
        )

    def evaluate_causal(
        self,
        *,
        partition: PartitionName | Literal["all"] = "validation",
        bootstrap_samples: int | None = None,
    ) -> CausalEvalResult:
        """Holdout nuisance predictive checks + ATE (not proof of identification)."""
        return causal_ops.evaluate_causal_op(
            self,
            partition=partition,
            bootstrap_samples=bootstrap_samples,
        )

    def refute_causal(
        self,
        *,
        kind: CausalRefuteKind = "placebo_treatment",
        random_state: int | None = 0,
    ) -> CausalRefuteResult:
        """Simple placebo / random-confounder sensitivity disclosure.

        Not a full DoWhy refutation suite.
        """
        return causal_ops.refute_causal_op(
            self,
            kind=kind,
            random_state=random_state,
        )

    @property
    def causal_assumptions(self) -> CausalAssumptions | None:
        """Last declared :class:`~buildml.causal.types.CausalAssumptions`."""
        return self._causal_assumptions

    @property
    def causal_plan(self) -> CausalPlan | None:
        """Last fitted :class:`~buildml.causal.results.CausalPlan`."""
        return self._causal_plan

    @property
    def causal_fit_result(self) -> CausalFitResult | None:
        """Last :class:`~buildml.causal.results.CausalFitResult`."""
        return self._causal_fit_result

    @property
    def causal_estimate_result(self) -> CausalEstimateResult | None:
        """Last :class:`~buildml.causal.results.CausalEstimateResult`."""
        return self._causal_estimate_result

    @property
    def causal_eval_result(self) -> CausalEvalResult | None:
        """Last :class:`~buildml.causal.results.CausalEvalResult`."""
        return self._causal_eval_result

    @property
    def causal_refute_result(self) -> CausalRefuteResult | None:
        """Last :class:`~buildml.causal.results.CausalRefuteResult`."""
        return self._causal_refute_result

    def save_causal_bundle(self, path: str | Path) -> Path:
        """Persist the active plan as ``buildml.causal_bundle.v1``.

        See :data:`buildml.causal.checkpoint.CHECKPOINT_BOUNDARY`."""
        return causal_ops.save_causal_bundle_op(self, path=path)

    def load_causal_bundle(self, path: str | Path) -> Session:
        """Load a causal bundle into this Session."""
        return causal_ops.load_causal_bundle_op(self, path=path)

    def set_graph(
        self,
        edges: Any,
        *,
        source_col: str = "source",
        target_col: str = "target",
        node_id_col: str = "node_id",
        directed: bool = False,
    ) -> GraphSpec:
        """Attach an edge list; Session rows are nodes.

        Conventions: one dataset row per node; ``node_id_col`` uniquely
        identifies nodes and must match edge endpoints. ``Session.split``
        creates **node** partitions. Not a Neo4j/KG product surface.

        Parameters
        ----------
        edges:
            Edge list as a DataFrame with ``source_col``/``target_col`` or a
            sequence of ``(source, target)`` pairs.
        """
        return graph_ops.set_graph_op(
            self,
            edges,
            source_col=source_col,
            target_col=target_col,
            node_id_col=node_id_col,
            directed=directed,
        )

    def fit_graph(
        self,
        *,
        method: GraphMethod = "classical",
        task: GraphTask = "node_classification",
        mode: GraphMode = "inductive",
        columns: Sequence[str] | None = None,
        classical_estimator: ClassicalEstimator = "logistic_regression",
        hidden_dim: int = 32,
        n_layers: int = 2,
        epochs: int = 80,
        learning_rate: float = 0.01,
        weight_decay: float = 5e-4,
        dropout: float = 0.1,
        random_state: int | None = 0,
        include_graph_metrics: bool = True,
        pyg_model: PyGModel = "gcn",
        heads: int = 4,
    ) -> GraphFitResult:
        """Fit graph node classification (classical, pure-Torch GCN, or PyG).

        **Leakage:** Requires a split and :meth:`set_graph`. Train labels only.
        Default ``mode='inductive'`` fits on the train-induced subgraph;
        ``transductive`` uses full topology with train-label-only supervision
        (disclosed). Classical path needs ``buildml[graph]`` (NetworkX);
        GCN needs ``buildml[torch]``; PyG needs ``buildml[graph-pyg]``
        with ``pyg_model`` in ``gcn`` / ``graphsage`` / ``gat``.
        """
        return graph_ops.fit_graph_op(
            self,
            method=method,
            task=task,
            mode=mode,
            columns=columns,
            classical_estimator=classical_estimator,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            dropout=dropout,
            random_state=random_state,
            include_graph_metrics=include_graph_metrics,
            pyg_model=pyg_model,
            heads=heads,
        )

    def predict_graph(
        self,
        *,
        partition: PartitionName | Literal["all"] = "validation",
    ) -> GraphPredictResult:
        """Predict node labels with the fitted :class:`GraphPlan`."""
        return graph_ops.predict_graph_op(self, partition=partition)

    def evaluate_graph(
        self,
        *,
        partition: PartitionName | Literal["all"] = "validation",
    ) -> GraphEvalResult:
        """Evaluate node classification on a holdout partition."""
        return graph_ops.evaluate_graph_op(self, partition=partition)

    @property
    def graph_spec(self) -> GraphSpec | None:
        """Last attached :class:`~buildml.graph.types.GraphSpec`."""
        return self._graph_spec

    @property
    def graph_plan(self) -> GraphPlan | None:
        """Last fitted :class:`~buildml.graph.results.GraphPlan`."""
        return self._graph_plan

    @property
    def graph_fit_result(self) -> GraphFitResult | None:
        """Last :class:`~buildml.graph.results.GraphFitResult`."""
        return self._graph_fit_result

    @property
    def graph_predict_result(self) -> GraphPredictResult | None:
        """Last :class:`~buildml.graph.results.GraphPredictResult`."""
        return self._graph_predict_result

    @property
    def graph_eval_result(self) -> GraphEvalResult | None:
        """Last :class:`~buildml.graph.results.GraphEvalResult`."""
        return self._graph_eval_result

    def save_graph_bundle(self, path: str | Path) -> Path:
        """Persist the active plan as ``buildml.graph_bundle.v1``.

        See :data:`buildml.graph.checkpoint.CHECKPOINT_BOUNDARY`."""
        return graph_ops.save_graph_bundle_op(self, path=path)

    def load_graph_bundle(self, path: str | Path) -> Session:
        """Load a graph bundle into this Session."""
        return graph_ops.load_graph_bundle_op(self, path=path)

    def fit_symbolic(
        self,
        *,
        backend: SymbolicBackend | None = None,
        source: SymbolicSource = "decision_tree",
        method: IndustrySymbolicMethod | None = None,
        task: SymbolicTask | None = None,
        rules: Sequence[Mapping[str, Any] | Rule] | None = None,
        columns: list[str] | None = None,
        random_state: int | None = 0,
        max_depth: int = 4,
        min_samples_leaf: int = 5,
        max_rules: int = 32,
        default_consequent: Any = None,
        prefer_reduce_components: bool = True,
        verify_constraints: bool = False,
    ) -> SymbolicFitResult:
        """Compile or induce a symbolic if-then rule base on train.

        Parameters
        ----------
        backend:
            ``sklearn`` (core tree/list/declared) or ``industry`` (skope-rules /
            imodels when ``buildml[symbolic-industry]`` is installed). Defaults
            to industry when installed, else sklearn.
        source:
            ``declared`` (expert rules via ``rules=``), ``decision_tree``
            (sklearn path export), or ``decision_list`` (sequential covering).
            Used when ``backend='sklearn'``.
        method:
            ``skope_rules``, ``rulefit``, or ``boosted_rules`` when
            ``backend='industry'``.
        verify_constraints:
            When True and z3-solver is installed, run a lite SAT check on hard
            constraint rules (not a full SMT product).

        Notes
        -----
        **Leakage:** Requires a split. Induction uses Session train only.
        Honesty: structured tabular rules with explanation traces — **not**
        an AGI symbolic reasoner, Prolog engine, or full Z3 SMT product.
        """
        return symbolic_ops.fit_symbolic_op(
            self,
            backend=backend,
            source=source,
            method=method,
            task=task,
            rules=rules,
            columns=columns,
            random_state=random_state,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_rules=max_rules,
            default_consequent=default_consequent,
            prefer_reduce_components=prefer_reduce_components,
            verify_constraints=verify_constraints,
        )

    def evaluate_symbolic(
        self,
        *,
        partition: PartitionName | Literal["all"] = "validation",
    ) -> SymbolicEvalResult:
        """Evaluate the symbolic rule base on a holdout partition."""
        return symbolic_ops.evaluate_symbolic_op(self, partition=partition)

    def predict_symbolic(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        return_traces: bool = True,
    ) -> SymbolicPredictResult:
        """Predict with rule-firing explanation traces (no update)."""
        return symbolic_ops.predict_symbolic_op(
            self,
            partition=partition,
            return_traces=return_traces,
        )

    def fit_neuro_symbolic(
        self,
        *,
        backend: NeuroSymbolicBackend | None = None,
        mode: NeuroSymbolicMode = "constraint_overlay",
        base_estimator: BaseEstimatorName = "logistic_regression",
        torch_method: str | None = None,
        task: SymbolicTask | None = None,
        rules: Sequence[Mapping[str, Any] | Rule] | None = None,
        rule_source: SymbolicSource = "decision_tree",
        columns: list[str] | None = None,
        random_state: int | None = 0,
        soft_strength: float = 0.5,
        max_depth: int = 3,
        min_samples_leaf: int = 5,
        max_rules: int = 24,
        prefer_reduce_components: bool = True,
        torch_epochs: int = 60,
        device: str = "cpu",
    ) -> NeuroSymbolicFitResult:
        """Fit a base model + symbolic hybrid in one Session API.

        Backends: ``sklearn`` (core) or ``torch`` (concept-bottleneck / NAM lite
        when ``buildml[torch]`` is installed). Modes: ``constraint_overlay``,
        ``rules_as_features``, ``constraint_repair``. Rules may be declared or
        train-induced (``rule_source``). Train-only for learning; holdout for
        eval/predict.
        """
        return symbolic_ops.fit_neuro_symbolic_op(
            self,
            backend=backend,
            mode=mode,
            base_estimator=base_estimator,
            torch_method=torch_method,
            task=task,
            rules=rules,
            rule_source=rule_source,
            columns=columns,
            random_state=random_state,
            soft_strength=soft_strength,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_rules=max_rules,
            prefer_reduce_components=prefer_reduce_components,
            torch_epochs=torch_epochs,
            device=device,
        )

    def evaluate_neuro_symbolic(
        self,
        *,
        partition: PartitionName | Literal["all"] = "validation",
    ) -> SymbolicEvalResult:
        """Evaluate the neuro-symbolic hybrid on a holdout partition."""
        return symbolic_ops.evaluate_neuro_symbolic_op(self, partition=partition)

    def predict_neuro_symbolic(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        return_traces: bool = True,
    ) -> SymbolicPredictResult:
        """Hybrid predict with neural + rule traces (no update)."""
        return symbolic_ops.predict_neuro_symbolic_op(
            self,
            partition=partition,
            return_traces=return_traces,
        )

    @property
    def symbolic_plan(self) -> SymbolicPlan | None:
        """Last fitted :class:`~buildml.symbolic.results.SymbolicPlan`."""
        return self._symbolic_plan

    @property
    def neuro_symbolic_plan(self) -> NeuroSymbolicPlan | None:
        """Last fitted :class:`~buildml.symbolic.results.NeuroSymbolicPlan`."""
        return self._neuro_symbolic_plan

    @property
    def symbolic_fit_result(self) -> SymbolicFitResult | None:
        """Last :class:`~buildml.symbolic.results.SymbolicFitResult`."""
        return self._symbolic_fit_result

    @property
    def neuro_symbolic_fit_result(self) -> NeuroSymbolicFitResult | None:
        """Last :class:`~buildml.symbolic.results.NeuroSymbolicFitResult`."""
        return self._neuro_symbolic_fit_result

    @property
    def symbolic_eval_result(self) -> SymbolicEvalResult | None:
        """Last symbolic / neuro-symbolic :class:`SymbolicEvalResult`."""
        return self._symbolic_eval_result

    @property
    def symbolic_predict_result(self) -> SymbolicPredictResult | None:
        """Last pure-symbolic :class:`SymbolicPredictResult`."""
        return self._symbolic_predict_result

    @property
    def neuro_symbolic_predict_result(self) -> SymbolicPredictResult | None:
        """Last neuro-symbolic :class:`SymbolicPredictResult`."""
        return self._neuro_symbolic_predict_result

    def save_symbolic_bundle(self, path: str | Path) -> Path:
        """Persist the active plan as ``buildml.symbolic_bundle.v1``.

        Prefers ``NeuroSymbolicPlan`` when both are present. See
        :data:`buildml.symbolic.checkpoint.CHECKPOINT_BOUNDARY`.
        """
        return symbolic_ops.save_symbolic_bundle_op(self, path=path)

    def load_symbolic_bundle(self, path: str | Path) -> Session:
        """Load a symbolic / neuro-symbolic bundle into this Session."""
        return symbolic_ops.load_symbolic_bundle_op(self, path=path)

    @staticmethod
    def symbolic_capability_matrix() -> dict[str, Any]:
        """Honest capability matrix for symbolic / neuro-symbolic backends."""
        return symbolic_ops.symbolic_capability_matrix_op()

    @staticmethod
    def cbr_capability_matrix() -> dict[str, Any]:
        """Honest capability matrix for CBR retrieval backends."""
        from buildml.cbr.catalog import cbr_capability_matrix

        return cbr_capability_matrix()

    @staticmethod
    def ranking_capability_matrix() -> dict[str, Any]:
        """Honest capability matrix for tabular LTR backends and methods."""
        from buildml.ranking.catalog import ranking_capability_matrix

        return ranking_capability_matrix()

    def fit_cbr(
        self,
        *,
        backend: str | None = None,
        task: CbrTask | None = None,
        metric: CbrMetric = "euclidean",
        reuse: CbrReuseMode = "distance_weighted",
        adapt: CbrAdaptMode = "none",
        k: int = 5,
        columns: list[str] | None = None,
        categorical_columns: list[str] | None = None,
        text_columns: list[str] | None = None,
        text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        standardize: bool = True,
        distance_eps: float = 1e-8,
        random_state: int | None = 0,
        prefer_reduce_components: bool = True,
        torch_epochs: int = 40,
        device: str = "cpu",
    ) -> CbrFitResult:
        """Build a tabular case memory from Session train.

        Parameters
        ----------
        backend:
            ``sklearn`` (exact kNN fallback), ``industry`` (hnswlib/faiss ANN when
            ``buildml[cbr-industry]``), ``embedding`` (sentence-transformers text
            cases when ``buildml[rag|ssl]``), ``torch`` (learned metric encoder).
        metric:
            ``euclidean`` / ``manhattan`` / ``cosine`` (numeric) or ``mixed``
            (Gower-style numeric + categorical).
        reuse:
            Classification: ``majority`` / ``distance_weighted``.
            Regression: ``distance_weighted`` / ``local_mean`` / ``local_ridge``.

        Notes
        -----
        **Leakage:** Requires a split. Case base uses train only.
        Honesty: case→solution CBR for supervised-style tasks — **not** RAG
        (document retrieval for generation) and not a vector DB product.
        """
        return cbr_ops.fit_cbr_op(
            self,
            backend=backend,
            task=task,
            metric=metric,
            reuse=reuse,
            adapt=adapt,
            k=k,
            columns=columns,
            categorical_columns=categorical_columns,
            text_columns=text_columns,
            text_model_name=text_model_name,
            standardize=standardize,
            distance_eps=distance_eps,
            random_state=random_state,
            prefer_reduce_components=prefer_reduce_components,
            torch_epochs=torch_epochs,
            device=device,
        )

    def retrieve_cases(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        k: int | None = None,
        backend: str | None = None,
    ) -> CbrRetrieveResult:
        """Retrieve k nearest cases (no reuse / no memory update)."""
        return cbr_ops.retrieve_cases_op(
            self, partition=partition, k=k, backend=backend
        )

    def predict_cbr(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        k: int | None = None,
        return_traces: bool = True,
        backend: str | None = None,
    ) -> CbrPredictResult:
        """Predict via retrieve + reuse with case-influence traces."""
        return cbr_ops.predict_cbr_op(
            self,
            partition=partition,
            k=k,
            return_traces=return_traces,
            backend=backend,
        )

    def evaluate_cbr(
        self,
        *,
        partition: PartitionName | Literal["all"] = "validation",
        k: int | None = None,
    ) -> CbrEvalResult:
        """Evaluate CBR on a holdout partition (no memory update)."""
        return cbr_ops.evaluate_cbr_op(self, partition=partition, k=k)

    def retain_cbr(
        self,
        *,
        labeled_frame: Any | None = None,
        row_indices: Sequence[Any] | None = None,
        solution_column: str | None = None,
        source_disclosure: str,
        allow_overlap_with_train: bool = True,
    ) -> CbrRetainResult:
        """Retain new labeled cases; refuses Session validation/test indices.

        Requires a non-empty ``source_disclosure``. Pass either
        ``labeled_frame`` or ``row_indices`` (not both).
        """
        return cbr_ops.retain_cbr_op(
            self,
            labeled_frame=labeled_frame,
            row_indices=row_indices,
            solution_column=solution_column,
            source_disclosure=source_disclosure,
            allow_overlap_with_train=allow_overlap_with_train,
        )

    @property
    def cbr_plan(self) -> CbrPlan | None:
        """Last fitted :class:`~buildml.cbr.results.CbrPlan`."""
        return self._cbr_plan

    @property
    def cbr_fit_result(self) -> CbrFitResult | None:
        """Last :class:`~buildml.cbr.results.CbrFitResult`."""
        return self._cbr_fit_result

    @property
    def cbr_eval_result(self) -> CbrEvalResult | None:
        """Last :class:`~buildml.cbr.results.CbrEvalResult`."""
        return self._cbr_eval_result

    @property
    def cbr_predict_result(self) -> CbrPredictResult | None:
        """Last :class:`~buildml.cbr.results.CbrPredictResult`."""
        return self._cbr_predict_result

    @property
    def cbr_retrieve_result(self) -> CbrRetrieveResult | None:
        """Last :class:`~buildml.cbr.results.CbrRetrieveResult`."""
        return self._cbr_retrieve_result

    @property
    def cbr_retain_result(self) -> CbrRetainResult | None:
        """Last :class:`~buildml.cbr.results.CbrRetainResult`."""
        return self._cbr_retain_result

    def save_cbr_bundle(self, path: str | Path) -> Path:
        """Persist the active plan as ``buildml.cbr_bundle.v1``.

        See :data:`buildml.cbr.checkpoint.CHECKPOINT_BOUNDARY`.
        """
        return cbr_ops.save_cbr_bundle_op(self, path=path)

    def load_cbr_bundle(self, path: str | Path) -> Session:
        """Load a CBR bundle into this Session."""
        return cbr_ops.load_cbr_bundle_op(self, path=path)

    def fit_imitation(
        self,
        *,
        backend: str | None = None,
        task: ImitationTask | None = None,
        estimator: ImitationEstimator | None = None,
        method: str | None = None,
        columns: list[str] | None = None,
        action_column: str | None = None,
        env_id: str | None = None,
        n_epochs: int = 40,
        random_state: int | None = 0,
        prefer_reduce_components: bool = True,
    ) -> ImitationFitResult:
        """Fit behavioral cloning from demonstration rows (train only).

        Parameters
        ----------
        action_column:
            Demonstrated action column. Defaults to the Dataset target.
        task / estimator:
            Classification or regression BC; inferred when omitted.

        Notes
        -----
        **Leakage:** Requires a split. Policy uses train only.
        Honesty: BC from tables — not inverse RL, not DAgger, not robotics.
        """
        return rl_ops.fit_imitation_op(
            self,
            backend=backend,
            task=task,
            estimator=estimator,
            method=method,
            columns=columns,
            action_column=action_column,
            env_id=env_id,
            n_epochs=n_epochs,
            random_state=random_state,
            prefer_reduce_components=prefer_reduce_components,
        )

    def predict_imitation_action(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
    ) -> ImitationPredictResult:
        """Predict actions under the fitted behavioral cloning policy."""
        return rl_ops.predict_imitation_action_op(self, partition=partition)

    def evaluate_imitation(
        self,
        *,
        partition: PartitionName | Literal["all"] = "validation",
    ) -> ImitationEvalResult:
        """Compare predicted actions to held-out demonstration actions."""
        return rl_ops.evaluate_imitation_op(self, partition=partition)

    @property
    def imitation_plan(self) -> ImitationPlan | None:
        """Last fitted :class:`~buildml.rl.results.ImitationPlan`."""
        return self._imitation_plan

    @property
    def imitation_fit_result(self) -> ImitationFitResult | None:
        """Last :class:`~buildml.rl.results.ImitationFitResult`."""
        return self._imitation_fit_result

    @property
    def imitation_eval_result(self) -> ImitationEvalResult | None:
        """Last :class:`~buildml.rl.results.ImitationEvalResult`."""
        return self._imitation_eval_result

    @property
    def imitation_predict_result(self) -> ImitationPredictResult | None:
        """Last :class:`~buildml.rl.results.ImitationPredictResult`."""
        return self._imitation_predict_result

    def save_imitation_bundle(self, path: str | Path) -> Path:
        """Persist the active plan as ``buildml.imitation_bundle.v1``."""
        return rl_ops.save_imitation_bundle_op(self, path=path)

    def load_imitation_bundle(self, path: str | Path) -> Session:
        """Load an imitation bundle into this Session."""
        return rl_ops.load_imitation_bundle_op(self, path=path)

    def fit_rl(
        self,
        *,
        backend: str | None = None,
        mode: RlMode | None = None,
        algorithm: BanditAlgorithm | str = "linucb",
        columns: list[str] | None = None,
        action_column: str | None = None,
        reward_column: str | None = None,
        alpha: float = 1.0,
        epsilon: float = 0.1,
        temperature: float = 1.0,
        random_state: int | None = 0,
        prefer_reduce_components: bool = True,
        env_id: str = "CartPole-v1",
        n_episodes: int = 200,
        max_steps: int = 500,
        learning_rate: float = 0.01,
        gamma: float = 0.99,
        total_timesteps: int = 20_000,
    ) -> RlFitResult:
        """Fit a contextual bandit (core) or Gymnasium REINFORCE-lite.

        Parameters
        ----------
        mode:
            ``contextual_bandit`` (train logged table), ``gym_reinforce``
            (``buildml[rl]``), or ``gym_sb3`` (``buildml[rl-industry]``).
        backend:
            ``sklearn`` (bandit), ``native`` (REINFORCE-lite), or ``industry`` (SB3).
        algorithm:
            Bandit: ``linucb`` / ``epsilon_greedy`` / ``softmax``.
            SB3: ``ppo`` / ``dqn`` / ``a2c``.

        Notes
        -----
        **Leakage (bandit):** Requires a split; policy updates use train only.
        Holdout metrics are **offline** (DM/IPS) and disclosed as such.
        Honesty: Session bandit / small-env RL — not MuJoCo / robotics.
        """
        return rl_ops.fit_rl_op(
            self,
            backend=backend,
            mode=mode,
            algorithm=algorithm,
            columns=columns,
            action_column=action_column,
            reward_column=reward_column,
            alpha=alpha,
            epsilon=epsilon,
            temperature=temperature,
            random_state=random_state,
            prefer_reduce_components=prefer_reduce_components,
            env_id=env_id,
            n_episodes=n_episodes,
            max_steps=max_steps,
            learning_rate=learning_rate,
            gamma=gamma,
            total_timesteps=total_timesteps,
        )

    def act_rl(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        observations: Sequence[Any] | Any | None = None,
        deterministic: bool = True,
        random_state: int | None = 0,
    ) -> RlActResult:
        """Choose actions under the fitted RL policy.

        For ``gym_reinforce``, pass ``observations=...`` (env observation vectors).
        """
        return rl_ops.act_rl_op(
            self,
            partition=partition,
            observations=observations,
            deterministic=deterministic,
            random_state=random_state,
        )

    def evaluate_rl(
        self,
        *,
        partition: PartitionName | Literal["all"] = "validation",
        n_episodes: int | None = None,
        max_steps: int | None = None,
        random_state: int | None = 0,
        deterministic: bool = True,
    ) -> RlEvalResult:
        """Evaluate RL (offline bandit metrics or Gymnasium episode returns)."""
        return rl_ops.evaluate_rl_op(
            self,
            partition=partition,
            n_episodes=n_episodes,
            max_steps=max_steps,
            random_state=random_state,
            deterministic=deterministic,
        )

    @property
    def rl_plan(self) -> RlPlan | None:
        """Last fitted :class:`~buildml.rl.results.RlPlan`."""
        return self._rl_plan

    @property
    def rl_fit_result(self) -> RlFitResult | None:
        """Last :class:`~buildml.rl.results.RlFitResult`."""
        return self._rl_fit_result

    @property
    def rl_eval_result(self) -> RlEvalResult | None:
        """Last :class:`~buildml.rl.results.RlEvalResult`."""
        return self._rl_eval_result

    @property
    def rl_act_result(self) -> RlActResult | None:
        """Last :class:`~buildml.rl.results.RlActResult`."""
        return self._rl_act_result

    def save_rl_bundle(self, path: str | Path) -> Path:
        """Persist the active plan as ``buildml.rl_bundle.v1``."""
        return rl_ops.save_rl_bundle_op(self, path=path)

    def load_rl_bundle(self, path: str | Path) -> Session:
        """Load an RL bundle into this Session."""
        return rl_ops.load_rl_bundle_op(self, path=path)

    def fit_tda(
        self,
        *,
        backend: TdaBackend | None = None,
        vectorization: Vectorization = "persistence_image",
        homology_dims: Sequence[int] = (0, 1),
        knn: int = 16,
        maxdim: int = 1,
        thresh: float | None = None,
        n_bins: int = 20,
        n_layers: int = 3,
        pixel_size: float | None = None,
        standardize: bool = True,
        head: TdaHead = "logistic_regression",
        task: TdaTask | None = None,
        columns: list[str] | None = None,
        random_state: int | None = 0,
        prefer_reduce_components: bool = True,
        max_points_guard: int = 4000,
        subsample_strategy: SubsampleStrategy = "error",
        mapper: bool = False,
    ) -> TdaFitResult:
        """Fit topological features (+ optional sklearn head) on train only.

        Requires ``buildml[tda]`` (native) or ``buildml[tda-industry]`` (giotto).
        Local Vietoris–Rips on kNN train neighborhoods; vectorizer ranges and
        head use train only.

        Notes
        -----
        **Leakage:** Requires a split. Holdout never updates the PH pipeline.
        Honesty: Session PH + vectorization → sklearn — not a Mapper suite.
        """
        return tda_ops.fit_tda_op(
            self,
            backend=backend,
            vectorization=vectorization,
            homology_dims=homology_dims,
            knn=knn,
            maxdim=maxdim,
            thresh=thresh,
            n_bins=n_bins,
            n_layers=n_layers,
            pixel_size=pixel_size,
            standardize=standardize,
            head=head,
            task=task,
            columns=columns,
            random_state=random_state,
            prefer_reduce_components=prefer_reduce_components,
            max_points_guard=max_points_guard,
            subsample_strategy=subsample_strategy,
            mapper=mapper,
        )

    @staticmethod
    def tda_capability_matrix() -> dict[str, Any]:
        """Honest capability matrix for native vs giotto TDA backends."""
        return tda_ops.tda_capability_matrix_op()

    @staticmethod
    def ssl_capability_matrix() -> dict[str, Any]:
        """Honest capability matrix for self-supervised backends."""
        from buildml.selfsupervised.torch.catalog import ssl_capability_matrix

        return ssl_capability_matrix()

    @staticmethod
    def unsupervised_capability_matrix() -> dict[str, Any]:
        """Honest capability matrix for clustering / reduction backends."""
        from buildml.unsupervised.catalog import unsupervised_capability_matrix

        return unsupervised_capability_matrix()

    @staticmethod
    def rag_capability_matrix() -> dict[str, Any]:
        """Honest capability matrix for RAG embed / retrieve stacks."""
        from buildml.rag.catalog import rag_capability_matrix

        return rag_capability_matrix()

    @staticmethod
    def forecast_capability_matrix() -> dict[str, Any]:
        """Honest capability matrix for forecasting backends."""
        from buildml.forecasting.catalog import forecast_capability_matrix

        return forecast_capability_matrix()

    @staticmethod
    def timeseries_capability_matrix() -> dict[str, Any]:
        """Honest capability matrix for time-series analysis backends."""
        from buildml.timeseries.catalog import timeseries_capability_matrix

        return timeseries_capability_matrix()

    def transform_tda(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        backend: TdaBackend | None = None,
    ) -> TdaTransformResult:
        """Transform a partition with the frozen train-fitted TDA pipeline."""
        return tda_ops.transform_tda_op(self, partition=partition, backend=backend)

    def predict_tda(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
    ) -> TdaPredictResult:
        """Predict with the optional TDA supervised head."""
        return tda_ops.predict_tda_op(self, partition=partition)

    def evaluate_tda(
        self,
        *,
        partition: PartitionName | Literal["all"] = "validation",
        backend: TdaBackend | None = None,
        compare_diagram_distances: bool = False,
        diagram_distance_metric: DiagramDistanceMetric = "wasserstein",
        diagram_distance_dim: int = 1,
    ) -> TdaEvalResult:
        """Score the TDA head on a holdout partition (frozen train pipeline)."""
        return tda_ops.evaluate_tda_op(
            self,
            partition=partition,
            backend=backend,
            compare_diagram_distances=compare_diagram_distances,
            diagram_distance_metric=diagram_distance_metric,
            diagram_distance_dim=diagram_distance_dim,
        )

    @property
    def tda_plan(self) -> TdaPlan | None:
        """Last fitted :class:`~buildml.tda.results.TdaPlan`."""
        return self._tda_plan

    @property
    def tda_fit_result(self) -> TdaFitResult | None:
        """Last :class:`~buildml.tda.results.TdaFitResult`."""
        return self._tda_fit_result

    @property
    def tda_eval_result(self) -> TdaEvalResult | None:
        """Last :class:`~buildml.tda.results.TdaEvalResult`."""
        return self._tda_eval_result

    @property
    def tda_transform_result(self) -> TdaTransformResult | None:
        """Last :class:`~buildml.tda.results.TdaTransformResult`."""
        return self._tda_transform_result

    @property
    def tda_predict_result(self) -> TdaPredictResult | None:
        """Last :class:`~buildml.tda.results.TdaPredictResult`."""
        return self._tda_predict_result

    def save_tda_bundle(self, path: str | Path) -> Path:
        """Persist the active plan as ``buildml.tda_bundle.v2``."""
        return tda_ops.save_tda_bundle_op(self, path=path)

    def load_tda_bundle(self, path: str | Path) -> Session:
        """Load a TDA bundle into this Session."""
        return tda_ops.load_tda_bundle_op(self, path=path)

    def fit_recommender(
        self,
        *,
        method: RecommenderMethod | None = None,
        backend: RecommenderBackend | None = None,
        user_column: str | None = None,
        item_column: str | None = None,
        rating_column: str | None = None,
        feedback: FeedbackMode = "explicit",
        n_neighbors: int = 40,
        n_factors: int = 32,
        min_rating: float | None = None,
        item_feature_columns: list[str] | None = None,
        user_feature_columns: list[str] | None = None,
        cold_start: ColdStartPolicy = "popularity",
        random_state: int | None = 0,
        n_iterations: int = 15,
        lightfm_epochs: int = 10,
    ) -> RecommenderFitResult:
        """Fit a recommender on train interactions only.

        Requires explicit ``user_column`` / ``item_column``. Rating defaults to
        the Session target for ``feedback='explicit'``. Core algorithms: item/user
        kNN CF, TruncatedSVD / NMF, content-based item features. With
        ``buildml[recommenders-industry]``: implicit ALS/BPR (default for
        ``feedback='implicit'``) and LightFM hybrid.

        Notes
        -----
        **Leakage:** Requires a split. Holdout interactions never update the
        model. Known-item protocol + cold-start disclosure on recommend/eval.
        Distinct from RAG and from EDA ``Recommendation`` Findings.
        """
        return recommender_ops.fit_recommender_op(
            self,
            method=method,
            backend=backend,
            user_column=user_column,
            item_column=item_column,
            rating_column=rating_column,
            feedback=feedback,
            n_neighbors=n_neighbors,
            n_factors=n_factors,
            min_rating=min_rating,
            item_feature_columns=item_feature_columns,
            user_feature_columns=user_feature_columns,
            cold_start=cold_start,
            random_state=random_state,
            n_iterations=n_iterations,
            lightfm_epochs=lightfm_epochs,
        )

    def recommend(
        self,
        *,
        partition: PartitionName | Literal["all"] | None = None,
        user_ids: Sequence[Any] | None = None,
        k: int = 10,
        exclude_train_items: bool = True,
    ) -> RecommendResult:
        """Top-K item recommendations (train catalog / known-item protocol)."""
        return recommender_ops.recommend_op(
            self,
            partition=partition,
            user_ids=user_ids,
            k=k,
            exclude_train_items=exclude_train_items,
        )

    def evaluate_recommender(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        k: int = 10,
    ) -> RecommenderEvalResult:
        """Holdout ranking metrics: Precision@K, Recall@K, nDCG@K, MAP@K."""
        return recommender_ops.evaluate_recommender_op(
            self, partition=partition, k=k
        )

    @property
    def recommender_plan(self) -> RecommenderPlan | None:
        """Last fitted :class:`~buildml.recommenders.results.RecommenderPlan`."""
        return self._recommender_plan

    @property
    def recommender_fit_result(self) -> RecommenderFitResult | None:
        """Last :class:`~buildml.recommenders.results.RecommenderFitResult`."""
        return self._recommender_fit_result

    @property
    def recommender_eval_result(self) -> RecommenderEvalResult | None:
        """Last :class:`~buildml.recommenders.results.RecommenderEvalResult`."""
        return self._recommender_eval_result

    @property
    def recommender_recommend_result(self) -> RecommendResult | None:
        """Last :class:`~buildml.recommenders.results.RecommendResult`."""
        return self._recommender_recommend_result

    def save_recommender_bundle(self, path: str | Path) -> Path:
        """Persist the active plan as ``buildml.recommender_bundle.v1``."""
        return recommender_ops.save_recommender_bundle_op(self, path=path)

    def load_recommender_bundle(self, path: str | Path) -> Session:
        """Load a recommender bundle into this Session."""
        return recommender_ops.load_recommender_bundle_op(self, path=path)

    def fit_ranker(
        self,
        *,
        backend: RankerBackend | None = None,
        method: RankerMethod | str | None = None,
        query_column: str | None = None,
        item_column: str | None = None,
        relevance_column: str | None = None,
        feature_columns: list[str] | None = None,
        pointwise_estimator: PointwiseEstimator = "ridge",
        pairwise_estimator: PairwiseEstimator = "ranksvm",
        max_pairs_per_query: int = 80,
        relevance_threshold: float = 0.0,
        alpha: float = 1.0,
        C: float = 1.0,
        n_estimators: int = 120,
        learning_rate: float = 0.08,
        hidden_dim: int = 64,
        epochs: int = 40,
        device: str = "cpu",
        random_state: int | None = 0,
    ) -> RankerFitResult:
        """Fit a tabular learning-to-rank model on train rows only.

        Requires explicit ``query_column`` / ``item_column``. Relevance defaults
        to the Session target. Prefer ``group_split`` on the query id so test
        queries' labels never enter training.

        Backends (see ``ranking_capability_matrix()``):
        ``sklearn`` pointwise/pairwise fallback; ``industry`` GBDT rankers
        (``buildml[ranking-industry]``); ``torch`` listwise-lite
        (``buildml[torch]``). Industry is the default when installed.

        Notes
        -----
        **Leakage:** Requires a split. Holdout rows never update the model.
        Distinct from RAG retrieve/generate and from recommender CF.
        Honesty: Session tabular LTR — not a search-engine product.
        """
        return ranking_ops.fit_ranker_op(
            self,
            backend=backend,
            method=method,
            query_column=query_column,
            item_column=item_column,
            relevance_column=relevance_column,
            feature_columns=feature_columns,
            pointwise_estimator=pointwise_estimator,
            pairwise_estimator=pairwise_estimator,
            max_pairs_per_query=max_pairs_per_query,
            relevance_threshold=relevance_threshold,
            alpha=alpha,
            C=C,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
            epochs=epochs,
            device=device,
            random_state=random_state,
        )

    def rank(
        self,
        *,
        partition: PartitionName | Literal["all"] | None = None,
        query_ids: Sequence[Any] | None = None,
        k: int = 10,
        backend: RankerBackend | None = None,
    ) -> RankResult:
        """Order items for queries (descending score from frozen RankerPlan)."""
        return ranking_ops.rank_op(
            self,
            partition=partition,
            query_ids=query_ids,
            k=k,
            backend=backend,
        )

    def evaluate_ranker(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        k: int = 10,
        backend: RankerBackend | None = None,
    ) -> RankerEvalResult:
        """Holdout per-query metrics: nDCG@K, MAP@K, MRR@K."""
        return ranking_ops.evaluate_ranker_op(
            self, partition=partition, k=k, backend=backend
        )

    @property
    def ranker_plan(self) -> RankerPlan | None:
        """Last fitted :class:`~buildml.ranking.results.RankerPlan`."""
        return self._ranker_plan

    @property
    def ranker_fit_result(self) -> RankerFitResult | None:
        """Last :class:`~buildml.ranking.results.RankerFitResult`."""
        return self._ranker_fit_result

    @property
    def ranker_eval_result(self) -> RankerEvalResult | None:
        """Last :class:`~buildml.ranking.results.RankerEvalResult`."""
        return self._ranker_eval_result

    @property
    def ranker_rank_result(self) -> RankResult | None:
        """Last :class:`~buildml.ranking.results.RankResult`."""
        return self._ranker_rank_result

    def save_ranker_bundle(self, path: str | Path) -> Path:
        """Persist the active plan as ``buildml.ranker_bundle.v1``."""
        return ranking_ops.save_ranker_bundle_op(self, path=path)

    def load_ranker_bundle(self, path: str | Path) -> Session:
        """Load a ranker bundle into this Session."""
        return ranking_ops.load_ranker_bundle_op(self, path=path)

    def fit_kg(
        self,
        *,
        backend: KgBackend | None = None,
        method: KgMethod = "transe",
        head_column: str | None = None,
        relation_column: str | None = None,
        tail_column: str | None = None,
        embedding_dim: int = 50,
        epochs: int = 40,
        batch_size: int = 256,
        learning_rate: float = 0.01,
        margin: float = 1.0,
        neg_ratio: int = 1,
        norm: KgNorm = "l1",
        random_state: int | None = 0,
    ) -> KgFitResult:
        """Fit a knowledge-graph embedding model on train triples only.

        Requires explicit ``head_column`` / ``relation_column`` /
        ``tail_column``. Backends: ``native`` (numpy TransE/DistMult) or
        ``pykeen`` (RotatE/ComplEx/TransE/DistMult when
        ``buildml[kg-industry]`` is installed).

        Notes
        -----
        **Leakage:** Requires a split. Holdout triples never update embeddings
        or the train adjacency used by ``query_kg``. Distinct from Graph ML
        (``set_graph`` / ``fit_graph`` node classification) and from RAG.
        Honesty: Session KG learning/query — not a Neo4j / graph-DB product.
        """
        return kg_ops.fit_kg_op(
            self,
            backend=backend,
            method=method,
            head_column=head_column,
            relation_column=relation_column,
            tail_column=tail_column,
            embedding_dim=embedding_dim,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            margin=margin,
            neg_ratio=neg_ratio,
            norm=norm,
            random_state=random_state,
        )

    def score_triples(
        self,
        *,
        partition: PartitionName | Literal["all"] | None = None,
        triples: Any | None = None,
    ) -> ScoreTriplesResult:
        """Score (head, relation, tail) triples with the frozen KgPlan."""
        return kg_ops.score_triples_op(
            self, partition=partition, triples=triples
        )

    def predict_links(
        self,
        *,
        mode: LinkPredictionMode = "tail",
        heads: Sequence[Any] | None = None,
        relations: Sequence[Any] | None = None,
        tails: Sequence[Any] | None = None,
        k: int = 10,
        filtered: bool = True,
    ) -> PredictLinksResult:
        """Predict missing link components (tail / head / relation)."""
        return kg_ops.predict_links_op(
            self,
            mode=mode,
            heads=heads,
            relations=relations,
            tails=tails,
            k=k,
            filtered=filtered,
        )

    def query_kg(
        self,
        *,
        mode: KgQueryMode = "neighbors",
        entity: Any | None = None,
        source: Any | None = None,
        target: Any | None = None,
        relation: Any | None = None,
        direction: Literal["out", "in", "both"] = "out",
        max_hops: int = 3,
    ) -> KgQueryResult:
        """Symbolic neighbors / path / typed query over the train KG.

        Not an LLM, not Neo4j/Cypher, not RAG — BFS / adjacency on train
        triples only.
        """
        return kg_ops.query_kg_op(
            self,
            mode=mode,
            entity=entity,
            source=source,
            target=target,
            relation=relation,
            direction=direction,
            max_hops=max_hops,
        )

    def evaluate_kg(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        k: int = 10,
    ) -> KgEvalResult:
        """Holdout filtered link-prediction metrics: MRR, Hits@1/3/K."""
        return kg_ops.evaluate_kg_op(self, partition=partition, k=k)

    @property
    def kg_plan(self) -> KgPlan | None:
        """Last fitted :class:`~buildml.kg.results.KgPlan`."""
        return self._kg_plan

    @property
    def kg_fit_result(self) -> KgFitResult | None:
        """Last :class:`~buildml.kg.results.KgFitResult`."""
        return self._kg_fit_result

    @property
    def kg_eval_result(self) -> KgEvalResult | None:
        """Last :class:`~buildml.kg.results.KgEvalResult`."""
        return self._kg_eval_result

    @property
    def kg_score_result(self) -> ScoreTriplesResult | None:
        """Last :class:`~buildml.kg.results.ScoreTriplesResult`."""
        return self._kg_score_result

    @property
    def kg_predict_result(self) -> PredictLinksResult | None:
        """Last :class:`~buildml.kg.results.PredictLinksResult`."""
        return self._kg_predict_result

    @property
    def kg_query_result(self) -> KgQueryResult | None:
        """Last :class:`~buildml.kg.results.KgQueryResult`."""
        return self._kg_query_result

    def save_kg_bundle(self, path: str | Path) -> Path:
        """Persist the active plan as ``buildml.kg_bundle.v1``."""
        return kg_ops.save_kg_bundle_op(self, path=path)

    def load_kg_bundle(self, path: str | Path) -> Session:
        """Load a knowledge-graph bundle into this Session."""
        return kg_ops.load_kg_bundle_op(self, path=path)

    def fit_decision_policy(
        self,
        *,
        method: DecisionMethod = "threshold",
        backend: str | None = None,
        partition: TuningPartition = "validation",
        allow_test_tuning: bool = False,
        fp_cost: float | None = None,
        fn_cost: float | None = None,
        tp_benefit: float = 0.0,
        tn_benefit: float = 0.0,
        cost_matrix: Sequence[Sequence[float]] | None = None,
        class_labels: list[str] | None = None,
        capacity: int | None = None,
        budget: float | None = None,
        score_source: ScoreSource = "model_proba",
        score_column: str | None = None,
        cost_column: str | None = None,
        value_column: str | None = None,
        id_column: str | None = None,
        knapsack_solver: KnapsackSolver = "dp",
        objective: AllocationObjective = "maximize_score",
        min_score: float | None = None,
        lp_max_fraction: float = 1.0,
    ) -> DecisionFitResult:
        """Fit a decision policy on train/validation (test requires opt-in).

        Methods: ``threshold`` (wraps classical ``tune_threshold`` engine),
        ``cost_matrix`` (multiclass Bayes), ``topk``, ``knapsack``,
        ``lp_allocate`` (scipy linprog or CVXPY). ``backend=`` selects industry
        solvers when installed (see ``decision_capability_matrix()``).
        Prefer ``partition='validation'``.

        Notes
        -----
        **Leakage:** Tuning on Session test requires ``allow_test_tuning=True``.
        Honesty: ML score/cost/allocation helpers — not a general OR platform.
        """
        return decision_ops.fit_decision_policy_op(
            self,
            method=method,
            backend=backend,
            partition=partition,
            allow_test_tuning=allow_test_tuning,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
            tp_benefit=tp_benefit,
            tn_benefit=tn_benefit,
            cost_matrix=cost_matrix,
            class_labels=class_labels,
            capacity=capacity,
            budget=budget,
            score_source=score_source,
            score_column=score_column,
            cost_column=cost_column,
            value_column=value_column,
            id_column=id_column,
            knapsack_solver=knapsack_solver,
            objective=objective,
            min_score=min_score,
            lp_max_fraction=lp_max_fraction,
        )

    def apply_decisions(
        self,
        *,
        partition: PartitionName | Literal["all"] | None = "test",
        candidates: pd.DataFrame | None = None,
    ) -> ApplyDecisionsResult:
        """Apply the frozen DecisionPlan to a partition or candidate frame."""
        return decision_ops.apply_decisions_op(
            self, partition=partition, candidates=candidates
        )

    def evaluate_decisions(
        self,
        *,
        partition: PartitionName = "test",
    ) -> DecisionEvalResult:
        """Evaluate the frozen DecisionPlan on a holdout partition."""
        return decision_ops.evaluate_decisions_op(self, partition=partition)

    @property
    def decision_plan(self) -> DecisionPlan | None:
        """Last fitted :class:`~buildml.optimize.results.DecisionPlan`."""
        return self._decision_plan

    @property
    def decision_fit_result(self) -> DecisionFitResult | None:
        """Last :class:`~buildml.optimize.results.DecisionFitResult`."""
        return self._decision_fit_result

    @property
    def decision_eval_result(self) -> DecisionEvalResult | None:
        """Last :class:`~buildml.optimize.results.DecisionEvalResult`."""
        return self._decision_eval_result

    @property
    def decision_apply_result(self) -> ApplyDecisionsResult | None:
        """Last :class:`~buildml.optimize.results.ApplyDecisionsResult`."""
        return self._decision_apply_result

    def save_decision_bundle(self, path: str | Path) -> Path:
        """Persist the active plan as ``buildml.decision_bundle.v1``."""
        return decision_ops.save_decision_bundle_op(self, path=path)

    def load_decision_bundle(self, path: str | Path) -> Session:
        """Load a decision-policy bundle into this Session."""
        return decision_ops.load_decision_bundle_op(self, path=path)

    @staticmethod
    def decision_capability_matrix() -> dict[str, Any]:
        """Honest capability matrix for decision-policy backends."""
        return decision_ops.decision_capability_matrix_op()

    @staticmethod
    def optimize_capability_matrix() -> dict[str, Any]:
        """Alias for :meth:`decision_capability_matrix`."""
        return decision_ops.decision_capability_matrix_op()

    def fit_synthesizer(
        self,
        *,
        backend: SyntheticBackend | None = None,
        method: SynthesizerMethod = "gaussian_copula",
        columns: Sequence[str] | None = None,
        random_state: int = 42,
        smooth_sigma: float = 0.0,
        correlation_ridge: float = 1e-3,
        target_column: str | None = None,
        k_neighbors: int = 5,
        sampling_strategy: str | float | dict[str, float] = "auto",
        epochs: int = 300,
        batch_size: int = 500,
    ) -> SynthesizerFitResult:
        """Fit a tabular synthesizer on Session **train** only.

        Backends (see ``synthetic_capability_matrix()``):
        native — bootstrap / Gaussian copula / SMOTE (``buildml[imbalanced]``).
        sdv — CTGAN / TVAE / CopulaGAN (``buildml[synthetic-industry]``).

        Notes
        -----
        **Leakage:** Always train-only. Distinct from :meth:`resample`
        (class-balance preprocess). **Privacy:** not differential privacy.
        """
        return synthetic_ops.fit_synthesizer_op(
            self,
            backend=backend,
            method=method,
            columns=columns,
            random_state=random_state,
            smooth_sigma=smooth_sigma,
            correlation_ridge=correlation_ridge,
            target_column=target_column,
            k_neighbors=k_neighbors,
            sampling_strategy=sampling_strategy,
            epochs=epochs,
            batch_size=batch_size,
        )

    def sample_synthetic(
        self,
        *,
        n: int | None = None,
        random_state: int | None = None,
        condition: dict[str, Any] | None = None,
        merge_mode: MergeMode = "none",
        provenance_column: str = "_synthetic",
        validate: bool = False,
    ) -> SyntheticSampleResult:
        """Sample from the frozen synthesizer; optionally extend train.

        Default ``merge_mode='none'`` returns a Frame without mutating roles.
        ``merge_mode='extend_train'`` appends to train with a provenance
        column (role=ignore); holdouts unchanged. ``validate=True`` runs
        built-in schema checks on the sample.
        """
        return synthetic_ops.sample_synthetic_op(
            self,
            n=n,
            random_state=random_state,
            condition=condition,
            merge_mode=merge_mode,
            provenance_column=provenance_column,
            validate=validate,
        )

    def evaluate_synthetic(
        self,
        *,
        mode: EvalMode = "fidelity",
        eval_backend: EvalBackend = "auto",
        partition: PartitionName = "test",
        n_synthetic: int | None = None,
        random_state: int = 0,
        estimator: Literal["auto", "logistic", "ridge"] = "auto",
    ) -> SyntheticEvalResult:
        """Evaluate the frozen synthesizer (fidelity metrics or TSTR utility)."""
        return synthetic_ops.evaluate_synthetic_op(
            self,
            mode=mode,
            eval_backend=eval_backend,
            partition=partition,
            n_synthetic=n_synthetic,
            random_state=random_state,
            estimator=estimator,
        )

    @staticmethod
    def synthetic_capability_matrix() -> dict[str, Any]:
        """Honest capability matrix for synthetic backends and eval paths."""
        return synthetic_ops.synthetic_capability_matrix_op()

    @property
    def synthesizer_plan(self) -> SynthesizerPlan | None:
        """Last fitted :class:`~buildml.synthetic.results.SynthesizerPlan`."""
        return self._synthesizer_plan

    @property
    def synthetic_fit_result(self) -> SynthesizerFitResult | None:
        """Last :class:`~buildml.synthetic.results.SynthesizerFitResult`."""
        return self._synthetic_fit_result

    @property
    def synthetic_eval_result(self) -> SyntheticEvalResult | None:
        """Last :class:`~buildml.synthetic.results.SyntheticEvalResult`."""
        return self._synthetic_eval_result

    @property
    def synthetic_sample_result(self) -> SyntheticSampleResult | None:
        """Last :class:`~buildml.synthetic.results.SyntheticSampleResult`."""
        return self._synthetic_sample_result

    def save_synthetic_bundle(self, path: str | Path) -> Path:
        """Persist the active plan as ``buildml.synthetic_bundle.v1``."""
        return synthetic_ops.save_synthetic_bundle_op(self, path=path)

    def load_synthetic_bundle(self, path: str | Path) -> Session:
        """Load a synthesizer bundle into this Session."""
        return synthetic_ops.load_synthetic_bundle_op(self, path=path)

    @classmethod
    def register_transform(
        cls,
        name: str,
        *,
        fit: Any,
        transform: Any,
        description: str = "",
        output_columns: Any | None = None,
        drop_input_columns: bool = False,
        serializable: bool = True,
        overwrite: bool = False,
    ) -> CustomTransformSpec:
        """Register a custom train-fit transform for :meth:`apply_custom_transform`.

        The ``fit`` callable receives only train rows for the selected columns.
        See :func:`buildml.preprocess.register_transform` for the full contract."""
        return preprocess_ops.register_transform(
            cls,
            name=name,
            fit=fit,
            transform=transform,
            description=description,
            output_columns=output_columns,
            drop_input_columns=drop_input_columns,
            serializable=serializable,
            overwrite=overwrite,
        )

    @classmethod
    def list_transforms(cls) -> tuple[CustomTransformSpec, ...]:
        """Return registered custom transforms in name order."""
        return preprocess_ops.list_transforms(cls)

    def apply_custom_transform(
        self,
        name: str,
        *,
        columns: list[str],
        params: Mapping[str, Any] | None = None,
    ) -> Session:
        """Fit a registered custom transform on train and apply it to all rows.

        Parameters
        ----------
        name:
            Name previously passed to :meth:`register_transform`.
        columns:
            Input columns passed to fit/transform.
        params:
            Optional parameters forwarded to the registered ``fit`` callable.

        Notes
        -----
        **Leakage:** Requires a split. Fit sees train rows only. Score-time
        replay requires the same name to remain registered in-process."""
        return preprocess_ops.apply_custom_transform(
            self, name=name, columns=columns, params=params
        )

    @property
    def custom_plan(self) -> CustomTransformPlan | None:
        """Last fitted custom-transform plan, if any."""
        return self._custom_plan

    def dry_run(
        self,
        operation: str | Sequence[str] | None = None,
        *,
        parameters: Mapping[str, Any] | None = None,
    ) -> DryRunReport:
        """Preview intended operations without mutating Session state.

        Parameters
        ----------
        operation:
            One operation name, a sequence of names, or ``None`` for a focused
            default preview of available/blocked next steps.
        parameters:
            Optional parameters attached to a single-operation preview.

        Notes
        -----
        Dry-run does not fit, transform, or append history. Availability means
        API prerequisites pass, not that the operation is appropriate."""
        return workflow_ops.dry_run(self, operation=operation, parameters=parameters)

    @property
    def last_dry_run(self) -> DryRunReport | None:
        """Most recent dry-run report, if any."""
        return self._last_dry_run

    def summarize_history(self) -> HistorySummary:
        """Summarize operation history and list unresolved risks.

        Notes
        -----
        Read-only. Does not append history. Risks are heuristic review cues,
        not proof of leakage or invalid results."""
        return workflow_ops.summarize_history(self)

    @property
    def last_history_summary(self) -> HistorySummary | None:
        """Most recent history summary, if any."""
        return self._last_history_summary

    def fit(
        self,
        estimator: Any,
        *,
        task: Literal["classification", "regression", "auto"] = "auto",
    ) -> Session:
        """Fit a sklearn-compatible estimator on the train partition.

        Parameters
        ----------
        estimator:
            Unfitted estimator instance.
        task:
            Task type or ``auto``.

        Notes
        -----
        **Leakage:** Fits on train only. Call after split and preparation."""
        return classical_ops.fit(self, estimator=estimator, task=task)

    @property
    def fit_result(self) -> FitResult | None:
        """Last fit result, if any."""
        return self._fit_result

    def predict(
        self,
        *,
        partition: Literal["train", "validation", "test"] = "test",
        return_proba: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """Predict labels or probabilities on a partition.

        Parameters
        ----------
        partition:
            Split partition to score.
        return_proba:
            If True and supported, return class probabilities."""
        return classical_ops.predict(self, partition=partition, return_proba=return_proba)

    def evaluate(
        self,
        *,
        partition: Literal["train", "validation", "test"] = "test",
        export_figures: str | Path | None = None,
        export_html: str | Path | None = None,
        include_plots: bool = False,
    ) -> EvaluateResult:
        """Evaluate the last fitted estimator on a partition.

        Returns metrics, diagnostics (confusion matrix / residuals), and
        recommendations — not a single score.

        Parameters
        ----------
        partition:
            Split partition to score.
        include_plots / export_figures / export_html:
            Optionally build the eval plot board (requires ``buildml[viz]``)
            and persist figures/HTML. Plot board is also stored on
            :attr:`last_plot_board`."""
        return classical_ops.evaluate(
            self,
            partition=partition,
            export_figures=export_figures,
            export_html=export_html,
            include_plots=include_plots,
        )

    def make_torch_loaders(
        self,
        *,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle_train: bool = True,
        drop_last: bool = False,
        normalize: bool = True,
        seed: int = 0,
        task: Literal["classification", "regression", "auto"] = "auto",
        apply_plans: bool = False,
    ) -> TorchLoaderBundle:
        """Build Torch DataLoaders from current roles and split partitions.

        Requires ``pip install 'buildml[torch]'`` (or ``buildml[dl]``). Shuffle
        applies to the train loader only. When ``normalize`` is True, mean/std
        are fit on train and frozen for validation/test. Attached classical
        plans are disclosed on the loader report; pass ``apply_plans=True`` to
        re-apply fitted plans before building tensors.

        Returns
        -------
        TorchLoaderBundle
            Loaders keyed by partition plus the feature contract."""
        return dl_ops.make_torch_loaders(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle_train=shuffle_train,
            drop_last=drop_last,
            normalize=normalize,
            seed=seed,
            task=task,
            apply_plans=apply_plans,
        )

    def make_text_torch_loaders(
        self,
        *,
        text_column: str | None = None,
        batch_size: int = 16,
        max_len: int = 64,
        max_vocab: int = 5000,
        min_freq: int = 1,
        shuffle_train: bool = True,
        seed: int = 0,
    ) -> TorchLoaderBundle:
        """Build token-id DataLoaders for text classification (sequence modality).

        Vocabulary is fit on train only. Requires ``buildml[torch]``."""
        return dl_ops.make_text_torch_loaders(
            self,
            text_column=text_column,
            batch_size=batch_size,
            max_len=max_len,
            max_vocab=max_vocab,
            min_freq=min_freq,
            shuffle_train=shuffle_train,
            seed=seed,
        )

    def fit_torch(
        self,
        module: Any | None = None,
        *,
        loss_fn: Any | None = None,
        optimizer_factory: Any | None = None,
        epochs: int = 5,
        learning_rate: float = 1e-3,
        device: Literal["cpu", "cuda", "mps", "auto"] = "auto",
        grad_clip_norm: float | None = None,
        log_every: int = 1,
        early_stopping_patience: int | None = None,
        early_stopping_monitor: str = "val_loss",
        scheduler: Literal["none", "step", "plateau", "cosine"] = "none",
        resume: bool = False,
        config: TrainConfig | None = None,
        hidden: tuple[int, ...] = (64, 32),
        dropout: float = 0.1,
        mixed_precision: bool = False,
    ) -> Session:
        """Train an ``nn.Module`` on the train Torch loader.

        Requires ``pip install 'buildml[torch]'``. When ``module`` is omitted,
        builds a tabular MLP, text classifier, or multimodal fusion module from
        the loader contract. Does not replace classical :meth:`fit`.

        Parameters
        ----------
        module:
            Optional ``torch.nn.Module``. When omitted, a built-in model is
            constructed from the active loader contract. When ``resume=True``,
            weights are restored from :attr:`dl_train_result`.
        loss_fn:
            Optional ``(module, xb, yb) -> loss``. Defaults to CrossEntropy
            (classification) or MSE (regression).
        optimizer_factory:
            Optional ``callable(params) -> optimizer``. Defaults to Adam.
        epochs / learning_rate / device / grad_clip_norm / log_every:
            Train-loop knobs used when ``config`` is omitted. With ``resume=True``,
            ``epochs`` are **additional** epochs.
        early_stopping_patience / early_stopping_monitor / scheduler:
            Patience requires a validation loader. Scheduler defaults to ``none``.
        resume:
            When True, continue from :attr:`dl_train_result`.
        config:
            Optional :class:`~buildml.dl.types.TrainConfig` overriding scalar knobs.
        hidden / dropout:
            Built-in MLP / text classifier knobs when ``module`` is omitted.
        mixed_precision:
            When True on CUDA, enables AMP; CPU/MPS is a documented no-op."""
        return dl_ops.fit_torch(
            self,
            module=module,
            loss_fn=loss_fn,
            optimizer_factory=optimizer_factory,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
            grad_clip_norm=grad_clip_norm,
            log_every=log_every,
            early_stopping_patience=early_stopping_patience,
            early_stopping_monitor=early_stopping_monitor,
            scheduler=scheduler,
            resume=resume,
            config=config,
            hidden=hidden,
            dropout=dropout,
            mixed_precision=mixed_precision,
        )

    def make_multimodal_torch_loaders(
        self,
        *,
        text_column: str | None = None,
        numeric_columns: list[str] | None = None,
        image_column: str | None = None,
        audio_column: str | None = None,
        batch_size: int = 16,
        max_len: int = 64,
        max_vocab: int = 5000,
        min_freq: int = 1,
        normalize: bool = True,
        normalize_images: bool = True,
        normalize_audio: bool = True,
        image_size: tuple[int, int] = (32, 32),
        image_channels: int = 3,
        audio_sample_rate: int = 16_000,
        audio_max_samples: int = 16_000,
        audio_source_sample_rate: int | None = None,
        shuffle_train: bool = True,
        seed: int = 0,
        task: Literal["classification", "regression", "auto"] = "auto",
        preprocess: Any | None = None,
        use_saved_preprocess: bool = False,
    ) -> TorchLoaderBundle:
        """Build fused multimodal DataLoaders (tabular/text/image/audio; train-only stats).

        Pass ``preprocess=`` or ``use_saved_preprocess=True`` to restore frozen
        multimodal fit stats from a prior trainer bundle.
        """
        return dl_ops.make_multimodal_torch_loaders(
            self,
            text_column=text_column,
            numeric_columns=numeric_columns,
            image_column=image_column,
            audio_column=audio_column,
            batch_size=batch_size,
            max_len=max_len,
            max_vocab=max_vocab,
            min_freq=min_freq,
            normalize=normalize,
            normalize_images=normalize_images,
            normalize_audio=normalize_audio,
            image_size=image_size,
            image_channels=image_channels,
            audio_sample_rate=audio_sample_rate,
            audio_max_samples=audio_max_samples,
            audio_source_sample_rate=audio_source_sample_rate,
            shuffle_train=shuffle_train,
            seed=seed,
            task=task,
            preprocess=preprocess,
            use_saved_preprocess=use_saved_preprocess,
        )

    def make_image_multimodal_torch_loaders(
        self,
        *,
        image_column: str,
        text_column: str | None = None,
        numeric_columns: list[str] | None = None,
        audio_column: str | None = None,
        batch_size: int = 16,
        max_len: int = 64,
        max_vocab: int = 5000,
        min_freq: int = 1,
        normalize: bool = True,
        normalize_images: bool = True,
        normalize_audio: bool = True,
        image_size: tuple[int, int] = (32, 32),
        image_channels: int = 3,
        audio_sample_rate: int = 16_000,
        audio_max_samples: int = 16_000,
        audio_source_sample_rate: int | None = None,
        shuffle_train: bool = True,
        seed: int = 0,
        task: Literal["classification", "regression", "auto"] = "auto",
    ) -> TorchLoaderBundle:
        """Build image multimodal loaders (image ⊕ tabular and/or text and/or audio).

        Path cells need Pillow (bundled in ``buildml[torch]``); array cells work
        with Torch alone.
        """
        return dl_ops.make_image_multimodal_torch_loaders(
            self,
            image_column=image_column,
            text_column=text_column,
            numeric_columns=numeric_columns,
            audio_column=audio_column,
            batch_size=batch_size,
            max_len=max_len,
            max_vocab=max_vocab,
            min_freq=min_freq,
            normalize=normalize,
            normalize_images=normalize_images,
            normalize_audio=normalize_audio,
            image_size=image_size,
            image_channels=image_channels,
            audio_sample_rate=audio_sample_rate,
            audio_max_samples=audio_max_samples,
            audio_source_sample_rate=audio_source_sample_rate,
            shuffle_train=shuffle_train,
            seed=seed,
            task=task,
        )

    def make_audio_multimodal_torch_loaders(
        self,
        *,
        audio_column: str,
        text_column: str | None = None,
        numeric_columns: list[str] | None = None,
        image_column: str | None = None,
        batch_size: int = 16,
        max_len: int = 64,
        max_vocab: int = 5000,
        min_freq: int = 1,
        normalize: bool = True,
        normalize_images: bool = True,
        normalize_audio: bool = True,
        image_size: tuple[int, int] = (32, 32),
        image_channels: int = 3,
        audio_sample_rate: int = 16_000,
        audio_max_samples: int = 16_000,
        audio_source_sample_rate: int | None = None,
        shuffle_train: bool = True,
        seed: int = 0,
        task: Literal["classification", "regression", "auto"] = "auto",
    ) -> TorchLoaderBundle:
        """Build audio multimodal loaders (audio ⊕ tabular and/or text and/or image).

        Path cells need soundfile (bundled in ``buildml[torch]`` /
        ``buildml[audio]``); waveform arrays work with Torch alone. Small 1D-CNN
        fusion branch — not a speech foundation model.
        """
        return dl_ops.make_audio_multimodal_torch_loaders(
            self,
            audio_column=audio_column,
            text_column=text_column,
            numeric_columns=numeric_columns,
            image_column=image_column,
            batch_size=batch_size,
            max_len=max_len,
            max_vocab=max_vocab,
            min_freq=min_freq,
            normalize=normalize,
            normalize_images=normalize_images,
            normalize_audio=normalize_audio,
            image_size=image_size,
            image_channels=image_channels,
            audio_sample_rate=audio_sample_rate,
            audio_max_samples=audio_max_samples,
            audio_source_sample_rate=audio_source_sample_rate,
            shuffle_train=shuffle_train,
            seed=seed,
            task=task,
        )

    def cross_validate_torch(
        self,
        *,
        n_folds: int = 3,
        epochs: int = 3,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        device: Literal["cpu", "cuda", "mps", "auto"] = "auto",
        normalize: bool = True,
        seed: int = 0,
        stratify: bool = True,
        task: Literal["classification", "regression", "auto"] = "auto",
        module_factory: Any | None = None,
    ) -> TorchCVResult:
        """Fold-local Torch CV (normalize fit per fold; not nested search)."""
        return dl_ops.cross_validate_torch(
            self,
            n_folds=n_folds,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            normalize=normalize,
            seed=seed,
            stratify=stratify,
            task=task,
            module_factory=module_factory,
        )

    def search_torch(
        self,
        *,
        param_grid: dict[str, list[Any]] | None = None,
        param_distributions: dict[str, Any] | None = None,
        inner_search: Literal["grid", "randomized", "auto"] = "auto",
        n_iter: int = 5,
        n_folds: int = 3,
        epochs: int = 2,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        device: Literal["cpu", "cuda", "mps", "auto"] = "auto",
        normalize: bool = True,
        seed: int = 0,
        stratify: bool = True,
        task: Literal["classification", "regression", "auto"] = "auto",
        scoring_metric: str | None = None,
        module_factory: Any | None = None,
    ) -> Any:
        """Inner-fold Torch hyperparameter search on the train universe."""
        return dl_ops.search_torch(
            self,
            param_grid=param_grid,
            param_distributions=param_distributions,
            inner_search=inner_search,
            n_iter=n_iter,
            n_folds=n_folds,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            normalize=normalize,
            seed=seed,
            stratify=stratify,
            task=task,
            scoring_metric=scoring_metric,
            module_factory=module_factory,
        )

    def nested_cv_torch(
        self,
        *,
        param_grid: dict[str, list[Any]] | None = None,
        param_distributions: dict[str, Any] | None = None,
        inner_search: Literal["grid", "randomized", "auto"] = "auto",
        n_iter: int = 5,
        outer_cv: int = 3,
        inner_cv: int = 2,
        epochs: int = 2,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        device: Literal["cpu", "cuda", "mps", "auto"] = "auto",
        normalize: bool = True,
        seed: int = 0,
        stratify: bool = True,
        task: Literal["classification", "regression", "auto"] = "auto",
        scoring_metric: str | None = None,
        module_factory: Any | None = None,
    ) -> Any:
        """Nested Torch CV with fold-local normalize and inner hyperparameter search."""
        return dl_ops.nested_cv_torch(
            self,
            param_grid=param_grid,
            param_distributions=param_distributions,
            inner_search=inner_search,
            n_iter=n_iter,
            outer_cv=outer_cv,
            inner_cv=inner_cv,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            normalize=normalize,
            seed=seed,
            stratify=stratify,
            task=task,
            scoring_metric=scoring_metric,
            module_factory=module_factory,
        )

    def export_torch(
        self,
        path: str | Path,
        *,
        format: Literal["torchscript", "onnx"] = "torchscript",
        opset: int = 17,
        dynamic_batch: bool = True,
        example_input: Any | None = None,
    ) -> Any:
        """Export the last Torch trainer to TorchScript or ONNX (alpha escape hatch)."""
        return dl_ops.export_torch(
            self,
            path=path,
            format=format,
            opset=opset,
            dynamic_batch=dynamic_batch,
            example_input=example_input,
        )

    def fit_torch_ddp(
        self,
        module_factory: Any,
        *,
        epochs: int = 5,
        learning_rate: float = 1e-3,
        mixed_precision: bool = False,
        world_size: int | None = None,
        allow_cpu_ddp: bool = False,
        multi_node: bool = False,
        config: TrainConfig | None = None,
    ) -> Any:
        """DDP training (single-node spawn or multi-node torchrun join).

        Single-node requires multi-GPU unless ``allow_cpu_ddp``. Multi-node
        joins ``WORLD_SIZE``/``RANK``/``LOCAL_RANK``/``MASTER_ADDR``/
        ``MASTER_PORT`` from torchrun.
        """
        return dl_ops.fit_torch_ddp(
            self,
            module_factory,
            epochs=epochs,
            learning_rate=learning_rate,
            mixed_precision=mixed_precision,
            world_size=world_size,
            allow_cpu_ddp=allow_cpu_ddp,
            multi_node=multi_node,
            config=config,
        )

    def make_speech_torch_loaders(
        self,
        *,
        audio_column: str | None = None,
        batch_size: int = 8,
        sample_rate: int = 16_000,
        max_samples: int = 16_000,
        source_sample_rate: int | None = None,
        normalize_audio: bool = True,
        encoder_dim: int = 64,
        shuffle_train: bool = True,
        seed: int = 0,
    ) -> TorchLoaderBundle:
        """Build speech classification loaders (finetune-lite; not FM-from-scratch)."""
        return dl_ops.make_speech_torch_loaders(
            self,
            audio_column=audio_column,
            batch_size=batch_size,
            sample_rate=sample_rate,
            max_samples=max_samples,
            source_sample_rate=source_sample_rate,
            normalize_audio=normalize_audio,
            encoder_dim=encoder_dim,
            shuffle_train=shuffle_train,
            seed=seed,
        )

    def fit_speech_torch(
        self,
        *,
        epochs: int = 5,
        learning_rate: float = 1e-3,
        device: Literal["cpu", "cuda", "mps", "auto"] = "auto",
        freeze_encoder: bool = False,
        audio_column: str | None = None,
        batch_size: int = 8,
        sample_rate: int = 16_000,
        max_samples: int = 16_000,
        source_sample_rate: int | None = None,
        normalize_audio: bool = True,
        encoder_dim: int = 64,
        seed: int = 0,
    ) -> Session:
        """Fine-tune tiny speech encoder + classifier (integration/finetune-lite)."""
        return dl_ops.fit_speech_torch(
            self,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
            freeze_encoder=freeze_encoder,
            audio_column=audio_column,
            batch_size=batch_size,
            sample_rate=sample_rate,
            max_samples=max_samples,
            source_sample_rate=source_sample_rate,
            normalize_audio=normalize_audio,
            encoder_dim=encoder_dim,
            seed=seed,
        )

    def transcribe_speech(
        self,
        *,
        audio_column: str,
        backend: Literal["stub", "transformers"] = "stub",
        model_id: str | None = None,
        sample_rate: int = 16_000,
        max_samples: int = 16_000,
        source_sample_rate: int | None = None,
        partition: Literal["train", "validation", "test", "all"] = "all",
    ) -> Any:
        """ASR transcription (stub CI-safe; transformers via ``buildml[speech]``)."""
        return dl_ops.transcribe_speech(
            self,
            audio_column=audio_column,
            backend=backend,
            model_id=model_id,
            sample_rate=sample_rate,
            max_samples=max_samples,
            source_sample_rate=source_sample_rate,
            partition=partition,
        )

    def serve_bundle(
        self,
        path: str | Path | None = None,
        *,
        kind: Literal["pipeline", "torchscript"] = "pipeline",
        host: str = "127.0.0.1",
        port: int = 8080,
        title: str = "BuildML Serve",
        blocking: bool = False,
        api_keys: str | list[str] | tuple[str, ...] | None = None,
        allow_insecure_public_bind: bool = False,
        ssl_certfile: str | Path | None = None,
        ssl_keyfile: str | Path | None = None,
    ) -> Any:
        """Launch managed local serving (``buildml[serve]``; optional API-key auth).

        Non-loopback binds require ``api_keys`` unless
        ``allow_insecure_public_bind=True``. Optional ``ssl_certfile`` /
        ``ssl_keyfile`` enable local HTTPS. Not an AI tool (CLI/Session-primary).
        """
        return dl_ops.serve_bundle(
            self,
            path,
            kind=kind,
            host=host,
            port=port,
            title=title,
            blocking=blocking,
            api_keys=api_keys,
            allow_insecure_public_bind=allow_insecure_public_bind,
            ssl_certfile=ssl_certfile,
            ssl_keyfile=ssl_keyfile,
        )

    def load_pretrained_backbone(
        self,
        modality: Literal["vision", "audio", "speech"],
        architecture: str | None = None,
        *,
        weights: Literal["none", "mock", "pretrained"] = "mock",
        freeze: bool = True,
        seed: int = 0,
        model_id: str | None = None,
    ) -> Any:
        """Load a curated vision/audio/speech backbone (mock weights default; not a full zoo)."""
        return dl_ops.load_pretrained_backbone(
            self,
            modality,
            architecture,
            weights=weights,
            freeze=freeze,
            seed=seed,
            model_id=model_id,
        )

    def attach_backbone_head(
        self,
        n_classes: int,
        *,
        freeze_backbone: bool | None = None,
    ) -> Any:
        """Attach a classification head to the last :meth:`load_pretrained_backbone` result."""
        return dl_ops.attach_backbone_head(
            self,
            n_classes,
            freeze_backbone=freeze_backbone,
        )

    def evaluate_asr(
        self,
        *,
        hypotheses: list[str] | None = None,
        references: list[str],
        lowercase: bool = True,
    ) -> Any:
        """Score ASR hypotheses vs references (WER/CER).

        When ``hypotheses`` is omitted, reuses texts from the last
        :meth:`transcribe_speech` result.
        """
        return dl_ops.evaluate_asr(
            self,
            hypotheses=hypotheses,
            references=references,
            lowercase=lowercase,
        )

    def pack_torchserve(
        self,
        output_dir: str | Path,
        *,
        torchscript_path: str | Path | None = None,
        model_name: str = "buildml_model",
    ) -> Any:
        """Pack TorchScript into a TorchServe model directory (does not run TorchServe)."""
        return dl_ops.pack_torchserve(
            self,
            output_dir,
            torchscript_path=torchscript_path,
            model_name=model_name,
        )

    def prepare_tensorrt_export(
        self,
        output_dir: str | Path,
        *,
        onnx_path: str | Path | None = None,
        engine_name: str = "model.engine",
        fp16: bool = True,
    ) -> Any:
        """Write a TensorRT trtexec plan for an ONNX file (does not build engines)."""
        return dl_ops.prepare_tensorrt_export(
            self,
            output_dir,
            onnx_path=onnx_path,
            engine_name=engine_name,
            fp16=fp16,
        )

    def emit_k8s_ddp_job(
        self,
        path: str | Path,
        *,
        job_name: str = "buildml-torchrun-ddp",
        namespace: str = "default",
        image: str = "pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime",
        nnodes: int = 2,
        nproc_per_node: int = 2,
        script_path: str = "/workspace/train.py",
        cpu_request: str = "2",
        memory_request: str = "4Gi",
        gpu_limit: int = 1,
        gpu_request: int | None = None,
        service_account: str | None = None,
        include_configmap: bool = True,
    ) -> Any:
        """Emit a Kubernetes Job YAML for torchrun DDP (template; not live orchestration)."""
        return dl_ops.emit_k8s_ddp_job(
            self,
            path,
            job_name=job_name,
            namespace=namespace,
            image=image,
            nnodes=nnodes,
            nproc_per_node=nproc_per_node,
            script_path=script_path,
            cpu_request=cpu_request,
            memory_request=memory_request,
            gpu_limit=gpu_limit,
            gpu_request=gpu_request,
            service_account=service_account,
            include_configmap=include_configmap,
        )

    def emit_k8s_serve_deployment(
        self,
        path: str | Path,
        *,
        name: str = "buildml-serve",
        namespace: str = "default",
        image: str = "python:3.12-slim",
        replicas: int = 1,
        port: int = 8080,
        cpu_request: str = "1",
        memory_request: str = "2Gi",
        gpu_limit: int | None = None,
        service_account: str | None = None,
    ) -> Any:
        """Emit a Kubernetes Deployment+Service YAML for managed serve (template only)."""
        return dl_ops.emit_k8s_serve_deployment(
            self,
            path,
            name=name,
            namespace=namespace,
            image=image,
            replicas=replicas,
            port=port,
            cpu_request=cpu_request,
            memory_request=memory_request,
            gpu_limit=gpu_limit,
            service_account=service_account,
        )

    def domain_adapt_speech_torch(
        self,
        *,
        epochs: int = 5,
        learning_rate: float = 1e-3,
        device: Literal["cpu", "cuda", "mps", "auto"] = "auto",
        freeze_encoder: bool = True,
        audio_column: str | None = None,
        batch_size: int = 8,
        sample_rate: int = 16_000,
        max_samples: int = 16_000,
        source_sample_rate: int | None = None,
        normalize_audio: bool = True,
        encoder_dim: int = 64,
        seed: int = 0,
    ) -> Session:
        """Domain-adapt speech classify (finetune-lite; not FM continued pretrain)."""
        return dl_ops.domain_adapt_speech_torch(
            self,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
            freeze_encoder=freeze_encoder,
            audio_column=audio_column,
            batch_size=batch_size,
            sample_rate=sample_rate,
            max_samples=max_samples,
            source_sample_rate=source_sample_rate,
            normalize_audio=normalize_audio,
            encoder_dim=encoder_dim,
            seed=seed,
        )

    def refuse_speech_foundation_pretrain(self) -> None:
        """Refuse FM-from-scratch / large continued-pretrain with an explicit error."""
        return dl_ops.refuse_speech_foundation_pretrain(self)

    @property
    def dl_speech_result(self) -> Any | None:
        """Last :meth:`transcribe_speech` result, if any."""
        return self._dl_speech_result

    @property
    def dl_backbone(self) -> Any | None:
        """Last :meth:`load_pretrained_backbone` result, if any."""
        return self._dl_backbone

    @property
    def dl_backbone_head(self) -> Any | None:
        """Last :meth:`attach_backbone_head` result, if any."""
        return self._dl_backbone_head

    @property
    def dl_asr_eval(self) -> Any | None:
        """Last :meth:`evaluate_asr` result, if any."""
        return self._dl_asr_eval

    @property
    def dl_train_result(self) -> TrainResult | None:
        """Last Torch :class:`~buildml.dl.results.TrainResult`, if any."""
        return self._dl_train_result

    @property
    def dl_cv_result(self) -> TorchCVResult | None:
        """Last :class:`~buildml.dl.cv.TorchCVResult`, if any."""
        return self._dl_cv_result

    @property
    def dl_search_result(self) -> Any | None:
        """Last :meth:`search_torch` result, if any."""
        return self._dl_search_result

    @property
    def dl_nested_cv_result(self) -> Any | None:
        """Last :meth:`nested_cv_torch` result, if any."""
        return self._dl_nested_cv_result

    @property
    def dl_export_result(self) -> Any | None:
        """Last :meth:`export_torch` result, if any."""
        return self._dl_export_result

    @property
    def dl_ddp_result(self) -> Any | None:
        """Last :meth:`fit_torch_ddp` result, if any."""
        return self._dl_ddp_result

    def torch_training_curve(self) -> TrainingCurveReport:
        """Return structured training-curve teaching data for the last Torch run.

        Requires a prior :meth:`fit_torch` / :meth:`load_torch_bundle`. Torch-free
        to read once :attr:`dl_train_result` exists."""
        return dl_ops.torch_training_curve(self)

    def evaluate_torch(
        self,
        *,
        partition: Literal["train", "validation", "test"] = "test",
        device: str | None = None,
    ) -> DLEvaluateResult:
        """Evaluate the last Torch trainer on a named partition.

        Requires ``pip install 'buildml[torch]'``. Uses loaders from
        :meth:`make_torch_loaders` (rebuilds them if missing)."""
        return dl_ops.evaluate_torch(self, partition=partition, device=device)

    def save_torch_bundle(self, path: str | Path) -> Path:
        """Persist the last Torch trainer as ``buildml.torch_bundle.v1``.

        Distinct from Session checkpoints and classical pipeline bundles.
        See :data:`buildml.dl.checkpoint.CHECKPOINT_BOUNDARY`."""
        return dl_ops.save_torch_bundle(self, path=path)

    def load_torch_bundle(
        self,
        path: str | Path,
        module: Any,
        *,
        map_location: str | None = None,
    ) -> Session:
        """Load a Torch trainer bundle into this Session.

        Restores weights plus optional multimodal preprocess meta. Does not
        rebuild DataLoaders — remake multimodal/text loaders before scoring.

        Parameters
        ----------
        path:
            Bundle directory with ``meta.json`` and ``trainer.pt``.
        module:
            Compatible ``nn.Module`` shell that receives ``load_state_dict``.
        map_location:
            Optional device for ``torch.load`` (default CPU)."""
        return dl_ops.load_torch_bundle(self, path=path, module=module, map_location=map_location)

    def rag_ingest_corpus(
        self,
        source: str | Path | Sequence[Any] | None = None,
        *,
        text_column: str | None = None,
        id_column: str | None = None,
        glob: str = "*.txt",
        encoding: str = "utf-8",
        role: Literal["index", "eval_only"] = "index",
    ) -> Session:
        """Load a text corpus for the RAG path (requires ``buildml[rag]``).

        Provide a file/directory ``source``, an in-memory document sequence, or
        ``text_column`` to bridge the current Session frame. Never silently
        indexes every column.

        Delegates to :mod:`buildml.rag.corpus`. Distinct from classical ingest."""
        return rag_ops.rag_ingest_corpus(
            self,
            source=source,
            text_column=text_column,
            id_column=id_column,
            glob=glob,
            encoding=encoding,
            role=role,
        )

    def rag_chunk(
        self,
        *,
        size: int = 512,
        overlap: int = 64,
        strategy: str = "fixed",
    ) -> Session:
        """Chunk the active RAG corpus (fixed or recursive strategy)."""
        return rag_ops.rag_chunk(self, size=size, overlap=overlap, strategy=strategy)

    def rag_embed_and_index(
        self,
        *,
        embedder: Any | None = "auto",
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        chunk_strategy: str | None = None,
        device: str | None = None,
    ) -> Session:
        """Embed chunks and build the default NumPy cosine index.

        Default embedder is ``auto`` (sentence-transformers when ``buildml[rag]``
        is installed, else hashing with disclosure). Pass ``embedder="hashing"``
        for explicit lexical/CI paths."""
        return rag_ops.rag_embed_and_index(
            self,
            embedder=embedder,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_strategy=chunk_strategy,
            device=device,
        )

    def rag_retrieve(
        self,
        query: str,
        *,
        k: int = 5,
        mode: str | None = None,
        fusion: str | None = None,
        filters: dict[str, Any] | None = None,
        rerank: bool | str | None = None,
        config: Any | None = None,
    ) -> Any:
        """Retrieve ranked chunks (dense / BM25 / hybrid) against the active RAG index.

        Defaults: ``mode="hybrid"`` when ``buildml[rag]`` is installed, else ``dense``.
        Metadata filters and cross-encoder rerank are opt-in."""
        return rag_ops.rag_retrieve(
            self,
            query=query,
            k=k,
            mode=mode,
            fusion=fusion,
            filters=filters,
            rerank=rerank,
            config=config,
        )

    def rag_evaluate(
        self,
        qrels: Any,
        *,
        k: int = 5,
        relevance_mode: str = "document",
        mode: str | None = None,
        retrieve_config: Any | None = None,
    ) -> Any:
        """Score retrieval with gold qrels (recall@k, MRR, nDCG@k, hit-rate@k).

        ``relevance_mode="document"`` (default) scores parent ``doc_id`` hits;
        ``"chunk"`` scores ``chunk_id`` labels. Requires ``buildml[rag]``."""
        return rag_ops.rag_evaluate(
            self,
            qrels=qrels,
            k=k,
            relevance_mode=relevance_mode,
            mode=mode,
            retrieve_config=retrieve_config,
        )

    def rag_generate(
        self,
        query: str,
        *,
        k: int = 5,
        provider: RagChatProvider | None = None,
        mode: str | None = None,
        fusion: str | None = None,
        filters: dict[str, Any] | None = None,
        rerank: bool | str | None = None,
        retrieve_config: RetrieveConfig | None = None,
        config: GenerateConfig | None = None,
        use_last_retrieve: bool = False,
    ) -> GenerateResult:
        """Retrieve context and generate a grounded answer with citations.

        Requires an active RAG index and a chat provider (explicit ``provider``
        or a prior :meth:`ai_configure`). Empty retrieval and provider failures
        raise :class:`~buildml.core.errors.ValidationError`."""
        return rag_ops.rag_generate(
            self,
            query=query,
            k=k,
            provider=provider,
            mode=mode,
            fusion=fusion,
            filters=filters,
            rerank=rerank,
            retrieve_config=retrieve_config,
            config=config,
            use_last_retrieve=use_last_retrieve,
        )

    def rag_upsert(
        self,
        documents: Sequence[Any] | None = None,
        *,
        chunks: Sequence[Any] | None = None,
        chunk: bool = True,
    ) -> Session:
        """Upsert documents or chunks into the active RAG index without a full rebuild.

        Replaces existing ``chunk_id`` rows and re-embeds only new/changed text."""
        return rag_ops.rag_upsert(self, documents=documents, chunks=chunks, chunk=chunk)

    def rag_delete(
        self,
        *,
        chunk_ids: Sequence[str] | None = None,
        doc_ids: Sequence[str] | None = None,
    ) -> Session:
        """Delete chunks by id and/or parent document id from the active RAG index."""
        return rag_ops.rag_delete(self, chunk_ids=chunk_ids, doc_ids=doc_ids)

    @property
    def rag_index_result(self) -> IndexResult | None:
        """Last :class:`~buildml.rag.results.IndexResult`, if any."""
        return self._rag_index_result

    @property
    def rag_retrieve_result(self) -> RetrieveResult | None:
        """Last :class:`~buildml.rag.results.RetrieveResult`, if any."""
        return self._rag_retrieve_result

    @property
    def rag_eval_result(self) -> RagEvalResult | None:
        """Last :class:`~buildml.rag.results.RagEvalResult`, if any."""
        return self._rag_eval_result

    @property
    def rag_generate_result(self) -> GenerateResult | None:
        """Last :class:`~buildml.rag.results.GenerateResult`, if any."""
        return self._rag_generate_result

    def save_rag_bundle(self, path: str | Path) -> Path:
        """Persist the active RAG index as ``buildml.rag_bundle.v1``.

        Distinct from Session checkpoints and Torch trainer bundles.
        See :data:`buildml.rag.checkpoint.CHECKPOINT_BOUNDARY`."""
        return rag_ops.save_rag_bundle(self, path=path)

    def load_rag_bundle(self, path: str | Path) -> Session:
        """Load a RAG bundle into this Session (requires ``buildml[rag]``)."""
        return rag_ops.load_rag_bundle(self, path=path)

    def ai_configure(
        self,
        *,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        api_key_env: str = "BUILDML_OPENAI_API_KEY",
        egress_level: str = "stats_only",
        max_iterations: int = 10,
        max_tokens: int | None = None,
        max_cost_usd: float | None = None,
    ) -> Session:
        """Configure an AI provider for LLM-assisted workflow guidance.

        API keys are read from environment variables by default. Keys are never
        logged, persisted in transcripts/checkpoints, or echoed in errors.

        Parameters
        ----------
        provider
            Provider name (currently ``"openai"`` for OpenAI-compatible APIs,
            or ``"mock"`` for CI testing without real keys).
        model
            Model identifier for the provider.
        api_key
            API key (if None, reads from ``api_key_env`` environment variable).
        api_key_env
            Environment variable name for the API key.
        egress_level
            Default egress level: ``"schema_only"``, ``"stats_only"`` (default),
            ``"redacted_sample"``, or ``"full_sample"``.
        max_iterations
            Maximum tool iterations per AI call (default 10).
        max_tokens
            Optional token budget limit across all AI calls.
        max_cost_usd
            Optional cost budget limit (USD) across all AI calls.

        Returns
        -------
        Session
            Self for chaining."""
        return ai_ops.ai_configure(
            self,
            provider=provider,
            model=model,
            api_key=api_key,
            api_key_env=api_key_env,
            egress_level=egress_level,
            max_iterations=max_iterations,
            max_tokens=max_tokens,
            max_cost_usd=max_cost_usd,
        )

    def ai_egress_preview(
        self,
        *,
        level: str | None = None,
        allow_columns: Sequence[str] | None = None,
        deny_columns: Sequence[str] | None = None,
    ) -> EgressManifest:
        """Preview what data will leave the machine before an LLM call.

        Returns an :class:`~buildml.ai.privacy.EgressManifest` showing columns,
        row counts, and estimated tokens that would be sent to the provider.

        Parameters
        ----------
        level
            Override egress level for this preview (``"schema_only"``,
            ``"stats_only"``, ``"redacted_sample"``, ``"full_sample"``).
        allow_columns
            Explicit allowlist of columns to include.
        deny_columns
            Explicit denylist of columns to exclude.

        Returns
        -------
        EgressManifest
            What would leave the machine at this egress level."""
        return ai_ops.ai_egress_preview(
            self, level=level, allow_columns=allow_columns, deny_columns=deny_columns
        )

    def ai_dry_run(
        self,
        question: str,
        *,
        level: str | None = None,
    ) -> dict[str, Any]:
        """Preview the full prompt payload without calling the provider.

        Returns the system prompt, user message, tools, and egress manifest
        that would be sent to the LLM.

        Parameters
        ----------
        question
            The question or goal to preview.
        level
            Override egress level for this preview.

        Returns
        -------
        dict
            Prompt payload including messages, tools, and egress manifest."""
        return ai_ops.ai_dry_run(self, question=question, level=level)

    def ai_advisor(
        self,
        question: str,
        *,
        level: str | None = None,
        confirm: bool = False,
    ) -> AdvisorResult:
        """Get advisory Q&A guidance about the current workflow (read-only).

        The advisor can describe data, explain operations, and suggest next
        steps, but cannot execute state-changing operations.

        Parameters
        ----------
        question
            The question to ask about the workflow.
        level
            Override egress level for this call.
        confirm
            Required True for FULL_SAMPLE egress (raw data). REDACTED_SAMPLE
            also requires explicit confirmation.

        Returns
        -------
        AdvisorResult
            Advisory response with evidence and recommendations.

        Raises
        ------
        ValidationError
            If FULL_SAMPLE or REDACTED_SAMPLE egress is requested without
            confirm=True."""
        return ai_ops.ai_advisor(self, question=question, level=level, confirm=confirm)

    def ai_plan(
        self,
        goal: str,
        *,
        level: str | None = None,
        confirm: bool = False,
    ) -> PlanResult:
        """Generate a structured workflow plan for a goal (read-only).

        Returns a plan with steps, prerequisites, and expected changes based
        on the current Session state.

        Parameters
        ----------
        goal
            The workflow goal to plan for.
        level
            Override egress level for this call.
        confirm
            Required True for FULL_SAMPLE or REDACTED_SAMPLE egress levels.

        Returns
        -------
        PlanResult
            Structured plan with steps, rationale, and limitations.

        Raises
        ------
        ValidationError
            If FULL_SAMPLE or REDACTED_SAMPLE egress is requested without
            confirm=True."""
        return ai_ops.ai_plan(self, goal=goal, level=level, confirm=confirm)

    def ai_execute(
        self,
        tool: str,
        params: dict[str, Any] | None = None,
        *,
        confirm: bool = False,
    ) -> ExecutorProposal | ExecutorResult:
        """Execute a single tool with propose-confirm-execute flow.

        Proposes the tool execution and requires explicit confirmation for
        write operations. Read-only tools may auto-confirm.

        Parameters
        ----------
        tool
            Name of the tool to execute (must be in the allowed registry).
        params
            Tool arguments as a dictionary.
        confirm
            If True, confirms and executes; otherwise returns a proposal.

        Returns
        -------
        ExecutorProposal or ExecutorResult
            Proposal (if not confirmed) or execution result (if confirmed)."""
        return ai_ops.ai_execute(self, tool=tool, params=params, confirm=confirm)

    def ai_run_plan(
        self,
        plan: Any | None = None,
        *,
        confirmations: dict[int, bool] | None = None,
        auto_confirm_read_only: bool = True,
        stop_on_error: bool = True,
        stop_on_unconfirmed: bool = True,
        max_steps: int | None = None,
    ) -> PlanExecutionResult:
        """Execute a multi-step plan with confirmation gating.

        Default behavior pauses at the first step requiring confirmation that
        hasn't been confirmed. Read-only steps auto-confirm by default.

        Parameters
        ----------
        plan
            The PlanResult to execute. If None, uses the last ai_plan result.
        confirmations
            Dict mapping step_index -> True/False for confirmation decisions.
            Steps not in the dict use default confirmation behavior.
        auto_confirm_read_only
            If True (default), auto-confirm read-only operations.
        stop_on_error
            If True (default), stop execution on first error.
        stop_on_unconfirmed
            If True (default), stop at steps requiring unconfirmed confirmation.
        max_steps
            Maximum number of steps to execute (None = no limit).

        Returns
        -------
        PlanExecutionResult
            Combined result of the plan execution with per-step details.

        Raises
        ------
        ValidationError
            If no plan is provided and no prior ai_plan result exists."""
        return ai_ops.ai_run_plan(
            self,
            plan=plan,
            confirmations=confirmations,
            auto_confirm_read_only=auto_confirm_read_only,
            stop_on_error=stop_on_error,
            stop_on_unconfirmed=stop_on_unconfirmed,
            max_steps=max_steps,
        )

    def ai_run_autonomous(
        self,
        goal: str,
        *,
        plan: Any | None = None,
        confirm_autonomy: bool = False,
        max_steps: int = 8,
        tool_allowlist: Sequence[str] | None = None,
        allow_destructive: bool = False,
        provider_plan: bool = True,
    ) -> Any:
        """Explicit autonomy mode: plan-and-execute allowlisted tools under hard caps.

        Default AI remains propose→confirm→execute. This path auto-confirms only
        after ``confirm_autonomy=True``, with max-steps, allowlist, blocked sample
        egress, destructive gating, and transcript audit. Operator automation —
        not unconstrained agency.
        """
        return ai_ops.ai_run_autonomous(
            self,
            goal,
            plan=plan,
            confirm_autonomy=confirm_autonomy,
            max_steps=max_steps,
            tool_allowlist=tool_allowlist,
            allow_destructive=allow_destructive,
            provider_plan=provider_plan,
        )

    def ai_status(self) -> dict[str, Any]:
        """Get AI operator status including provider, egress, budget, and autonomy.

        Returns factual walkthrough disclosure about the current AI configuration
        and residual autonomy risks when a prior autonomous run exists.

        Returns
        -------
        dict
            Status including provider, egress level, budget, and transcript info."""
        return ai_ops.ai_status(self)

    def save_ai_transcript(self, path: str | Path, *, redact: bool = True) -> Path:
        """Save the AI transcript to a JSON file (secrets redacted by default).

        Transcripts record conversation history, tool calls, and egress
        manifests. API keys and raw data are redacted before saving.

        Parameters
        ----------
        path
            Output file path.
        redact
            If True (default), redact potential secrets before saving.

        Returns
        -------
        Path
            The resolved output path."""
        return ai_ops.save_ai_transcript(self, path=path, redact=redact)

    def load_ai_transcript(self, path: str | Path) -> Session:
        """Load an AI transcript for resume or audit.

        Parameters
        ----------
        path
            Input file path.

        Returns
        -------
        Session
            Self for chaining."""
        return ai_ops.load_ai_transcript(self, path=path)

    @property
    def ai_result(
        self,
    ) -> AdvisorResult | PlanResult | ExecutorResult | PlanExecutionResult | None:
        """Last AI result (AdvisorResult, PlanResult, or ExecutorResult)."""
        return self._ai_result

    @property
    def ai_transcript(self) -> TranscriptStore | None:
        """Active AI transcript store, if any."""
        return self._ai_transcript

    def eval_plots(
        self,
        *,
        partition: Literal["train", "validation", "test"] = "test",
        include_learning_curve: bool = True,
        include_importance: bool = True,
        n_importance_repeats: int = 6,
        learning_curve_cv: int = 3,
        export_figures: str | Path | None = None,
        export_html: str | Path | None = None,
        show: bool = False,
    ) -> PlotBoardReport:
        """Build an evaluation plot board for the fitted estimator.

        Adaptive panels include confusion/residuals, ROC/PR, calibration,
        threshold tradeoffs, learning curves, and permutation importance.
        Panels degrade gracefully when ``predict_proba`` or binary targets
        are unavailable.

        Notes
        -----
        Requires ``pip install 'buildml[viz]'``. Delegates to
        :func:`buildml.model.plot_boards.build_eval_plot_board`."""
        return classical_ops.eval_plots(
            self,
            partition=partition,
            include_learning_curve=include_learning_curve,
            include_importance=include_importance,
            n_importance_repeats=n_importance_repeats,
            learning_curve_cv=learning_curve_cv,
            export_figures=export_figures,
            export_html=export_html,
            show=show,
        )

    @property
    def last_plot_board(self) -> PlotBoardReport | None:
        """Most recent :meth:`eval_plots` / evaluate plot board, if any."""
        return self._last_plot_board

    def compare_models(
        self,
        estimators: dict[str, Any],
        *,
        task: Literal["classification", "regression", "auto"] = "auto",
        partition: Literal["train", "validation", "test"] = "test",
        ranking_metric: str | None = None,
    ) -> ModelComparison:
        """Fit/evaluate multiple estimators and return a ranked comparison card."""
        return classical_ops.compare_models(
            self,
            estimators=estimators,
            task=task,
            partition=partition,
            ranking_metric=ranking_metric,
        )

    def cv_score(
        self,
        estimator: Any,
        *,
        task: Literal["classification", "regression", "auto"] = "auto",
        cv: int | Any = 5,
        cv_strategy: Literal[
            "auto", "kfold", "stratified", "group", "stratified_group", "time"
        ] = "auto",
        scoring_metric: str | None = None,
        groups: pd.Series | None = None,
        preprocess: PreprocessRecipe | None = None,
        allow_session_global_preprocess: bool = False,
    ) -> CVScoreResult:
        """Cross-validate an estimator on the train partition only.

        Returns mean±std fold metrics, interpretation, limitations, and
        recommendations. The test partition is never used for fold membership
        or scoring.

        Parameters
        ----------
        estimator:
            Unfitted sklearn-compatible estimator.
        cv / cv_strategy:
            Fold count or splitter; strategy selects k-fold, stratified,
            group, or time-aware folds when ``cv`` is an integer.
        scoring_metric:
            Primary metric for summaries (defaults by task).
        groups:
            Optional group labels aligned to train rows.
        preprocess:
            Optional fold-local :class:`PreprocessRecipe` refit each fold.
        allow_session_global_preprocess:
            Explicit opt-in when Session-global preprocess already ran.
            Default ``False`` refuses that path even if a fold-local recipe is
            passed (recipes do not rebuild from raw/unpoisoned rows).

        Notes
        -----
        **Leakage:** If Session impute/encode/scale/text/reduce already ran, CV
        refuses unless ``allow_session_global_preprocess=True``. Prefer
        re-ingesting unpoisoned data, then fold-local recipes (including
        ``text`` and ``reduce``) for selection claims that include
        preprocessing. Custom transforms and resample stay Session-global."""
        return classical_ops.cv_score(
            self,
            estimator=estimator,
            task=task,
            cv=cv,
            cv_strategy=cv_strategy,
            scoring_metric=scoring_metric,
            groups=groups,
            preprocess=preprocess,
            allow_session_global_preprocess=allow_session_global_preprocess,
        )

    def nested_cv_score(
        self,
        estimator: Any,
        *,
        param_grid: dict[str, list[Any]] | None = None,
        param_distributions: dict[str, Any] | None = None,
        recipe_grid: dict[str, list[Any]] | None = None,
        recipe_distributions: dict[str, Any] | None = None,
        param_space: Any | None = None,
        recipe_space: Any | None = None,
        inner_search: Literal[
            "auto", "grid", "randomized", "optuna", "evolutionary"
        ] = "auto",
        n_iter: int = 10,
        n_trials: int = 20,
        population_size: int = 8,
        n_generations: int = 3,
        random_state: int | None = 42,
        task: Literal["classification", "regression", "auto"] = "auto",
        outer_cv: int | Any = 5,
        inner_cv: int | Any = 3,
        cv_strategy: Literal[
            "auto", "kfold", "stratified", "group", "stratified_group", "time"
        ] = "auto",
        scoring_metric: str | None = None,
        groups: pd.Series | None = None,
        preprocess: PreprocessRecipe | None = None,
        allow_session_global_preprocess: bool = False,
        warm_start_studies: bool = False,
    ) -> NestedCVResult:
        """Outer-loop estimate after inner hyperparameter / recipe-knob search.

        Each outer fold chooses estimator params and/or fold-local recipe knobs
        (``select_k``, ``n_bins``, …) with inner CV on that fold's training rows
        only, then scores the winner on the outer-eval rows. Session test and
        validation partitions never enter either loop.

        Parameters
        ----------
        param_grid / param_distributions:
            Estimator search space (at most one). Optional when a recipe space
            is provided.
        recipe_grid / recipe_distributions:
            Fold-local recipe knob space (at most one). Requires ``preprocess``.
        param_space / recipe_space:
            Optuna spaces when ``inner_search='optuna'`` (or ``auto`` with these
            args). Declare-style dicts for ``inner_search='evolutionary'``.
            Optuna requires ``pip install 'buildml[optuna]'``.
        inner_search:
            ``auto``, ``grid``, ``randomized``, ``optuna``, or ``evolutionary``.
        n_trials:
            Optuna inner trials per outer fold; evolutionary ``max_evaluations``.
        population_size / n_generations:
            Evolutionary GA knobs when ``inner_search='evolutionary'``.
        outer_cv / inner_cv:
            Outer and inner fold counts or sklearn splitters.
        preprocess:
            Fold-local :class:`PreprocessRecipe` refit in both loops.
        warm_start_studies:
            Opt-in Optuna study sharing across outer folds (default False).
            Safe for Session test/validation (never scored); see nested CV notes.

        Notes
        -----
        Prefer this over reporting :meth:`grid_search` mean CV as a
        post-selection generalization claim. Read ``mean_metrics`` /
        ``std_metrics`` for the outer estimate and
        ``outer_folds[*].best_params`` / ``best_recipe_knobs`` for chosen
        configs (including Optuna / evolutionary winners)."""
        return classical_ops.nested_cv_score(
            self,
            estimator=estimator,
            param_grid=param_grid,
            param_distributions=param_distributions,
            recipe_grid=recipe_grid,
            recipe_distributions=recipe_distributions,
            param_space=param_space,
            recipe_space=recipe_space,
            inner_search=inner_search,
            n_iter=n_iter,
            n_trials=n_trials,
            population_size=population_size,
            n_generations=n_generations,
            random_state=random_state,
            task=task,
            outer_cv=outer_cv,
            inner_cv=inner_cv,
            cv_strategy=cv_strategy,
            scoring_metric=scoring_metric,
            groups=groups,
            preprocess=preprocess,
            allow_session_global_preprocess=allow_session_global_preprocess,
            warm_start_studies=warm_start_studies,
        )

    def grid_search(
        self,
        estimator: Any,
        param_grid: dict[str, list[Any]] | None = None,
        *,
        recipe_grid: dict[str, list[Any]] | None = None,
        task: Literal["classification", "regression", "auto"] = "auto",
        cv: int | Any = 5,
        cv_strategy: Literal[
            "auto", "kfold", "stratified", "group", "stratified_group", "time"
        ] = "auto",
        ranking_metric: str | None = None,
        groups: pd.Series | None = None,
        preprocess: PreprocessRecipe | None = None,
        allow_session_global_preprocess: bool = False,
        refit: bool = True,
    ) -> SearchResult:
        """Grid-search estimator params and/or fold-local recipe knobs.

        Ranks configurations by mean CV score, never peeking at test. When
        ``refit=True`` (default), the winning params/knobs are refit on full
        train and become the active :attr:`fit_result`."""
        return classical_ops.grid_search(
            self,
            estimator=estimator,
            param_grid=param_grid,
            recipe_grid=recipe_grid,
            task=task,
            cv=cv,
            cv_strategy=cv_strategy,
            ranking_metric=ranking_metric,
            groups=groups,
            preprocess=preprocess,
            allow_session_global_preprocess=allow_session_global_preprocess,
            refit=refit,
        )

    def randomized_search(
        self,
        estimator: Any,
        param_distributions: dict[str, Any] | None = None,
        *,
        recipe_distributions: dict[str, Any] | None = None,
        n_iter: int = 10,
        random_state: int | None = 42,
        task: Literal["classification", "regression", "auto"] = "auto",
        cv: int | Any = 5,
        cv_strategy: Literal[
            "auto", "kfold", "stratified", "group", "stratified_group", "time"
        ] = "auto",
        ranking_metric: str | None = None,
        groups: pd.Series | None = None,
        preprocess: PreprocessRecipe | None = None,
        allow_session_global_preprocess: bool = False,
        refit: bool = True,
    ) -> SearchResult:
        """Randomized search over estimator params and/or recipe knobs.

        Same leakage contract as :meth:`grid_search`: folds stay inside train;
        the winner may be refit onto the full training partition."""
        return classical_ops.randomized_search(
            self,
            estimator=estimator,
            param_distributions=param_distributions,
            recipe_distributions=recipe_distributions,
            n_iter=n_iter,
            random_state=random_state,
            task=task,
            cv=cv,
            cv_strategy=cv_strategy,
            ranking_metric=ranking_metric,
            groups=groups,
            preprocess=preprocess,
            allow_session_global_preprocess=allow_session_global_preprocess,
            refit=refit,
        )

    def optuna_search(
        self,
        estimator: Any,
        *,
        param_space: Any | None = None,
        recipe_space: Any | None = None,
        n_trials: int = 20,
        random_state: int | None = 42,
        task: Literal["classification", "regression", "auto"] = "auto",
        cv: int | Any = 5,
        cv_strategy: Literal[
            "auto", "kfold", "stratified", "group", "stratified_group", "time"
        ] = "auto",
        ranking_metric: str | None = None,
        groups: pd.Series | None = None,
        preprocess: PreprocessRecipe | None = None,
        allow_session_global_preprocess: bool = False,
        refit: bool = True,
    ) -> SearchResult:
        """Optuna TPE search with leakage-safe train-fold CV.

        Requires ``pip install 'buildml[optuna]'``. ``param_space`` may be a
        ``trial -> dict`` callable or a declare-style mapping
        (``float`` / ``int`` / ``categorical``). ``recipe_space`` sweeps
        fold-local recipe knobs and requires ``preprocess``."""
        return classical_ops.optuna_search(
            self,
            estimator=estimator,
            param_space=param_space,
            recipe_space=recipe_space,
            n_trials=n_trials,
            random_state=random_state,
            task=task,
            cv=cv,
            cv_strategy=cv_strategy,
            ranking_metric=ranking_metric,
            groups=groups,
            preprocess=preprocess,
            allow_session_global_preprocess=allow_session_global_preprocess,
            refit=refit,
        )

    def evolutionary_search(
        self,
        estimator: Any,
        *,
        param_space: dict[str, Any] | None = None,
        recipe_space: dict[str, Any] | None = None,
        population_size: int = 12,
        n_generations: int = 5,
        elite_size: int = 2,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.2,
        tournament_size: int = 3,
        max_evaluations: int | None = None,
        random_state: int | None = 42,
        task: Literal["classification", "regression", "auto"] = "auto",
        cv: int | Any = 5,
        cv_strategy: Literal[
            "auto", "kfold", "stratified", "group", "stratified_group", "time"
        ] = "auto",
        ranking_metric: str | None = None,
        groups: pd.Series | None = None,
        preprocess: PreprocessRecipe | None = None,
        allow_session_global_preprocess: bool = False,
        refit: bool = True,
    ) -> SearchResult:
        """Genetic-algorithm HPO with leakage-safe train-fold CV.

        In-tree NumPy GA (population, tournament selection, crossover/mutation,
        elitism) — not random search renamed, not NAS, not a swarm zoo.
        ``param_space`` / ``recipe_space`` use declare-style float/int/
        categorical mappings (dicts only)."""
        return classical_ops.evolutionary_search(
            self,
            estimator=estimator,
            param_space=param_space,
            recipe_space=recipe_space,
            population_size=population_size,
            n_generations=n_generations,
            elite_size=elite_size,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            tournament_size=tournament_size,
            max_evaluations=max_evaluations,
            random_state=random_state,
            task=task,
            cv=cv,
            cv_strategy=cv_strategy,
            ranking_metric=ranking_metric,
            groups=groups,
            preprocess=preprocess,
            allow_session_global_preprocess=allow_session_global_preprocess,
            refit=refit,
        )

    @property
    def last_cv(self) -> CVScoreResult | None:
        """Most recent :meth:`cv_score` result, if any."""
        return self._last_cv

    @property
    def last_nested_cv(self) -> NestedCVResult | None:
        """Most recent :meth:`nested_cv_score` result, if any."""
        return self._last_nested_cv

    @property
    def last_search(self) -> SearchResult | None:
        """Most recent grid/randomized/optuna/evolutionary search result, if any."""
        return self._last_search

    def extract_dates(
        self,
        columns: list[str] | tuple[str, ...] | None = None,
        *,
        include_time: bool = False,
        drop_original: bool = False,
    ) -> Session:
        """Expand datetime columns into calendar/time parts (``.dt``-correct)."""
        return preprocess_ops.extract_dates(
            self, columns=columns, include_time=include_time, drop_original=drop_original
        )

    @property
    def date_plan(self) -> DateFeaturePlan | None:
        """Last date-feature plan, if any."""
        return self._date_plan

    def save_model(self, path: str | Path) -> Path:
        """Persist the last fitted estimator bundle.

        This stores the estimator and feature contract only. Prefer
        :meth:`save_pipeline` when impute/encode/scale plans must travel with
        the model."""
        return classical_ops.save_model(self, path=path)

    def load_model(self, path: str | Path) -> Session:
        """Load a previously saved fitted estimator bundle into this session."""
        return classical_ops.load_model(self, path=path)

    def save_pipeline(
        self,
        path: str | Path,
        *,
        evaluate_partition: Literal["train", "validation", "test"] | None = "test",
        title: str | None = None,
    ) -> Path:
        """Persist fitted preprocess plans, estimator, and a model card.

        Layout includes ``model.joblib``, ``plans.joblib``, ``meta.json``, and
        ``model_card`` JSON/Markdown. Persists impute, encode, scale, dates,
        outliers, binning, feature selection, and resample (lineage) plans when
        present. This is not a Session checkpoint: data, splits, and full
        history remain checkpoint concerns.

        Parameters
        ----------
        path:
            Destination directory.
        evaluate_partition:
            If set and a split exists, attach metrics from that partition to
            the model card. Use ``None`` to skip evaluation at save time.
        title:
            Optional model-card title."""
        return classical_ops.save_pipeline(
            self, path=path, evaluate_partition=evaluate_partition, title=title
        )

    def load_pipeline(self, path: str | Path) -> Session:
        """Load a pipeline bundle (estimator + preprocess plans + model card).

        Restores :attr:`fit_result`, preprocess plan attributes, and
        :attr:`model_card`. Does not replace the dataset or split; attach
        compatible data separately (or via :meth:`checkpoint_load`)."""
        return classical_ops.load_pipeline(self, path=path)

    def apply_preprocess_plans(
        self,
        data: Dataset | pd.DataFrame | None = None,
        plans: dict[str, Any] | None = None,
        *,
        inplace: bool = True,
        use_session_plans: bool = True,
    ) -> ApplyPlansResult:
        """Re-apply fitted preprocess plans in score-time order.

        Parameters
        ----------
        data:
            Optional Dataset or DataFrame to transform. Defaults to this
            session's dataset.
        plans:
            Optional plan mapping (checkpoint/pipeline ``plans.joblib`` payload
            or short keys). When omitted and ``use_session_plans=True``, uses
            plans currently attached to the session.
        inplace:
            When ``True`` and ``data`` is omitted (or is this session's
            dataset), replace the session dataset and update the split plan if
            outlier drop rewrote membership.
        use_session_plans:
            Merge session-attached plans under any explicit ``plans`` mapping.

        Returns
        -------
        ApplyPlansResult
            Transformed dataset plus applied/skipped steps and warnings.

        Notes
        -----
        **Order:** dates → impute → outliers → encode → binning → scale →
        feature_select. Resample plans are lineage-only and are never
        reapplied at score time.

        **Leakage:** Plans must already be train-fitted; this method does not
        fit. Missing columns raise :class:`~buildml.core.errors.ValidationError`."""
        return preprocess_ops.apply_preprocess_plans(
            self, data=data, plans=plans, inplace=inplace, use_session_plans=use_session_plans
        )

    def predict_from_pipeline(
        self,
        path: str | Path,
        data: Dataset | pd.DataFrame | None = None,
        *,
        roles: dict[str, ColumnRole | str] | None = None,
        return_proba: bool = False,
        apply_plans: bool = True,
    ) -> PipelinePredictResult:
        """Score a frame through a saved pipeline bundle in one call.

        Parameters
        ----------
        path:
            Pipeline bundle directory.
        data:
            Score frame. Defaults to this session's dataset when omitted.
        roles:
            Optional roles when ``data`` is a bare DataFrame.
        return_proba:
            Request class probabilities when the estimator supports them.
        apply_plans:
            Replay fitted preprocess plans from the bundle before predict
            (default True).

        Notes
        -----
        Does not mutate this session's dataset or fit_result. Prefer this for
        inference-only scoring of new frames."""
        return classical_ops.predict_from_pipeline(
            self,
            path=path,
            data=data,
            roles=roles,
            return_proba=return_proba,
            apply_plans=apply_plans,
        )

    def prepare_design_matrix(
        self,
        *,
        partition: Literal["train", "validation", "test"] = "train",
        columns: list[str] | tuple[str, ...] | None = None,
        sample_rows: int | None = None,
        random_state: int | None = 0,
    ) -> MaterializePrepResult:
        """Project/sample columns via the active engine before sklearn materialize.

        When ``columns`` is omitted and a split exists, prepares the partition
        feature+target design matrix. Disclosures record projection and any
        sampling; sklearn still requires an in-memory matrix."""
        return classical_ops.prepare_design_matrix(
            self,
            partition=partition,
            columns=columns,
            sample_rows=sample_rows,
            random_state=random_state,
        )

    @property
    def model_card(self) -> ModelCard | None:
        """Model card from the last :meth:`save_pipeline` / :meth:`load_pipeline`."""
        return self._model_card

    def eda(
        self,
        *,
        include_plots: bool = False,
        show: bool = False,
        sample_rows: int | None = None,
        max_columns: int = 100,
        max_plots: int = 36,
        export_html: str | Path | None = None,
        export_figures: str | Path | None = None,
        html_format: Literal["studio", "research"] = "studio",
    ) -> EDAReport:
        """Run exploratory analysis.

        Includes quality/pattern screens, distributional tests, correlations,
        mutual information, VIF/PCA, target-aware tests, outlier screens,
        train/test drift (if split exists), adaptive visualization planning,
        narrative generation, and optional HTML/figure export.

        Parameters
        ----------
        include_plots:
            Render adaptive plots (requires ``pip install 'buildml[viz]'``).
        show:
            Print the narrative summary.
        sample_rows:
            Optional analysis sample size for large datasets.
        max_columns:
            Maximum columns used by detailed analyzers. Dataset-wide quality
            checks still cover the full schema.
        max_plots:
            Cap on adaptive plot specifications.
        export_html:
            Optional path for a self-contained HTML artifact. Default format is
            an offline Teaching Studio snapshot (same surface as ``eda_app``).
        export_figures:
            Optional directory for saved PNG figures.
        html_format:
            ``"studio"`` (default) writes the offline Teaching Studio; ``"research"``
            writes the layered research HTML shell with matplotlib embeds."""
        return eda_ops.eda(
            self,
            include_plots=include_plots,
            show=show,
            sample_rows=sample_rows,
            max_columns=max_columns,
            max_plots=max_plots,
            export_html=export_html,
            export_figures=export_figures,
            html_format=html_format,
        )

    def eda_app(
        self,
        *,
        report: EDAReport | None = None,
        host: str = "127.0.0.1",
        port: int = 8765,
        open_browser: bool = True,
        title: str = "BuildML EDA Studio",
        sample_rows: int | None = None,
        max_columns: int = 100,
        blocking: bool = False,
    ) -> EDAAppHandle:
        """Launch the local EDA Teaching Studio web app.

        Runs a FastAPI process on the local host and opens a browser to an
        interactive product UI (domain boards, Teaching Studio, Concept Academy,
        Plotly charts, PDF/CSV export). Requires ``pip install 'buildml[dashboard]'``.

        Parameters
        ----------
        report:
            Optional existing :class:`~buildml.eda.report.EDAReport`. When omitted,
            uses the last ``eda()`` result or runs a fresh analysis.
        host, port:
            Local bind address for the ASGI server.
        open_browser:
            Open the system browser when the server is ready.
        title:
            App header title.
        sample_rows, max_columns:
            Forwarded to ``eda()`` when a fresh report must be computed.
        blocking:
            If True, serve on the current thread until interrupted.

        Returns
        -------
        EDAAppHandle
            Handle with ``url``, ``stop()``, and ``is_running``."""
        return eda_ops.eda_app(
            self,
            report=report,
            host=host,
            port=port,
            open_browser=open_browser,
            title=title,
            sample_rows=sample_rows,
            max_columns=max_columns,
            blocking=blocking,
        )

    def open_eda_dashboard(
        self,
        *,
        report: EDAReport | None = None,
        host: str = "127.0.0.1",
        port: int = 8765,
        open_browser: bool = True,
        title: str = "BuildML EDA Studio",
        sample_rows: int | None = None,
        max_columns: int = 100,
        blocking: bool = False,
    ) -> EDAAppHandle:
        """Alias for :meth:`eda_app`."""
        return eda_ops.open_eda_dashboard(
            self,
            report=report,
            host=host,
            port=port,
            open_browser=open_browser,
            title=title,
            sample_rows=sample_rows,
            max_columns=max_columns,
            blocking=blocking,
        )

    @property
    def last_eda(self) -> EDAReport | None:
        """Most recent EDA report produced by :meth:`eda` or :meth:`eda_app`."""
        return self._last_eda

    def calibration(
        self,
        *,
        partition: Literal["train", "validation", "test"] = "test",
        export_figures: str | Path | None = None,
        export_html: str | Path | None = None,
    ) -> DiagnosticReport:
        """Probability calibration diagnostics for the fitted classifier.

        Returns Brier/ECE, reliability curve points, and interpretation tips.
        Optional figure/HTML export uses the viz extra."""
        return classical_ops.calibration(
            self, partition=partition, export_figures=export_figures, export_html=export_html
        )

    def tune_threshold(
        self,
        *,
        partition: Literal["train", "validation", "test"] = "test",
        fp_cost: float | None = None,
        fn_cost: float | None = None,
        tp_benefit: float = 0.0,
        tn_benefit: float = 0.0,
        export_figures: str | Path | None = None,
        export_html: str | Path | None = None,
    ) -> DiagnosticReport:
        """Sweep binary decision thresholds with precision/recall/F1 and optional costs.

        Parameters
        ----------
        partition:
            Rows used for the sweep. Prefer ``validation`` when selecting a
            policy; use ``test`` only to confirm a fixed threshold.
        fp_cost, fn_cost:
            Non-negative false-positive / false-negative costs. Provide both to
            minimize expected cost on the scored partition.
        tp_benefit, tn_benefit:
            Optional benefits subtracted from cost for true positives / negatives."""
        return classical_ops.tune_threshold(
            self,
            partition=partition,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
            tp_benefit=tp_benefit,
            tn_benefit=tn_benefit,
            export_figures=export_figures,
            export_html=export_html,
        )

    def learning_curve(
        self,
        estimator: Any,
        *,
        task: Literal["classification", "regression", "auto"] = "auto",
        cv: int = 5,
        export_figures: str | Path | None = None,
        export_html: str | Path | None = None,
    ) -> DiagnosticReport:
        """Compute learning curves on the training partition."""
        return classical_ops.learning_curve(
            self,
            estimator=estimator,
            task=task,
            cv=cv,
            export_figures=export_figures,
            export_html=export_html,
        )

    def feature_importance(
        self,
        *,
        partition: Literal["train", "validation", "test"] = "test",
        n_repeats: int = 8,
        export_figures: str | Path | None = None,
        export_html: str | Path | None = None,
    ) -> DiagnosticReport:
        """Permutation feature importance on a holdout partition."""
        return classical_ops.feature_importance(
            self,
            partition=partition,
            n_repeats=n_repeats,
            export_figures=export_figures,
            export_html=export_html,
        )

    def error_slices(
        self,
        *,
        by: str | Sequence[str],
        partition: Literal["train", "validation", "test"] = "test",
        max_segments: int = 20,
        min_segment_n: int = 5,
        export_html: str | Path | None = None,
    ) -> DiagnosticReport:
        """Slice prediction errors by one or more columns on a partition.

        Notes
        -----
        Observational only: segment gaps are not fairness proof. Prefer
        validation for exploration and keep test for a final estimate.
        Segments with ``n < min_segment_n`` are listed under ``small_segments``."""
        return classical_ops.error_slices(
            self,
            by=by,
            partition=partition,
            max_segments=max_segments,
            min_segment_n=min_segment_n,
            export_html=export_html,
        )

    def resample(
        self,
        *,
        sampler: Literal[
            "smote",
            "random_oversample",
            "random_undersample",
            "adasyn",
            "borderline_smote",
        ] = "smote",
        random_state: int = 42,
        sampling_strategy: str | float | dict[str, float] = "auto",
    ) -> Session:
        """Resample the **train** partition only (requires ``buildml[imbalanced]``).

        Validation/test rows are never altered. See
        :meth:`resample_strategies` for strategy guidance."""
        return preprocess_ops.resample(
            self, sampler=sampler, random_state=random_state, sampling_strategy=sampling_strategy
        )

    def resample_strategies(self) -> list[dict[str, Any]]:
        """List imbalance resampling strategies and when to use them."""
        return preprocess_ops.resample_strategies(self)

    @property
    def resample_plan(self) -> ResamplePlan | None:
        """Last train-only resample plan, if any."""
        return self._resample_plan

    def to_engine(self, engine: EngineName | str | None = None) -> Any:
        """Materialize the current dataset in a selected engine's native type.

        Parameters
        ----------
        engine:
            Target engine. Defaults to the dataset's current engine setting."""
        return data_ops.to_engine(self, engine=engine)

    def checkpoint_save(
        self,
        path: str | Path,
        *,
        sidecar_partition_rows: int | None = None,
        sidecar_compression: str | None = None,
        sidecar_layout: str | None = None,
    ) -> Path:
        """Save a resumable checkpoint bundle for mid-loop exit.

        Parameters
        ----------
        path:
            Destination directory.
        sidecar_partition_rows:
            Optional rows-per-partition for native sidecars (default 25_000).
            Ignored when ``sidecar_layout='single'``.
        sidecar_compression:
            Optional Parquet compression for native sidecars (default ``zstd``).
        sidecar_layout:
            ``'auto'`` (default; partition at ≥50_000 rows), ``'single'``, or
            ``'partitioned'``."""
        return data_ops.checkpoint_save(
            self,
            path=path,
            sidecar_partition_rows=sidecar_partition_rows,
            sidecar_compression=sidecar_compression,
            sidecar_layout=sidecar_layout,
        )

    @classmethod
    def checkpoint_load(cls, path: str | Path, *, data_only: bool = False) -> Session:
        """Load a checkpoint bundle and validate reattach conditions.

        Parameters
        ----------
        path:
            Checkpoint directory.
        data_only:
            If True, ignore metadata and treat data as a fresh ingest.

        Notes
        -----
        When ``plans.joblib`` is present, preprocess plan objects are restored
        for mid-loop resume. Checkpoints still do not embed a fitted estimator;
        use :meth:`load_pipeline` for inference artifacts."""
        return data_ops.checkpoint_load_session(cls, path=path, data_only=data_only)

    def reattach(self, path: str | Path, *, data_only: bool = False) -> Session:
        """Replace this session state from a checkpoint path (instance helper)."""
        return data_ops.reattach(self, path=path, data_only=data_only)

    def to_pandas(self) -> pd.DataFrame:
        """Escape hatch: copy the current dataset as a Pandas DataFrame."""
        return data_ops.to_pandas(self)

    def to_parquet(self, path: str | Path) -> Path:
        """Write the current dataset to Parquet."""
        return data_ops.to_parquet(self, path=path)

    def head(self, n: int = 5) -> pd.DataFrame:
        """Preview the first rows."""
        return data_ops.head(self, n=n)

    def with_mode(self, mode: DataMode | str) -> Session:
        """Record a mode override on the dataset metadata.

        Accepted values are ``memory`` and ``lazy``. Legacy ``out_of_core`` is
        coerced to ``lazy`` (there is no separate out-of-core fit mode)."""
        return data_ops.with_mode(self, mode=mode)

    def with_engine(self, engine: EngineName | str) -> Session:
        """Select a compute engine and attach a native handle when applicable.

        Parameters
        ----------
        engine:
            ``pandas``, ``polars``, or ``duckdb``.

        Notes
        -----
        Polars/DuckDB attach a persistent ``Dataset.native`` table used by
        :meth:`prepare_design_matrix`, :meth:`~buildml.data.dataset.Dataset.project`,
        and sample/filter helpers before Pandas materialization. Sklearn fit
        still requires an in-memory design matrix. Missing extras raise
        :class:`~buildml.core.errors.MissingExtraError`."""
        return data_ops.with_engine(self, engine=engine)

    def sync_native(self) -> Session:
        """Rebuild ``Dataset.native`` from the current Pandas frame (eager).

        Session preprocess transforms already sync when ``engine`` is Polars or
        DuckDB. Call this after external Pandas mutation of ``dataset.frame``,
        or after a transform that opted out of sync. This is not a lazy plan
        of prior steps — it converts the full current frame into the engine
        table."""
        return data_ops.sync_native(self)

    def metadata(self) -> dict[str, Any]:
        """Session/dataset metadata snapshot."""
        return data_ops.metadata(self)

    def workflow(self) -> tuple[WorkflowStep, ...]:
        """Resolve every public operation against current workflow state."""
        return workflow_ops.workflow(self)

    def walkthrough(
        self,
        *,
        export_html: str | Path | None = None,
    ) -> WorkflowWalkthroughReport:
        """Build a workflow walkthrough from resolver state and history."""
        return workflow_ops.walkthrough(self, export_html=export_html)

    @property
    def last_walkthrough(self) -> WorkflowWalkthroughReport | None:
        """Most recently generated workflow walkthrough, if any."""
        return self._last_walkthrough

    def explain(
        self,
        operation: str | None = None,
        *,
        moment: Literal["before", "after"] = "before",
    ) -> Any:
        """Explain an operation before/after execution, or return the workflow."""
        return workflow_ops.explain(self, operation=operation, moment=moment)

    def _session_preprocess_applied(self) -> bool:
        """True when Session-level train-global preprocess plans exist."""
        return state.session_preprocess_applied(self)

    def _plan_objects(self) -> dict[str, Any]:
        return state.plan_objects(self)

    def _preprocess_summary(self) -> dict[str, Any]:
        return state.preprocess_summary(self)

    def _restore_plans(self, plans: dict[str, Any] | None) -> None:
        return state.restore_plans(self, plans=plans)

    def _clear_plans(self) -> None:
        return state.clear_plans(self)

    def _record(
        self,
        action: str,
        details: dict[str, Any] | None = None,
        *,
        decision_origin: Literal["automatic", "recommended", "explicit"] = "explicit",
        warnings: list[str] | tuple[str, ...] = (),
        result_summary: dict[str, Any] | None = None,
    ) -> None:
        return state.record(
            self,
            action=action,
            details=details,
            decision_origin=decision_origin,
            warnings=warnings,
            result_summary=result_summary,
        )
