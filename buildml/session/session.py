"""BuildML Session: public facade composed from domain mixins.

Implementation lives in ``buildml.session.mixins`` (signatures/docstrings) and
``buildml.session.*_ops`` (orchestration logic). ``Session`` remains the sole
public class users import; method names and signatures are unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from buildml.checkpoint.validate import ReattachResult
from buildml.core.errors import ValidationError
from buildml.core.results import IngestReport
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.explain.history import normalize_history
from buildml.session import state
from buildml.session.mixins import (
    ActivelearningSessionMixin,
    AiSessionMixin,
    AnomalySessionMixin,
    AutomlSessionMixin,
    CausalSessionMixin,
    CbrSessionMixin,
    ClassicalSessionMixin,
    DataSessionMixin,
    DecisionSessionMixin,
    DlSessionMixin,
    EdaSessionMixin,
    EnsembleSessionMixin,
    FairnessSessionMixin,
    FederatedSessionMixin,
    ForecastSessionMixin,
    GraphSessionMixin,
    KgSessionMixin,
    MetalearningSessionMixin,
    MultitaskSessionMixin,
    NlpSessionMixin,
    OnlineSessionMixin,
    PreprocessSessionMixin,
    ProbabilisticSessionMixin,
    RagSessionMixin,
    RankingSessionMixin,
    RecommenderSessionMixin,
    RlSessionMixin,
    SelfsupervisedSessionMixin,
    SemisupervisedSessionMixin,
    SymbolicSessionMixin,
    SyntheticSessionMixin,
    TdaSessionMixin,
    TimeseriesSessionMixin,
    UnsupervisedSessionMixin,
    WorkflowSessionMixin,
)

if TYPE_CHECKING:

    from pathlib import Path

    from buildml.eda.report import EDAReport
    from buildml.model.compare import ModelComparison
    from buildml.model.diagnostics import DiagnosticReport
    from buildml.model.plot_boards import PlotBoardReport
    from buildml.model.selection import CVScoreResult, NestedCVResult, SearchResult
    from buildml.model.supervised import FitResult
    from buildml.pipeline.card import ModelCard
    from buildml.preprocess.binning import BinningPlan
    from buildml.preprocess.custom import CustomTransformPlan
    from buildml.preprocess.dates import DateFeaturePlan
    from buildml.preprocess.encode import EncodePlan
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


class Session(
    ActivelearningSessionMixin,
    AiSessionMixin,
    AnomalySessionMixin,
    AutomlSessionMixin,
    CausalSessionMixin,
    CbrSessionMixin,
    ClassicalSessionMixin,
    DataSessionMixin,
    DecisionSessionMixin,
    DlSessionMixin,
    EdaSessionMixin,
    EnsembleSessionMixin,
    FairnessSessionMixin,
    FederatedSessionMixin,
    ForecastSessionMixin,
    GraphSessionMixin,
    KgSessionMixin,
    MetalearningSessionMixin,
    MultitaskSessionMixin,
    NlpSessionMixin,
    OnlineSessionMixin,
    PreprocessSessionMixin,
    ProbabilisticSessionMixin,
    RagSessionMixin,
    RankingSessionMixin,
    RecommenderSessionMixin,
    RlSessionMixin,
    SelfsupervisedSessionMixin,
    SemisupervisedSessionMixin,
    SymbolicSessionMixin,
    SyntheticSessionMixin,
    TdaSessionMixin,
    TimeseriesSessionMixin,
    UnsupervisedSessionMixin,
    WorkflowSessionMixin,
):
    """Primary user-facing object for BuildML 2.x workflows.

    A ``Session`` is a workflow that remembers itself. Instead of juggling
    loose DataFrames, fitted scalers, and index arrays, you attach data to one
    object and call steps on it. The session tracks four things you would
    otherwise have to track by hand:

    **The data and what each column means.** :meth:`ingest` attaches a table;
    :meth:`set_roles` labels each column as a ``feature``, the ``target``, an
    ``id``, a ``group``, a ``time`` stamp, a sample ``weight``, or ``ignore``.
    Every later step reads those roles, which is why you never re-list your
    feature columns.

    **Which rows may be learned from.** A split (:meth:`split`,
    :meth:`group_split`, :meth:`time_split`, :meth:`inject_split`) records
    train/validation/test membership once. Preprocessing steps then fit their
    statistics on the train rows alone and apply them everywhere: the single
    most common source of silently optimistic scores, handled for you.

    **A record of every decision.** Each call appends to :attr:`history` with
    its parameters and whether the choice was yours or a default. That history
    drives :meth:`summarize_history`, :meth:`walkthrough`, :meth:`workflow`,
    and the model card, so a finished session can explain itself.

    **Fitted plans and results.** Transforms return reusable plan objects
    (:attr:`scale_plan`, :attr:`encode_plan`, …) and trainers store their
    outputs on ``*_result`` properties, so scoring new data later reproduces
    exactly what training did.

    Most methods return ``self``, so steps chain. Methods that produce
    something you inspect: frames, reports, fitted results: return that
    instead.

    The classical path is ingest, roles, split, preprocess, fit, evaluate. The
    same session also carries deep learning, forecasting, anomaly detection,
    ranking, recommenders, causal inference, RL, and other domains.

    Prefer namespaced facades for domains (``session.fairness.evaluate``,
    ``session.anomaly.fit``, ``session.rag.retrieve``, …). Flat
    ``fit_<domain>`` / ``evaluate_<domain>`` aliases remain until BuildML 3.0
    and emit ``DeprecationWarning`` for domain actions. Classical core flat
    methods (``ingest`` / ``fit`` / ``evaluate`` / …) stay dual first-class;
    see ``docs/session-facade-migration.md``.

    Examples
    --------
    The full classical workflow, end to end:

    >>> import pandas as pd
    >>> from buildml import Session
    >>> frame = pd.DataFrame({"a": [1, 2, 3, 4], "y": [0, 1, 0, 1]})
    >>> session = Session.ingest(frame)
    >>> _ = session.set_roles({"a": "feature", "y": "target"})
    >>> _ = session.split(test_size=0.5, stratify=True)
    >>> session.partition("train").shape[0] > 0
    True

    Steps chain, because each returns the session:

    >>> from sklearn.ensemble import RandomForestClassifier
    >>> report = (
    ...     Session.ingest("customers.csv")
    ...     .set_roles({"churned": "target", "customer_id": "id"})
    ...     .split(test_size=0.2, stratify=True)
    ...     .impute()
    ...     .encode()
    ...     .scale()
    ...     .fit(RandomForestClassifier())
    ...     .evaluate()
    ... )  # doctest: +SKIP

    Notes
    -----
    **Leakage:** Split before you preprocess. Fitting a scaler or encoder on
    all rows lets test-set information reach the model and inflates your
    scores. :meth:`assert_can_fit` turns that rule into an error rather than a
    convention.

    ``with session:`` calls :meth:`close_native` on exit so owned DuckDB
    connections on the Session dataset are released safely.

    See Also
    --------
    Session.ingest : Entry point that creates a session from data.
    Session.explain : Plain-language explanation of any BuildML concept.
    Session.walkthrough : Narrated report of everything this session did.
    """

    def __init__(
        self,
        dataset: Dataset | None = None,
        ingest_report: IngestReport | None = None,
        split_plan: SplitPlan | None = None,
        history: list[dict[str, Any]] | None = None,
        reattach_result: ReattachResult | None = None,
    ) -> None:
        """Construct a session directly from already-prepared state.

        Prefer :meth:`ingest` for new work and :meth:`checkpoint_load` for
        resumed work. This constructor exists for those two paths and for
        tests that need to place a session in a specific state.

        Parameters
        ----------
        dataset:
            Data handle to attach. ``None`` creates an empty session, which is
            what :meth:`ingest` produces for a dry run: every data-dependent
            method then raises until a dataset arrives.
        ingest_report:
            Findings from the automated ingest scan (detected format, chosen
            engine, size warnings). ``None`` when the session was not created
            by :meth:`ingest`.
        split_plan:
            Pre-existing train/validation/test membership. ``None`` means no
            split yet, and fit-capable steps will refuse to run.
        history:
            Prior operation records to continue appending to, used when
            resuming from a checkpoint so the audit trail survives the restart.
        reattach_result:
            Validation outcome from a checkpoint load, recording whether the
            restored data still matches what the checkpoint expected.
        """
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
        self._nlp_text_plan: NlpTextPlan | None = None
        self._nlp_topic_plan: NlpTopicPlan | None = None
        self._nlp_fit_result: NlpFitResult | None = None
        self._nlp_eval_result: NlpEvalResult | None = None
        self._nlp_predict_result: NlpPredictResult | None = None
        self._nlp_interpret_result: NlpInterpretResult | None = None
        self._nlp_topic_result: NlpTopicResult | None = None
        self._nlp_topic_assign_result: NlpTopicAssignResult | None = None
        self._nlp_keyphrase_result: NlpKeyphraseResult | None = None
        self._nlp_sentiment_result: NlpSentimentResult | None = None
        self._nlp_entity_result: NlpEntityResult | None = None
        self._nlp_summary_result: NlpSummaryResult | None = None
        self._nlp_language_result: NlpLanguageResult | None = None
        self._nlp_profile_result: NlpCorpusProfile | None = None
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
        self._fairness_report: Any | None = None
        self._fairness_mitigation_suggestion: Any | None = None
        self._last_evaluate_partition: str | None = None
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
        """Enter a ``with session:`` block and return the session itself.

        Using a session as a context manager guarantees that native database
        connections are closed when the block ends, even if an exception is
        raised inside it. This matters when :meth:`with_engine` has attached a
        DuckDB connection, which holds a file handle.

        Returns
        -------
        Session
            This same session, so ``with Session.ingest(path) as session:``
            binds the session to the loop variable.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        """Leave a ``with session:`` block, releasing native resources.

        Calls :meth:`close_native`. Exceptions are not suppressed: returning
        ``None`` lets any error propagate to the caller as normal.

        Parameters
        ----------
        exc_type:
            Class of the exception that ended the block, or ``None`` on a
            clean exit.
        exc:
            The exception instance itself, or ``None`` on a clean exit.
        tb:
            Traceback for the exception, or ``None`` on a clean exit.
        """
        self.close_native()

    @property
    def dataset(self) -> Dataset:
        """The data this session is working on.

        A :class:`~buildml.data.dataset.Dataset` wraps the table together with
        its schema, column roles, chosen engine, and any native engine handle.
        Reach for it when you need the underlying frame
        (``session.dataset.frame``) or want to check what roles are currently
        assigned.

        Returns
        -------
        ~buildml.data.dataset.Dataset
            The attached data handle. Never ``None``: the accessor raises
            rather than handing back an empty value, so downstream code does
            not have to guard.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            No data is attached. This happens on a session built by a dry-run
            :meth:`ingest`, which carries a report but no table.
        """
        if self._dataset is None:
            raise ValidationError("Session has no dataset. Call Session.ingest(...) first.")
        return self._dataset

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


# Additive namespaced facades + domain flat-method deprecation wrappers.
from buildml.session.facades import install_session_facades

install_session_facades(Session)
