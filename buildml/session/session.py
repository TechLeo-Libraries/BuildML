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
    nlp_ops,
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
    statistics on the train rows alone and apply them everywhere — the single
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
    something you inspect — frames, reports, fitted results — return that
    instead.

    The classical path is ingest, roles, split, preprocess, fit, evaluate. The
    same session also carries deep learning, forecasting, anomaly detection,
    ranking, recommenders, causal inference, RL, and other domains; each
    follows a ``fit_<domain>`` / ``evaluate_<domain>`` / ``save_<domain>_bundle``
    naming pattern so learning one domain teaches you the rest.

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
            what :meth:`ingest` produces for a dry run — every data-dependent
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

    def close_native(self) -> None:
        """Close the DuckDB connection this session owns, if it has one.

        :meth:`with_engine` with ``'duckdb'`` opens a connection that stays
        alive so later queries can reuse it. That connection holds an operating
        system handle, so it should be released when you are finished — on
        Windows especially, an open handle can block deleting or overwriting
        the underlying file.

        Calling this is always safe. It does nothing when no dataset is
        attached, when the engine is Pandas or Polars, or when the connection
        has already been closed. Datasets derived from a parent share the
        parent's connection and are not owners, so closing a derived handle
        does not pull the connection out from under the original.

        Notes
        -----
        You rarely need to call this by hand — prefer ``with session:``, which
        calls it for you on exit.
        """
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
        """Create a session by loading a table, and inspect it while loading.

        This is where every BuildML workflow starts. Give it a DataFrame you
        already have or a path to a file on disk, and you get back a
        :class:`Session` holding the data plus an
        :class:`~buildml.core.results.IngestReport` describing what was found:
        the detected file format, the column schema, row and byte estimates,
        which compute engines are installed, and any warnings worth reading.

        Ingest does not just read the file — it decides *how* to read it. A
        small CSV loads straight into Pandas. A large one does not, because
        quietly pulling a multi-gigabyte file into memory is how notebooks die.
        Instead BuildML refuses and tells you the four ways forward: force it
        with ``mode='memory'``, look before you leap with ``dry_run=True``,
        sample with ``read_nrows``, or install ``buildml[engines]`` and load
        natively through Polars or DuckDB.

        Parameters
        ----------
        source:
            The data to load: a :class:`pandas.DataFrame` you already hold in
            memory, or a path to a ``.csv``, ``.tsv``, ``.parquet``, or
            ``.arrow``/``.feather`` file. Format is detected from the file, not
            assumed from the extension alone.
        mode:
            How the data should live in memory — ``'memory'`` for a fully
            materialised frame, ``'lazy'`` to keep an engine handle and defer
            work until something forces materialisation. Leave as ``None`` to
            let the size heuristic decide, and pass ``'memory'`` explicitly to
            override a refusal on a large file.
        engine:
            Which compute engine backs the data: ``'pandas'``, ``'polars'``, or
            ``'duckdb'``. ``None`` picks the best available for the estimated
            size. Polars and DuckDB read the file natively with no Pandas-first
            pass, which is what makes large sources tractable, but they require
            ``pip install 'buildml[engines]'``.
        dry_run:
            When True, inspect without loading: you get a session carrying the
            report but no dataset. Use this to see the schema, size estimate,
            and warnings before committing memory to a file you are unsure
            about.
        mock_byte_estimate:
            Pretend the source is this many bytes when applying the size
            heuristics. Exists so tests and demonstrations can trigger the
            large-file path without producing a large file.
        read_nrows:
            Read at most this many rows from a CSV. A quick way to work on a
            representative slice of something too big to load whole — but note
            that statistics from a truncated read describe the slice, not the
            file.

        Returns
        -------
        Session
            A new session. It carries a dataset unless ``dry_run=True`` (or a
            large-source refusal under dry run), and always carries
            :attr:`ingest_report`. Read that report's ``warnings`` before
            continuing; it is where scale and engine advice appears.

        Raises
        ------
        ~buildml.core.errors.IngestError
            The path does not exist, the format is not one BuildML reads, or
            the source is large enough that loading it blindly into Pandas
            would be reckless. The message names the specific way out.
        ~buildml.core.errors.MissingExtraError
            You asked for ``engine='polars'`` or ``'duckdb'`` without the
            matching extra installed.

        Notes
        -----
        **Scale:** Large paths are not silently loaded into Pandas. Use
        ``dry_run=True``, ``read_nrows``, ``mode='memory'`` (force), or engine
        extras.

        **Leakage:** Call :meth:`split` before fit-capable operations. Use
        :meth:`assert_can_fit` to enforce train-only fit scope.

        Examples
        --------
        The ordinary case — a DataFrame already in hand:

        >>> import pandas as pd
        >>> from buildml import Session
        >>> session = Session.ingest(pd.DataFrame({"a": [1, 2], "y": [0, 1]}))
        >>> session.dataset.frame.shape
        (2, 2)

        Inspect a file before deciding how to load it:

        >>> probe = Session.ingest("events.parquet", dry_run=True)  # doctest: +SKIP
        >>> probe.ingest_report.row_estimate  # doctest: +SKIP
        4210332
        >>> probe.ingest_report.warnings  # doctest: +SKIP
        ['Source looks large (812993024 bytes estimated). ...']

        Then load it natively rather than through Pandas:

        >>> session = Session.ingest(
        ...     "events.parquet", engine="duckdb", mode="lazy"
        ... )  # doctest: +SKIP

        See Also
        --------
        Session.set_roles : The next step — tell BuildML what the columns mean.
        Session.with_engine : Switch engines after ingest.
        Session.checkpoint_load : Resume a saved session instead of re-reading.
        """
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
        """The data this session is working on.

        A :class:`~buildml.data.dataset.Dataset` wraps the table together with
        its schema, column roles, chosen engine, and any native engine handle.
        Reach for it when you need the underlying frame
        (``session.dataset.frame``) or want to check what roles are currently
        assigned.

        Returns
        -------
        ~buildml.data.dataset.Dataset
            The attached data handle. Never ``None`` — the accessor raises
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

    @property
    def ingest_report(self) -> IngestReport | None:
        """What the loader found when reading the source.

        Holds the detected format, column schema, row and byte estimates, the
        mode and engine BuildML recommended versus the ones actually used, and
        a ``warnings`` list. Those warnings are the useful part: they flag
        oversized sources, engines that would help if installed, and lazy
        handles that have not yet been materialised.

        ``None`` when the session was not created by :meth:`ingest` — for
        example one restored by :meth:`checkpoint_load`, where the original
        read happened in an earlier run.
        """
        return self._ingest_report

    @property
    def split_plan(self) -> SplitPlan | None:
        """Which rows belong to train, validation, and test.

        A :class:`~buildml.data.splits.SplitPlan` records the strategy used
        (``random``, ``stratified``, ``group``, ``time``, or ``injected``) plus
        the exact row positions in each partition. Because membership is stored
        as indices rather than copied frames, every later step agrees on what
        "train" means without passing data around.

        ``None`` until you call :meth:`split`, :meth:`group_split`,
        :meth:`time_split`, or :meth:`inject_split`. While it is ``None``,
        fit-capable steps refuse to run.
        """
        return self._split_plan

    @property
    def history(self) -> list[dict[str, Any]]:
        """Every operation this session has performed, in order.

        Each entry records the operation name, the parameters it ran with,
        whether each choice was explicit or a BuildML default, a before/after
        state snapshot, and any warnings raised. This is the raw material
        behind :meth:`summarize_history`, :meth:`walkthrough`,
        :meth:`workflow`, and the model card — and it is what lets a finished
        session answer "why is this number what it is?".

        Returns
        -------
        list of dict
            A shallow copy, so appending to the returned list does not corrupt
            the session's own record. The dictionaries inside are shared; treat
            them as read-only.
        """
        return list(self._history)

    @property
    def reattach_result(self) -> ReattachResult | None:
        """Whether restored checkpoint data still matched what was expected.

        When :meth:`checkpoint_load` or :meth:`reattach` restores a session, it
        re-checks the data against the fingerprint stored in the checkpoint and
        records a :class:`~buildml.checkpoint.validate.ReattachResult` with a
        ``status`` and human-readable ``messages``. Check it after resuming: a
        non-clean status means the underlying data drifted since the checkpoint
        was written, so plans fitted then may no longer be valid now.

        ``None`` when this session was never restored from a checkpoint.
        """
        return self._reattach_result

    def set_roles(self, mapping: dict[str, str | ColumnRole]) -> Session:
        """Declare what each column means, so later steps can act on it.

        A role is BuildML's answer to "which column is the answer, and which
        ones are allowed to help predict it?". Assigning roles once removes the
        need to pass column lists into every later call — :meth:`scale` knows
        to leave your identifier alone, :meth:`split` knows what to stratify
        on, and :meth:`fit` knows what it is predicting.

        The roles are:

        ``target``
            The column being predicted. Supervised methods require exactly one.
        ``feature``
            An input the model may learn from. Columns default to this.
        ``id``
            A row identifier. Carried through but never used as a predictor,
            and never modified by preprocessing.
        ``group``
            An entity that owns several rows — a customer, a patient, a
            document. :meth:`group_split` keeps all of a group's rows on the
            same side of the split.
        ``time``
            The timestamp that orders the data. Used by :meth:`time_split` and
            the forecasting methods.
        ``weight``
            Per-row importance passed to estimators that accept sample weights.
        ``ignore``
            Kept in the table, excluded from everything.

        Parameters
        ----------
        mapping:
            Column name to role. Roles may be given as strings (``'target'``)
            or as :class:`~buildml.core.types.ColumnRole` members. Only the
            columns you name change; anything you leave out keeps its current
            role.

        Returns
        -------
        Session
            ``self``, so this call chains into the next step.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            A named column is not in the dataset, or the role is not one of the
            values above.

        Examples
        --------
        >>> import pandas as pd
        >>> from buildml import Session
        >>> frame = pd.DataFrame({"id": [1, 2], "x": [0.5, 0.9], "y": [0, 1]})
        >>> session = Session.ingest(frame)
        >>> _ = session.set_roles({"id": "id", "y": "target"})
        >>> session.dataset.roles["y"].value
        'target'

        Marking a grouping column is what makes a leakage-safe split possible:

        >>> _ = session.set_roles({"customer_id": "group"})  # doctest: +SKIP
        >>> _ = session.group_split(test_size=0.2)  # doctest: +SKIP

        See Also
        --------
        Session.split : The next step for independent rows.
        Session.group_split : Use when rows share an entity.
        Session.time_split : Use when rows are ordered in time.
        """
        return data_ops.set_roles(self, mapping=mapping)

    def split(
        self,
        *,
        test_size: float | int = 0.2,
        validation_size: float | int | None = None,
        random_state: int | None = 42,
        stratify: bool = False,
    ) -> Session:
        """Randomly hold back rows so you can measure honest performance.

        A model that has seen a row can usually predict it. To find out whether
        it learned anything general, you must score it on rows it never saw.
        This method decides, once, which rows those are, and records the
        decision on the session so every later step respects it.

        Rows are shuffled and cut into a train partition (the model learns
        here), an optional validation partition (you tune here), and a test
        partition (you measure here, once, at the end). Nothing is copied —
        only row positions are stored — so the split costs almost nothing and
        stays consistent no matter how the data is transformed afterwards.

        Use this when rows are independent. If several rows describe the same
        customer or patient, use :meth:`group_split`; if the rows form a time
        series, use :meth:`time_split`. Random splitting in either of those
        cases leaks information and produces scores you cannot trust.

        Parameters
        ----------
        test_size:
            How much data to hold back for the final measurement. A float is a
            proportion (``0.2`` means 20% of rows); an integer is an absolute
            row count. Larger test sets give a more stable estimate of
            performance but leave less to learn from.
        validation_size:
            How much to carve out of the remaining rows for tuning — again a
            proportion or a count. Set this when you plan to compare models or
            search hyperparameters, so the test set stays untouched until the
            end. ``None`` produces just train and test.
        random_state:
            Seed for the shuffle. Keeping the default ``42`` means the same
            rows land in the same partitions on every run, so your results are
            reproducible. Pass ``None`` for a different split each time.
        stratify:
            When True, preserve the target's class proportions in every
            partition. Turn this on for classification, particularly when one
            class is rare — an unstratified split can leave a rare class almost
            absent from test, making the score meaningless.

        Returns
        -------
        Session
            ``self``, so this call chains into preprocessing.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            No dataset is attached, the requested sizes leave a partition
            empty, or ``stratify=True`` was passed without exactly one target
            column (or with a class too rare to appear in every partition).

        Notes
        -----
        **Leakage:** After splitting, fit-capable operations must use the train
        partition only (:meth:`assert_can_fit`).

        Splitting before preprocessing is deliberate. BuildML's transforms fit
        their statistics on train rows alone, which is only possible if the
        split already exists — so ordering the calls this way is what makes the
        leakage guarantee real rather than aspirational.

        Examples
        --------
        >>> import pandas as pd
        >>> from buildml import Session
        >>> frame = pd.DataFrame({"a": [1, 2, 3, 4], "y": [0, 1, 0, 1]})
        >>> session = Session.ingest(frame).set_roles({"y": "target"})
        >>> _ = session.split(test_size=0.5, stratify=True)
        >>> len(session.partition("train")), len(session.partition("test"))
        (2, 2)

        Reserve a validation partition when you intend to tune:

        >>> _ = session.split(test_size=0.2, validation_size=0.2)  # doctest: +SKIP

        See Also
        --------
        Session.group_split : Keep an entity's rows together.
        Session.time_split : Respect chronological order.
        Session.inject_split : Reuse a split decided elsewhere.
        Session.cv_score : Rotate the holdout instead of fixing it.
        """
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
        """Adopt a split that was decided outside BuildML.

        Sometimes the partitioning is not yours to choose: a benchmark ships
        with an official train/test division, a colleague's split must be
        reproduced exactly, or the boundary follows domain logic no generic
        splitter encodes (everything before the regulation changed is train,
        everything after is test). Pass the row positions directly and BuildML
        treats them exactly as it would treat a split it generated — the
        leakage guards, partition accessors, and history record all apply.

        The plan is recorded with kind ``'injected'``, so :meth:`walkthrough`
        and the model card report honestly that the split was supplied rather
        than derived.

        Parameters
        ----------
        train_indices:
            Positional row numbers (``0`` to ``n_rows - 1``, not DataFrame
            index labels) the model may learn from.
        test_indices:
            Positional row numbers reserved for the final measurement.
        validation_indices:
            Optional positional row numbers for tuning. ``None`` produces just
            train and test.

        Returns
        -------
        Session
            ``self``, so this call chains into preprocessing.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            The partitions overlap, an index falls outside the dataset, or
            train or test would be empty. Overlap is rejected rather than
            silently deduplicated, because a row in both train and test defeats
            the entire point of the split.

        Examples
        --------
        Reproduce a chronological cut-off decided by domain knowledge:

        >>> import pandas as pd
        >>> from buildml import Session
        >>> frame = pd.DataFrame({"a": [1, 2, 3, 4], "y": [0, 1, 0, 1]})
        >>> session = Session.ingest(frame).set_roles({"y": "target"})
        >>> _ = session.inject_split(train_indices=[0, 1], test_indices=[2, 3])
        >>> session.split_plan.kind
        'injected'

        See Also
        --------
        Session.split : Let BuildML choose the rows.
        Session.split_plan : Inspect the resulting membership.
        """
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
        """Split by entity, so no customer appears on both sides.

        When several rows describe the same thing — twelve monthly records for
        one customer, forty sensor readings from one machine, every sentence of
        one document — a random row split scatters that entity across train and
        test. The model then sees eleven of the customer's months in training
        and is asked to predict the twelfth. It does well, and the score is a
        lie: in production the customer is entirely new.

        This method splits whole groups instead of individual rows. Every row
        belonging to a group lands in exactly one partition, so a test score
        answers the question you actually care about — how well does this work
        on someone we have never seen?

        Because groups are the unit, ``test_size`` counts groups rather than
        rows. Partitions therefore rarely come out at exactly the requested row
        proportion, and that is expected: groups differ in size.

        Parameters
        ----------
        test_size:
            Proportion (float) or number (int) of *groups* held back for the
            final measurement.
        validation_size:
            Optional proportion or count of groups for tuning, taken from the
            groups not already assigned to test. ``None`` produces just train
            and test.
        random_state:
            Seed controlling which groups go where. The default keeps the
            assignment stable across runs.
        group_column:
            Which column identifies the entity. ``None`` uses the single column
            you marked with the ``group`` role via :meth:`set_roles`, which is
            the usual path; name a column here to override it for one call.

        Returns
        -------
        Session
            ``self``, so this call chains into preprocessing.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            No group column was resolved (none assigned and none passed, or
            several assigned), or there are too few distinct groups to fill the
            requested partitions.

        Notes
        -----
        **Leakage:** Prefer this over :meth:`split` when rows share entities
        (customers, sites, documents). Random row splits leak across groups.

        A useful check: if you can imagine two rows in your data that a model
        could match to each other by memorising an identity rather than by
        learning a pattern, you need this method.

        Examples
        --------
        >>> import pandas as pd
        >>> from buildml import Session
        >>> frame = pd.DataFrame(
        ...     {
        ...         "customer": ["a", "a", "b", "b", "c", "c"],
        ...         "spend": [10, 12, 90, 84, 41, 38],
        ...         "churn": [0, 0, 1, 1, 0, 0],
        ...     }
        ... )
        >>> session = Session.ingest(frame).set_roles(
        ...     {"customer": "group", "churn": "target"}
        ... )
        >>> _ = session.group_split(test_size=1)
        >>> train = set(session.partition("train")["customer"])
        >>> test = set(session.partition("test")["customer"])
        >>> train & test
        set()

        See Also
        --------
        Session.split : Use when rows are independent.
        Session.cv_score : Cross-validation with the same group awareness.
        """
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
        """Train on the past and test on the future, as deployment will.

        Shuffling a time series lets the model learn from Thursday to predict
        Wednesday. Nothing in the mathematics objects, and the score comes out
        excellent, but the arrangement is impossible in production — you never
        have next month's data when making this month's prediction. Models
        validated that way routinely collapse on release.

        This method sorts rows by their timestamp and cuts chronologically. The
        most recent rows become test, an optional validation block is taken
        from the end of what remains, and the earliest rows are train. The
        result mirrors reality: every evaluation row is later than every row
        the model learned from.

        Parameters
        ----------
        test_size:
            Proportion (float) or number of rows (int) at the end of the
            timeline to hold back. Make this long enough to span the seasonal
            cycle you care about — a two-week test set says little about a
            model with yearly seasonality.
        validation_size:
            Optional proportion or row count for tuning, taken from the end of
            the remaining rows so it still sits before test in time. ``None``
            produces just train and test.
        time_column:
            Which column orders the data. ``None`` uses the single column you
            marked with the ``time`` role via :meth:`set_roles`.

        Returns
        -------
        Session
            ``self``, so this call chains into preprocessing.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            No time column was resolved (none assigned and none passed, or
            several assigned), or the requested sizes leave a partition empty.

        Notes
        -----
        **Leakage:** Prefer this over shuffled splits for temporal processes.
        The splitter does not add a calendar embargo beyond strict ordering.

        The embargo point matters if your target is measured over a window. A
        label like "churned within 30 days" computed for the last training row
        depends on outcomes that fall inside the test period. Strict ordering
        does not catch that; drop a gap of rows yourself, or build the label so
        its window closes before the boundary.

        Examples
        --------
        >>> import pandas as pd
        >>> from buildml import Session
        >>> frame = pd.DataFrame(
        ...     {
        ...         "day": pd.date_range("2024-01-01", periods=6, freq="D"),
        ...         "demand": [10, 12, 15, 14, 19, 21],
        ...     }
        ... )
        >>> session = Session.ingest(frame).set_roles(
        ...     {"day": "time", "demand": "target"}
        ... )
        >>> _ = session.time_split(test_size=2)
        >>> session.partition("train")["day"].max() < session.partition("test")["day"].min()
        True

        See Also
        --------
        Session.fit_forecast : Forecasting models for temporal targets.
        Session.analyze_timeseries : Inspect trend and seasonality first.
        """
        return data_ops.time_split(
            self, test_size=test_size, validation_size=validation_size, time_column=time_column
        )

    def partition(
        self,
        name: PartitionName | Literal["train", "validation", "test"],
    ) -> pd.DataFrame:
        """Pull out the rows belonging to one partition, as a DataFrame.

        The split stores row positions, not data. This materialises one of
        those partitions so you can look at it — check class balance, sanity
        check a transform, or hand the frame to code outside BuildML.

        Parameters
        ----------
        name:
            Which partition to extract: ``'train'``, ``'validation'``, or
            ``'test'``.

        Returns
        -------
        pandas.DataFrame
            A copy of those rows with all current columns, reflecting every
            transform applied so far. Because it is a copy, editing it does not
            change the session's data.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            No split has been created yet, or ``name`` is not one of the three
            partition names. Asking for ``'validation'`` when the split was
            made without one also raises.

        Examples
        --------
        >>> import pandas as pd
        >>> from buildml import Session
        >>> frame = pd.DataFrame({"a": [1, 2, 3, 4], "y": [0, 1, 0, 1]})
        >>> session = Session.ingest(frame).set_roles({"y": "target"})
        >>> _ = session.split(test_size=0.5)
        >>> session.partition("test").shape[0]
        2

        Check that stratification did what you asked:

        >>> session.partition("train")["y"].value_counts(normalize=True)  # doctest: +SKIP
        """
        return data_ops.partition(self, name=name)

    def assert_can_fit(self, partition: PartitionName = "train") -> Session:
        """Refuse to continue unless fitting is confined to the train rows.

        BuildML's own transforms already fit on train alone. This method is for
        the code you write around them: drop it in front of a custom fit step
        and it turns "we should only fit on train" from a comment into a
        runtime guarantee. It raises if no split exists, and it raises if the
        partition you name is anything other than train.

        It is a checkpoint, not a transform. Nothing about the data changes.

        Parameters
        ----------
        partition:
            The partition you are about to fit on. Only ``'train'`` is
            permitted; naming ``'validation'`` or ``'test'`` is precisely the
            mistake this call exists to catch.

        Returns
        -------
        Session
            ``self``, so the guard sits inline in a chain.

        Raises
        ------
        ~buildml.core.errors.LeakageError
            No split has been created, or ``partition`` is not ``'train'``.

        Examples
        --------
        >>> import pandas as pd
        >>> from buildml import Session
        >>> frame = pd.DataFrame({"a": [1, 2, 3, 4], "y": [0, 1, 0, 1]})
        >>> session = Session.ingest(frame).set_roles({"y": "target"})
        >>> _ = session.split(test_size=0.5)
        >>> train = session.assert_can_fit().partition("train")

        Fitting anything of your own on a holdout is stopped outright:

        >>> session.assert_can_fit("test")
        Traceback (most recent call last):
            ...
        buildml.core.errors.LeakageError: ...

        See Also
        --------
        Session.split : Create the split this guard checks for.
        """
        return data_ops.assert_can_fit(self, partition=partition)

    def drop_columns(self, columns: list[str] | tuple[str, ...]) -> Session:
        """Remove columns you do not want the model to see.

        Use this for columns that would leak the answer (a field populated only
        after the outcome is known), free-text notes you are not vectorising,
        or duplicated identifiers. Marking a column ``'ignore'`` with
        :meth:`set_roles` keeps it in the table for later reference; dropping
        it removes it entirely and reclaims the memory.

        Parameters
        ----------
        columns:
            Names of the columns to remove.

        Returns
        -------
        Session
            ``self``, so this call chains into the next step.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            A named column is not present in the dataset.

        Notes
        -----
        Split membership is preserved (row order unchanged). Roles for dropped
        columns are removed.

        Dropping columns does not disturb an existing split, because splits are
        defined over rows. You can therefore drop before or after splitting
        with the same result.

        Examples
        --------
        >>> import pandas as pd
        >>> from buildml import Session
        >>> frame = pd.DataFrame({"a": [1, 2], "scratch": [9, 9], "y": [0, 1]})
        >>> session = Session.ingest(frame)
        >>> _ = session.drop_columns(["scratch"])
        >>> session.dataset.columns
        ['a', 'y']

        See Also
        --------
        Session.set_roles : Exclude a column without deleting it.
        Session.select_features : Let a scoring rule choose what to keep.
        """
        return preprocess_ops.drop_columns(self, columns=columns)

    def impute(
        self,
        *,
        columns: list[str] | None = None,
        strategy: Literal["mean", "median", "most_frequent", "constant"] = "median",
        fill_value: Any | None = None,
    ) -> Session:
        """Fill in missing values, using only what the training rows reveal.

        Most estimators cannot accept a missing value, so gaps have to be
        filled with a stand-in before fitting. The stand-in is computed from
        the training rows and then applied everywhere — that ordering is the
        whole point. If you filled from all rows, the median would encode a
        little of the test set into every training row, and your score would
        drift upward for no real reason.

        Which stand-in to use depends on the column. The median resists
        outliers, so it is the default and the safe choice for skewed
        quantities like income. The mean suits roughly symmetric measurements.
        The most frequent value is the sensible fallback for categoricals. A
        constant is right when the gap itself is meaningful — "no prior claim"
        is information, not an accident, and filling it with the median would
        erase that.

        Parameters
        ----------
        columns:
            Which columns to fill. ``None`` selects numeric columns with the
            ``feature`` role and deliberately leaves ``ignore``, ``id``,
            ``target``, ``group``, ``time``, and ``weight`` alone, so
            identifiers and labels are never quietly altered. Name columns
            explicitly to override that protection.
        strategy:
            How to compute the stand-in: ``'median'`` (the default, robust to
            extreme values), ``'mean'``, ``'most_frequent'``, or ``'constant'``
            for a fixed value you supply.
        fill_value:
            The value used when ``strategy='constant'``. Ignored by the other
            strategies.

        Returns
        -------
        Session
            ``self``, so this call chains into the next transform.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            No split exists yet, a named column is absent, or
            ``strategy='constant'`` was chosen without a ``fill_value``.

        Notes
        -----
        **Leakage:** Requires an existing split. Statistics are learned from
        the train partition only, then applied to all rows.

        Filling a gap invents data. When a column is mostly missing, or when
        missingness is itself predictive, consider adding an indicator column
        or dropping the column instead. :attr:`last_preprocess` reports how
        many values were filled per column so you can judge.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from buildml import Session
        >>> frame = pd.DataFrame(
        ...     {"income": [30.0, np.nan, 50.0, 70.0], "y": [0, 1, 0, 1]}
        ... )
        >>> session = Session.ingest(frame).set_roles({"y": "target"})
        >>> _ = session.split(test_size=0.5)
        >>> _ = session.impute(strategy="median")
        >>> session.dataset.frame["income"].isna().sum()
        np.int64(0)

        Treat an absent value as a fact rather than a gap:

        >>> _ = session.impute(
        ...     columns=["prior_claims"], strategy="constant", fill_value=0
        ... )  # doctest: +SKIP

        See Also
        --------
        Session.impute_plan : The fitted statistics, for reuse at score time.
        Session.encode : Run after imputing when categoricals have gaps.
        """
        return preprocess_ops.impute(
            self, columns=columns, strategy=strategy, fill_value=fill_value
        )

    @property
    def impute_plan(self) -> SimpleImputePlan | None:
        """The fill values learned by the last :meth:`impute` call.

        A :class:`~buildml.preprocess.impute.SimpleImputePlan` holds the
        columns it covers, the strategy used, and ``statistics_`` — the actual
        per-column value that was substituted. Keeping the plan is what makes
        scoring reproducible: new data must be filled with the *training*
        median, not with its own.

        ``None`` until :meth:`impute` runs. Pass it to
        :meth:`apply_preprocess_plans` to replay the same fill on fresh rows.
        """
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
        """Turn category labels into numbers a model can work with.

        Estimators do arithmetic, and ``"Ireland"`` is not a number. Encoding
        is how a category becomes something computable — but the choice of
        encoding changes what the model is able to learn, so it is worth
        understanding rather than accepting the default blindly.

        ``'onehot'`` gives each level its own 0/1 column. It makes no claim
        about order or distance between levels, which is the honest
        representation for genuinely unordered categories, and it is the
        default. Its cost is width: a column with a thousand levels becomes a
        thousand columns.

        ``'ordinal'`` maps levels to ``0, 1, 2, …``. Compact, but it asserts
        that level 2 sits between level 1 and level 3. That is right for
        ``small < medium < large`` and wrong for country names — a linear model
        will happily conclude that Ireland is halfway between Iceland and
        Italy. Tree models are largely immune, which is why ordinal encoding is
        often fine with them and dangerous without them.

        ``'infrequent'`` pools every level that is rare in training into a
        single ``other`` bucket before one-hot encoding. This is the practical
        answer to high-cardinality columns: rare levels carry too few examples
        to learn from and generate columns that are almost entirely zero.

        ``'target'`` replaces each level with the average target for that level
        — extremely compact and often the strongest encoder, but the one that
        leaks most eagerly, since the target is being folded into a feature.
        BuildML defends against that on two fronts: training rows receive
        out-of-fold averages (a row never contributes to the mean it is given),
        and rare levels are pulled toward the overall average by ``smoothing``.

        Parameters
        ----------
        columns:
            Which columns to encode. ``None`` selects categorical columns with
            the ``feature`` role, skipping ``ignore``, ``id``, ``target``,
            ``group``, ``time``, and ``weight``. Name columns explicitly to
            override.
        method:
            One of ``'onehot'``, ``'ordinal'``, ``'infrequent'``, or
            ``'target'``, as described above.
        min_frequency:
            For ``'infrequent'``, the line between "keep" and "pool". A float
            is a share of training rows (``0.05`` pools any level appearing in
            under 5% of them); an integer is a raw count. Raise it to compress
            harder, lower it to keep more distinct levels.
        n_folds:
            For ``'target'``, how many folds generate the out-of-fold averages.
            More folds mean each average is computed from more data and the
            encoding is less noisy, at proportionally more work.
        random_state:
            For ``'target'``, the seed for fold assignment, so the encoding is
            reproducible run to run.
        smoothing:
            For ``'target'``, how strongly rare levels are pulled toward the
            overall target mean. A level seen twice should not be trusted as
            much as one seen two thousand times; raising this trusts the global
            average more and the level-specific average less.

        Returns
        -------
        Session
            ``self``, so this call chains into the next transform.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            No split exists, a named column is absent, or ``'target'`` encoding
            was requested without a target column assigned.

        Notes
        -----
        **Leakage:** Requires a split. Vocabularies and target means are learned
        on train only. Target encoding writes out-of-fold values on train and
        full-train means on holdouts.

        Levels that appear only in test are unseen at fit time and encode to
        the all-zero row (one-hot) or the global prior (target). This is
        correct behaviour, not a bug: the model has no evidence about a
        category it never saw.

        Examples
        --------
        >>> import pandas as pd
        >>> from buildml import Session
        >>> frame = pd.DataFrame(
        ...     {"city": ["dublin", "cork", "dublin", "cork"], "y": [0, 1, 0, 1]}
        ... )
        >>> session = Session.ingest(frame).set_roles({"y": "target"})
        >>> _ = session.inject_split(train_indices=[0, 1], test_indices=[2, 3])
        >>> _ = session.encode(method="onehot")
        >>> sorted(c for c in session.dataset.columns if c.startswith("city"))
        ['city_cork', 'city_dublin']

        Compress a wide identifier-like column instead of exploding it:

        >>> _ = session.encode(
        ...     columns=["merchant"], method="infrequent", min_frequency=0.01
        ... )  # doctest: +SKIP

        See Also
        --------
        Session.encode_plan : The learned vocabulary, for reuse at score time.
        Session.text_features : For free text rather than discrete labels.
        """
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
        """The category vocabulary learned by the last :meth:`encode` call.

        An :class:`~buildml.preprocess.encode.EncodePlan` records the method
        used, the fitted encoder, the output ``feature_names_``, and — for the
        methods that need them — the pooling maps for ``'infrequent'`` and the
        per-level target means plus prior for ``'target'``.

        Score-time data must be encoded with this exact vocabulary, in this
        exact column order, or the model receives features that mean something
        different from what it was trained on. Keeping the plan is how that is
        guaranteed.

        ``None`` until :meth:`encode` runs.
        """
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
        """Find extreme numeric values, and decide what to do about them.

        A handful of very large values can dominate a model. Linear
        regressions chase them, scalers stretch to accommodate them, and
        distance-based methods let them distort every neighbourhood. This
        method locates them and applies the treatment you choose.

        Two detectors are available. ``'iqr'`` uses Tukey fences: the middle
        half of the training data defines a range, and anything more than
        ``iqr_multiplier`` times that range beyond it is flagged. It makes no
        assumption about the distribution's shape, which is why it is the
        default. ``'zscore'`` flags values more than ``zscore_threshold``
        standard deviations from the mean; that is cheaper to reason about but
        assumes roughly normal data, and it is self-defeating on heavy tails
        because the outliers themselves inflate the standard deviation.

        The treatment matters more than the detector. ``'detect'`` changes
        nothing and simply reports — always start here. ``'cap'`` pulls flagged
        values back to the fence, keeping the row and its other columns while
        removing the extreme's leverage. ``'drop'`` deletes the row entirely
        and rebuilds split membership around the loss.

        Parameters
        ----------
        columns:
            Which numeric columns to screen. ``None`` selects numeric
            ``feature``-role columns, leaving identifiers, targets, and weights
            untouched.
        method:
            ``'iqr'`` for Tukey fences or ``'zscore'`` for standard-deviation
            distance.
        action:
            ``'detect'`` to report only, ``'cap'`` to clip to the fences, or
            ``'drop'`` to remove flagged rows.
        iqr_multiplier:
            How far beyond the middle-half range counts as extreme, for
            ``'iqr'``. The conventional ``1.5`` marks the usual boxplot
            whiskers; ``3.0`` flags only the genuinely far-out values.
        zscore_threshold:
            How many standard deviations from the mean counts as extreme, for
            ``'zscore'``.

        Returns
        -------
        Session
            ``self``, so this call chains into the next transform.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            No split exists, a named column is absent or non-numeric, or
            ``'drop'`` would empty a partition.

        Notes
        -----
        **Leakage:** Fence statistics are learned on train only, then applied
        with the frozen bounds. Heuristic screens are not proof of error.

        That last sentence is the important one. An outlier detector finds
        values that are unusual, not values that are wrong. In fraud, churn,
        and equipment failure the extreme rows are frequently the signal.
        Dropping them can quietly delete the very thing you set out to predict
        — run ``'detect'`` first and look at what was flagged before removing
        anything.

        Examples
        --------
        >>> import pandas as pd
        >>> from buildml import Session
        >>> frame = pd.DataFrame(
        ...     {"amount": [10.0, 12.0, 11.0, 9000.0], "y": [0, 1, 0, 1]}
        ... )
        >>> session = Session.ingest(frame).set_roles({"y": "target"})
        >>> _ = session.split(test_size=0.5)
        >>> _ = session.handle_outliers(action="detect")
        >>> session.outlier_plan.action
        'detect'

        Then, if the extremes really are recording errors, clip them:

        >>> _ = session.handle_outliers(method="iqr", action="cap")  # doctest: +SKIP

        See Also
        --------
        Session.fit_anomaly : When the extremes are what you want to find.
        Session.scale : Run after capping, so the scaler is not stretched.
        """
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
        """The fences and counts from the last :meth:`handle_outliers` call.

        An :class:`~buildml.preprocess.outliers.OutlierPlan` carries the
        per-column ``lower_`` and ``upper_`` bounds learned on the training
        rows, the detector and action used, and how many values were flagged
        (``n_flagged_train``) or rows removed (``n_dropped``).

        Read those counts before trusting the result: flagging a third of your
        data means the threshold is wrong, not that a third of your data is
        wrong. The frozen bounds are also what score-time capping reuses, so
        new data is clipped to training fences rather than its own.

        ``None`` until :meth:`handle_outliers` runs.
        """
        return self._outlier_plan

    def bin(
        self,
        *,
        columns: list[str] | None = None,
        strategy: Literal["quantile", "uniform"] = "quantile",
        n_bins: int = 5,
        encode_as: Literal["ordinal", "onehot"] = "ordinal",
    ) -> Session:
        """Group a continuous column into bands, trading detail for shape.

        Binning turns ``age = 34`` into ``age is in the 30–40 band``. You lose
        resolution, and in exchange you gain two things: the model can express
        a relationship that is not a straight line without you specifying its
        form, and the result is far easier to explain to someone who has to act
        on it. Risk bands, price tiers, and age brackets are how most people
        already think about these quantities.

        ``'quantile'`` places the edges so each band holds roughly the same
        number of training rows. Bands end up narrow where the data is dense
        and wide where it is sparse, so every band has enough examples to
        support an estimate. ``'uniform'`` makes every band the same width
        instead, which preserves the real spacing of the values but can leave
        some bands nearly empty when the distribution is skewed.

        Parameters
        ----------
        columns:
            Which numeric columns to band. ``None`` selects numeric
            ``feature``-role columns.
        strategy:
            ``'quantile'`` for equal-population bands or ``'uniform'`` for
            equal-width bands.
        n_bins:
            How many bands to create. Fewer bands generalise more strongly and
            explain more cleanly; more bands retain more of the original
            signal. Past a point extra bands simply reintroduce the noise you
            were binning away.
        encode_as:
            ``'ordinal'`` writes the band number, which keeps one column and
            preserves the natural ordering of the bands. ``'onehot'`` writes an
            indicator per band, letting a linear model give each band its own
            independent effect at the cost of extra columns.

        Returns
        -------
        Session
            ``self``, so this call chains into the next transform.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            No split exists, a named column is absent or non-numeric, or
            ``n_bins`` exceeds the number of distinct training values
            available.

        Notes
        -----
        **Leakage:** Edges are learned on train only. End bins use open
        ``±inf`` edges so score-time extremes remain defined.

        That open-ended detail prevents a common production failure. If the
        outermost edges were closed at the training minimum and maximum, a new
        row beyond that range would fall into no band at all and produce a
        missing value at score time. Unbounded end bins mean every future value
        lands somewhere.

        Gradient-boosted trees already discover their own thresholds, so
        binning before them usually costs accuracy and gains nothing. Reach for
        it with linear models, or when the banded output is itself the
        deliverable.

        Examples
        --------
        >>> import pandas as pd
        >>> from buildml import Session
        >>> frame = pd.DataFrame(
        ...     {"age": [21, 34, 47, 63], "y": [0, 1, 0, 1]}
        ... )
        >>> session = Session.ingest(frame).set_roles({"y": "target"})
        >>> _ = session.split(test_size=0.5)
        >>> _ = session.bin(columns=["age"], n_bins=2, strategy="quantile")
        >>> session.binning_plan.n_bins
        2

        See Also
        --------
        Session.encode : Encode the resulting bands as independent indicators.
        Session.tune_threshold : Choose a cut-off on predictions, not inputs.
        """
        return preprocess_ops.bin(
            self, columns=columns, strategy=strategy, n_bins=n_bins, encode_as=encode_as
        )

    @property
    def binning_plan(self) -> BinningPlan | None:
        """The band edges learned by the last :meth:`bin` call.

        A :class:`~buildml.preprocess.binning.BinningPlan` holds the per-column
        cut points in ``edges_``, the band ``labels_``, and the strategy and
        output encoding used. Inspect ``edges_`` to see where the boundaries
        actually fell — with ``'quantile'`` they follow the data's shape rather
        than round numbers, which is often surprising and always worth a look
        before the bands are shown to anyone.

        Reused at score time so new rows are assigned to training bands rather
        than to bands recomputed from themselves.

        ``None`` until :meth:`bin` runs.
        """
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
        """Keep the columns that carry signal and drop the rest.

        Fewer features usually means a model that trains faster, generalises
        better, and can be explained to someone. The difficulty is deciding
        which columns to lose. Three strategies are offered, in increasing
        order of how much they know about your problem.

        ``'variance'`` drops columns that barely change. A field that is the
        same value in 99% of rows cannot distinguish those rows, whatever the
        target is. This strategy never looks at the target, so it is cheap,
        safe, and a reasonable first pass — but it cannot tell a constant-ish
        column that matters from one that does not.

        ``'univariate'`` scores each column against the target on its own and
        keeps the best ``k``. It sees relevance, but only one column at a time:
        two features that are useless alone and powerful together will both be
        discarded, and ten copies of the same strong signal will all be kept.

        ``'model'`` fits an estimator and keeps the features it actually
        relied on. This is the only option that accounts for interactions and
        redundancy, because the model weighs features against each other. It
        costs a fit, and the selection inherits that model's biases —
        tree-based importances, for instance, favour high-cardinality columns.

        Parameters
        ----------
        strategy:
            ``'variance'``, ``'univariate'``, or ``'model'``, as described
            above.
        columns:
            Restrict selection to these columns. ``None`` considers all
            ``feature``-role columns. Columns you exclude are kept
            unconditionally.
        threshold:
            For ``'variance'``, the minimum variance a column must have to
            survive. ``0.0`` removes only perfectly constant columns.
        k:
            For ``'univariate'``, how many top-scoring features to keep.
        score_func:
            For ``'univariate'``, how relevance is measured:
            ``'f_classif'`` for classification targets, ``'f_regression'`` for
            continuous ones, or ``'mutual_info'``, which detects non-linear
            relationships the F-tests miss but takes longer and is noisier on
            small samples.
        estimator:
            For ``'model'``, the fitted-on-train estimator whose importances
            drive selection. ``None`` uses a sensible default for the task.

        Returns
        -------
        Session
            ``self``, so this call chains into the next step.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            No split exists, features are still non-numeric or contain missing
            values, or ``k`` exceeds the number of available features.

        Notes
        -----
        **Leakage:** Selection fits on train only. Encode categoricals and
        impute before calling when features are non-numeric or contain nulls.

        Selection is itself a fitted decision. Choosing features on the whole
        dataset and then cross-validating is a classic way to produce scores
        that cannot be reproduced — the columns were already chosen with
        knowledge of the held-out rows. Selecting on train alone, as this does,
        avoids that.

        Examples
        --------
        >>> import pandas as pd
        >>> from buildml import Session
        >>> frame = pd.DataFrame(
        ...     {
        ...         "useful": [1.0, 5.0, 2.0, 8.0],
        ...         "constant": [7.0, 7.0, 7.0, 7.0],
        ...         "y": [0, 1, 0, 1],
        ...     }
        ... )
        >>> session = Session.ingest(frame).set_roles({"y": "target"})
        >>> _ = session.split(test_size=0.5)
        >>> _ = session.select_features(strategy="variance", threshold=0.0)
        >>> "constant" in session.feature_select_plan.dropped_features_
        True

        See Also
        --------
        Session.feature_importance : Explain a fitted model's reliance.
        Session.reduce_dimensions : Compress features instead of discarding.
        """
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
        """What the last :meth:`select_features` call kept, and why.

        A :class:`~buildml.preprocess.select.FeatureSelectPlan` lists
        ``selected_features_`` and ``dropped_features_`` and, where the
        strategy produces them, the per-feature ``scores_`` behind the
        decision. Reading the dropped list is worth the minute it takes:
        selection is automated, but noticing that a column you know to be
        important was discarded tells you the scoring rule does not match your
        problem.

        ``None`` until :meth:`select_features` runs.
        """
        return self._feature_select_plan

    @property
    def last_preprocess(self) -> PreprocessResult | None:
        """The narrated outcome of the most recent preprocessing step.

        Every transform returns ``self`` for chaining, so the detail of what it
        did lands here instead. A
        :class:`~buildml.preprocess.result.PreprocessResult` carries the
        ``evidence`` gathered (counts, statistics, affected columns), the
        ``findings`` drawn from it, an ``interpretation`` in plain language,
        the ``limitations`` of what was done, and ``recommendations`` for what
        to consider next.

        This is the difference between knowing that imputation ran and knowing
        that it filled 42% of one column — the second tells you to go back and
        think again.

        Holds only the latest step. ``None`` before any preprocessing has run.
        """
        return self._last_preprocess

    def scale(
        self,
        *,
        columns: list[str] | None = None,
        method: Literal["standard", "minmax"] = "standard",
    ) -> Session:
        """Put numeric columns on a comparable footing.

        Income runs to tens of thousands; a satisfaction rating runs from one
        to five. Any method that adds features together or measures distance
        between rows will let income drown out the rating purely because its
        numbers are bigger. Scaling removes that accident of units so the model
        weighs columns on evidence rather than magnitude.

        ``'standard'`` subtracts each column's training mean and divides by its
        training standard deviation, so the column ends up centred at zero with
        unit spread. Values are unbounded, which is the right behaviour when
        extremes are real, and it is the default.

        ``'minmax'`` squeezes each column into the ``[0, 1]`` range using the
        training minimum and maximum. Useful when a bounded input is required —
        some neural network layers, some visualisations — but fragile: one
        extreme training value compresses everything else into a narrow band,
        and a larger value at score time lands outside ``[0, 1]`` entirely.

        Scaling is essential for linear and logistic regression with
        regularisation, support vector machines, k-nearest neighbours,
        k-means, and PCA. Decision trees and their ensembles split one feature
        at a time and are completely indifferent to it.

        Parameters
        ----------
        columns:
            Which columns to scale. ``None`` selects numeric ``feature``-role
            columns and skips ``ignore``, ``id``, ``target``, ``group``,
            ``time``, and ``weight`` — so monetary amounts you are predicting
            and identifiers you need to read back stay in their original units.
            Name columns explicitly to override.
        method:
            ``'standard'`` for zero mean and unit variance, or ``'minmax'`` for
            a ``[0, 1]`` range.

        Returns
        -------
        Session
            ``self``, so this call chains into the fit.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            No split exists, a named column is absent or non-numeric, or a
            column still contains missing values — impute first.

        Notes
        -----
        **Leakage:** Requires a split. Scaler is fit on train only.

        Scale last, after imputing, encoding, and any outlier treatment. Each
        of those changes the distribution, and the scaler should learn from the
        distribution the model will actually receive.

        Examples
        --------
        >>> import pandas as pd
        >>> from buildml import Session
        >>> frame = pd.DataFrame(
        ...     {"income": [30000.0, 52000.0, 41000.0, 78000.0],
        ...      "rating": [4.0, 2.0, 5.0, 3.0],
        ...      "y": [0, 1, 0, 1]}
        ... )
        >>> session = Session.ingest(frame).set_roles({"y": "target"})
        >>> _ = session.split(test_size=0.5)
        >>> _ = session.scale(method="standard")
        >>> session.scale_plan.method
        'standard'

        See Also
        --------
        Session.impute : Run before scaling; scalers reject missing values.
        Session.reduce_dimensions : Scale first, or PCA follows the units.
        """
        return preprocess_ops.scale(self, columns=columns, method=method)

    @property
    def scale_plan(self) -> ScalePlan | None:
        """The fitted scaler from the last :meth:`scale` call.

        A :class:`~buildml.preprocess.scale.ScalePlan` holds the columns it
        covers, the method used, and the fitted scikit-learn ``scaler`` object
        carrying the training means, spreads, or ranges.

        This one must be reused at score time. A single new row has no
        meaningful mean or standard deviation of its own, so re-fitting on it
        is not merely inaccurate but undefined — the training statistics are
        the only correct choice.

        ``None`` until :meth:`scale` runs.
        """
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
        """Turn free-text columns into numeric features.

        A product review or a support ticket is a string, and a model needs
        numbers. These vectorisers convert each document into a row of counts
        or weights over words, replacing one text column with many numeric
        ones.

        ``'tfidf'`` counts each word, then discounts words that appear in many
        documents. "The" appears everywhere and distinguishes nothing;
        "refund" appears in a few documents and says a great deal. Weighting by
        that inverse document frequency is why TF-IDF is the sensible default.

        ``'count'`` keeps the raw counts with no discounting. Simpler to
        interpret, and the natural input to Naive Bayes, which expects counts.

        ``'hashing'`` skips the vocabulary altogether and hashes each word into
        a fixed number of slots. Nothing needs to be stored, so it handles
        streaming text and vocabularies too large to hold — at the price of
        collisions (two unrelated words can share a slot) and features you
        cannot map back to words.

        Parameters
        ----------
        columns:
            Which text columns to vectorise. ``None`` selects string-valued
            ``feature``-role columns.
        method:
            ``'tfidf'``, ``'count'``, or ``'hashing'``, as described above.
        max_features:
            How wide the output is: the vocabulary size kept for ``'tfidf'``
            and ``'count'`` (the most frequent terms in training win), or the
            number of hash slots for ``'hashing'``. Larger retains more
            distinctions and costs more columns; too small for hashing means
            frequent collisions.
        ngram_range:
            The inclusive span of word-group sizes to include. ``(1, 1)`` uses
            single words only. ``(1, 2)`` adds adjacent pairs, which is how the
            vectoriser can tell "not good" from "good" — worth the extra
            columns for sentiment-like problems.
        drop_input_columns:
            When True (the default), remove the original text column once its
            numeric features exist, since most estimators cannot consume the
            raw string anyway. Set False to keep the text for inspection.

        Returns
        -------
        Session
            ``self``, so this call chains into the fit.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            No split exists, or a named column is absent or not text-like.

        Notes
        -----
        **Leakage:** Requires a split. Vocabularies and IDF weights are learned
        from train documents only. Missing text becomes empty strings.

        Words that appear only in test documents are outside the fitted
        vocabulary and are ignored. That is correct — the model has no
        evidence about a word it never saw during training.

        These are bag-of-words representations: they record which words occur,
        not the order they occur in. When word order and meaning matter, reach
        for :meth:`fit_ssl_pretext` with a text backbone or the RAG methods
        instead.

        Examples
        --------
        >>> import pandas as pd
        >>> from buildml import Session
        >>> frame = pd.DataFrame(
        ...     {
        ...         "review": ["great value", "poor value", "great service", "poor service"],
        ...         "y": [1, 0, 1, 0],
        ...     }
        ... )
        >>> session = Session.ingest(frame).set_roles({"y": "target"})
        >>> _ = session.split(test_size=0.5)
        >>> _ = session.text_features(method="tfidf", max_features=8)
        >>> "review" in session.dataset.columns
        False

        Capture short phrases rather than isolated words:

        >>> _ = session.text_features(ngram_range=(1, 2))  # doctest: +SKIP

        See Also
        --------
        Session.encode : For discrete labels rather than free text.
        Session.rag_embed_and_index : For semantic search over documents.
        """
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
        """The fitted vectorisers from the last :meth:`text_features` call.

        A :class:`~buildml.preprocess.text.TextFeaturePlan` holds the source
        columns, the method and settings used, the per-column fitted
        ``vectorizers_``, the generated ``feature_names_``, and how many
        features each column produced.

        ``feature_names_`` is the useful part when interpreting a model: it maps
        each numeric column back to the word or phrase it counts, which is what
        turns a coefficient into "the word *refund* pushes this prediction
        up".

        ``None`` until :meth:`text_features` runs.
        """
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
        """Compress many numeric columns into a few informative ones.

        Where :meth:`select_features` discards columns, this one blends them.
        Each output column is built from all the inputs, so information spread
        thinly across fifty correlated measurements can survive in five
        columns. The cost is interpretability: ``pc_1`` is a mixture, not a
        measurement, and no business user will recognise it.

        ``'pca'`` finds the directions along which the training data varies
        most and projects onto them. It is linear, fast, deterministic, and
        genuinely reusable — a new row can be projected with the same fitted
        rotation, which is what makes it safe in a scoring pipeline.

        ``'umap'`` learns a non-linear embedding that tries to preserve local
        neighbourhood structure. It captures curved structure PCA cannot, and
        it can project new rows. Requires ``pip install
        'buildml[unsupervised]'``.

        ``'tsne'`` is transductive: it embeds the rows it was given and has no
        natural way to place a new one. BuildML transfers holdout rows by
        nearest neighbour and records that compromise in the plan's
        ``disclosures``. Treat t-SNE as a tool for looking at your training
        data, not as a step in a pipeline that will score fresh rows.

        Parameters
        ----------
        columns:
            Which numeric columns to compress. ``None`` selects numeric
            ``feature``-role columns.
        method:
            ``'pca'``, ``'umap'``, or ``'tsne'``, as described above.
        n_components:
            How many output columns to produce. An integer sets the count
            directly. For PCA a float in ``(0, 1]`` instead names a target —
            ``0.95`` keeps however many components are needed to retain 95% of
            the training variance, which is usually the more meaningful way to
            ask. ``None`` uses the method's default.
        drop_input_columns:
            When True (the default), replace the source columns with the new
            ones. Keeping both is rarely useful, since the outputs are built
            from the inputs and the two are heavily redundant.
        prefix:
            Naming stem for the output columns, giving ``pc_1``, ``pc_2``, and
            so on.
        random_state:
            Seed for the methods with a stochastic component (UMAP and t-SNE),
            so repeated runs agree.
        umap_n_neighbors:
            For UMAP, how much of the neighbourhood each point considers. Small
            values preserve fine local detail; large values favour the global
            shape.
        umap_min_dist:
            For UMAP, how tightly points may be packed together in the output.
            Lower values produce tighter, more separated clumps.
        tsne_perplexity:
            For t-SNE, roughly how many neighbours each point is balanced
            against. It must be well below the number of rows, and the
            resulting picture changes noticeably with it — try several before
            drawing conclusions.
        tsne_learning_rate:
            For t-SNE, the optimiser step size. ``'auto'`` scales it to the
            sample size and is almost always the right choice.

        Returns
        -------
        Session
            ``self``, so this call chains into the fit.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            No split exists, columns are non-numeric or contain missing values,
            or ``n_components`` exceeds the number of available columns.
        ~buildml.core.errors.MissingExtraError
            ``method='umap'`` without ``buildml[unsupervised]`` installed.

        Notes
        -----
        **Leakage:** Requires a split. The transform is learned on train only.
        Explained variance / embedding quality is unsupervised — not predictive utility.
        Scale numeric inputs first when magnitudes differ.

        That middle sentence is the trap. PCA maximises variance, and variance
        is not relevance: a component that explains 60% of the spread in your
        features can be irrelevant to the target, while the component
        explaining 2% carries the signal. Retaining 95% of variance does not
        retain 95% of predictive power.

        Scaling first is not optional advice. PCA works on covariance, so a
        column measured in thousands dominates one measured in units, and the
        leading components end up describing your choice of units rather than
        your data.

        Examples
        --------
        >>> import pandas as pd
        >>> from buildml import Session
        >>> frame = pd.DataFrame(
        ...     {
        ...         "a": [1.0, 2.0, 3.0, 4.0],
        ...         "b": [2.0, 4.1, 5.9, 8.0],
        ...         "y": [0, 1, 0, 1],
        ...     }
        ... )
        >>> session = Session.ingest(frame).set_roles({"y": "target"})
        >>> _ = session.split(test_size=0.5)
        >>> _ = session.scale()
        >>> _ = session.reduce_dimensions(method="pca", n_components=1)
        >>> session.reduce_plan.n_components
        1

        Keep as many components as it takes to retain most of the variance:

        >>> _ = session.reduce_dimensions(method="pca", n_components=0.95)  # doctest: +SKIP

        See Also
        --------
        Session.select_features : Keep original columns instead of blending.
        Session.fit_clusters : Grouping, which often follows reduction.
        """
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
        """The fitted projection from the last :meth:`reduce_dimensions` call.

        A :class:`~buildml.preprocess.reduce.ReducePlan` holds the fitted
        ``reducer_``, the source and output column names, and — for PCA —
        ``explained_variance_ratio_`` per component plus the running
        ``cumulative_explained_variance_``. Reading the cumulative figure tells
        you whether the components you kept were enough.

        ``disclosures`` is where honesty about the method lives. For t-SNE it
        records that holdout rows were placed by nearest-neighbour transfer
        rather than by a true learned mapping, so nobody later mistakes the
        embedding for something it is not.

        ``None`` until :meth:`reduce_dimensions` runs.
        """
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
        
        Delegates to :meth:`buildml.unsupervised.cluster.fit_clusterer`, stores the
        :class:`~buildml.unsupervised.results.ClusterPlan` on this Session, and records
        the fit. Follow with :meth:`assign_clusters` or :meth:`evaluate_clusters`.
        
        Parameters
        ----------
        method:
            Clustering method key (``kmeans``, ``gmm``, ``hdbscan``, etc.).
        n_clusters:
            Target cluster count for parametric methods; ignored for density methods.
        columns:
            Optional explicit feature columns; ``None`` auto-selects numerics.
        random_state:
            Seed for stochastic initialization and sampling.
        n_init:
            Number of k-means restarts (``auto`` uses sklearn default).
        max_iter:
            Maximum iterations for iterative clusterers.
        linkage:
            Linkage criterion for hierarchical clustering.
        eps:
            Neighborhood radius for DBSCAN.
        min_samples:
            Minimum samples per core point for DBSCAN/OPTICS.
        gmm_covariance_type:
            Covariance structure for Gaussian mixture models.
        gmm_max_components:
            Upper bound on components when ``auto_k`` selects GMM k.
        gmm_select_by:
            Model-selection score for GMM component count (``bic`` or ``aic``).
        hdbscan_min_cluster_size:
            Minimum cluster size for HDBSCAN.
        hdbscan_min_samples:
            Core distance samples for HDBSCAN; defaults to min cluster size.
        spectral_affinity:
            Affinity matrix type for spectral clustering.
        spectral_n_neighbors:
            Neighbors for spectral nearest-neighbors affinity.
        optics_min_samples:
            Minimum samples for OPTICS core distances.
        optics_xi:
            Steepness threshold for OPTICS cluster extraction.
        optics_min_cluster_size:
            Minimum cluster size for OPTICS extraction.
        bandwidth:
            Kernel bandwidth for mean-shift; ``None`` estimates from data.
        latent_dim:
            Embedding dimension for deep clustering backend.
        pretrain_epochs:
            Pretraining epochs for deep clustering autoencoder.
        finetune_epochs:
            Fine-tuning epochs for deep clustering head.
        batch_size:
            Minibatch size for deep clustering backend.
        learning_rate:
            Optimizer learning rate for deep clustering backend.
        prefer_reduce_components:
            Prefer reduced component columns when a reduce plan exists on this Session.
        label_column:
            Output column name for cluster assignments when attaching.
        auto_k:
            When True, search ``auto_k_min``..``auto_k_max`` for k-means/GMM.
        auto_k_min:
            Lower bound for automatic k search.
        auto_k_max:
            Upper bound for automatic k search.
        
        Returns
        -------
        ClusterFitResult
            Serializable fit summary including cluster count and method disclosures.
        """
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
        """Assign cluster labels with the train-fitted plan without refitting.
        
        Delegates to :meth:`buildml.unsupervised.cluster.assign_clusters`. When
        ``attach=True``, cluster labels are merged into Session dataset.
        
        Parameters
        ----------
        partition:
            Partition to assign (``train``, ``validation``, ``test``, or ``all``).
        attach:
            When True, attach cluster label column to this Session dataset frame.
        
        Returns
        -------
        ClusterAssignResult
            Cluster assignments and optional attached column metadata.
        
        Raises
        ------
        ValidationError
            When no cluster plan exists on this Session.
        """
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
        """Evaluate train-fitted clusters on a holdout partition.
        
        Delegates to :meth:`buildml.unsupervised.evaluate.evaluate_clustering`.
        Computes internal metrics and optional external alignment when labels exist.
        
        Parameters
        ----------
        partition:
            Holdout partition to score. Validation falls back to test when absent.
        external_label_column:
            Optional column for external cluster-quality metrics (e.g. ARI).
        sample_size:
            Optional subsample size for expensive metrics; ``None`` uses all rows.
        random_state:
            Seed for subsampling and stability bootstraps.
        compute_stability:
            When True, run bootstrap stability diagnostics.
        stability_runs:
            Number of bootstrap runs for stability analysis.
        stability_sample_fraction:
            Fraction of rows sampled per stability bootstrap.
        compute_elbow:
            When True, compute elbow curve for k-means k selection diagnostics.
        elbow_k_min:
            Minimum k for elbow diagnostics.
        elbow_k_max:
            Maximum k for elbow diagnostics.
        
        Returns
        -------
        ClusterEvalResult
            Internal and optional external clustering metrics.
        
        Raises
        ------
        ValidationError
            When no cluster plan exists on this Session.
        """
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
        """Return the last unsupervised cluster plan, if any.
        
        Stored on this Session after :meth:`fit_clusters` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        ClusterPlan or None
            ``None`` before the first :meth:`fit_clusters` call on this session.
        """
        return self._cluster_plan

    @property
    def cluster_fit_result(self) -> ClusterFitResult | None:
        """Return the last cluster fit result, if any.
        
        Stored on this Session after :meth:`fit_clusters` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        ClusterFitResult or None
            ``None`` before the first :meth:`fit_clusters` call on this session.
        """
        return self._cluster_fit_result

    @property
    def cluster_assign_result(self) -> ClusterAssignResult | None:
        """Return the last cluster assignment result, if any.
        
        Stored on this Session after :meth:`assign_clusters` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        ClusterAssignResult or None
            ``None`` before the first :meth:`assign_clusters` call on this session.
        """
        return self._cluster_assign_result

    @property
    def cluster_eval_result(self) -> ClusterEvalResult | None:
        """Return the last cluster evaluation result, if any.
        
        Stored on this Session after :meth:`evaluate_clusters` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        ClusterEvalResult or None
            ``None`` before the first :meth:`evaluate_clusters` call on this session.
        """
        return self._cluster_eval_result

    def save_unsupervised_bundle(self, path: str | Path) -> Path:
        """Persist the active cluster plan as ``buildml.unsupervised_bundle.v2``.
        
        Delegates to :meth:`buildml.unsupervised.checkpoint.save_unsupervised_bundle`.
        Reload with :meth:`load_unsupervised_bundle`.
        
        Parameters
        ----------
        path:
            Destination directory for the bundle (created if missing).
        
        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.
        
        Raises
        ------
        ValidationError
            When no cluster plan exists on this Session.
        """
        return unsupervised_ops.save_unsupervised_bundle_op(self, path=path)

    def load_unsupervised_bundle(self, path: str | Path) -> Session:
        """Load an unsupervised clustering bundle into this Session.
        
        Delegates to :meth:`buildml.unsupervised.checkpoint.load_unsupervised_bundle`
        and clears prior fit/assign/eval results.
        
        Parameters
        ----------
        path:
            Path to a ``buildml.unsupervised_bundle.v2`` directory.
        
        Returns
        -------
        Session
            this Session with cluster plan attached for chaining.
        """
        return unsupervised_ops.load_unsupervised_bundle_op(self, path=path)

    def fit_voting(
        self,
        estimators: Mapping[str, Any] | Sequence[tuple[str, Any]],
        *,
        voting: VotingMethod = "hard",
        weights: Sequence[float] | None = None,
        task: Literal["classification", "regression", "auto"] = "auto",
    ) -> EnsembleFitResult:
        """Fit a voting ensemble on the train partition only.
        
        Delegates to :meth:`buildml.ensemble.fit.fit_voting_ensemble`, stores the
        plan on this Session, and sets ``fit_result`` so classical evaluate/predict work.
        
        Parameters
        ----------
        estimators:
            Base estimators as a mapping or ``(name, estimator)`` sequence.
        voting:
            Voting strategy (``hard`` or ``soft`` for classifiers).
        weights:
            Optional per-estimator vote weights.
        task:
            Task type override (``classification``, ``regression``, or ``auto``).
        
        Returns
        -------
        EnsembleFitResult
            Serializable fit summary including base estimator names.
        
        Notes
        -----
        **Leakage:** Requires a split. Fits on train only. Sets Session ``fit_result``
        so classical ``evaluate`` / ``predict`` / ``save_pipeline`` work.
        """
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
        """Fit a stacking ensemble on the train partition only.
        
        Delegates to :meth:`buildml.ensemble.fit.fit_stacking_ensemble` with
        out-of-fold meta features computed inside train only.
        
        Parameters
        ----------
        estimators:
            Base estimators as a mapping or ``(name, estimator)`` sequence.
        final_estimator:
            Meta-learner fitted on out-of-fold base predictions.
        cv:
            Number of cross-validation folds inside train for OOF features.
        passthrough:
            When True, include original features in meta-learner input.
        stack_method:
            Base prediction method (``auto``, ``predict_proba``, etc.).
        task:
            Task type override (``classification``, ``regression``, or ``auto``).
        
        Returns
        -------
        EnsembleFitResult
            Serializable fit summary including CV fold count and base names.
        
        Notes
        -----
        **Leakage:** Stacking CV folds stay inside train. Session test is never used
        for out-of-fold meta features.
        """
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
        
        Delegates to :meth:`buildml.ensemble.fit.fit_blending_ensemble` with a
        holdout carved from train for meta-learner fitting.
        
        Parameters
        ----------
        estimators:
            Base estimators as a mapping or ``(name, estimator)`` sequence.
        final_estimator:
            Meta-learner fitted on holdout base predictions.
        holdout_fraction:
            Fraction of train rows reserved for blend holdout.
        blend_method:
            Base prediction method for blending (``predict_proba``, etc.).
        random_state:
            Seed for holdout split and base estimator initialization.
        refit_bases_on_full_train:
            When True, refit base estimators on all train rows after blending.
        passthrough:
            When True, include original features in meta-learner input.
        task:
            Task type override (``classification``, ``regression``, or ``auto``).
        
        Returns
        -------
        EnsembleFitResult
            Serializable fit summary including holdout fraction disclosures.
        
        Notes
        -----
        **Leakage:** The blend holdout is carved from train. Session validation/test
        never enter meta-learner fitting. Prefer stacking when you want CV OOF
        meta features instead of a single holdout.
        """
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
        
        Delegates to the same metric path as ``Session.evaluate``. Requires a
        prior :meth:`fit_voting`, :meth:`fit_stacking`, or :meth:`fit_blending`.
        
        Parameters
        ----------
        partition:
            Partition to evaluate (``train``, ``validation``, or ``test``).
        
        Returns
        -------
        EvaluateResult
            Classical metrics plus ensemble strategy disclosures.
        
        Raises
        ------
        ValidationError
            When no fitted ensemble exists on this Session.
        """
        return ensemble_ops.evaluate_ensemble(self, partition=partition)

    @property
    def ensemble_plan(self) -> EnsemblePlan | None:
        """Return the last native ensemble plan, if any.
        
        Stored on this Session after :meth:`fit_voting` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        EnsemblePlan or None
            ``None`` before the first :meth:`fit_voting` call on this session.
        """
        return self._ensemble_plan

    @property
    def ensemble_fit_result(self) -> EnsembleFitResult | None:
        """Return the last ensemble fit result, if any.
        
        Stored on this Session after :meth:`fit_voting` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        EnsembleFitResult or None
            ``None`` before the first :meth:`fit_voting` call on this session.
        """
        return self._ensemble_fit_result

    def save_ensemble_bundle(self, path: str | Path) -> Path:
        """Persist the active EnsemblePlan as ``buildml.ensemble_bundle.v1``.
        
        Delegates to :meth:`buildml.ensemble.checkpoint.save_ensemble_bundle`.
        Reload with :meth:`load_ensemble_bundle`.
        
        Parameters
        ----------
        path:
            Destination directory for the bundle (created if missing).
        
        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.
        
        Raises
        ------
        ValidationError
            When no ensemble plan exists on this Session.
        """
        return ensemble_ops.save_ensemble_bundle_op(self, path=path)

    def load_ensemble_bundle(self, path: str | Path) -> Session:
        """Load an ensemble bundle into this Session.
        
        Delegates to :meth:`buildml.ensemble.checkpoint.load_ensemble_bundle`
        and restores ``fit_result`` for classical evaluate/predict.
        
        Parameters
        ----------
        path:
            Path to a ``buildml.ensemble_bundle.v1`` directory.
        
        Returns
        -------
        Session
            this Session with EnsemblePlan and ``fit_result`` attached.
        """
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
        """Run AutoML model-family and recipe-strategy search on the train partition.
        
        Delegates to :meth:`buildml.automl.search.run_automl`, stores the
        :class:`~buildml.automl.results.AutoMLPlan` and winner on this Session, and
        optionally refits the best candidate. Follow with :meth:`evaluate_automl`
        for classical supervised metrics on a holdout partition.
        
        Parameters
        ----------
        backend:
            ``native``, ``optuna``, ``flaml``, or ``autogluon`` search backend.
        task:
            ``classification``, ``regression``, or ``auto`` to infer from target.
        method:
            Search method (``randomized``, ``grid``, ``optuna``, ``evolutionary``).
        selection:
            How to rank trials: ``cv``, ``nested``, or ``validation``.
        n_trials:
            Maximum candidate trials under the trial budget.
        cv:
            Inner CV folds or splitter for ``selection='cv'`` / ``'nested'``.
        outer_cv:
            Outer CV folds when ``selection='nested'``.
        cv_strategy:
            CV splitter strategy (``auto``, ``kfold``, ``stratified``, etc.).
        ranking_metric:
            Metric to rank candidates; defaults to task-appropriate score.
        families:
            Optional subset of model family names to search.
        include_recipe_search:
            When True, search discrete fold-local recipe strategies.
        include_industry_families:
            When True, extend catalog with GBDT families when extras installed.
        include_ensembles:
            When True, evaluate voting ensembles from diverse top families.
        ensemble_mode:
            Ensemble types to score when ``include_ensembles=True``.
        max_ensemble_bases:
            Maximum base families combined in one ensemble trial.
        preprocess:
            Fixed fold-local recipe when ``include_recipe_search=False``.
        allow_session_global_preprocess:
            Allow search when Session-global preprocess was already applied.
        refit:
            When True, refit the best candidate on all train rows after selection.
        random_state:
            Seed for randomized search and CV splitters.
        groups:
            Optional group labels for grouped CV strategies.
        budget:
            Structured trial/time budget caps for the search loop.
        time_budget:
            Optional wall-clock seconds cap for the search.
        
        Returns
        -------
        AutoMLResult
            Ranked trial table, winner metadata, and search disclosures.
            Session ``_fit_result`` is set when ``refit=True``.
        
        Notes
        -----
        **Leakage:** Same refusal as classical CV/search when Session-global
        preprocess already poisoned the frame. Session test never enters selection.
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
        
        Delegates to :meth:`buildml.model.supervised.evaluate_estimator` on the
        refitted winner stored in Session ``_fit_result``. Annotates diagnostics
        with AutoML plan metadata when available.
        
        Parameters
        ----------
        partition:
            Split partition to score (``train``, ``validation``, or ``test``).
        
        Returns
        -------
        EvaluateResult
            Metrics, diagnostics, and recommendations for the winning estimator.
        
        Raises
        ------
        ValidationError
            When no refitted AutoML winner exists on this Session.
        """
        return automl_ops.evaluate_automl(self, partition=partition)

    @property
    def automl_plan(self) -> AutoMLPlan | None:
        """Return the last selected AutoML plan, if any.
        
        Stored on this Session after :meth:`run_automl` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        AutoMLPlan or None
            ``None`` before the first :meth:`run_automl` call on this session.
        """
        return self._automl_plan

    @property
    def automl_result(self) -> AutoMLResult | None:
        """Return the last AutoML search result, if any.
        
        Stored on this Session after :meth:`run_automl` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        AutoMLResult or None
            ``None`` before the first :meth:`run_automl` call on this session.
        """
        return self._automl_result

    def save_automl_bundle(self, path: str | Path) -> Path:
        """Persist the active AutoML plan as ``buildml.automl_bundle.v1``.
        
        Delegates to :meth:`buildml.automl.checkpoint.save_automl_bundle`.
        Reload with :meth:`load_automl_bundle`.
        
        Parameters
        ----------
        path:
            Destination directory for the bundle (created if missing).
        
        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.
        
        Raises
        ------
        ValidationError
            When no AutoML plan exists on this Session.
        """
        return automl_ops.save_automl_bundle_op(self, path=path)

    def load_automl_bundle(self, path: str | Path) -> Session:
        """Load an AutoML bundle into this Session.
        
        Delegates to :meth:`buildml.automl.checkpoint.load_automl_bundle`,
        restores plan and refitted winner, and clears search result cache.
        
        Parameters
        ----------
        path:
            Path to a ``buildml.automl_bundle.v1`` directory.
        
        Returns
        -------
        Session
            this Session with AutoML plan and fit result attached for chaining.
        """
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
        """Fit a classical forecaster on the train partition only.
        
        Delegates to :meth:`buildml.forecasting.fit.fit_forecaster`, stores the
        :class:`~buildml.forecasting.results.ForecastPlan` on this Session, and records
        the fit. Follow with :meth:`generate_forecast` or
        :meth:`evaluate_forecast`.
        
        Parameters
        ----------
        method:
            Forecasting method (``lag_ridge``, ``arima``, ``nbeats``, etc.).
        horizon:
            Default forecast horizon in steps.
        lags:
            Explicit lag indices for lag-based methods.
        seasonal_period:
            Seasonal period for seasonal methods.
        exog_columns:
            Optional exogenous regressor columns.
        target_column:
            Target series column; defaults to target role.
        time_column:
            Timestamp column; inferred when omitted.
        random_state:
            Seed for stochastic estimators.
        alpha:
            Regularization strength for ridge-style methods.
        max_iter:
            Maximum iterations for iterative solvers.
        max_depth:
            Tree depth for tree-based forecasters.
        learning_rate:
            Learning rate for gradient boosting forecasters.
        order:
            ARIMA ``(p, d, q)`` order tuple.
        seasonal_order:
            Seasonal ARIMA ``(P, D, Q, s)`` order tuple.
        nbeats_input_size:
            Input window size for N-BEATS backend.
        nbeats_horizon:
            N-BEATS forecast horizon override.
        
        Returns
        -------
        ForecastFitResult
            Serializable fit summary including method and horizon disclosures.
        
        Notes
        -----
        **Leakage:** Requires ``time_split`` (or chronologically ordered
        ``inject_split``). Random/stratified/group splits are refused. Lag features
        use only past target values.
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
        """Generate an H-step forecast from the train-fitted ForecastPlan.
        
        Delegates to :meth:`buildml.forecasting.predict.generate_forecast` without
        refitting. History is taken from train end or extended through validation/test
        when ``origin`` requests it.
        
        Parameters
        ----------
        horizon:
            Forecast steps; defaults to the plan horizon when ``None``.
        origin:
            History cutoff (``train_end``, ``validation_end``, or ``test_end``).
        future_exog:
            Optional exogenous values for the forecast horizon.
        
        Returns
        -------
        ForecastGenerateResult
            Point forecasts and optional intervals for the requested horizon.
        
        Raises
        ------
        ValidationError
            When no forecast plan exists or ``origin`` requires a missing split.
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
        """Evaluate the train-fitted ForecastPlan on a holdout partition.
        
        Delegates to :meth:`buildml.forecasting.evaluate.evaluate_forecast` using
        rolling or static evaluation strategies. Falls back to ``test`` when no
        validation partition exists.
        
        Parameters
        ----------
        partition:
            Holdout partition (default ``test``).
        strategy:
            Evaluation strategy (``rolling_one_step`` or ``static_multi_step``).
        
        Returns
        -------
        ForecastEvalResult
            Holdout error metrics for the frozen forecast plan.
        
        Raises
        ------
        ValidationError
            When no forecast plan exists on this Session.
        """
        return forecast_ops.evaluate_forecast_op(
            self, partition=partition, strategy=strategy
        )

    @property
    def forecast_plan(self) -> ForecastPlan | None:
        """Return the last forecast plan, if any.
        
        Stored on this Session after :meth:`fit_forecast` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        ForecastPlan or None
            ``None`` before the first :meth:`fit_forecast` call on this session.
        """
        return self._forecast_plan

    @property
    def forecast_fit_result(self) -> ForecastFitResult | None:
        """Return the last forecast fit result, if any.
        
        Stored on this Session after :meth:`fit_forecast` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        ForecastFitResult or None
            ``None`` before the first :meth:`fit_forecast` call on this session.
        """
        return self._forecast_fit_result

    @property
    def forecast_generate_result(self) -> ForecastGenerateResult | None:
        """Return the last forecast generation result, if any.
        
        Stored on this Session after :meth:`generate_forecast` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        ForecastGenerateResult or None
            ``None`` before the first :meth:`generate_forecast` call on this session.
        """
        return self._forecast_generate_result

    @property
    def forecast_eval_result(self) -> ForecastEvalResult | None:
        """Return the last forecast evaluation result, if any.
        
        Stored on this Session after :meth:`evaluate_forecast` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        ForecastEvalResult or None
            ``None`` before the first :meth:`evaluate_forecast` call on this session.
        """
        return self._forecast_eval_result

    def save_forecast_bundle(self, path: str | Path) -> Path:
        """Persist the active ForecastPlan as ``buildml.forecast_bundle.v2``.
        
        Delegates to :meth:`buildml.forecasting.checkpoint.save_forecast_bundle`.
        Reload with :meth:`load_forecast_bundle`.
        
        Parameters
        ----------
        path:
            Destination directory for the bundle (created if missing).
        
        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.
        
        Raises
        ------
        ValidationError
            When no forecast plan exists on this Session.
        """
        return forecast_ops.save_forecast_bundle_op(self, path=path)

    def load_forecast_bundle(self, path: str | Path) -> Session:
        """Load a forecast bundle into this Session.
        
        Delegates to :meth:`buildml.forecasting.checkpoint.load_forecast_bundle`
        and clears prior generate/eval results.
        
        Parameters
        ----------
        path:
            Path to a ``buildml.forecast_bundle.v2`` directory.
        
        Returns
        -------
        Session
            this Session with ForecastPlan attached for chaining.
        """
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
        """Run time-series analysis on train-only or full-dataset scope.
        
        Delegates to :meth:`buildml.timeseries.analyze.analyze_timeseries`, stores
        the result on this Session, and records the operation. Default scope is
        ``train`` to avoid peeking at holdout data during EDA.
        
        Parameters
        ----------
        target_column:
            Series to analyze; defaults to the target role column.
        time_column:
            Timestamp or index column; inferred when omitted.
        scope:
            ``train`` (default) restricts to train indices; ``all`` uses full data.
        seasonal_period:
            Seasonal period for decomposition and diagnostics.
        decompose_method:
            Decomposition algorithm (STL, classical, etc.).
        include_decompose:
            When True, run seasonal decomposition.
        include_diagnostics:
            When True, run stationarity and autocorrelation diagnostics.
        include_changepoints:
            When True, detect structural changepoints.
        include_features:
            When True, extract lag/rolling/spectral features.
        acf_lags:
            Maximum lag for autocorrelation function plots.
        pacf_lags:
            Maximum lag for partial autocorrelation function plots.
        adf_regression:
            Regression term for Augmented Dickey-Fuller test.
        kpss_regression:
            Regression term for KPSS stationarity test.
        changepoint_method:
            Changepoint detection algorithm override.
        changepoint_penalty:
            Penalty controlling changepoint count.
        rolling_window:
            Window size for rolling statistics.
        spectral_n_fft:
            FFT size for spectral analysis (``None`` uses series length).
        
        Returns
        -------
        TimeseriesAnalysisResult
            Decomposition, diagnostics, changepoints, and feature summaries.
            Use :meth:`ts_decompose` or :meth:`ts_diagnostics` for focused runs.
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

    def ts_decompose(
        self,
        *,
        target_column: str | None = None,
        time_column: str | None = None,
        scope: str = "train",
        seasonal_period: int | None = None,
        decompose_method: str | None = None,
    ) -> Any:
        """Run decomposition-only time-series analysis on Session data.

        Convenience wrapper around :meth:`analyze_timeseries` that enables
        seasonal decomposition and disables diagnostics, changepoints, and
        feature extraction. Use this when you only need trend/seasonal/residual
        components before choosing a forecast method.

        Parameters
        ----------
        target_column:
            Series to decompose; defaults to the target role column.
        time_column:
            Timestamp or index column; inferred when omitted.
        scope:
            ``train`` (default) restricts to train indices; ``all`` uses full data.
        seasonal_period:
            Seasonal period for decomposition.
        decompose_method:
            Decomposition algorithm (STL, classical, etc.).

        Returns
        -------
        TimeseriesAnalysisResult
            Result with decomposition components populated.
        """
        return timeseries_ops.ts_decompose_op(
            self,
            target_column=target_column,
            time_column=time_column,
            scope=scope,
            seasonal_period=seasonal_period,
            decompose_method=decompose_method,
        )

    def ts_diagnostics(
        self,
        *,
        target_column: str | None = None,
        time_column: str | None = None,
        scope: str = "train",
        acf_lags: int = 40,
        pacf_lags: int = 40,
    ) -> Any:
        """Run diagnostics-only time-series analysis on Session data.

        Convenience wrapper around :meth:`analyze_timeseries` that runs ACF/PACF
        and ADF/KPSS stationarity tests while skipping decomposition,
        changepoints, and feature extraction.

        Parameters
        ----------
        target_column:
            Series to diagnose; defaults to the target role column.
        time_column:
            Timestamp or index column; inferred when omitted.
        scope:
            ``train`` (default) restricts to train indices; ``all`` uses full data.
        acf_lags:
            Maximum lag for autocorrelation function plots.
        pacf_lags:
            Maximum lag for partial autocorrelation function plots.

        Returns
        -------
        TimeseriesAnalysisResult
            Result with diagnostic tests and ACF/PACF summaries populated.
        """
        return timeseries_ops.ts_diagnostics_op(
            self,
            target_column=target_column,
            time_column=time_column,
            scope=scope,
            acf_lags=acf_lags,
            pacf_lags=pacf_lags,
        )

    @property
    def ts_analysis_result(self) -> Any | None:
        """Return the last time-series analysis result, if any.
        
        Stored on this Session after :meth:`analyze_timeseries` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        TimeseriesAnalysisResult or None
            ``None`` before the first :meth:`analyze_timeseries` call on this session.
        """
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
        """Fit an anomaly detector on the train partition only.
        
        Delegates to :meth:`buildml.anomaly.fit.fit_detector`, stores the
        :class:`~buildml.anomaly.results.AnomalyPlan` on this Session, and records
        the fit. ``backend`` selects sklearn (core), pyod
        (``buildml[anomaly-industry]``), or torch (``buildml[torch]``). ``method``
        must belong to the backend catalog — see :meth:`anomaly_capability_matrix`.
        
        Parameters
        ----------
        backend:
            Optional backend override (see capability matrix for identifiers).
        method:
            Method or strategy identifier for the resolved backend.
        mode:
            Anomaly detection mode (``unsupervised`` or ``supervised``).
        columns:
            Optional explicit feature column list; ``None`` auto-selects numerics.
        random_state:
            Seed for stochastic steps (sampling, initialization, bagging).
        contamination:
            Expected outlier fraction for sklearn-style detectors.
        threshold_policy:
            How the decision threshold is chosen from train scores.
        score_threshold:
            Fixed score cutoff when threshold policy is ``fixed``.
        quantile:
            Quantile for score threshold when policy is ``quantile``.
        n_estimators:
            Number of trees for forest-based detectors.
        max_samples:
            Subsample size for isolation forest (``auto``, int, or float).
        n_neighbors:
            Neighborhood size for LOF and similar methods.
        nu:
            Upper bound on training-error fraction for one-class SVM.
        kernel:
            Kernel type for one-class SVM.
        gamma:
            Kernel coefficient for one-class SVM.
        latent_dim:
            Bottleneck width for torch autoencoder backend.
        ae_epochs:
            Training epochs for torch autoencoder backend.
        ae_batch_size:
            Minibatch size for torch autoencoder backend.
        normal_label_column:
            Optional column marking normal rows in semi-supervised setups.
        normal_label_value:
            Value in ``normal_label_column`` indicating normal rows.
        positive_label:
            Positive class label for supervised fraud scorers.
        prefer_reduce_components:
            Prefer reduced component columns when a reduce plan exists on this Session.
        flag_column:
            Output column name for boolean anomaly flags when attaching scores.
        score_column:
            Output column name for continuous anomaly scores when attaching.
        
        Returns
        -------
        AnomalyFitResult
            Serializable fit summary including threshold and alert-rate disclosures.
            Follow with :meth:`score_anomalies` or :meth:`tune_anomaly_threshold`.
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
        """Tune the anomaly decision threshold on validation labels without refitting.
        
        Delegates to :meth:`buildml.anomaly.threshold.tune_anomaly_threshold` and
        optionally writes the tuned threshold back to this Session plan. Test
        tuning requires explicit ``allow_test_tuning=True``.
        
        Parameters
        ----------
        partition:
            Labeled holdout partition for threshold search (default ``validation``).
        label_column:
            Optional explicit label column; defaults to target role column.
        positive_label:
            Positive/anomaly label value for supervised tuning metrics.
        metric:
            Metric to optimize (``f1``, ``fbeta``, ``precision``, ``recall``).
        fbeta:
            Beta for F-beta when ``metric='fbeta'``.
        allow_test_tuning:
            When False, refuse tuning on the test partition.
        update_plan:
            When True, apply the tuned threshold to the active Session plan.
        
        Returns
        -------
        AnomalyThresholdTuneResult
            Tuned threshold, metric value, and partition used for search.
        
        Raises
        ------
        ValidationError
            When no anomaly plan exists or tuning preconditions fail.
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
        """Return the anomaly backend/method capability matrix for this install.
        
        Delegates to :meth:`buildml.anomaly.catalog.anomaly_capability_matrix`.
        Use before :meth:`fit_anomaly` to confirm ``backend`` and ``method`` pairs
        available with current extras.
        
        Returns
        -------
        dict
            Nested map of backend identifiers to supported methods and modes.
        """
        return anomaly_ops.anomaly_capability_matrix_op()

    def score_anomalies(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        attach: bool = False,
        override_threshold: float | None = None,
    ) -> AnomalyScoreResult:
        """Score and flag rows with the train-fitted anomaly plan without refitting.
        
        Delegates to :meth:`buildml.anomaly.score.score_anomalies`. When
        ``attach=True``, score and flag columns are merged into Session dataset.
        
        Parameters
        ----------
        partition:
            Partition to score (``train``, ``validation``, ``test``, or ``all``).
        attach:
            When True, attach score and flag columns to this Session dataset frame.
        override_threshold:
            Optional score cutoff overriding the plan threshold for this call.
        
        Returns
        -------
        AnomalyScoreResult
            Scores, flags, and optional alert-rate summary for the partition.
        
        Raises
        ------
        ValidationError
            When no anomaly plan exists on this Session.
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
        """Evaluate train-fitted anomaly scores on a labeled holdout partition.
        
        Delegates to :meth:`buildml.anomaly.evaluate.evaluate_anomaly`. Requires
        labels on the holdout partition; does not refit the detector.
        
        Parameters
        ----------
        partition:
            Holdout partition to score. Validation falls back to test when absent.
        label_column:
            Optional explicit label column; defaults to target role column.
        positive_label:
            Positive/anomaly label value for supervised metrics.
        k:
            Optional top-k for precision@k / recall@k style metrics.
        override_threshold:
            Optional score cutoff overriding the plan threshold for this evaluation.
        
        Returns
        -------
        AnomalyEvalResult
            Holdout classification metrics and ranking diagnostics when labels exist.
        
        Raises
        ------
        ValidationError
            When no anomaly plan exists on this Session.
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
        """Return the last anomaly plan, if any.
        
        Stored on this Session after :meth:`fit_anomaly` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        AnomalyPlan or None
            ``None`` before the first :meth:`fit_anomaly` call on this session.
        """
        return self._anomaly_plan

    @property
    def anomaly_fit_result(self) -> AnomalyFitResult | None:
        """Return the last anomaly fit result, if any.
        
        Stored on this Session after :meth:`fit_anomaly` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        AnomalyFitResult or None
            ``None`` before the first :meth:`fit_anomaly` call on this session.
        """
        return self._anomaly_fit_result

    @property
    def anomaly_score_result(self) -> AnomalyScoreResult | None:
        """Return the last anomaly scoring result, if any.
        
        Stored on this Session after :meth:`score_anomalies` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        AnomalyScoreResult or None
            ``None`` before the first :meth:`score_anomalies` call on this session.
        """
        return self._anomaly_score_result

    @property
    def anomaly_eval_result(self) -> AnomalyEvalResult | None:
        """Return the last anomaly evaluation result, if any.
        
        Stored on this Session after :meth:`evaluate_anomaly` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        AnomalyEvalResult or None
            ``None`` before the first :meth:`evaluate_anomaly` call on this session.
        """
        return self._anomaly_eval_result

    def save_anomaly_bundle(self, path: str | Path) -> Path:
        """Persist the active anomaly plan as ``buildml.anomaly_bundle.v1``.
        
        Delegates to :meth:`buildml.anomaly.checkpoint.save_anomaly_bundle`.
        Reload with :meth:`load_anomaly_bundle`.
        
        Parameters
        ----------
        path:
            Destination directory for the bundle (created if missing).
        
        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.
        
        Raises
        ------
        ValidationError
            When no anomaly plan exists on this Session.
        """
        return anomaly_ops.save_anomaly_bundle_op(self, path=path)

    def load_anomaly_bundle(self, path: str | Path) -> Session:
        """Load an anomaly bundle into this Session.
        
        Delegates to :meth:`buildml.anomaly.checkpoint.load_anomaly_bundle` and
        clears prior score/eval/threshold-tune results.
        
        Parameters
        ----------
        path:
            Path to a ``buildml.anomaly_bundle.v1`` directory.
        
        Returns
        -------
        Session
            this Session with anomaly plan attached for chaining.
        """
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
        """Fit a semi-supervised classifier on labeled and unlabeled train rows.
        
        Delegates to :meth:`buildml.semisupervised.fit.fit_semisupervised`, stores
        the :class:`~buildml.semisupervised.results.SemiSupervisedPlan` on this Session,
        and records the fit. Follow with :meth:`predict_semisupervised` or
        :meth:`evaluate_semisupervised`.
        
        Parameters
        ----------
        backend:
            Optional backend override (``sklearn``, ``industry``, ``torch``, ``text``).
        method:
            Semi-supervised method key (``label_propagation``, ``self_training``, etc.).
        columns:
            Optional explicit feature columns for tabular backends.
        random_state:
            Seed for stochastic steps.
        kernel:
            Kernel or affinity type for graph-based methods.
        n_neighbors:
            Neighborhood size for kNN graph construction.
        max_iter:
            Maximum iterations for iterative label propagation methods.
        alpha:
            Clamping factor for label propagation (labeled vs propagated mass).
        base_estimator:
            Base classifier for self-training and pseudo-label methods.
        threshold:
            Confidence threshold for pseudo-label acceptance.
        criterion:
            Pseudo-label selection criterion (``threshold`` or ``k_best``).
        k_best:
            Top-k pseudo-labels per iteration when ``criterion='k_best'``.
        max_self_train_iter:
            Maximum self-training rounds for pseudo-label methods.
        epochs:
            Training epochs for torch consistency-regularization backend.
        batch_size:
            Minibatch size for torch backend.
        learning_rate:
            Optimizer learning rate for torch backend.
        consistency_weight:
            Weight on unlabeled consistency loss for torch backend.
        mixup_alpha:
            Mixup alpha for torch consistency backend.
        device:
            Torch device string (``cpu`` or ``cuda``).
        text_column:
            Text column for HF embedding semi-supervised backend.
        text_model_name:
            Sentence-transformer model name for text backend.
        unlabeled_marker:
            Value treated as unlabeled in the train target column.
        prefer_reduce_components:
            Prefer reduced component columns when a reduce plan exists on this Session.
        
        Returns
        -------
        SemiSupervisedFitResult
            Serializable fit summary including labeled/unlabeled train counts.
        
        Notes
        -----
        **Leakage:** Requires a split. Fit uses train only. Unlabeled rows are
        target NaNs (default). Validation/test never invent labels for selection.
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
        """Predict with the train-fitted semi-supervised plan without refitting.
        
        Delegates to :meth:`buildml.semisupervised.predict.predict_semisupervised`.
        When ``attach=True``, predictions are merged into Session dataset.
        
        Parameters
        ----------
        partition:
            Partition to score (``train``, ``validation``, ``test``, or ``all``).
        attach:
            When True, attach prediction column to this Session dataset frame.
        prediction_column:
            Column name used when ``attach=True``.
        
        Returns
        -------
        SemiSupervisedPredictResult
            Predictions and optional probabilities for the requested partition.
        
        Raises
        ------
        ValidationError
            When no semi-supervised plan exists on this Session.
        """
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
        """Evaluate the semi-supervised plan on labeled rows of a holdout partition.
        
        Delegates to :meth:`buildml.semisupervised.evaluate.evaluate_semisupervised`.
        Unlabeled holdout rows are skipped; holdout data is never used during fit.
        
        Parameters
        ----------
        partition:
            Holdout partition to score. Validation falls back to test when absent.
        unlabeled_marker:
            Value treated as unlabeled when scoring labeled rows only.
        
        Returns
        -------
        SemiSupervisedEvalResult
            Holdout metrics computed on labeled rows only.
        
        Raises
        ------
        ValidationError
            When no semi-supervised plan exists on this Session.
        """
        return semisupervised_ops.evaluate_semisupervised_op(
            self,
            partition=partition,
            unlabeled_marker=unlabeled_marker,
        )

    @property
    def semisupervised_plan(self) -> SemiSupervisedPlan | None:
        """Return the last semi-supervised plan, if any.
        
        Stored on this Session after :meth:`fit_semisupervised` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        SemiSupervisedPlan or None
            ``None`` before the first :meth:`fit_semisupervised` call on this session.
        """
        return self._semisupervised_plan

    @property
    def semisupervised_fit_result(self) -> SemiSupervisedFitResult | None:
        """Return the last semi-supervised fit result, if any.
        
        Stored on this Session after :meth:`fit_semisupervised` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        SemiSupervisedFitResult or None
            ``None`` before the first :meth:`fit_semisupervised` call on this session.
        """
        return self._semisupervised_fit_result

    @property
    def semisupervised_predict_result(self) -> SemiSupervisedPredictResult | None:
        """Return the last semi-supervised prediction result, if any.
        
        Stored on this Session after :meth:`predict_semisupervised` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        SemiSupervisedPredictResult or None
            ``None`` before the first :meth:`predict_semisupervised` call on this session.
        """
        return self._semisupervised_predict_result

    @property
    def semisupervised_eval_result(self) -> SemiSupervisedEvalResult | None:
        """Return the last semi-supervised evaluation result, if any.
        
        Stored on this Session after :meth:`evaluate_semisupervised` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        SemiSupervisedEvalResult or None
            ``None`` before the first :meth:`evaluate_semisupervised` call on this session.
        """
        return self._semisupervised_eval_result

    def save_semisupervised_bundle(self, path: str | Path) -> Path:
        """Persist the semi-supervised plan as ``buildml.semisupervised_bundle.v1``.
        
        Delegates to :meth:`buildml.semisupervised.checkpoint.save_semisupervised_bundle`.
        Reload with :meth:`load_semisupervised_bundle`.
        
        Parameters
        ----------
        path:
            Destination directory for the bundle (created if missing).
        
        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.
        
        Raises
        ------
        ValidationError
            When no semi-supervised plan exists on this Session.
        """
        return semisupervised_ops.save_semisupervised_bundle_op(self, path=path)

    def load_semisupervised_bundle(self, path: str | Path) -> Session:
        """Load a semi-supervised bundle into this Session.
        
        Delegates to :meth:`buildml.semisupervised.checkpoint.load_semisupervised_bundle`
        and clears prior fit/predict/eval results.
        
        Parameters
        ----------
        path:
            Path to a ``buildml.semisupervised_bundle.v1`` directory.
        
        Returns
        -------
        Session
            this Session with semi-supervised plan attached for chaining.
        """
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
        """Fit a self-supervised pretext encoder on the train partition only.
        
        Delegates to :meth:`buildml.selfsupervised.fit.fit_ssl_pretext`, stores the
        :class:`~buildml.selfsupervised.results.SSLPlan` on this Session, and records
        the fit. Follow with :meth:`transform_ssl` or :meth:`finetune_ssl_head`.
        
        Parameters
        ----------
        method:
            Self-supervised method override; inferred from modality when ``None``.
        columns:
            Tabular feature columns for pretext training.
        text_column:
            Text column for language-model or contrastive text methods.
        image_column:
            Image path/bytes column for vision methods.
        random_state:
            Seed for augmentations and initialization.
        latent_dim:
            Output embedding dimensionality.
        hidden:
            Hidden layer sizes for tabular encoders.
        mask_ratio:
            Fraction of features masked in masked-modeling pretext tasks.
        n_mask_views:
            Number of masked views per sample for contrastive objectives.
        max_iter:
            Maximum iterations for sklearn-style encoders.
        epochs:
            Training epochs for torch backends.
        batch_size:
            Minibatch size for torch training.
        learning_rate:
            Optimizer learning rate for torch backends.
        temperature:
            Temperature for contrastive loss scaling.
        projector_dim:
            Projector head dimension for contrastive methods.
        projector_hidden:
            Hidden sizes for the contrastive projector MLP.
        prefer_reduce_components:
            Prefer reduced component columns when a reduce plan exists on this Session.
        representation_prefix:
            Column prefix when attaching embeddings to the dataset.
        backbone:
            Vision backbone architecture name for image methods.
        weight_mode:
            Weight initialization mode for mock/demo vision backends.
        hf_model_name:
            HuggingFace model name for text embedding methods.
        device:
            Torch device string (``cpu`` or ``cuda``).
        
        Returns
        -------
        SSLFitResult
            Serializable fit summary including method, modality, and disclosures.
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
        """Export SSL representations with the train-fitted pretext encoder.
        
        Delegates to :meth:`buildml.selfsupervised.transform.transform_ssl`
        without refitting. Optionally attaches embedding columns to Session dataset.
        
        Parameters
        ----------
        partition:
            Split partition to encode (default ``train``).
        attach:
            When True, merge embedding columns into this Session dataset frame.
        
        Returns
        -------
        SSLTransformResult
            Embedding matrix metadata and optional attached column names.
        
        Raises
        ------
        ValidationError
            When no SSL plan exists on this Session.
        """
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
        """Fit a supervised head on frozen SSL embeddings using labeled train rows.
        
        Delegates to :meth:`buildml.selfsupervised.finetune.finetune_ssl_head`.
        Requires a prior :meth:`fit_ssl_pretext`.
        
        Parameters
        ----------
        estimator:
            Supervised head estimator (``logistic_regression``, etc.).
        random_state:
            Seed for head fitting.
        unlabeled_marker:
            Value marking unlabeled rows to exclude from head training.
        
        Returns
        -------
        SSLHeadFitResult
            Head fit summary including labeled row counts and disclosures.
        
        Raises
        ------
        ValidationError
            When no SSL plan exists on this Session.
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
        """Evaluate frozen SSL pretext encoder and head on a labeled partition.
        
        Delegates to :meth:`buildml.selfsupervised.evaluate.evaluate_ssl`.
        Requires both :meth:`fit_ssl_pretext` and :meth:`finetune_ssl_head`.
        Falls back to ``test`` when no validation partition exists.
        
        Parameters
        ----------
        partition:
            Holdout partition for evaluation (default ``validation``).
        unlabeled_marker:
            Value marking unlabeled rows to exclude from evaluation.
        
        Returns
        -------
        SSLEvalResult
            Holdout metrics for the frozen pretext + head pipeline.
        
        Raises
        ------
        ValidationError
            When SSL or head plans are missing on this Session.
        """
        return selfsupervised_ops.evaluate_ssl_op(
            self,
            partition=partition,
            unlabeled_marker=unlabeled_marker,
        )

    @property
    def ssl_plan(self) -> SelfSupervisedPlan | None:
        """Return the last self-supervised pretext plan, if any.
        
        Stored on this Session after :meth:`fit_ssl_pretext` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        SelfSupervisedPlan or None
            ``None`` before the first :meth:`fit_ssl_pretext` call on this session.
        """
        return self._ssl_plan

    @property
    def ssl_fit_result(self) -> SelfSupervisedFitResult | None:
        """Return the last self-supervised fit result, if any.
        
        Stored on this Session after :meth:`fit_ssl_pretext` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        SelfSupervisedFitResult or None
            ``None`` before the first :meth:`fit_ssl_pretext` call on this session.
        """
        return self._ssl_fit_result

    @property
    def ssl_transform_result(self) -> SelfSupervisedTransformResult | None:
        """Return the last SSL transform result, if any.
        
        Stored on this Session after :meth:`transform_ssl` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        SelfSupervisedTransformResult or None
            ``None`` before the first :meth:`transform_ssl` call on this session.
        """
        return self._ssl_transform_result

    @property
    def ssl_head_plan(self) -> SSLHeadPlan | None:
        """Return the last SSL head plan, if any.
        
        Stored on this Session after :meth:`finetune_ssl_head` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        SSLHeadPlan or None
            ``None`` before the first :meth:`finetune_ssl_head` call on this session.
        """
        return self._ssl_head_plan

    @property
    def ssl_head_fit_result(self) -> SSLHeadFitResult | None:
        """Return the last SSL head fit result, if any.
        
        Stored on this Session after :meth:`finetune_ssl_head` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        SSLHeadFitResult or None
            ``None`` before the first :meth:`finetune_ssl_head` call on this session.
        """
        return self._ssl_head_fit_result

    @property
    def ssl_eval_result(self) -> SelfSupervisedEvalResult | None:
        """Return the last SSL evaluation result, if any.
        
        Stored on this Session after :meth:`evaluate_ssl` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        SelfSupervisedEvalResult or None
            ``None`` before the first :meth:`evaluate_ssl` call on this session.
        """
        return self._ssl_eval_result

    def save_ssl_bundle(self, path: str | Path) -> Path:
        """Persist the active SSL plan as ``buildml.ssl_bundle.v2``.
        
        Delegates to :meth:`buildml.selfsupervised.checkpoint.save_ssl_bundle`.
        Reload with :meth:`load_ssl_bundle`.
        
        Parameters
        ----------
        path:
            Destination directory for the bundle (created if missing).
        
        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.
        
        Raises
        ------
        ValidationError
            When no SSL plan exists on this Session.
        """
        return selfsupervised_ops.save_ssl_bundle_op(self, path=path)

    def load_ssl_bundle(self, path: str | Path) -> Session:
        """Load a self-supervised bundle into this Session.
        
        Delegates to :meth:`buildml.selfsupervised.checkpoint.load_ssl_bundle`
        and clears prior transform/head/eval results.
        
        Parameters
        ----------
        path:
            Path to a ``buildml.ssl_bundle.v2`` directory.
        
        Returns
        -------
        Session
            this Session with SSLPlan and optional head plan attached.
        """
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
        """Fit or initialize the active learner on labeled train rows only.
        
        Delegates to :meth:`buildml.activelearning.fit.fit_active_learner`, stores
        the :class:`~buildml.activelearning.results.ActiveLearningPlan` on this Session,
        and records the fit. Follow with :meth:`suggest_query` and
        :meth:`label_rows` in a human-in-the-loop loop.
        
        Parameters
        ----------
        backend:
            Optional backend override (``sklearn``, ``industry``, ``torch``).
        strategy:
            Query strategy (``margin``, ``entropy``, ``committee``, etc.).
        base_estimator:
            Base estimator key for sklearn/industry backends.
        columns:
            Optional explicit feature columns.
        random_state:
            Seed for stochastic steps and committee members.
        batch_size:
            Default number of rows to suggest per query round.
        label_budget:
            Optional cap on total labels before query stops.
        unlabeled_marker:
            Value treated as unlabeled in the train target column.
        prefer_reduce_components:
            Prefer reduced component columns when a reduce plan exists on this Session.
        committee_size:
            Number of committee members for query-by-committee strategies.
        auto_refit:
            When True, refit after each labeling round by default.
        epochs:
            Training epochs for torch uncertainty backend.
        learning_rate:
            Optimizer learning rate for torch backend.
        mc_samples:
            Monte Carlo dropout samples for torch uncertainty strategies.
        device:
            Torch device string (``cpu`` or ``cuda``).
        
        Returns
        -------
        ActiveLearningFitResult
            Serializable fit summary including labeled/unlabeled pool sizes.
        
        Notes
        -----
        **Leakage:** Requires a split. Fit uses labeled train rows only. The
        unlabeled pool is train target missingness (NaN by default). Validation/test
        are never the query pool. Labels come from the user — no oracle in core.
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
        """Suggest unlabeled train-pool indices for human labeling without an oracle.
        
        Delegates to :meth:`buildml.activelearning.query.suggest_query` and stores
        suggested indices on this Session. User labels must be supplied via
        :meth:`label_rows`.
        
        Parameters
        ----------
        batch_size:
            Optional override for rows to suggest this round.
        strategy:
            Optional override for the query strategy this round.
        
        Returns
        -------
        ActiveLearningQueryResult
            Suggested train-pool indices, scores, and strategy metadata.
        
        Raises
        ------
        ValidationError
            When no active-learning plan exists on this Session.
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
        """Incorporate user-provided labels on train-pool rows and optionally refit.
        
        Delegates to :meth:`buildml.activelearning.label.label_rows`, mutates
        Session dataset labels, updates the plan, and optionally refits the learner.
        
        Parameters
        ----------
        indices:
            Train-pool dataset indices to label (from :meth:`suggest_query`).
        labels:
            User-supplied labels aligned with ``indices``.
        refit:
            When True/False, override plan ``auto_refit`` for this labeling round.
        
        Returns
        -------
        ActiveLearningLabelResult
            Labeling summary including whether a refit occurred.
        
        Raises
        ------
        ValidationError
            When no active-learning plan exists on this Session.
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
        """Evaluate the active learner on labeled rows of a holdout partition.
        
        Delegates to :meth:`buildml.activelearning.evaluate.evaluate_active_learning`.
        Unlabeled holdout rows are skipped; holdout data is never queried.
        
        Parameters
        ----------
        partition:
            Holdout partition to score. Validation falls back to test when absent.
        unlabeled_marker:
            Value treated as unlabeled when scoring labeled rows only.
        
        Returns
        -------
        ActiveLearningEvalResult
            Holdout metrics computed on labeled rows only.
        
        Raises
        ------
        ValidationError
            When no active-learning plan exists on this Session.
        """
        return activelearning_ops.evaluate_active_learning_op(
            self,
            partition=partition,
            unlabeled_marker=unlabeled_marker,
        )

    @property
    def activelearning_plan(self) -> ActiveLearningPlan | None:
        """Return the last active-learning plan, if any.
        
        Stored on this Session after :meth:`fit_active_learner` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        ActiveLearningPlan or None
            ``None`` before the first :meth:`fit_active_learner` call on this session.
        """
        return self._activelearning_plan

    @property
    def activelearning_fit_result(self) -> ActiveLearningFitResult | None:
        """Return the last active-learning fit result, if any.
        
        Stored on this Session after :meth:`fit_active_learner` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        ActiveLearningFitResult or None
            ``None`` before the first :meth:`fit_active_learner` call on this session.
        """
        return self._activelearning_fit_result

    @property
    def activelearning_query_result(self) -> ActiveLearningQueryResult | None:
        """Return the last active-learning query result, if any.
        
        Stored on this Session after :meth:`suggest_query` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        ActiveLearningQueryResult or None
            ``None`` before the first :meth:`suggest_query` call on this session.
        """
        return self._activelearning_query_result

    @property
    def activelearning_label_result(self) -> ActiveLearningLabelResult | None:
        """Return the last active-learning labeling result, if any.
        
        Stored on this Session after :meth:`label_rows` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        ActiveLearningLabelResult or None
            ``None`` before the first :meth:`label_rows` call on this session.
        """
        return self._activelearning_label_result

    @property
    def activelearning_eval_result(self) -> ActiveLearningEvalResult | None:
        """Return the last active-learning evaluation result, if any.
        
        Stored on this Session after :meth:`evaluate_active_learning` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        ActiveLearningEvalResult or None
            ``None`` before the first :meth:`evaluate_active_learning` call on this session.
        """
        return self._activelearning_eval_result

    def save_active_learning_bundle(self, path: str | Path) -> Path:
        """Persist the active-learning plan as ``buildml.activelearning_bundle.v1``.
        
        Delegates to
        :meth:`buildml.activelearning.checkpoint.save_active_learning_bundle`.
        Reload with :meth:`load_active_learning_bundle`.
        
        Parameters
        ----------
        path:
            Destination directory for the bundle (created if missing).
        
        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.
        
        Raises
        ------
        ValidationError
            When no active-learning plan exists on this Session.
        """
        return activelearning_ops.save_active_learning_bundle_op(self, path=path)

    def load_active_learning_bundle(self, path: str | Path) -> Session:
        """Load an active-learning bundle into this Session.
        
        Delegates to
        :meth:`buildml.activelearning.checkpoint.load_active_learning_bundle`
        and clears prior fit/query/label/eval results.
        
        Parameters
        ----------
        path:
            Path to a ``buildml.activelearning_bundle.v1`` directory.
        
        Returns
        -------
        Session
            this Session with active-learning plan attached for chaining.
        """
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
        """Warm-start an incremental estimator on the first train chunk.
        
        Delegates to :meth:`buildml.online.fit.fit_online`, stores the resulting
        :class:`~buildml.online.results.OnlinePlan` on this Session, and records
        the operation. Follow with :meth:`partial_fit_online` for train-only
        updates, then :meth:`evaluate_online` or :meth:`predict_online`
        on holdout partitions.
        
        Parameters
        ----------
        backend:
            Optional backend override (``sklearn``, ``industry``, ``torch``).
        estimator:
            Online estimator key (see the online capability matrix).
        task:
            Optional task override (``classification`` or ``regression``).
        columns:
            Optional explicit feature columns.
        random_state:
            Seed for stochastic estimators.
        chunk_size:
            Default rows per subsequent partial_fit chunk.
        n_init:
            Init chunk size; defaults to ``chunk_size`` when ``None``.
        indices:
            Optional explicit train-partition indices for the init chunk.
        classes:
            Full label vocabulary for classifiers (discovered from train if omitted).
        prefer_reduce_components:
            Prefer reduced component columns when a reduce plan exists on this Session.
        allow_refit_fallback:
            Permit disclosed full refits when an estimator lacks partial_fit.
        drift_disclose:
            Enable mean-shift drift disclosure on updates.
        drift_detector:
            Drift detector key (``mean_shift``, ``adwin``, ``page_hinkley``, ``none``).
        buffer_size:
            Replay buffer size for torch continual backends.
        epochs_per_update:
            Training epochs per partial_fit for torch backends.
        batch_size:
            Minibatch size for torch backends.
        learning_rate:
            Optimizer learning rate for torch backends.
        ewc_lambda:
            EWC penalty weight for ``ewc_mlp``.
        hidden_dim:
            MLP hidden width for torch backends.
        device:
            Torch device string (``cpu`` or ``cuda``).
        
        Returns
        -------
        OnlineFitResult
            Serializable fit summary including warnings and init-chunk stats.
            Use :meth:`partial_fit_online` next for incremental updates.
        
        Notes
        -----
        **Leakage:** Requires a split. Init and later updates use train chunks only
        (or role-aligned external frames). Validation/test are never used for
        updates. Classifiers need a ``classes`` vocabulary (explicit or discovered
        from the full train target column — labels only).
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
        """Apply one incremental partial_fit update on the next train chunk or frame.
        
        Delegates to :meth:`buildml.online.update.partial_fit_online`, advances the
        Session online plan cursor, and records the update. Requires a prior call
        to :meth:`fit_online`.
        
        Parameters
        ----------
        n_rows:
            Rows to take from unused train indices; defaults to plan ``chunk_size``.
        indices:
            Optional explicit train-partition dataset indices for this update.
        frame:
            Optional user-provided incremental frame with role-aligned columns.
        
        Returns
        -------
        OnlineUpdateResult
            Serializable update summary including drift notes and refit mode.
            Repeat for more chunks or call :meth:`evaluate_online`.
        
        Raises
        ------
        ValidationError
            When no online plan exists or chunk source preconditions fail.
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
        """Evaluate the online learner on a holdout partition without updating it.
        
        Delegates to :meth:`buildml.online.evaluate.evaluate_online` and stores
        the result on this Session. Holdout partitions are never used for partial_fit
        updates.
        
        Parameters
        ----------
        partition:
            Holdout partition to score (``validation``, ``test``, or ``all``).
            Validation falls back to test when no validation split exists.
        drift_check:
            When True, compare holdout feature means to the training stream.
        
        Returns
        -------
        OnlineEvalResult
            Holdout metrics and optional drift flags. Does not mutate the estimator.
        
        Raises
        ------
        ValidationError
            When no online plan exists on this Session.
        """
        return online_ops.evaluate_online_op(
            self, partition=partition, drift_check=drift_check
        )

    def predict_online(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
    ) -> OnlinePredictResult:
        """Predict with the incremental online estimator without updating it.
        
        Delegates to :meth:`buildml.online.predict.predict_online` and stores
        predictions on this Session. Use after :meth:`fit_online` and optional
        :meth:`partial_fit_online` calls.
        
        Parameters
        ----------
        partition:
            Partition to score (``train``, ``validation``, ``test``, or ``all``).
        
        Returns
        -------
        OnlinePredictResult
            Predictions and optional probabilities for the requested partition.
        
        Raises
        ------
        ValidationError
            When no online plan exists on this Session.
        """
        return online_ops.predict_online_op(self, partition=partition)

    @property
    def online_plan(self) -> OnlinePlan | None:
        """Return the last online-learning plan, if any.
        
        Stored on this Session after :meth:`fit_online` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        OnlinePlan or None
            ``None`` before the first :meth:`fit_online` call on this session.
        """
        return self._online_plan

    @property
    def online_fit_result(self) -> OnlineFitResult | None:
        """Return the last online fit result, if any.
        
        Stored on this Session after :meth:`fit_online` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        OnlineFitResult or None
            ``None`` before the first :meth:`fit_online` call on this session.
        """
        return self._online_fit_result

    @property
    def online_update_result(self) -> OnlineUpdateResult | None:
        """Return the last online update result, if any.
        
        Stored on this Session after :meth:`partial_fit_online` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        OnlineUpdateResult or None
            ``None`` before the first :meth:`partial_fit_online` call on this session.
        """
        return self._online_update_result

    @property
    def online_eval_result(self) -> OnlineEvalResult | None:
        """Return the last online evaluation result, if any.
        
        Stored on this Session after :meth:`evaluate_online` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        OnlineEvalResult or None
            ``None`` before the first :meth:`evaluate_online` call on this session.
        """
        return self._online_eval_result

    @property
    def online_predict_result(self) -> OnlinePredictResult | None:
        """Return the last online prediction result, if any.
        
        Stored on this Session after :meth:`predict_online` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        OnlinePredictResult or None
            ``None`` before the first :meth:`predict_online` call on this session.
        """
        return self._online_predict_result

    def save_online_bundle(self, path: str | Path) -> Path:
        """Persist the active online plan as ``buildml.online_bundle.v1``.
        
        Delegates to :meth:`buildml.online.checkpoint.save_online_bundle`.
        Distinct from Session checkpoints — reload the learner with
        :meth:`load_online_bundle`.
        
        Parameters
        ----------
        path:
            Destination directory for the bundle (created if missing).
        
        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.
        
        Raises
        ------
        ValidationError
            When no online plan exists on this Session.
        """
        return online_ops.save_online_bundle_op(self, path=path)

    def load_online_bundle(self, path: str | Path) -> Session:
        """Load an online-learning bundle into this Session.
        
        Delegates to :meth:`buildml.online.checkpoint.load_online_bundle`,
        replaces Session online state, and clears prior fit/eval/predict results.
        
        Parameters
        ----------
        path:
            Path to a ``buildml.online_bundle.v1`` directory.
        
        Returns
        -------
        Session
            this Session with online plan attached for chaining.
        """
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
        """Fit a multi-target estimator on the train partition only.
        
        Delegates to :meth:`buildml.multitask.fit.fit_multitask`, stores the
        :class:`~buildml.multitask.results.MultiTaskPlan` on this Session, and records
        the fit. Follow with :meth:`predict_multitask` or
        :meth:`evaluate_multitask`.
        
        Parameters
        ----------
        backend:
            Optional backend override (``sklearn``, ``industry``, ``torch``).
        method:
            Multi-task strategy (``multi_output``, ``chain``, ``torch_multihead``).
        task:
            Task mix (``auto``, ``classification``, ``regression``, ``mixed``).
        targets:
            Optional explicit target column names (roles or list).
        columns:
            Optional explicit feature columns.
        base_estimator:
            Base estimator key for sklearn/industry backends.
        random_state:
            Seed for stochastic steps.
        order:
            Optional target column order for chained strategies.
        prefer_reduce_components:
            Prefer reduced component columns when a reduce plan exists on this Session.
        prediction_prefix:
            Prefix for attached prediction column names.
        epochs:
            Training epochs for torch multi-head backend.
        batch_size:
            Minibatch size for torch backend.
        learning_rate:
            Optimizer learning rate for torch backend.
        device:
            Torch device string (``cpu`` or ``cuda``).
        
        Returns
        -------
        MultiTaskFitResult
            Serializable fit summary per target and backend disclosures.
        
        Notes
        -----
        **Leakage:** Requires a split. Fit uses train only. Validation/test are
        never used for fitting. Needs ``>= 2`` target columns (roles or
        ``targets=``). Sklearn/industry require same-type tasks; torch supports
        mixed cls+reg. Classical ``Session.fit`` remains single-target.
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
        """Predict all targets with the frozen multi-task plan without refitting.
        
        Delegates to :meth:`buildml.multitask.predict.predict_multitask`. When
        ``attach=True``, prediction columns are merged into Session dataset.
        
        Parameters
        ----------
        partition:
            Partition to score (``train``, ``validation``, ``test``, or ``all``).
        attach:
            When True, attach prediction columns to this Session dataset frame.
        prediction_prefix:
            Optional override for attached column name prefix.
        
        Returns
        -------
        MultiTaskPredictResult
            Per-target predictions and optional attached column metadata.
        
        Raises
        ------
        ValidationError
            When no multi-task plan exists on this Session.
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
        """Evaluate the multi-task plan on a holdout partition without refitting.
        
        Delegates to :meth:`buildml.multitask.evaluate.evaluate_multitask`.
        Holdout partitions are never used during fit.
        
        Parameters
        ----------
        partition:
            Holdout partition to score. Validation falls back to test when absent.
        
        Returns
        -------
        MultiTaskEvalResult
            Per-target and aggregated holdout metrics.
        
        Raises
        ------
        ValidationError
            When no multi-task plan exists on this Session.
        """
        return multitask_ops.evaluate_multitask_op(self, partition=partition)

    @property
    def multitask_plan(self) -> MultiTaskPlan | None:
        """Return the last multi-task plan, if any.
        
        Stored on this Session after :meth:`fit_multitask` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        MultiTaskPlan or None
            ``None`` before the first :meth:`fit_multitask` call on this session.
        """
        return self._multitask_plan

    @property
    def multitask_fit_result(self) -> MultiTaskFitResult | None:
        """Return the last multi-task fit result, if any.
        
        Stored on this Session after :meth:`fit_multitask` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        MultiTaskFitResult or None
            ``None`` before the first :meth:`fit_multitask` call on this session.
        """
        return self._multitask_fit_result

    @property
    def multitask_predict_result(self) -> MultiTaskPredictResult | None:
        """Return the last multi-task prediction result, if any.
        
        Stored on this Session after :meth:`predict_multitask` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        MultiTaskPredictResult or None
            ``None`` before the first :meth:`predict_multitask` call on this session.
        """
        return self._multitask_predict_result

    @property
    def multitask_eval_result(self) -> MultiTaskEvalResult | None:
        """Return the last multi-task evaluation result, if any.
        
        Stored on this Session after :meth:`evaluate_multitask` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        MultiTaskEvalResult or None
            ``None`` before the first :meth:`evaluate_multitask` call on this session.
        """
        return self._multitask_eval_result

    def save_multitask_bundle(self, path: str | Path) -> Path:
        """Persist the active multi-task plan as ``buildml.multitask_bundle.v1``.
        
        Delegates to :meth:`buildml.multitask.checkpoint.save_multitask_bundle`.
        Reload with :meth:`load_multitask_bundle`.
        
        Parameters
        ----------
        path:
            Destination directory for the bundle (created if missing).
        
        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.
        
        Raises
        ------
        ValidationError
            When no multi-task plan exists on this Session.
        """
        return multitask_ops.save_multitask_bundle_op(self, path=path)

    def load_multitask_bundle(self, path: str | Path) -> Session:
        """Load a multi-task bundle into this Session.
        
        Delegates to :meth:`buildml.multitask.checkpoint.load_multitask_bundle`
        and clears prior fit/eval/predict results.
        
        Parameters
        ----------
        path:
            Path to a ``buildml.multitask_bundle.v1`` directory.
        
        Returns
        -------
        Session
            this Session with multi-task plan attached for chaining.
        """
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
        """Meta-train on episodic tasks carved from the train partition only.
        
        Delegates to :meth:`buildml.metalearning.fit.fit_metalearning`, stores the
        :class:`~buildml.metalearning.results.MetaLearningPlan` on this Session, and
        records the fit. Follow with :meth:`adapt_to_task` or
        :meth:`evaluate_metalearning`.
        
        Parameters
        ----------
        backend:
            Optional backend override (``native`` or ``torch``).
        method:
            Meta-learning method (``prototypical``, ``maml``, etc.).
        task_column:
            Column identifying tasks/episodes; inferred from roles when omitted.
        columns:
            Explicit feature columns for episodic sampling.
        n_way:
            Classes per episode; inferred from data when ``None``.
        k_shot:
            Support examples per class in each episode.
        n_query:
            Query examples per class in each episode.
        n_episodes:
            Number of meta-training episodes per epoch.
        base_estimator:
            Fallback sklearn estimator for non-torch backends.
        random_state:
            Seed for episode sampling and initialization.
        prefer_reduce_components:
            Prefer reduced component columns when a reduce plan exists on this Session.
        task_holdout_fraction:
            Fraction of train tasks held out during meta-training.
        meta_epochs:
            Number of meta-training epochs (torch backends).
        inner_lr:
            Inner-loop learning rate for MAML-style methods.
        inner_steps:
            Inner-loop gradient steps per episode.
        meta_lr:
            Outer/meta learning rate.
        embed_dim:
            Embedding dimension for torch encoders.
        hidden_dim:
            Hidden layer width for torch encoders.
        device:
            Torch device string (``cpu`` or ``cuda``).
        
        Returns
        -------
        MetaLearningFitResult
            Serializable fit summary including task counts and disclosures.
        
        Notes
        -----
        **Leakage:** Requires a split. Meta-train uses train only. Validation/test
        are never used for meta-training. Needs a task/group column (role or
        ``task_column=``) and exactly one ``role='target'``. Honesty: tabular
        few-shot / episodic protocols — not foundation-model meta-learning.
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
        
        Delegates to :meth:`buildml.metalearning.adapt.adapt_to_task` using the
        plan from :meth:`fit_metalearning`. No meta-training occurs here.
        
        Parameters
        ----------
        task_id:
            Task identifier to adapt to; required when multiple tasks exist.
        partition:
            Partition containing support labels (default ``train``).
        support_frame:
            Optional explicit support DataFrame instead of a partition slice.
        max_support_per_class:
            Cap on support rows sampled per class.
        random_state:
            Seed for support sampling.
        
        Returns
        -------
        MetaLearningAdaptResult
            Adapted predictions and support-set summary for the task.
        
        Raises
        ------
        ValidationError
            When no meta-learning plan exists on this Session.
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
        """Run episodic holdout evaluation without meta-training on holdout.
        
        Delegates to :meth:`buildml.metalearning.evaluate.evaluate_metalearning`.
        Falls back to ``test`` when no validation partition exists.
        
        Parameters
        ----------
        partition:
            Holdout partition for episodic evaluation (default ``validation``).
        k_shot:
            Support examples per class override for evaluation episodes.
        n_query:
            Query examples per class override for evaluation episodes.
        n_way:
            Classes per episode override.
        prefer_novel_tasks:
            When True, prefer tasks not seen during meta-training.
        random_state:
            Seed for episode construction.
        
        Returns
        -------
        MetaLearningEvalResult
            Episodic accuracy metrics on the holdout partition.
        
        Raises
        ------
        ValidationError
            When no meta-learning plan exists on this Session.
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
        """Return the last meta-learning plan, if any.
        
        Stored on this Session after :meth:`fit_metalearning` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        MetaLearningPlan or None
            ``None`` before the first :meth:`fit_metalearning` call on this session.
        """
        return self._metalearning_plan

    @property
    def metalearning_fit_result(self) -> MetaLearningFitResult | None:
        """Return the last meta-learning fit result, if any.
        
        Stored on this Session after :meth:`fit_metalearning` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        MetaLearningFitResult or None
            ``None`` before the first :meth:`fit_metalearning` call on this session.
        """
        return self._metalearning_fit_result

    @property
    def metalearning_adapt_result(self) -> MetaAdaptResult | None:
        """Return the last meta-learning adaptation result, if any.
        
        Stored on this Session after :meth:`adapt_to_task` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        MetaAdaptResult or None
            ``None`` before the first :meth:`adapt_to_task` call on this session.
        """
        return self._metalearning_adapt_result

    @property
    def metalearning_eval_result(self) -> MetaLearningEvalResult | None:
        """Return the last meta-learning evaluation result, if any.
        
        Stored on this Session after :meth:`evaluate_metalearning` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        MetaLearningEvalResult or None
            ``None`` before the first :meth:`evaluate_metalearning` call on this session.
        """
        return self._metalearning_eval_result

    def save_metalearning_bundle(self, path: str | Path) -> Path:
        """Persist the active MetaLearningPlan as ``buildml.metalearning_bundle.v1``.
        
        Delegates to :meth:`buildml.metalearning.checkpoint.save_metalearning_bundle`.
        Reload with :meth:`load_metalearning_bundle`.
        
        Parameters
        ----------
        path:
            Destination directory for the bundle (created if missing).
        
        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.
        
        Raises
        ------
        ValidationError
            When no meta-learning plan exists on this Session.
        """
        return metalearning_ops.save_metalearning_bundle_op(self, path=path)

    def load_metalearning_bundle(self, path: str | Path) -> Session:
        """Load a meta-learning bundle into this Session.
        
        Delegates to :meth:`buildml.metalearning.checkpoint.load_metalearning_bundle`
        and clears prior adapt/eval results.
        
        Parameters
        ----------
        path:
            Path to a ``buildml.metalearning_bundle.v1`` directory.
        
        Returns
        -------
        Session
            this Session with MetaLearningPlan attached for chaining.
        """
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
        """Simulate federated averaging on this Session train clients.
        
        Delegates to :meth:`buildml.federated.fit.fit_federated`, stores the global
        :class:`~buildml.federated.results.FederatedPlan` on this Session, and records
        the fit. Follow with :meth:`evaluate_federated` or
        :meth:`predict_federated` on holdout partitions.
        
        Parameters
        ----------
        backend:
            Optional backend override (``native`` or ``flower``).
        method:
            Federated aggregation method (``fedavg`` or ``fedprox``).
        estimator:
            Sklearn linear/SGD estimator key for local and global models.
        task:
            Optional task override; inferred from ``estimator`` when ``None``.
        client_column:
            Optional explicit client/group column.
        columns:
            Optional explicit feature columns.
        n_rounds:
            Number of federated communication rounds.
        local_epochs:
            Local training epochs per selected client per round.
        client_fraction:
            Fraction of eligible clients sampled each round.
        mu:
            FedProx proximal strength (required when ``method='fedprox'``).
        random_state:
            Seed for client sampling and estimator initialization.
        prefer_reduce_components:
            Prefer reduced component columns when a reduce plan exists on this Session.
        min_client_rows:
            Minimum train rows required for a client to participate.
        
        Returns
        -------
        FederatedFitResult
            Serializable fit summary including rounds, clients, and disclosures.
            Use :meth:`evaluate_federated` or :meth:`predict_federated` next.
        
        Notes
        -----
        **Leakage:** Requires a split. Local client updates use train only.
        Validation/test are never used for training. Needs a client/group column
        (role or ``client_column=``) and exactly one ``role='target'``. Honesty:
        local FedAvg-style simulation — ``backend='flower'`` uses Flower libraries
        but still runs in-process unless you deploy Flower separately; not
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
        
        Delegates to :meth:`buildml.federated.evaluate.evaluate_federated`.
        Holdout data is never used for federated training rounds.
        
        Parameters
        ----------
        backend:
            Optional backend override for evaluation adapters.
        partition:
            Holdout partition to score. Validation falls back to test when absent.
        per_client:
            When True, include per-client holdout metrics in the result.
        
        Returns
        -------
        FederatedEvalResult
            Global and optional per-client holdout metrics.
        
        Raises
        ------
        ValidationError
            When no federated plan exists on this Session.
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
        """Predict with the global federated model without local updates.
        
        Delegates to :meth:`buildml.federated.predict.predict_federated` and
        stores predictions on this Session.
        
        Parameters
        ----------
        backend:
            Optional backend override for prediction adapters.
        partition:
            Partition to score (``train``, ``validation``, ``test``, or ``all``).
        
        Returns
        -------
        FederatedPredictResult
            Predictions from the aggregated global model.
        
        Raises
        ------
        ValidationError
            When no federated plan exists on this Session.
        """
        return federated_ops.predict_federated_op(
            self,
            backend=backend,
            partition=partition,
        )

    @property
    def federated_plan(self) -> FederatedPlan | None:
        """Return the last federated plan, if any.
        
        Stored on this Session after :meth:`fit_federated` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        FederatedPlan or None
            ``None`` before the first :meth:`fit_federated` call on this session.
        """
        return self._federated_plan

    @property
    def federated_fit_result(self) -> FederatedFitResult | None:
        """Return the last federated fit result, if any.
        
        Stored on this Session after :meth:`fit_federated` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        FederatedFitResult or None
            ``None`` before the first :meth:`fit_federated` call on this session.
        """
        return self._federated_fit_result

    @property
    def federated_eval_result(self) -> FederatedEvalResult | None:
        """Return the last federated evaluation result, if any.
        
        Stored on this Session after :meth:`evaluate_federated` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        FederatedEvalResult or None
            ``None`` before the first :meth:`evaluate_federated` call on this session.
        """
        return self._federated_eval_result

    @property
    def federated_predict_result(self) -> FederatedPredictResult | None:
        """Return the last federated prediction result, if any.
        
        Stored on this Session after :meth:`predict_federated` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        FederatedPredictResult or None
            ``None`` before the first :meth:`predict_federated` call on this session.
        """
        return self._federated_predict_result

    def save_federated_bundle(self, path: str | Path) -> Path:
        """Persist the active federated plan as ``buildml.federated_bundle.v1``.
        
        Delegates to :meth:`buildml.federated.checkpoint.save_federated_bundle`.
        Reload with :meth:`load_federated_bundle`.
        
        Parameters
        ----------
        path:
            Destination directory for the bundle (created if missing).
        
        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.
        
        Raises
        ------
        ValidationError
            When no federated plan exists on this Session.
        """
        return federated_ops.save_federated_bundle_op(self, path=path)

    def export_round_history(
        self,
        path: str | Path,
        *,
        include_disclosures: bool = False,
    ) -> Path:
        """Export federated round metrics to JSON for audit and teaching overlays.

        Delegates to :func:`buildml.federated.results.export_round_history` using
        the active :class:`~buildml.federated.results.FederatedPlan` on this Session.

        Parameters
        ----------
        path:
            Destination JSON file path (parent directories are created).
        include_disclosures:
            When ``True``, embed plan disclosures and warnings in the payload.

        Returns
        -------
        pathlib.Path
            Resolved output file path.

        Raises
        ------
        ValidationError
            When no federated plan exists on this Session.
        """
        return federated_ops.export_round_history_op(
            self,
            path,
            include_disclosures=include_disclosures,
        )

    def load_federated_bundle(self, path: str | Path) -> Session:
        """Load a federated-learning bundle into this Session.
        
        Delegates to :meth:`buildml.federated.checkpoint.load_federated_bundle`
        and clears prior fit/eval/predict results.
        
        Parameters
        ----------
        path:
            Path to a ``buildml.federated_bundle.v1`` directory.
        
        Returns
        -------
        Session
            this Session with federated plan attached for chaining.
        """
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
        """Fit a Bayesian or probabilistic estimator on this Session train only.
        
        Delegates to :meth:`buildml.probabilistic.fit.fit_probabilistic`, stores
        the :class:`~buildml.probabilistic.results.ProbabilisticPlan` on this Session,
        and records the fit. Follow with :meth:`predict_probabilistic` or
        :meth:`predict_interval`.
        
        Parameters
        ----------
        backend:
            Optional backend override (``native``, ``mapie``, ``ngboost``).
        estimator:
            Probabilistic estimator key (``bayesian_ridge``, etc.).
        task:
            Task override; inferred from target when ``None``.
        columns:
            Explicit feature columns; ``None`` auto-selects numerics.
        random_state:
            Seed for stochastic steps.
        alpha:
            Significance level for intervals (e.g. 0.1 for 90% intervals).
        conformal:
            When True, apply split-conformal calibration on train carve-out.
        conformal_calibration_fraction:
            Fraction of train reserved for conformal calibration.
        interval_method:
            Interval construction method override.
        prefer_reduce_components:
            Prefer reduced component columns when a reduce plan exists on this Session.
        n_restarts_optimizer:
            Restarts for Bayesian ridge optimizer.
        n_estimators:
            Tree count for NGBoost backend.
        learning_rate:
            Learning rate for NGBoost backend.
        
        Returns
        -------
        ProbabilisticFitResult
            Serializable fit summary including backend and conformal disclosures.
        
        Notes
        -----
        **Backends:** ``native`` (sklearn + in-tree conformal), ``mapie`` and
        ``ngboost`` when ``buildml[probabilistic-industry]`` is installed.
        
        **Leakage:** Requires a split. Fit and optional split-conformal calibration
        use train only (conformal carve never touches validation/test). Honesty:
        uncertainty quantification for tabular estimators — not PyMC/Stan MCMC.
        Classical ``Session.calibration()`` is unchanged.
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
        """Evaluate the probabilistic plan on a holdout partition.
        
        Delegates to :meth:`buildml.probabilistic.evaluate.evaluate_probabilistic`
        for calibration and interval coverage metrics. Falls back to ``test`` when
        no validation partition exists.
        
        Parameters
        ----------
        partition:
            Holdout partition (default ``validation``).
        alpha:
            Significance level override for interval metrics.
        
        Returns
        -------
        ProbabilisticEvalResult
            Calibration, coverage, and sharpness metrics on the partition.
        
        Raises
        ------
        ValidationError
            When no probabilistic plan exists on this Session.
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
        """Predict with the probabilistic estimator without updating the plan.
        
        Delegates to :meth:`buildml.probabilistic.predict.predict_probabilistic`
        and optionally returns standard deviations or class probabilities.
        
        Parameters
        ----------
        partition:
            Split partition to predict on (default ``test``).
        return_std:
            When True, include predictive standard deviations (regression).
        return_proba:
            When True, include class probabilities (classification).
        
        Returns
        -------
        ProbabilisticPredictResult
            Point predictions with optional uncertainty outputs.
        
        Raises
        ------
        ValidationError
            When no probabilistic plan exists on this Session.
        """
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
        """Predict predictive intervals or conformal prediction sets on a partition.
        
        Delegates to :meth:`buildml.probabilistic.predict.predict_interval`.
        Regression returns lower/upper bounds; classification returns prediction sets.
        
        Parameters
        ----------
        partition:
            Split partition to score (default ``test``).
        alpha:
            Significance level override for interval width.
        method:
            Interval method override (conformal, native, etc.).
        
        Returns
        -------
        ProbabilisticIntervalResult
            Interval bounds or conformal sets per row.
        
        Raises
        ------
        ValidationError
            When no probabilistic plan exists on this Session.
        """
        return probabilistic_ops.predict_interval_op(
            self,
            partition=partition,
            alpha=alpha,
            method=method,
        )

    @property
    def probabilistic_plan(self) -> ProbabilisticPlan | None:
        """Return the last probabilistic plan, if any.
        
        Stored on this Session after :meth:`fit_probabilistic` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        ProbabilisticPlan or None
            ``None`` before the first :meth:`fit_probabilistic` call on this session.
        """
        return self._probabilistic_plan

    @property
    def probabilistic_fit_result(self) -> ProbabilisticFitResult | None:
        """Return the last probabilistic fit result, if any.
        
        Stored on this Session after :meth:`fit_probabilistic` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        ProbabilisticFitResult or None
            ``None`` before the first :meth:`fit_probabilistic` call on this session.
        """
        return self._probabilistic_fit_result

    @property
    def probabilistic_eval_result(self) -> ProbabilisticEvalResult | None:
        """Return the last probabilistic evaluation result, if any.
        
        Stored on this Session after :meth:`evaluate_probabilistic` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        ProbabilisticEvalResult or None
            ``None`` before the first :meth:`evaluate_probabilistic` call on this session.
        """
        return self._probabilistic_eval_result

    @property
    def probabilistic_predict_result(self) -> ProbabilisticPredictResult | None:
        """Return the last probabilistic prediction result, if any.
        
        Stored on this Session after :meth:`predict_probabilistic` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        ProbabilisticPredictResult or None
            ``None`` before the first :meth:`predict_probabilistic` call on this session.
        """
        return self._probabilistic_predict_result

    @property
    def probabilistic_interval_result(self) -> ProbabilisticIntervalResult | None:
        """Return the last probabilistic interval result, if any.
        
        Stored on this Session after :meth:`predict_interval` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        ProbabilisticIntervalResult or None
            ``None`` before the first :meth:`predict_interval` call on this session.
        """
        return self._probabilistic_interval_result

    def save_probabilistic_bundle(self, path: str | Path) -> Path:
        """Persist the active ProbabilisticPlan as ``buildml.probabilistic_bundle.v1``.
        
        Delegates to :meth:`buildml.probabilistic.checkpoint.save_probabilistic_bundle`.
        Reload with :meth:`load_probabilistic_bundle`.
        
        Parameters
        ----------
        path:
            Destination directory for the bundle (created if missing).
        
        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.
        
        Raises
        ------
        ValidationError
            When no probabilistic plan exists on this Session.
        """
        return probabilistic_ops.save_probabilistic_bundle_op(self, path=path)

    def load_probabilistic_bundle(self, path: str | Path) -> Session:
        """Load a probabilistic bundle into this Session.
        
        Delegates to :meth:`buildml.probabilistic.checkpoint.load_probabilistic_bundle`
        and clears prior eval/predict/interval results.
        
        Parameters
        ----------
        path:
            Path to a ``buildml.probabilistic_bundle.v1`` directory.
        
        Returns
        -------
        Session
            this Session with ProbabilisticPlan attached for chaining.
        """
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
        """Declare identification assumptions required before causal estimation.
        
        Captures treatment, outcome, confounders, and explicit acknowledgements
        of unconfoundedness and positivity. Stores a validated
        :class:`~buildml.causal.types.CausalAssumptions` object on this Session for
        downstream fit and estimate APIs.
        
        Parameters
        ----------
        treatment:
            Column name for the treatment or exposure variable.
        outcome:
            Column name for the outcome of interest.
        confounders:
            Confounder column names; pass ``[]`` only with
            ``allow_empty_confounders=True``.
        estimand:
            Target estimand (``ATE`` by default).
        identification:
            Identification strategy (``backdoor`` by default).
        instruments:
            Optional instrument column names for IV-style paths.
        acknowledge_unconfoundedness:
            Explicit acknowledgement that unconfoundedness holds.
        acknowledge_positivity:
            Explicit acknowledgement that positivity/overlap holds.
        allow_empty_confounders:
            When True, permit an empty confounder list after validation.
        
        Returns
        -------
        CausalAssumptions
            Validated assumptions object stored on this Session.
        
        Raises
        ------
        ValidationError
            When ``confounders`` is ``None`` or validation fails.
        
        Notes
        -----
        EDA / association / feature-importance results never satisfy these
        acknowledgements. Estimation APIs refuse to run without a validated
        :class:`CausalAssumptions` object on this Session (or passed explicitly).
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
        """Fit causal models on this Session train and estimate ATE.
        
        Delegates to :meth:`buildml.causal.fit.fit_causal`, stores the
        :class:`~buildml.causal.results.CausalPlan` on this Session, and records the
        fit. Follow with :meth:`estimate_causal` or :meth:`evaluate_causal`.
        
        Parameters
        ----------
        backend:
            Optional backend override (``native``, ``dowhy``, ``econml``).
        method:
            Estimator method (``aipw`` by default).
        assumptions:
            Optional assumptions override; uses Session-stored assumptions when
            omitted.
        bootstrap_samples:
            Number of bootstrap draws for uncertainty intervals.
        random_state:
            Seed for stochastic nuisance-model steps.
        clip_propensity:
            Min/max propensity clipping bounds for IPW-style methods.
        outcome_model:
            Outcome nuisance model identifier.
        propensity_model:
            Propensity nuisance model identifier.
        
        Returns
        -------
        CausalFitResult
            Serializable fit summary including ATE point estimate and warnings.
        
        Notes
        -----
        **Leakage:** Requires a split. Nuisance models fit on train only.
        **Assumptions:** Requires validated CausalAssumptions — refused otherwise.
        Backends: native (T-learner/IPW/AIPW), dowhy, econml when
        ``buildml[causal-industry]`` is installed. Not causal discovery; EDA
        remains associational.
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
        """Estimate ATE on a partition using the fitted CausalPlan.
        
        Delegates to :meth:`buildml.causal.estimate.estimate_causal` without
        refitting nuisance models. Useful for re-scoring train or scoring
        holdout partitions with bootstrap overrides.
        
        Parameters
        ----------
        partition:
            Partition to score (``train``, ``validation``, ``test``, or ``all``).
        bootstrap_samples:
            Optional bootstrap override; uses plan default when omitted.
        random_state:
            Optional seed override for bootstrap resampling.
        
        Returns
        -------
        CausalEstimateResult
            Partition ATE estimate with optional bootstrap interval.
        
        Raises
        ------
        ValidationError
            When no causal plan exists on this Session.
        """
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
        """Evaluate nuisance predictive quality and ATE on a holdout partition.
        
        Delegates to :meth:`buildml.causal.evaluate.evaluate_causal` to report
        outcome/propensity model quality alongside partition-level ATE checks.
        
        Parameters
        ----------
        partition:
            Holdout partition for evaluation (``validation`` by default).
        bootstrap_samples:
            Optional bootstrap override for partition ATE uncertainty.
        
        Returns
        -------
        CausalEvalResult
            Nuisance metrics and partition ATE evaluation summary.
        
        Raises
        ------
        ValidationError
            When no causal plan exists on this Session.
        """
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
        
        Delegates to :meth:`buildml.causal.refute.refute_causal` to stress-test
        the fitted plan with placebo treatments or random confounders.
        
        Parameters
        ----------
        kind:
            Refutation kind (``placebo_treatment`` by default).
        random_state:
            Seed for stochastic refutation steps.
        
        Returns
        -------
        CausalRefuteResult
            Refutation outcome and sensitivity disclosures.
        
        Raises
        ------
        ValidationError
            When no causal plan exists on this Session.
        """
        return causal_ops.refute_causal_op(
            self,
            kind=kind,
            random_state=random_state,
        )

    @property
    def causal_assumptions(self) -> CausalAssumptions | None:
        """Return the last declared causal assumptions, if any.
        
        Stored on this Session after :meth:`declare_causal_assumptions` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        CausalAssumptions or None
            ``None`` before the first :meth:`declare_causal_assumptions` call on this session.
        """
        return self._causal_assumptions

    @property
    def causal_plan(self) -> CausalPlan | None:
        """Return the last causal plan, if any.
        
        Stored on this Session after :meth:`fit_causal` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        CausalPlan or None
            ``None`` before the first :meth:`fit_causal` call on this session.
        """
        return self._causal_plan

    @property
    def causal_fit_result(self) -> CausalFitResult | None:
        """Return the last causal fit result, if any.
        
        Stored on this Session after :meth:`fit_causal` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        CausalFitResult or None
            ``None`` before the first :meth:`fit_causal` call on this session.
        """
        return self._causal_fit_result

    @property
    def causal_estimate_result(self) -> CausalEstimateResult | None:
        """Return the last causal estimate result, if any.
        
        Stored on this Session after :meth:`estimate_causal` so later calls can reuse
        the same plan without refitting.
        
        Returns
        -------
        CausalEstimateResult or None
            ``None`` before the first :meth:`estimate_causal` call on this session.
        """
        return self._causal_estimate_result

    @property
    def causal_eval_result(self) -> CausalEvalResult | None:
        """
        Return the metrics from the most recent causal evaluation.

        Stored on Session after :meth:`evaluate_causal` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.causal.results.CausalEvalResult` or None
            ``None`` until :meth:`evaluate_causal` has run.
        """
        return self._causal_eval_result

    @property
    def causal_refute_result(self) -> CausalRefuteResult | None:
        """
        Return the refutation from the most recent refute_causal call.

        Stored on Session after :meth:`refute_causal` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.causal.results.CausalRefuteResult` or None
            ``None`` until :meth:`refute_causal` has run.
        """
        return self._causal_refute_result

    def save_causal_bundle(self, path: str | Path) -> Path:
        """
        Persist the active CausalPlan as ``buildml.causal_bundle.v1``.

        Delegates to :func:`buildml.causal.checkpoint.save_causal_bundle`.
        Reload with :meth:`load_causal_bundle`.

        Parameters
        ----------
        path:
            Destination directory for the bundle (created if missing).

        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.

        Raises
        ------
        ValidationError
            When no causal plan exists on the Session.
        """
        return causal_ops.save_causal_bundle_op(self, path=path)

    def load_causal_bundle(self, path: str | Path) -> Session:
        """
        Load a causal bundle into this Session.

        Delegates to :func:`buildml.causal.checkpoint.load_causal_bundle` and
        clears prior estimate/eval/refute results.

        Parameters
        ----------
            Session instance to populate with the loaded causal plan.
        path:
            Path to a ``buildml.causal_bundle.v1`` directory.

        Returns
        -------
        Session
            This Session with causal plan attached for chaining.
        """
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
        """
        Attach an edge list to the Session with dataset rows as nodes.

        Delegates to :func:`buildml.graph.data.build_graph_spec` and validates
        node identifiers against the dataset. Call before :meth:`fit_graph`.

        Parameters
        ----------
        edges:
            Edge list as a DataFrame or sequence of ``(source, target)`` tuples.
        source_col:
            Column name for edge source endpoints.
        target_col:
            Column name for edge target endpoints.
        node_id_col:
            Column uniquely identifying dataset rows as graph nodes.
        directed:
            When True, treat edges as directed.

        Returns
        -------
        GraphSpec
            Validated graph specification stored on Session as ``_graph_spec``.

        Raises
        ------
        ValidationError
            When no dataset is attached or node ids are invalid.

        Notes
        -----
        Dataset rows are nodes. ``node_id_col`` must uniquely identify rows and
        match edge endpoints. Splits created with :meth:`Session.split` are node
        partitions. Call this before :meth:`Session.fit_graph`.
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
        """
        Fit graph node classification on Session train nodes.

        Delegates to :func:`buildml.graph.fit.fit_graph`, stores the
        :class:`~buildml.graph.results.GraphPlan` on Session, and records the fit.
        Follow with :meth:`predict_graph` or :meth:`evaluate_graph`.

        Parameters
        ----------
        method:
            Graph learning method (``classical`` or ``pyg``).
        task:
            Graph task type (currently ``node_classification``).
        mode:
            ``inductive`` (train subgraph) or ``transductive`` (full topology).
        columns:
            Node feature columns; ``None`` auto-selects numerics.
        classical_estimator:
            Sklearn estimator for classical graph method.
        hidden_dim:
            Hidden dimension for GNN layers.
        n_layers:
            Number of message-passing layers.
        epochs:
            Training epochs for GNN backends.
        learning_rate:
            Optimizer learning rate.
        weight_decay:
            L2 regularization for GNN training.
        dropout:
            Dropout rate between GNN layers.
        random_state:
            Seed for weight initialization and sampling.
        include_graph_metrics:
            When True, compute graph-level structural metrics.
        pyg_model:
            PyG architecture (``gcn``, ``graphsage``, ``gat``).
        heads:
            Attention heads for GAT when ``pyg_model='gat'``.

        Returns
        -------
        GraphFitResult
            Serializable fit summary including method and mode disclosures.

        Raises
        ------
        ValidationError
            When no GraphSpec exists on the Session.

        Notes
        -----
        **Leakage:** Requires a split. Labels from train only.
        **Inductive (default):** train-induced subgraph for fit.
        **Transductive:** full topology; train-label-only supervision (disclosed).
        Classical needs ``buildml[graph]``; GCN needs ``buildml[torch]``;
        PyG needs ``buildml[graph-pyg]`` (``pyg_model``: gcn/graphsage/gat).
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
        """
        Predict node labels with the fitted GraphPlan on a partition.

        Delegates to :func:`buildml.graph.predict.predict_graph` without refitting.

        Parameters
        ----------
        partition:
            Node partition to predict on (default ``validation``).

        Returns
        -------
        GraphPredictResult
            Node predictions and optional probabilities for the partition.

        Raises
        ------
        ValidationError
            When no graph plan exists on the Session.
        """
        return graph_ops.predict_graph_op(self, partition=partition)

    def evaluate_graph(
        self,
        *,
        partition: PartitionName | Literal["all"] = "validation",
    ) -> GraphEvalResult:
        """
        Evaluate node classification on a holdout graph partition.

        Delegates to :func:`buildml.graph.evaluate.evaluate_graph` and stores
        metrics on Session.

        Parameters
        ----------
        partition:
            Holdout node partition (default ``validation``).

        Returns
        -------
        GraphEvalResult
            Classification metrics for nodes in the partition.

        Raises
        ------
        ValidationError
            When no graph plan exists on the Session.
        """
        return graph_ops.evaluate_graph_op(self, partition=partition)

    @property
    def graph_spec(self) -> GraphSpec | None:
        """
        Return the graph specification attached by the most recent set_graph call.

        Stored on Session after :meth:`set_graph` or :meth:`load_graph_bundle` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.graph.types.GraphSpec` or None
            ``None`` until :meth:`set_graph` or :meth:`load_graph_bundle` has run.
        """
        return self._graph_spec

    @property
    def graph_plan(self) -> GraphPlan | None:
        """
        Return the graph plan built by the most recent graph fit.

        Stored on Session after :meth:`fit_graph` or :meth:`load_graph_bundle` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.graph.results.GraphPlan` or None
            ``None`` until :meth:`fit_graph` or :meth:`load_graph_bundle` has run.
        """
        return self._graph_plan

    @property
    def graph_fit_result(self) -> GraphFitResult | None:
        """
        Return the report from the most recent graph fit.

        Stored on Session after :meth:`fit_graph` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.graph.results.GraphFitResult` or None
            ``None`` until :meth:`fit_graph` has run.
        """
        return self._graph_fit_result

    @property
    def graph_predict_result(self) -> GraphPredictResult | None:
        """
        Return the node predictions from the most recent graph scoring call.

        Stored on Session after :meth:`predict_graph` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.graph.results.GraphPredictResult` or None
            ``None`` until :meth:`predict_graph` has run.
        """
        return self._graph_predict_result

    @property
    def graph_eval_result(self) -> GraphEvalResult | None:
        """
        Return the metrics from the most recent graph evaluation.

        Stored on Session after :meth:`evaluate_graph` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.graph.results.GraphEvalResult` or None
            ``None`` until :meth:`evaluate_graph` has run.
        """
        return self._graph_eval_result

    def save_graph_bundle(self, path: str | Path) -> Path:
        """
        Persist the active GraphPlan as ``buildml.graph_bundle.v1``.

        Delegates to :func:`buildml.graph.checkpoint.save_graph_bundle`.
        Reload with :meth:`load_graph_bundle`.

        Parameters
        ----------
        path:
            Destination directory for the bundle (created if missing).

        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.

        Raises
        ------
        ValidationError
            When no graph plan exists on the Session.
        """
        return graph_ops.save_graph_bundle_op(self, path=path)

    def load_graph_bundle(self, path: str | Path) -> Session:
        """
        Load a graph bundle into this Session.

        Delegates to :func:`buildml.graph.checkpoint.load_graph_bundle`,
        restores GraphSpec from the plan, and clears prior predict/eval results.

        Parameters
        ----------
            Session instance to populate with the loaded GraphPlan.
        path:
            Path to a ``buildml.graph_bundle.v1`` directory.

        Returns
        -------
        Session
            This Session with GraphPlan and GraphSpec attached for chaining.
        """
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
        """
        Compile or induce a symbolic rule base on Session train.

        Delegates to :func:`buildml.symbolic.fit.fit_symbolic`, stores the
        :class:`~buildml.symbolic.results.SymbolicPlan` on Session, and records
        the fit. Follow with :meth:`predict_symbolic` or
        :meth:`evaluate_symbolic`.

        Parameters
        ----------
        backend:
            Optional symbolic backend override.
        source:
            Rule source (``decision_tree`` induction or declared rules).
        method:
            Optional industry backend method override.
        task:
            Optional task override (classification/regression).
        rules:
            Optional pre-declared rules to compile instead of inducing.
        columns:
            Optional explicit feature column list.
        random_state:
            Seed for stochastic induction steps.
        max_depth:
            Maximum tree depth when inducing from a decision tree.
        min_samples_leaf:
            Minimum leaf size for tree induction.
        max_rules:
            Cap on emitted rules after induction.
        default_consequent:
            Fallback prediction when no rule fires.
        prefer_reduce_components:
            Prefer reduced component columns when a reduce plan exists.
        verify_constraints:
            When True, run optional Z3 constraint verification when available.

        Returns
        -------
        SymbolicFitResult
            Serializable fit summary including rule count and disclosures.

        Notes
        -----
        **Leakage:** Requires a split. Induction / compile statistics use train
        only. Honesty: structured tabular if-then rules — not Prolog/Z3/AGI.
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
        """
        Evaluate the symbolic plan on a holdout partition.

        Delegates to :func:`buildml.symbolic.evaluate.evaluate_symbolic` using the
        frozen train rule base. Falls back to ``test`` when validation is empty.

        Parameters
        ----------
        partition:
            Holdout partition for evaluation (``validation`` by default).

        Returns
        -------
        SymbolicEvalResult
            Holdout metrics and rule-coverage disclosures.

        Raises
        ------
        ValidationError
            When no symbolic plan exists on the Session.
        """
        return symbolic_ops.evaluate_symbolic_op(self, partition=partition)

    def predict_symbolic(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        return_traces: bool = True,
    ) -> SymbolicPredictResult:
        """
        Predict with the symbolic rule base (no update).

        Delegates to :func:`buildml.symbolic.predict.predict_symbolic` without
        modifying the induced or compiled rules.

        Parameters
        ----------
        partition:
            Partition to predict on (``test`` by default).
        return_traces:
            When True, include fired-rule traces per row.

        Returns
        -------
        SymbolicPredictResult
            Predictions and optional rule traces for the partition.

        Raises
        ------
        ValidationError
            When no symbolic plan exists on the Session.
        """
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
        """
        Fit a sklearn + symbolic hybrid on Session train.

        Delegates to :func:`buildml.symbolic.fit.fit_neuro_symbolic`, stores the
        :class:`~buildml.symbolic.results.NeuroSymbolicPlan` on Session, and
        records the fit. Follow with :meth:`predict_neuro_symbolic` or
        :meth:`evaluate_neuro_symbolic`.

        Parameters
        ----------
        backend:
            Optional neuro-symbolic backend override.
        mode:
            Hybrid mode (``constraint_overlay`` by default).
        base_estimator:
            Sklearn base estimator identifier.
        torch_method:
            Optional torch method when backend is torch.
        task:
            Optional task override (classification/regression).
        rules:
            Optional pre-declared rules for the symbolic overlay.
        rule_source:
            Rule induction source when ``rules`` is omitted.
        columns:
            Optional explicit feature column list.
        random_state:
            Seed for stochastic base-estimator and induction steps.
        soft_strength:
            Soft constraint strength for overlay modes.
        max_depth:
            Maximum tree depth for rule induction.
        min_samples_leaf:
            Minimum leaf size for rule induction.
        max_rules:
            Cap on induced rules for the overlay.
        prefer_reduce_components:
            Prefer reduced component columns when a reduce plan exists.
        torch_epochs:
            Training epochs for torch hybrid backend.
        device:
            Torch device string.

        Returns
        -------
        NeuroSymbolicFitResult
            Serializable fit summary including hybrid disclosures.

        Notes
        -----
        **Leakage:** Requires a split. Base estimator fit and any rule induction
        use train only. This is a real Session-integrated hybrid — not a
        disconnected "fit then apply rules" pair without shared state.
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
        """
        Evaluate the neuro-symbolic plan on a holdout partition.

        Delegates to :func:`buildml.symbolic.evaluate.evaluate_neuro_symbolic`
        using the frozen hybrid plan. Falls back to ``test`` when validation is
        empty.

        Parameters
        ----------
            :meth:`fit_neuro_symbolic`.
        partition:
            Holdout partition for evaluation (``validation`` by default).

        Returns
        -------
        SymbolicEvalResult
            Holdout metrics and hybrid overlay disclosures.

        Raises
        ------
        ValidationError
            When no neuro-symbolic plan exists on the Session.
        """
        return symbolic_ops.evaluate_neuro_symbolic_op(self, partition=partition)

    def predict_neuro_symbolic(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        return_traces: bool = True,
    ) -> SymbolicPredictResult:
        """
        Predict with the neuro-symbolic hybrid (no update).

        Delegates to :func:`buildml.symbolic.predict.predict_neuro_symbolic`
        without refitting the base estimator or rules.

        Parameters
        ----------
            :meth:`fit_neuro_symbolic`.
        partition:
            Partition to predict on (``test`` by default).
        return_traces:
            When True, include overlay/rule traces per row.

        Returns
        -------
        NeuroSymbolicPredictResult
            Predictions and optional traces for the partition.

        Raises
        ------
        ValidationError
            When no neuro-symbolic plan exists on the Session.
        """
        return symbolic_ops.predict_neuro_symbolic_op(
            self,
            partition=partition,
            return_traces=return_traces,
        )

    @property
    def symbolic_plan(self) -> SymbolicPlan | None:
        """
        Return the symbolic rule plan built by the most recent symbolic fit.

        Stored on Session after :meth:`fit_symbolic` or :meth:`load_symbolic_bundle` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.symbolic.results.SymbolicPlan` or None
            ``None`` until :meth:`fit_symbolic` or :meth:`load_symbolic_bundle` has run.
        """
        return self._symbolic_plan

    @property
    def neuro_symbolic_plan(self) -> NeuroSymbolicPlan | None:
        """
        Return the neuro-symbolic plan built by the most recent hybrid fit.

        Stored on Session after :meth:`fit_neuro_symbolic` or :meth:`load_symbolic_bundle` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.symbolic.results.NeuroSymbolicPlan` or None
            ``None`` until :meth:`fit_neuro_symbolic` or :meth:`load_symbolic_bundle` has run.
        """
        return self._neuro_symbolic_plan

    @property
    def symbolic_fit_result(self) -> SymbolicFitResult | None:
        """
        Return the report from the most recent pure-symbolic fit.

        Stored on Session after :meth:`fit_symbolic` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.symbolic.results.SymbolicFitResult` or None
            ``None`` until :meth:`fit_symbolic` has run.
        """
        return self._symbolic_fit_result

    @property
    def neuro_symbolic_fit_result(self) -> NeuroSymbolicFitResult | None:
        """
        Return the report from the most recent neuro-symbolic fit.

        Stored on Session after :meth:`fit_neuro_symbolic` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.symbolic.results.NeuroSymbolicFitResult` or None
            ``None`` until :meth:`fit_neuro_symbolic` has run.
        """
        return self._neuro_symbolic_fit_result

    @property
    def symbolic_eval_result(self) -> SymbolicEvalResult | None:
        """
        Return the metrics from the most recent symbolic or neuro-symbolic evaluation.

        Stored on Session after :meth:`evaluate_symbolic` or :meth:`evaluate_neuro_symbolic` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.symbolic.results.SymbolicEvalResult` or None
            ``None`` until :meth:`evaluate_symbolic` or :meth:`evaluate_neuro_symbolic` has run.
        """
        return self._symbolic_eval_result

    @property
    def symbolic_predict_result(self) -> SymbolicPredictResult | None:
        """
        Return the predictions from the most recent pure-symbolic scoring call.

        Stored on Session after :meth:`predict_symbolic` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.symbolic.results.SymbolicPredictResult` or None
            ``None`` until :meth:`predict_symbolic` has run.
        """
        return self._symbolic_predict_result

    @property
    def neuro_symbolic_predict_result(self) -> SymbolicPredictResult | None:
        """
        Return the predictions from the most recent neuro-symbolic scoring call.

        Stored on Session after :meth:`predict_neuro_symbolic` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.symbolic.results.SymbolicPredictResult` or None
            ``None`` until :meth:`predict_neuro_symbolic` has run.
        """
        return self._neuro_symbolic_predict_result

    def save_symbolic_bundle(self, path: str | Path) -> Path:
        """
        Persist the active symbolic or neuro-symbolic plan.

        Delegates to :func:`buildml.symbolic.checkpoint.save_symbolic_bundle`.
        Reload with :meth:`load_symbolic_bundle`.

        Parameters
        ----------
        path:
            Destination directory for the bundle (created if missing).

        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.

        Raises
        ------
        ValidationError
            When no symbolic / neuro-symbolic plan exists on the Session.
        """
        return symbolic_ops.save_symbolic_bundle_op(self, path=path)

    def load_symbolic_bundle(self, path: str | Path) -> Session:
        """
        Load a symbolic bundle into this Session.

        Delegates to :func:`buildml.symbolic.checkpoint.load_symbolic_bundle` and
        clears prior eval/predict results. Restores either a pure symbolic or a
        neuro-symbolic plan based on bundle contents.

        Parameters
        ----------
            Session instance to populate with the loaded plan.
        path:
            Path to a ``buildml.symbolic_bundle.v1`` directory.

        Returns
        -------
        Session
            This Session with symbolic plan attached for chaining.
        """
        return symbolic_ops.load_symbolic_bundle_op(self, path=path)

    @staticmethod
    def symbolic_capability_matrix() -> dict[str, Any]:
        """
        Honest capability matrix for symbolic / neuro-symbolic backends.

        Delegates to :func:`buildml.symbolic.catalog.symbolic_capability_matrix`.
        Use before :meth:`fit_symbolic` or :meth:`fit_neuro_symbolic` to
        confirm available backends, sources, and methods for the current install.

        Returns
        -------
        dict
            Nested map of backend identifiers to supported sources and methods.
        """
        return symbolic_ops.symbolic_capability_matrix_op()

    @staticmethod
    def cbr_capability_matrix() -> dict[str, Any]:
        """
        Report which case-based retrieval backends are available on this machine.

        Call before :meth:`fit_cbr` when choosing among sklearn kNN, ANN industry
        extras, text embeddings, or torch metric encoders. Read-only introspection.

        Returns
        -------
        dict[str, Any]
            Retrieval backends, metrics, and install hints from
            :func:`buildml.cbr.catalog.cbr_capability_matrix`.
        """
        from buildml.cbr.catalog import cbr_capability_matrix

        return cbr_capability_matrix()

    @staticmethod
    def ranking_capability_matrix() -> dict[str, Any]:
        """
        Report which learning-to-rank backends and objectives are available here.

        Call before :meth:`fit_ranker` to confirm LightGBM/XGBoost/CatBoost or
        sklearn fallbacks before writing a fit call that will fail on this install.
        Read-only — no dataset required.

        Returns
        -------
        dict[str, Any]
            Ranker backends, supported objectives, and install hints from
            :func:`buildml.ranking.catalog.ranking_capability_matrix`.
        """
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
        """
        Build a case base from Session train.

        Delegates to :func:`buildml.cbr.fit.fit_cbr`, stores the
        :class:`~buildml.cbr.results.CbrPlan` on Session, and records the fit.
        Follow with :meth:`retrieve_cases` or :meth:`predict_cbr`.

        Parameters
        ----------
        backend:
            Optional backend override (see CBR capability matrix).
        task:
            Optional task override (classification/regression).
        metric:
            Case distance metric (``euclidean`` by default).
        reuse:
            Reuse mode for combining retrieved cases.
        adapt:
            Adaptation mode applied after retrieval.
        k:
            Default number of neighbors to retrieve.
        columns:
            Optional explicit feature column list.
        categorical_columns:
            Optional categorical feature columns for mixed distances.
        text_columns:
            Optional text columns for embedding-based retrieval.
        text_model_name:
            Sentence-transformer model for text columns.
        standardize:
            When True, standardize numeric features on train.
        distance_eps:
            Epsilon added to distances for numerical stability.
        random_state:
            Seed for stochastic steps.
        prefer_reduce_components:
            Prefer reduced component columns when a reduce plan exists.
        torch_epochs:
            Training epochs for torch embedding backend.
        device:
            Torch device string for embedding backend.

        Returns
        -------
        CbrFitResult
            Serializable fit summary including case-base size and disclosures.

        Notes
        -----
        **Leakage:** Requires a split. Case memory uses train only. Honesty:
        tabular case→solution CBR — not RAG document retrieval.
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
        """
        Retrieve k nearest cases for a partition (no reuse).

        Delegates to :func:`buildml.cbr.retrieve.retrieve_cases` for inspection
        without applying a reuse/adapt policy.

        Parameters
        ----------
        partition:
            Partition to retrieve against (``test`` by default).
        k:
            Optional neighbor override; uses plan default when omitted.
        backend:
            Optional backend override for retrieval.

        Returns
        -------
        CbrRetrieveResult
            Retrieved cases and distance traces for each query row.

        Raises
        ------
        ValidationError
            When no CBR plan exists on the Session.
        """
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
        """
        Predict via retrieve + reuse (no case-base update).

        Delegates to :func:`buildml.cbr.predict.predict_cbr` using the fitted
        reuse/adapt policy without modifying the case base.

        Parameters
        ----------
        partition:
            Partition to predict on (``test`` by default).
        k:
            Optional neighbor override; uses plan default when omitted.
        return_traces:
            When True, include retrieval/reuse traces in the result.
        backend:
            Optional backend override for prediction.

        Returns
        -------
        CbrPredictResult
            Predictions and optional retrieval traces for the partition.

        Raises
        ------
        ValidationError
            When no CBR plan exists on the Session.
        """
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
        """
        Evaluate CBR on a holdout partition.

        Delegates to :func:`buildml.cbr.evaluate.evaluate_cbr` using the frozen
        train case base. Falls back to ``test`` when validation is empty.

        Parameters
        ----------
        partition:
            Holdout partition for evaluation (``validation`` by default).
        k:
            Optional neighbor override; uses plan default when omitted.

        Returns
        -------
        CbrEvalResult
            Holdout metrics and retrieval disclosures.

        Raises
        ------
        ValidationError
            When no CBR plan exists on the Session.
        """
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
        """
        Retain new labeled cases (refuses Session validation/test indices).

        Delegates to :func:`buildml.cbr.retain.retain_cbr` or
        :func:`buildml.cbr.retain.retain_from_indices` to grow the case base
        with explicit source disclosure.

        Parameters
        ----------
        labeled_frame:
            Optional frame of new labeled rows to retain.
        row_indices:
            Optional dataset row indices to retain (mutually exclusive with
            ``labeled_frame``).
        solution_column:
            Solution column when ``labeled_frame`` is supplied.
        source_disclosure:
            Required provenance string for retained cases.
        allow_overlap_with_train:
            When True, permit overlap between retained rows and train indices.

        Returns
        -------
        CbrRetainResult
            Retain summary including updated case-base size.

        Raises
        ------
        ValidationError
            When no CBR plan exists or retain inputs are invalid.
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
        """
        Return the case memory built by the most recent CBR fit.

        Stored on Session after :meth:`fit_cbr` or :meth:`load_cbr_bundle` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.cbr.results.CbrPlan` or None
            ``None`` until :meth:`fit_cbr` or :meth:`load_cbr_bundle` has run.
        """
        return self._cbr_plan

    @property
    def cbr_fit_result(self) -> CbrFitResult | None:
        """
        Return the report from the most recent CBR fit.

        Stored on Session after :meth:`fit_cbr` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.cbr.results.CbrFitResult` or None
            ``None`` until :meth:`fit_cbr` has run.
        """
        return self._cbr_fit_result

    @property
    def cbr_eval_result(self) -> CbrEvalResult | None:
        """
        Return the metrics from the most recent CBR evaluation.

        Stored on Session after :meth:`evaluate_cbr` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.cbr.results.CbrEvalResult` or None
            ``None`` until :meth:`evaluate_cbr` has run.
        """
        return self._cbr_eval_result

    @property
    def cbr_predict_result(self) -> CbrPredictResult | None:
        """
        Return the predictions from the most recent CBR scoring call.

        Stored on Session after :meth:`predict_cbr` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.cbr.results.CbrPredictResult` or None
            ``None`` until :meth:`predict_cbr` has run.
        """
        return self._cbr_predict_result

    @property
    def cbr_retrieve_result(self) -> CbrRetrieveResult | None:
        """
        Return the nearest cases from the most recent retrieval call.

        Stored on Session after :meth:`retrieve_cases` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.cbr.results.CbrRetrieveResult` or None
            ``None`` until :meth:`retrieve_cases` has run.
        """
        return self._cbr_retrieve_result

    @property
    def cbr_retain_result(self) -> CbrRetainResult | None:
        """
        Return the report from the most recent case retention call.

        Stored on Session after :meth:`retain_cbr` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.cbr.results.CbrRetainResult` or None
            ``None`` until :meth:`retain_cbr` has run.
        """
        return self._cbr_retain_result

    def save_cbr_bundle(self, path: str | Path) -> Path:
        """
        Persist the active CbrPlan.

        Delegates to :func:`buildml.cbr.checkpoint.save_cbr_bundle`.
        Reload with :meth:`load_cbr_bundle`.

        Parameters
        ----------
        path:
            Destination directory for the bundle (created if missing).

        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.

        Raises
        ------
        ValidationError
            When no CBR plan exists on the Session.
        """
        return cbr_ops.save_cbr_bundle_op(self, path=path)

    def load_cbr_bundle(self, path: str | Path) -> Session:
        """
        Load a CBR bundle into this Session.

        Delegates to :func:`buildml.cbr.checkpoint.load_cbr_bundle` and clears
        prior eval/predict/retrieve/retain results.

        Parameters
        ----------
            Session instance to populate with the loaded CBR plan.
        path:
            Path to a ``buildml.cbr_bundle.v1`` directory.

        Returns
        -------
        Session
            This Session with CBR plan attached for chaining.
        """
        return cbr_ops.load_cbr_bundle_op(self, path=path)

    @staticmethod
    def nlp_capability_matrix() -> dict[str, Any]:
        """
        Honest capability matrix for NLP backends and task surfaces.

        Thin Session facade over ``buildml.nlp``; records the call and stores NLP artifacts on the Session for follow-up steps.

        Returns
        -------
        dict[str, Any]
            Structured domain result recorded on the Session for follow-up evaluate/explain/export steps.
        """
        return nlp_ops.nlp_capability_matrix_op()

    def profile_text_corpus(
        self,
        *,
        text_column: str | None = None,
        top_tokens: int = 25,
        near_duplicate_threshold: float = 0.9,
        detect_languages: bool = True,
        stopword_language: str | None = None,
    ) -> NlpCorpusProfile:
        """
        Profile corpus health and screen the split for text contamination.

        Thin Session facade over ``buildml.nlp``; records the call and stores NLP artifacts on the Session for follow-up steps.

        Parameters
        ----------
        text_column:
            Text column name; defaults to the sole text-role column.
        top_tokens:
            How many highest-frequency tokens to surface in the corpus profile. Raise it for a broader vocabulary snapshot; lower it for a short health check.
        near_duplicate_threshold:
            Similarity cutoff above which two documents are flagged as near-duplicates. Closer to 1.0 is stricter; lower values catch paraphrases but add noise.
        detect_languages:
            When True, run language identification during profiling and report per-language counts. Turn off for monolingual corpora to skip that cost.
        stopword_language:
            Language key for the stopword list used in token stats (for example ``"english"``). ``None`` skips stopword filtering.

        Returns
        -------
        NlpCorpusProfile
            Structured domain result recorded on the Session for follow-up evaluate/explain/export steps.
        """
        return nlp_ops.profile_text_corpus_op(
            self,
            text_column=text_column,
            top_tokens=top_tokens,
            near_duplicate_threshold=near_duplicate_threshold,
            detect_languages=detect_languages,
            stopword_language=stopword_language,
        )

    def detect_language(
        self,
        *,
        partition: PartitionName | Literal["all"] = "all",
        backend: str | None = "native",
        text_column: str | None = None,
        min_characters: int = 12,
    ) -> NlpLanguageResult:
        """
        Identify the language of every document in a partition.

        Thin Session facade over ``buildml.nlp``; records the call and stores NLP artifacts on the Session for follow-up steps.

        Parameters
        ----------
        partition:
            Split partition to read or score.
        backend:
            Backend identifier; see capability matrix for valid values.
        text_column:
            Text column name; defaults to the sole text-role column.
        min_characters:
            Minimum character length before a document is language-classified. Short strings are skipped because language ID is unreliable on tiny text.

        Returns
        -------
        NlpLanguageResult
            Structured domain result recorded on the Session for follow-up evaluate/explain/export steps.
        """
        return nlp_ops.detect_language_op(
            self,
            partition=partition,
            backend=backend,
            text_column=text_column,
            min_characters=min_characters,
        )

    def fit_text_classifier(
        self,
        *,
        backend: str | None = None,
        estimator: str | None = None,
        text_column: str | None = None,
        vectorizer: str = "tfidf",
        analyzer: str = "word",
        ngram_range: tuple[int, int] = (1, 2),
        max_features: int | None = 20000,
        min_df: int | float = 1,
        max_df: int | float = 1.0,
        sublinear_tf: bool = True,
        binary: bool = False,
        n_hash_features: int = 2**18,
        normalize_steps: list[str] | None = None,
        stopwords: list[str] | None = None,
        stopword_language: str | None = None,
        min_token_length: int = 1,
        max_token_length: int = 40,
        stem: bool = False,
        lemmatize: bool = False,
        class_weight: str | None = None,
        C: float = 1.0,
        alpha: float = 1.0,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_seq_tokens: int = 256,
        device: str = "cpu",
        random_state: int | None = 0,
    ) -> NlpFitResult:
        """
        Fit a single-label document classifier on Session train.

        Thin Session facade over ``buildml.nlp``; records the call and stores NLP artifacts on the Session for follow-up steps.

        Parameters
        ----------
        backend:
            Backend identifier; see capability matrix for valid values.
        estimator:
            Unfitted sklearn-compatible estimator instance.
        text_column:
            Text column name; defaults to the sole text-role column.
        vectorizer:
            Bag vectorizer family for bag-of-n-grams paths (for example ``"tfidf"`` or ``"count"``). TF-IDF down-weights corpus-wide tokens; count keeps raw frequencies.
        analyzer:
            Tokenization unit for the vectorizer: ``"word"`` (default) or ``"char"`` / ``"char_wb"`` for character n-grams when morphology or misspellings matter.
        ngram_range:
            Inclusive ``(min_n, max_n)`` n-gram window. ``(1, 1)`` is unigrams only; ``(1, 2)`` adds bigrams and usually helps short-text classification at higher sparsity cost.
        max_features:
            Cap on vocabulary size after frequency ranking. ``None`` keeps all terms above ``min_df``; a finite cap controls memory and noise from rare tokens.
        min_df:
            Minimum document frequency for a term to enter the vocabulary (integer count or fraction of documents). Higher values drop rare/noisy tokens.
        max_df:
            Maximum document frequency for a term (count or fraction). Lower it to strip overly common tokens that behave like stopwords.
        sublinear_tf:
            When True (TF-IDF), replace raw term frequency with ``1 + log(tf)`` to dampen very frequent tokens.
        binary:
            When True, term values become 0/1 presence indicators instead of counts or TF-IDF weights.
        n_hash_features:
            Number of feature-hashing buckets for high-cardinality text/categorical hashing encoders.
        normalize_steps:
            Ordered text-normalization steps to apply before vectorization (lowercase, unicode cleanup, whitespace collapse, …).
        stopwords:
            Stopword list name, custom iterable, or ``None``. Removing common function words often helps bag-of-words models.
        stopword_language:
            Language key for the stopword list used in token stats (for example ``"english"``). ``None`` skips stopword filtering.
        min_token_length:
            Minimum token length retained after tokenization. Shorter tokens are discarded as noise.
        max_token_length:
            Maximum token length retained after tokenization. Longer tokens are discarded or truncated.
        stem:
            When True, apply stemming after tokenization. Useful for bag-of-words; usually off for embeddings.
        lemmatize:
            When True, lemmatize tokens (requires the NLP morphology extra/backend). Usually preferable to crude stemming when available.
        class_weight:
            Class reweighting strategy (``"balanced"``, a per-class dict, or ``None``). Use ``"balanced"`` when minority classes are underrepresented in train.
        C:
            Inverse regularization strength for linear / SVM-style models. Larger ``C`` fits training data more tightly; smaller ``C`` prefers simpler boundaries.
        alpha:
            Regularization strength for penalized linear models (larger = stronger penalty).
        embedding_model_name:
            Sentence-transformer / embedding model id used to encode documents.
        max_seq_tokens:
            Maximum tokens per document for transformer encode/fit loops. Longer inputs are truncated.
        device:
            Compute device string (``"cpu"``, ``"cuda"``, ``"mps"``, …) for torch-backed paths.
        random_state:
            Seed for randomized fitting steps so re-runs are comparable. ``None`` leaves RNG undeterministic.

        Returns
        -------
        NlpFitResult
            Structured domain result recorded on the Session for follow-up evaluate/explain/export steps.

        Notes
        -----
        **Leakage:** Requires a split. Normalization, vocabulary, document
        frequencies, and the head are all fitted on train only. Honesty: document
        classification - not sequence labelling, not generation, not RAG.
        """
        return nlp_ops.fit_text_classifier_op(
            self,
            backend=backend,
            estimator=estimator,
            text_column=text_column,
            vectorizer=vectorizer,
            analyzer=analyzer,
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            binary=binary,
            n_hash_features=n_hash_features,
            normalize_steps=normalize_steps,
            stopwords=stopwords,
            stopword_language=stopword_language,
            min_token_length=min_token_length,
            max_token_length=max_token_length,
            stem=stem,
            lemmatize=lemmatize,
            class_weight=class_weight,
            C=C,
            alpha=alpha,
            embedding_model_name=embedding_model_name,
            max_seq_tokens=max_seq_tokens,
            device=device,
            random_state=random_state,
        )

    def predict_text(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        return_probabilities: bool = True,
    ) -> NlpPredictResult:
        """
        Score a partition with the train-fitted text plan.

        Thin Session facade over ``buildml.nlp``; records the call and stores NLP artifacts on the Session for follow-up steps.

        Parameters
        ----------
        partition:
            Split partition to read or score.
        return_probabilities:
            When True, include class probabilities in the prediction/result payload.

        Returns
        -------
        NlpPredictResult
            Structured domain result recorded on the Session for follow-up evaluate/explain/export steps.
        """
        return nlp_ops.predict_text_op(
            self,
            partition=partition,
            return_probabilities=return_probabilities,
        )

    def evaluate_text_classifier(
        self,
        *,
        partition: PartitionName | Literal["all"] = "validation",
    ) -> NlpEvalResult:
        """
        Evaluate the text classifier on a holdout partition.

        Thin Session facade over ``buildml.nlp``; records the call and stores NLP artifacts on the Session for follow-up steps.

        Parameters
        ----------
        partition:
            Split partition to read or score.

        Returns
        -------
        NlpEvalResult
            Structured domain result recorded on the Session for follow-up evaluate/explain/export steps.
        """
        return nlp_ops.evaluate_text_classifier_op(self, partition=partition)

    def interpret_text_prediction(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        target_class: Any = None,
        top_k: int = 12,
        max_documents: int = 10,
        include_global: bool = True,
    ) -> NlpInterpretResult:
        """
        Explain document decisions with per-token contributions.

        Thin Session facade over ``buildml.nlp``; records the call and stores NLP artifacts on the Session for follow-up steps.

        Parameters
        ----------
        partition:
            Split partition to read or score.
        target_class:
            Class label treated as the positive / focus class for binary metrics or explanations.
        top_k:
            How many retrieved chunks / candidates to keep. Higher recall costs more context and noise.
        max_documents:
            Maximum documents to profile or process in one call. Lower it for a cheap health check on huge corpora.
        include_global:
            When True, attach global explanation summaries in addition to local ones.

        Returns
        -------
        NlpInterpretResult
            Structured domain result recorded on the Session for follow-up evaluate/explain/export steps.
        """
        return nlp_ops.interpret_text_prediction_op(
            self,
            partition=partition,
            target_class=target_class,
            top_k=top_k,
            max_documents=max_documents,
            include_global=include_global,
        )

    def fit_topics(
        self,
        *,
        method: str = "nmf",
        n_topics: int = 6,
        text_column: str | None = None,
        top_terms: int = 10,
        max_features: int | None = 20000,
        min_df: int | float = 2,
        max_df: int | float = 0.95,
        ngram_range: tuple[int, int] = (1, 1),
        normalize_steps: list[str] | None = None,
        stopwords: list[str] | None = None,
        stopword_language: str | None = "en",
        stem: bool = False,
        max_iter: int = 300,
        random_state: int | None = 0,
    ) -> NlpTopicResult:
        """
        Fit an unsupervised topic model on Session train documents.

        Thin Session facade over ``buildml.nlp``; records the call and stores NLP artifacts on the Session for follow-up steps.

        Parameters
        ----------
        method:
            Algorithm or method identifier for the resolved backend.
        n_topics:
            Number of topics for topic-model fits. Too many fragments themes; too few collapses them.
        text_column:
            Text column name; defaults to the sole text-role column.
        top_terms:
            How many top terms/features to surface in explanations or topic summaries.
        max_features:
            Cap on vocabulary size after frequency ranking. ``None`` keeps all terms above ``min_df``; a finite cap controls memory and noise from rare tokens.
        min_df:
            Minimum document frequency for a term to enter the vocabulary (integer count or fraction of documents). Higher values drop rare/noisy tokens.
        max_df:
            Maximum document frequency for a term (count or fraction). Lower it to strip overly common tokens that behave like stopwords.
        ngram_range:
            Inclusive ``(min_n, max_n)`` n-gram window. ``(1, 1)`` is unigrams only; ``(1, 2)`` adds bigrams and usually helps short-text classification at higher sparsity cost.
        normalize_steps:
            Ordered text-normalization steps to apply before vectorization (lowercase, unicode cleanup, whitespace collapse, …).
        stopwords:
            Stopword list name, custom iterable, or ``None``. Removing common function words often helps bag-of-words models.
        stopword_language:
            Language key for the stopword list used in token stats (for example ``"english"``). ``None`` skips stopword filtering.
        stem:
            When True, apply stemming after tokenization. Useful for bag-of-words; usually off for embeddings.
        max_iter:
            Hard cap on solver iterations or boosting rounds. Raise it when fits do not converge.
        random_state:
            Seed for randomized fitting steps so re-runs are comparable. ``None`` leaves RNG undeterministic.

        Returns
        -------
        NlpTopicResult
            Structured domain result recorded on the Session for follow-up evaluate/explain/export steps.

        Notes
        -----
        **Leakage:** Requires a split. The vectorizer and decomposition are fitted on
        train only, so ``assign_topics`` on holdout is a pure transform.
        """
        return nlp_ops.fit_topics_op(
            self,
            method=method,
            n_topics=n_topics,
            text_column=text_column,
            top_terms=top_terms,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            normalize_steps=normalize_steps,
            stopwords=stopwords,
            stopword_language=stopword_language,
            stem=stem,
            max_iter=max_iter,
            random_state=random_state,
        )

    def assign_topics(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
    ) -> NlpTopicAssignResult:
        """
        Transform a partition into per-document topic weights.

        Thin Session facade over ``buildml.nlp``; records the call and stores NLP artifacts on the Session for follow-up steps.

        Parameters
        ----------
        partition:
            Split partition to read or score.

        Returns
        -------
        NlpTopicAssignResult
            Structured domain result recorded on the Session for follow-up evaluate/explain/export steps.
        """
        return nlp_ops.assign_topics_op(self, partition=partition)

    def extract_keyphrases(
        self,
        *,
        partition: PartitionName | Literal["all"] = "train",
        method: str = "tfidf",
        text_column: str | None = None,
        top_n: int = 15,
        max_phrase_words: int = 3,
        per_document: bool = True,
        max_documents: int = 25,
        stopword_language: str | None = "en",
        stopwords: list[str] | None = None,
        min_df: int | float = 1,
        max_df: int | float = 1.0,
        window: int = 4,
        random_state: int | None = 0,
    ) -> NlpKeyphraseResult:
        """
        Rank keyphrases for a partition with an unsupervised scorer.

        Thin Session facade over ``buildml.nlp``; records the call and stores NLP artifacts on the Session for follow-up steps.

        Parameters
        ----------
        partition:
            Split partition to read or score.
        method:
            Algorithm or method identifier for the resolved backend.
        text_column:
            Text column name; defaults to the sole text-role column.
        top_n:
            How many top items/rows/terms to keep in the result table.
        max_phrase_words:
            Maximum words allowed in extracted keyphrases.
        per_document:
            When True, return per-document outputs instead of a corpus-level aggregate.
        max_documents:
            Maximum documents to profile or process in one call. Lower it for a cheap health check on huge corpora.
        stopword_language:
            Language key for the stopword list used in token stats (for example ``"english"``). ``None`` skips stopword filtering.
        stopwords:
            Stopword list name, custom iterable, or ``None``. Removing common function words often helps bag-of-words models.
        min_df:
            Minimum document frequency for a term to enter the vocabulary (integer count or fraction of documents). Higher values drop rare/noisy tokens.
        max_df:
            Maximum document frequency for a term (count or fraction). Lower it to strip overly common tokens that behave like stopwords.
        window:
            Sliding window size (in tokens/characters as documented by the NLP backend)
            used when scoring local keyphrase or co-occurrence context.
        random_state:
            Seed for randomized fitting steps so re-runs are comparable. ``None`` leaves RNG undeterministic.

        Returns
        -------
        NlpKeyphraseResult
            Structured domain result recorded on the Session for follow-up evaluate/explain/export steps.
        """
        return nlp_ops.extract_keyphrases_op(
            self,
            partition=partition,
            method=method,
            text_column=text_column,
            top_n=top_n,
            max_phrase_words=max_phrase_words,
            per_document=per_document,
            max_documents=max_documents,
            stopword_language=stopword_language,
            stopwords=stopwords,
            min_df=min_df,
            max_df=max_df,
            window=window,
            random_state=random_state,
        )

    def analyze_sentiment(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        backend: str = "lexicon",
        text_column: str | None = None,
        threshold: float = 0.05,
        compare_to_target: bool = False,
        transformer_model: str = "distilbert-base-uncased-finetuned-sst-2-english",
        device: str = "cpu",
    ) -> NlpSentimentResult:
        """
        Score a partition's documents for sentiment.

        Thin Session facade over ``buildml.nlp``; records the call and stores NLP artifacts on the Session for follow-up steps.

        Parameters
        ----------
        partition:
            Split partition to read or score.
        backend:
            Backend identifier; see capability matrix for valid values.
        text_column:
            Text column name; defaults to the sole text-role column.
        threshold:
            Decision cutoff on scores or probabilities. Raise it to flag fewer positives.
        compare_to_target:
            When True, compare predictions or retrieved labels against the Session target for metrics.
        transformer_model:
            Hugging Face transformer model id for sequence-classification / NER / generative NLP paths.
        device:
            Compute device string (``"cpu"``, ``"cuda"``, ``"mps"``, …) for torch-backed paths.

        Returns
        -------
        NlpSentimentResult
            Structured domain result recorded on the Session for follow-up evaluate/explain/export steps.
        """
        return nlp_ops.analyze_sentiment_op(
            self,
            partition=partition,
            backend=backend,
            text_column=text_column,
            threshold=threshold,
            compare_to_target=compare_to_target,
            transformer_model=transformer_model,
            device=device,
        )

    def extract_entities(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        backend: str | None = "rules",
        text_column: str | None = None,
        labels: list[str] | None = None,
        gazetteers: dict[str, list[str]] | None = None,
        spacy_model: str = "en_core_web_sm",
        max_documents: int = 25,
        batch_size: int = 32,
    ) -> NlpEntityResult:
        """
        Extract entity mentions from a partition's documents.

        Thin Session facade over ``buildml.nlp``; records the call and stores NLP artifacts on the Session for follow-up steps.

        Parameters
        ----------
        partition:
            Split partition to read or score.
        backend:
            Backend identifier; see capability matrix for valid values.
        text_column:
            Text column name; defaults to the sole text-role column.
        labels:
            Explicit label order for metrics/confusion matrices. ``None`` uses labels observed in the scored partition.
        gazetteers:
            Optional gazetteer dictionaries used by rule/NER backends to boost entity matches.
        spacy_model:
            spaCy pipeline name for industry NLP backends. Must be installed separately from BuildML.
        max_documents:
            Maximum documents to profile or process in one call. Lower it for a cheap health check on huge corpora.
        batch_size:
            Mini-batch size for transformer/embedding encode or train loops.

        Returns
        -------
        NlpEntityResult
            Structured domain result recorded on the Session for follow-up evaluate/explain/export steps.
        """
        return nlp_ops.extract_entities_op(
            self,
            partition=partition,
            backend=backend,
            text_column=text_column,
            labels=labels,
            gazetteers=gazetteers,
            spacy_model=spacy_model,
            max_documents=max_documents,
            batch_size=batch_size,
        )

    def summarize_text(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        method: str = "textrank",
        text_column: str | None = None,
        n_sentences: int = 3,
        max_documents: int = 25,
        max_input_sentences: int = 200,
        stopword_language: str | None = "en",
        stopwords: list[str] | None = None,
    ) -> NlpSummaryResult:
        """
        Build extractive summaries for a partition's documents.

        Thin Session facade over ``buildml.nlp``; records the call and stores NLP artifacts on the Session for follow-up steps.

        Parameters
        ----------
        partition:
            Split partition to read or score.
        method:
            Algorithm or method identifier for the resolved backend.
        text_column:
            Text column name; defaults to the sole text-role column.
        n_sentences:
            Number of sentences to keep in extractive summaries.
        max_documents:
            Maximum documents to profile or process in one call. Lower it for a cheap health check on huge corpora.
        max_input_sentences:
            Maximum sentences considered from each document before summarization.
        stopword_language:
            Language key for the stopword list used in token stats (for example ``"english"``). ``None`` skips stopword filtering.
        stopwords:
            Stopword list name, custom iterable, or ``None``. Removing common function words often helps bag-of-words models.

        Returns
        -------
        NlpSummaryResult
            Structured domain result recorded on the Session for follow-up evaluate/explain/export steps.
        """
        return nlp_ops.summarize_text_op(
            self,
            partition=partition,
            method=method,
            text_column=text_column,
            n_sentences=n_sentences,
            max_documents=max_documents,
            max_input_sentences=max_input_sentences,
            stopword_language=stopword_language,
            stopwords=stopwords,
        )

    @property
    def nlp_text_plan(self) -> NlpTextPlan | None:
        """Return the text plan built by the most recent classifier fit.

        The plan holds the normalization recipe, the train-fitted
        representation, and the fitted head, and is what ``predict_text``,
        ``evaluate_text_classifier``, and ``interpret_text_prediction``
        replay. It is what a bundle stores.

        Returns
        -------
        :class:`~buildml.nlp.results.NlpTextPlan` or None
            ``None`` until ``fit_text_classifier`` or ``load_nlp_bundle`` has
            run, which is why those operations refuse before it exists.
        """
        return self._nlp_text_plan

    @property
    def nlp_topic_plan(self) -> NlpTopicPlan | None:
        """Return the topic model built by the most recent topic fit.

        Holds the vectorizer and the decomposition together, so
        ``assign_topics`` stays a pure transform on holdout rows.

        Returns
        -------
        :class:`~buildml.nlp.results.NlpTopicPlan` or None
            ``None`` until ``fit_topics`` or ``load_nlp_bundle`` has run.
        """
        return self._nlp_topic_plan

    @property
    def nlp_fit_result(self) -> NlpFitResult | None:
        """Return the report from the most recent classifier fit.

        Records the resolved backend and head, training size, vocabulary size,
        class counts, the in-sample score, and any disclosures — the audit
        trail for how the model was built.

        Returns
        -------
        :class:`~buildml.nlp.results.NlpFitResult` or None
            ``None`` until ``fit_text_classifier`` has run in this Session; a
            reloaded bundle restores the plan without replaying the fit report.
        """
        return self._nlp_fit_result

    @property
    def nlp_eval_result(self) -> NlpEvalResult | None:
        """Return the metrics from the most recent classifier evaluation.

        Returns
        -------
        :class:`~buildml.nlp.results.NlpEvalResult` or None
            ``None`` until ``evaluate_text_classifier`` has run. Each call
            replaces the previous result, so read it before scoring another
            partition if you need both.
        """
        return self._nlp_eval_result

    @property
    def nlp_predict_result(self) -> NlpPredictResult | None:
        """Return the predictions from the most recent scoring call.

        Returns
        -------
        :class:`~buildml.nlp.results.NlpPredictResult` or None
            ``None`` until ``predict_text`` has run.
        """
        return self._nlp_predict_result

    @property
    def nlp_interpret_result(self) -> NlpInterpretResult | None:
        """Return the token attributions from the most recent interpretation.

        Returns
        -------
        :class:`~buildml.nlp.results.NlpInterpretResult` or None
            ``None`` until ``interpret_text_prediction`` has run — which it
            cannot for hashing or dense representations.
        """
        return self._nlp_interpret_result

    @property
    def nlp_topic_result(self) -> NlpTopicResult | None:
        """Return the report from the most recent topic fit.

        Carries the discovered topics with their terms plus the quality
        signals — NPMI coherence, reconstruction error, perplexity — that you
        read to decide whether the decomposition is worth keeping.

        Returns
        -------
        :class:`~buildml.nlp.results.NlpTopicResult` or None
            ``None`` until ``fit_topics`` has run in this Session.
        """
        return self._nlp_topic_result

    @property
    def nlp_topic_assign_result(self) -> NlpTopicAssignResult | None:
        """Return the topic assignment from the most recent assignment call.

        Returns
        -------
        :class:`~buildml.nlp.results.NlpTopicAssignResult` or None
            ``None`` until ``assign_topics`` has run.
        """
        return self._nlp_topic_assign_result

    @property
    def nlp_keyphrase_result(self) -> NlpKeyphraseResult | None:
        """Return the phrases from the most recent keyphrase extraction.

        Returns
        -------
        :class:`~buildml.nlp.results.NlpKeyphraseResult` or None
            ``None`` until ``extract_keyphrases`` has run.
        """
        return self._nlp_keyphrase_result

    @property
    def nlp_sentiment_result(self) -> NlpSentimentResult | None:
        """Return the scores from the most recent sentiment analysis.

        Returns
        -------
        :class:`~buildml.nlp.results.NlpSentimentResult` or None
            ``None`` until ``analyze_sentiment`` has run. Check
            ``matched_term_rate`` before quoting any rate from it.
        """
        return self._nlp_sentiment_result

    @property
    def nlp_entity_result(self) -> NlpEntityResult | None:
        """Return the mentions from the most recent entity extraction.

        Returns
        -------
        :class:`~buildml.nlp.results.NlpEntityResult` or None
            ``None`` until ``extract_entities`` has run.
        """
        return self._nlp_entity_result

    @property
    def nlp_summary_result(self) -> NlpSummaryResult | None:
        """Return the summaries from the most recent summarisation call.

        Returns
        -------
        :class:`~buildml.nlp.results.NlpSummaryResult` or None
            ``None`` until ``summarize_text`` has run.
        """
        return self._nlp_summary_result

    @property
    def nlp_language_result(self) -> NlpLanguageResult | None:
        """Return the languages from the most recent detection call.

        Returns
        -------
        :class:`~buildml.nlp.results.NlpLanguageResult` or None
            ``None`` until ``detect_language`` has run. ``profile_text_corpus``
            reports a language mix too, but does not populate this accessor.
        """
        return self._nlp_language_result

    @property
    def nlp_profile_result(self) -> NlpCorpusProfile | None:
        """Return the report from the most recent corpus profile.

        The contamination screen behind every honest text metric: read
        ``train_holdout_exact_overlap`` and ``findings`` before quoting a
        holdout score.

        Returns
        -------
        :class:`~buildml.nlp.results.NlpCorpusProfile` or None
            ``None`` until ``profile_text_corpus`` has run.
        """
        return self._nlp_profile_result

    def save_nlp_bundle(self, path: str | Path) -> Path:
        """Persist the active NLP plan(s) as ``buildml.nlp_bundle.v1``.

        The bundle carries the normalization recipe alongside the train-fitted
        representation and head, which is what lets a reload reproduce a
        holdout score exactly instead of approximately. A topic plan is
        included when one has been fitted.

        Parameters
        ----------
        path:
            Directory to write. Created if missing, and overwritten if it
            already holds a bundle.

        Returns
        -------
        pathlib.Path
            The directory written.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            Neither a text plan nor a topic plan has been fitted, so there is
            nothing to save.

        Notes
        -----
        Bundles and Session checkpoints are complementary, not
        interchangeable — see
        :data:`buildml.nlp.checkpoint.CHECKPOINT_BOUNDARY`. A checkpoint stores
        data, roles, splits, and history; it does not embed the vectorizer or
        the head.
        """
        return nlp_ops.save_nlp_bundle_op(self, path=path)

    def load_nlp_bundle(self, path: str | Path) -> Session:
        """Restore a saved text plan, and topic plan, into this Session.

        Scoring resumes without refitting, because the normalization recipe
        travelled with the representation. The Session still needs its own
        dataset, roles, and split — the bundle carries the model, not the
        workflow.

        Parameters
        ----------
        path:
            Directory previously written by ``save_nlp_bundle``.

        Returns
        -------
        Session
            This Session, so the call chains.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            The directory is not a readable ``buildml.nlp_bundle.v1``.
        """
        return nlp_ops.load_nlp_bundle_op(self, path=path)

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
        """
        Fit behavioral cloning on Session train demonstrations.

        Delegates to :func:`buildml.rl.imitation.fit_imitation`, stores the
        :class:`~buildml.rl.results.ImitationPlan` on Session, and records the
        fit. Follow with :meth:`predict_imitation_action` or
        :meth:`evaluate_imitation`.

        Parameters
        ----------
        backend:
            Optional backend override (sklearn, torch).
        task:
            Optional task override (classification/regression).
        estimator:
            Optional BC estimator identifier.
        method:
            Optional method alias for the resolved backend.
        columns:
            Optional explicit state feature columns.
        action_column:
            Optional action column override.
        env_id:
            Optional Gymnasium environment id for env-backed demos.
        n_epochs:
            Training epochs for torch BC backend.
        random_state:
            Seed for stochastic training steps.
        prefer_reduce_components:
            Prefer reduced component columns when a reduce plan exists.

        Returns
        -------
        ImitationFitResult
            Serializable fit summary including action-space disclosures.

        Notes
        -----
        **Leakage:** Requires a split. Policy uses train only. Honesty: BC from
        tables — not inverse RL / DAgger / robotics.
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
        """
        Predict actions under the fitted BC policy.

        Delegates to :func:`buildml.rl.imitation.predict_imitation_action` on a
        named partition without refitting the policy.

        Parameters
        ----------
        partition:
            Partition to predict on (``test`` by default).

        Returns
        -------
        ImitationPredictResult
            Predicted actions and optional quality disclosures.

        Raises
        ------
        ValidationError
            When no imitation plan exists on the Session.
        """
        return rl_ops.predict_imitation_action_op(self, partition=partition)

    def evaluate_imitation(
        self,
        *,
        partition: PartitionName | Literal["all"] = "validation",
    ) -> ImitationEvalResult:
        """
        Evaluate BC against held-out demonstration actions.

        Delegates to :func:`buildml.rl.imitation.evaluate_imitation` on a holdout
        partition. Falls back to ``test`` when validation is empty.

        Parameters
        ----------
        partition:
            Holdout partition for evaluation (``validation`` by default).

        Returns
        -------
        ImitationEvalResult
            Held-out action prediction metrics and disclosures.

        Raises
        ------
        ValidationError
            When no imitation plan exists on the Session.
        """
        return rl_ops.evaluate_imitation_op(self, partition=partition)

    @property
    def imitation_plan(self) -> ImitationPlan | None:
        """
        Return the behavioral-cloning plan built by the most recent imitation fit.

        Stored on Session after :meth:`fit_imitation` or :meth:`load_imitation_bundle` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.rl.results.ImitationPlan` or None
            ``None`` until :meth:`fit_imitation` or :meth:`load_imitation_bundle` has run.
        """
        return self._imitation_plan

    @property
    def imitation_fit_result(self) -> ImitationFitResult | None:
        """
        Return the report from the most recent imitation fit.

        Stored on Session after :meth:`fit_imitation` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.rl.results.ImitationFitResult` or None
            ``None`` until :meth:`fit_imitation` has run.
        """
        return self._imitation_fit_result

    @property
    def imitation_eval_result(self) -> ImitationEvalResult | None:
        """
        Return the metrics from the most recent imitation evaluation.

        Stored on Session after :meth:`evaluate_imitation` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.rl.results.ImitationEvalResult` or None
            ``None`` until :meth:`evaluate_imitation` has run.
        """
        return self._imitation_eval_result

    @property
    def imitation_predict_result(self) -> ImitationPredictResult | None:
        """
        Return the actions from the most recent imitation prediction call.

        Stored on Session after :meth:`predict_imitation_action` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.rl.results.ImitationPredictResult` or None
            ``None`` until :meth:`predict_imitation_action` has run.
        """
        return self._imitation_predict_result

    def save_imitation_bundle(self, path: str | Path) -> Path:
        """
        Persist the active ImitationPlan as ``buildml.imitation_bundle.v1``.

        Delegates to :func:`buildml.rl.checkpoint.save_imitation_bundle`.
        Reload with :meth:`load_imitation_bundle`.

        Parameters
        ----------
        path:
            Destination directory for the bundle (created if missing).

        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.

        Raises
        ------
        ValidationError
            When no imitation plan exists on the Session.
        """
        return rl_ops.save_imitation_bundle_op(self, path=path)

    def load_imitation_bundle(self, path: str | Path) -> Session:
        """
        Load an imitation bundle into this Session.

        Delegates to :func:`buildml.rl.checkpoint.load_imitation_bundle` and
        clears prior eval/predict results.

        Parameters
        ----------
            Session instance to populate with the loaded imitation plan.
        path:
            Path to a ``buildml.imitation_bundle.v1`` directory.

        Returns
        -------
        Session
            This Session with imitation plan attached for chaining.
        """
        return rl_ops.load_imitation_bundle_op(self, path=path)

    @staticmethod
    def rl_capability_matrix() -> dict[str, Any]:
        """
        Return the RL / imitation capability matrix for this installation.

        Delegates to :func:`buildml.rl.catalog.rl_capability_matrix`. Use before
        :meth:`fit_rl` or :meth:`fit_imitation` to confirm available
        backends, modes, and algorithms for the current extras install.

        Returns
        -------
        dict
            Nested map of backend identifiers to supported modes and methods.
        """
        return rl_ops.rl_capability_matrix_op()

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
        n_bins: int = 8,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ) -> RlFitResult:
        """
        Fit a contextual bandit (core) or a Gymnasium env policy (``buildml[rl]``).

        Delegates to :func:`buildml.rl.fit.fit_rl`, stores the
        :class:`~buildml.rl.results.RlPlan` on Session, and records the fit.
        Follow with :meth:`act_rl` or :meth:`evaluate_rl`.

        Parameters
        ----------
        backend:
            Optional backend override.
        mode:
            RL mode (``contextual_bandit`` or gym-style modes).
        algorithm:
            Bandit or policy algorithm identifier.
        columns:
            Optional state feature columns for bandit mode.
        action_column:
            Logged action column for bandit mode.
        reward_column:
            Logged reward column for bandit mode.
        alpha:
            Exploration/strength parameter for LinUCB-style bandits.
        epsilon:
            Epsilon for epsilon-greedy exploration.
        temperature:
            Softmax temperature for stochastic action selection.
        random_state:
            Seed for stochastic training and exploration.
        prefer_reduce_components:
            Prefer reduced component columns when a reduce plan exists.
        env_id:
            Gymnasium environment id for env-loop modes.
        n_episodes:
            Number of training episodes for tabular/env modes.
        max_steps:
            Maximum steps per episode.
        learning_rate:
            Optimizer learning rate for policy updates.
        gamma:
            Discount factor for temporal-difference methods.
        total_timesteps:
            Total timesteps for SB3-style trainers.
        n_bins:
            Discretization bins for tabular Q-learning.
        epsilon_min:
            Minimum epsilon for decay schedules.
        epsilon_decay:
            Per-episode epsilon decay multiplier.

        Returns
        -------
        RlFitResult
            Serializable fit summary including mode and algorithm disclosures.

        Notes
        -----
        **Leakage (bandit):** Requires a split; updates use train logged data only.
        **gym_reinforce / tabular_q / gym_sb3:** Env loop; does not fit on Session
        tabular partitions. Honesty: not MuJoCo / robotics / multi-agent.
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
            n_bins=n_bins,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
        )

    def act_rl(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        observations: Sequence[Any] | Any | None = None,
        deterministic: bool = True,
        random_state: int | None = 0,
    ) -> RlActResult:
        """
        Choose actions under the fitted RL policy.

        Delegates to :func:`buildml.rl.act.act_rl` for bandit rows or env
        observations without refitting the policy.

        Parameters
        ----------
        partition:
            Partition for bandit action selection (``test`` by default).
        observations:
            Optional explicit observation batch for env/bandit modes.
        deterministic:
            When True, disable exploratory sampling where supported.
        random_state:
            Seed for stochastic action selection.

        Returns
        -------
        RlActResult
            Selected actions and policy disclosures.

        Raises
        ------
        ValidationError
            When no RL plan exists on the Session.
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
        """
        Evaluate RL (offline bandit metrics or Gymnasium rollouts).

        Delegates to :func:`buildml.rl.evaluate.evaluate_rl` on a holdout
        partition or env rollouts. Falls back to ``test`` for bandits when
        validation is empty.

        Parameters
        ----------
        partition:
            Holdout partition for bandit evaluation (``validation`` by default).
        n_episodes:
            Optional episode override for env evaluation.
        max_steps:
            Optional per-episode step cap for env evaluation.
        random_state:
            Seed for stochastic rollouts.
        deterministic:
            When True, disable exploratory sampling during evaluation.

        Returns
        -------
        RlEvalResult
            Offline or env evaluation metrics and disclosures.

        Raises
        ------
        ValidationError
            When no RL plan exists on the Session.
        """
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
        """
        Return the RL plan built by the most recent fit_rl call.

        Stored on Session after :meth:`fit_rl` or :meth:`load_rl_bundle` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.rl.results.RlPlan` or None
            ``None`` until :meth:`fit_rl` or :meth:`load_rl_bundle` has run.
        """
        return self._rl_plan

    @property
    def rl_fit_result(self) -> RlFitResult | None:
        """
        Return the report from the most recent RL fit.

        Stored on Session after :meth:`fit_rl` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.rl.results.RlFitResult` or None
            ``None`` until :meth:`fit_rl` has run.
        """
        return self._rl_fit_result

    @property
    def rl_eval_result(self) -> RlEvalResult | None:
        """
        Return the metrics from the most recent RL evaluation.

        Stored on Session after :meth:`evaluate_rl` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.rl.results.RlEvalResult` or None
            ``None`` until :meth:`evaluate_rl` has run.
        """
        return self._rl_eval_result

    @property
    def rl_act_result(self) -> RlActResult | None:
        """
        Return the actions from the most recent act_rl call.

        Stored on Session after :meth:`act_rl` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.rl.results.RlActResult` or None
            ``None`` until :meth:`act_rl` has run.
        """
        return self._rl_act_result

    def save_rl_bundle(self, path: str | Path) -> Path:
        """
        Persist the active RlPlan as ``buildml.rl_bundle.v1``.

        Delegates to :func:`buildml.rl.checkpoint.save_rl_bundle`.
        Reload with :meth:`load_rl_bundle`.

        Parameters
        ----------
        path:
            Destination directory for the bundle (created if missing).

        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.

        Raises
        ------
        ValidationError
            When no RL plan exists on the Session.
        """
        return rl_ops.save_rl_bundle_op(self, path=path)

    def load_rl_bundle(self, path: str | Path) -> Session:
        """
        Load an RL bundle into this Session.

        Delegates to :func:`buildml.rl.checkpoint.load_rl_bundle` and clears
        prior eval/act results.

        Parameters
        ----------
            Session instance to populate with the loaded RL plan.
        path:
            Path to a ``buildml.rl_bundle.v1`` directory.

        Returns
        -------
        Session
            This Session with RL plan attached for chaining.
        """
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
        """
        Fit TDA features and optional supervised head on Session train only.

        Delegates to :func:`buildml.tda.fit.fit_tda`, stores the
        :class:`~buildml.tda.results.TdaPlan` on Session, and records the fit.
        Follow with :meth:`transform_tda`, :meth:`predict_tda`, or
        :meth:`evaluate_tda`.

        Parameters
        ----------
        backend:
            Optional backend override (``native`` or ``giotto``).
        vectorization:
            Persistence diagram vectorization method.
        homology_dims:
            Homology dimensions to compute (e.g. H0, H1).
        knn:
            Neighborhood size for Vietoris-Rips / kNN graph construction.
        maxdim:
            Maximum homology dimension for persistent homology.
        thresh:
            Optional distance threshold for filtration truncation.
        n_bins:
            Bin count for persistence images and landscapes.
        n_layers:
            Layer count for multi-scale vectorizations.
        pixel_size:
            Pixel size for persistence images.
        standardize:
            Standardize vectorized features before the optional head.
        head:
            Optional supervised classifier/regressor head on TDA features.
        task:
            Task override when head is supervised (classification/regression).
        columns:
            Explicit feature columns; ``None`` auto-selects numerics.
        random_state:
            Seed for subsampling and stochastic steps.
        prefer_reduce_components:
            Prefer reduced component columns when a reduce plan exists on Session.
        max_points_guard:
            Maximum point count before subsample/error guard triggers.
        subsample_strategy:
            Behavior when point count exceeds ``max_points_guard``.
        mapper:
            When True, also compute a Mapper graph summary.

        Returns
        -------
        TdaFitResult
            Serializable fit summary including homology and vectorizer state.

        Notes
        -----
        **Leakage:** Requires a split. NN index, vectorizer ranges, and head use
        train only. Requires ``buildml[tda]`` (native) or ``buildml[tda-industry]``
        (giotto backend).
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
        """
        Return the TDA backend and vectorization capability matrix.

        Delegates to :func:`buildml.tda.catalog.tda_capability_matrix`.
        Use before :meth:`fit_tda` to confirm backend and method availability.

        Returns
        -------
        dict
            Nested map of backend identifiers to supported vectorizations.
        """
        return tda_ops.tda_capability_matrix_op()

    @staticmethod
    def ssl_capability_matrix() -> dict[str, Any]:
        """
        Report which self-supervised learning backends are available on this machine.

        Call before contrastive or masked-model fit methods to confirm torch and
        industry SSL extras. Read-only — no Session state is changed.

        Returns
        -------
        dict[str, Any]
            SSL backends, pretext tasks, and install hints from
            :func:`buildml.selfsupervised.torch.catalog.ssl_capability_matrix`.
        """
        from buildml.selfsupervised.torch.catalog import ssl_capability_matrix

        return ssl_capability_matrix()

    @staticmethod
    def dl_capability_matrix() -> dict[str, Any]:
        """
        Report which deep-learning modalities and backends are available here.

        Call before ``fit_torch``, speech transcription, or backbone loading to
        confirm torch/speech extras and weight-mode defaults. Read-only.

        Returns
        -------
        dict[str, Any]
            Modalities, weight modes, speech backends, and install hints from
            :func:`buildml.dl.catalog.dl_capability_matrix`.
        """
        from buildml.dl.catalog import dl_capability_matrix

        return dl_capability_matrix()

    @staticmethod
    def unsupervised_capability_matrix() -> dict[str, Any]:
        """
        Report which clustering and dimensionality-reduction backends are available here.

        Call before :meth:`fit_clusters` or :meth:`reduce_dimensions` to choose among
        sklearn, HDBSCAN, torch, or industry extras on this install. Read-only.

        Returns
        -------
        dict[str, Any]
            Clustering backends, methods, and install hints from
            :func:`buildml.unsupervised.catalog.unsupervised_capability_matrix`.
        """
        from buildml.unsupervised.catalog import unsupervised_capability_matrix

        return unsupervised_capability_matrix()

    @staticmethod
    def rag_capability_matrix() -> dict[str, Any]:
        """
        Report which retrieval-augmented generation stacks are available here.

        Call before :meth:`build_rag_index` or generate helpers to confirm embed,
        store, rerank, and LLM extras on this machine. Read-only introspection.

        Returns
        -------
        dict[str, Any]
            RAG backends, embedders, and install hints from
            :func:`buildml.rag.catalog.rag_capability_matrix`.
        """
        from buildml.rag.catalog import rag_capability_matrix

        return rag_capability_matrix()

    @staticmethod
    def forecast_capability_matrix() -> dict[str, Any]:
        """
        Report which forecasting backends and model families are available here.

        Call before :meth:`fit_forecast` to see whether statsmodels, Prophet,
        neuralforecast, or core fallbacks imported successfully. Read-only.

        Returns
        -------
        dict[str, Any]
            Forecast backends, horizons, and install hints from
            :func:`buildml.forecasting.catalog.forecast_capability_matrix`.
        """
        from buildml.forecasting.catalog import forecast_capability_matrix

        return forecast_capability_matrix()

    @staticmethod
    def timeseries_capability_matrix() -> dict[str, Any]:
        """
        Report which time-series analysis backends are available on this machine.

        Call before :meth:`analyze_timeseries` to confirm statsmodels STL/ADF and
        ruptures changepoint extras versus core fallbacks. Read-only introspection.

        Returns
        -------
        dict[str, Any]
            Decomposition and changepoint backends from
            :func:`buildml.timeseries.catalog.timeseries_capability_matrix`.
        """
        from buildml.timeseries.catalog import timeseries_capability_matrix

        return timeseries_capability_matrix()

    @staticmethod
    def causal_capability_matrix() -> dict[str, Any]:
        """
        Report which causal inference backends and estimators are available here.

        Call before :meth:`estimate_causal_effect` or related causal fit methods to
        confirm DoWhy, EconML, or native paths on this install. Read-only.

        Returns
        -------
        dict[str, Any]
            Causal backends, identification methods, and install hints from
            :func:`buildml.causal.catalog.causal_capability_matrix`.
        """
        from buildml.causal.catalog import causal_capability_matrix

        return causal_capability_matrix()

    @staticmethod
    def federated_capability_matrix() -> dict[str, Any]:
        """
        Report which federated learning backends are available on this machine.

        Call before federated fit or aggregation helpers to confirm Flower, sklearn
        FedAvg, or native simulation paths. Read-only introspection.

        Returns
        -------
        dict[str, Any]
            Federated backends and install hints from
            :func:`buildml.federated.catalog.federated_capability_matrix`.
        """
        from buildml.federated.catalog import federated_capability_matrix

        return federated_capability_matrix()

    @staticmethod
    def graph_capability_matrix() -> dict[str, Any]:
        """
        Report which graph machine-learning backends are available here.

        Call before :meth:`fit_graph` to see whether PyTorch Geometric, DGL, or
        sklearn graph kernels imported successfully. Read-only — no dataset required.

        Returns
        -------
        dict[str, Any]
            Graph backends, tasks, and install hints from
            :func:`buildml.graph.catalog.graph_capability_matrix`.
        """
        from buildml.graph.catalog import graph_capability_matrix

        return graph_capability_matrix()

    @staticmethod
    def kg_capability_matrix() -> dict[str, Any]:
        """
        Report which knowledge-graph learning backends are available on this machine.

        Call before link-prediction or embedding fit methods to confirm PyKEEN,
        DGL-KE, or native paths on this install. Read-only introspection.

        Returns
        -------
        dict[str, Any]
            KG backends, tasks, and install hints from
            :func:`buildml.kg.catalog.kg_capability_matrix`.
        """
        from buildml.kg.catalog import kg_capability_matrix

        return kg_capability_matrix()

    @staticmethod
    def metalearning_capability_matrix() -> dict[str, Any]:
        """
        Report which meta-learning backends and algorithms are available here.

        Call before few-shot or MAML-style fit methods to confirm learn2learn,
        torch meta modules, or sklearn fallbacks. Read-only introspection.

        Returns
        -------
        dict[str, Any]
            Meta-learning backends and install hints from
            :func:`buildml.metalearning.catalog.metalearning_capability_matrix`.
        """
        from buildml.metalearning.catalog import metalearning_capability_matrix

        return metalearning_capability_matrix()

    @staticmethod
    def multitask_capability_matrix() -> dict[str, Any]:
        """
        Report which multi-task learning backends are available on this machine.

        Call before :meth:`fit_multitask` to confirm chained sklearn, torch
        shared-trunk, or industry extras on this install. Read-only.

        Returns
        -------
        dict[str, Any]
            Multi-task backends, heads, and install hints from
            :func:`buildml.multitask.catalog.multitask_capability_matrix`.
        """
        from buildml.multitask.catalog import multitask_capability_matrix

        return multitask_capability_matrix()

    @staticmethod
    def online_capability_matrix() -> dict[str, Any]:
        """
        Report which online and incremental learning backends are available here.

        Call before :meth:`fit_online` to see whether River, sklearn partial_fit,
        or torch streaming paths imported successfully. Read-only introspection.

        Returns
        -------
        dict[str, Any]
            Online backends, update modes, and install hints from
            :func:`buildml.online.catalog.online_capability_matrix`.
        """
        from buildml.online.catalog import online_capability_matrix

        return online_capability_matrix()

    @staticmethod
    def probabilistic_capability_matrix() -> dict[str, Any]:
        """
        Report which probabilistic prediction backends are available on this machine.

        Call before conformal or distributional fit methods to confirm mapie,
        torch quantile heads, or sklearn fallbacks on this install. Read-only.

        Returns
        -------
        dict[str, Any]
            Probabilistic backends, interval methods, and install hints from
            :func:`buildml.probabilistic.catalog.probabilistic_capability_matrix`.
        """
        from buildml.probabilistic.catalog import probabilistic_capability_matrix

        return probabilistic_capability_matrix()

    @staticmethod
    def recommender_capability_matrix() -> dict[str, Any]:
        """
        Report which recommender-system backends are available on this machine.

        Call before :meth:`fit_recommender` to confirm implicit, LightFM, or sklearn
        matrix-factorization paths on this install. Read-only introspection.

        Returns
        -------
        dict[str, Any]
            Recommender backends, interaction models, and install hints from
            :func:`buildml.recommenders.catalog.recommender_capability_matrix`.
        """
        from buildml.recommenders.catalog import recommender_capability_matrix

        return recommender_capability_matrix()

    @staticmethod
    def semisupervised_capability_matrix() -> dict[str, Any]:
        """
        Report which semi-supervised learning backends are available here.

        Call before label-propagation or pseudo-label fit methods to confirm sklearn,
        torch, or industry SSL hybrids on this install. Read-only.

        Returns
        -------
        dict[str, Any]
            Semi-supervised backends and install hints from
            :func:`buildml.semisupervised.catalog.semisupervised_capability_matrix`.
        """
        from buildml.semisupervised.catalog import semisupervised_capability_matrix

        return semisupervised_capability_matrix()

    @staticmethod
    def activelearning_capability_matrix() -> dict[str, Any]:
        """
        Report which active-learning query strategies are available on this machine.

        Call before pool-based query loops to confirm modAL, sklearn uncertainty
        samplers, or native strategies on this install. Read-only introspection.

        Returns
        -------
        dict[str, Any]
            Query strategies, backends, and install hints from
            :func:`buildml.activelearning.catalog.activelearning_capability_matrix`.
        """
        from buildml.activelearning.catalog import activelearning_capability_matrix

        return activelearning_capability_matrix()

    @staticmethod
    def automl_capability_matrix() -> dict[str, Any]:
        """
        Report which AutoML search backends and model families are available here.

        Call before :meth:`run_automl` to confirm FLAML, Optuna HPO, or native
        sklearn search paths on this install. Read-only — no dataset required.

        Returns
        -------
        dict[str, Any]
            AutoML backends, search spaces, and install hints from
            :func:`buildml.automl.catalog.automl_capability_matrix`.
        """
        from buildml.automl.catalog import automl_capability_matrix

        return automl_capability_matrix()

    def transform_tda(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        backend: TdaBackend | None = None,
    ) -> TdaTransformResult:
        """
        Transform a partition with the frozen train-fitted TDA pipeline.

        Delegates to :func:`buildml.tda.transform.transform_tda` using the plan
        from :meth:`fit_tda`. No refit occurs on holdout partitions.

        Parameters
        ----------
        partition:
            Split partition to transform (default ``test``).
        backend:
            Optional backend override for transform step.

        Returns
        -------
        TdaTransformResult
            Vectorized persistence features for the requested partition.

        Raises
        ------
        ValidationError
            When no TdaPlan exists on the Session.
        """
        return tda_ops.transform_tda_op(self, partition=partition, backend=backend)

    def predict_tda(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
    ) -> TdaPredictResult:
        """
        Predict with the optional TDA supervised head on a partition.

        Delegates to :func:`buildml.tda.predict.predict_tda`. Requires a
        supervised head fitted during :meth:`fit_tda`.

        Parameters
        ----------
        partition:
            Split partition to predict on (default ``test``).

        Returns
        -------
        TdaPredictResult
            Predictions and optional probabilities from the TDA head.

        Raises
        ------
        ValidationError
            When no TdaPlan exists on the Session.
        """
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
        """
        Evaluate the TDA head on a holdout partition.

        Delegates to :func:`buildml.tda.evaluate.evaluate_tda` and optionally
        compares persistence diagram distances between partitions.

        Parameters
        ----------
        partition:
            Holdout partition for evaluation (default ``validation``).
        backend:
            Optional backend override for evaluation.
        compare_diagram_distances:
            When True, compute diagram distance metrics between partitions.
        diagram_distance_metric:
            Distance metric for persistence diagrams (e.g. Wasserstein).
        diagram_distance_dim:
            Homology dimension for diagram distance comparison.

        Returns
        -------
        TdaEvalResult
            Holdout metrics for the supervised TDA head and optional distances.

        Raises
        ------
        ValidationError
            When no TdaPlan exists on the Session.
        """
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
        """
        Return the TDA plan built by the most recent fit_tda call.

        Stored on Session after :meth:`fit_tda` or :meth:`load_tda_bundle` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.tda.results.TdaPlan` or None
            ``None`` until :meth:`fit_tda` or :meth:`load_tda_bundle` has run.
        """
        return self._tda_plan

    @property
    def tda_fit_result(self) -> TdaFitResult | None:
        """
        Return the report from the most recent TDA fit.

        Stored on Session after :meth:`fit_tda` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.tda.results.TdaFitResult` or None
            ``None`` until :meth:`fit_tda` has run.
        """
        return self._tda_fit_result

    @property
    def tda_eval_result(self) -> TdaEvalResult | None:
        """
        Return the metrics from the most recent TDA evaluation.

        Stored on Session after :meth:`evaluate_tda` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.tda.results.TdaEvalResult` or None
            ``None`` until :meth:`evaluate_tda` has run.
        """
        return self._tda_eval_result

    @property
    def tda_transform_result(self) -> TdaTransformResult | None:
        """
        Return the topological features from the most recent transform_tda call.

        Stored on Session after :meth:`transform_tda` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.tda.results.TdaTransformResult` or None
            ``None`` until :meth:`transform_tda` has run.
        """
        return self._tda_transform_result

    @property
    def tda_predict_result(self) -> TdaPredictResult | None:
        """
        Return the predictions from the most recent predict_tda call.

        Stored on Session after :meth:`predict_tda` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.tda.results.TdaPredictResult` or None
            ``None`` until :meth:`predict_tda` has run.
        """
        return self._tda_predict_result

    def save_tda_bundle(self, path: str | Path) -> Path:
        """
        Persist the active TdaPlan as ``buildml.tda_bundle.v2``.

        Delegates to :func:`buildml.tda.checkpoint.save_tda_bundle`.
        Reload with :meth:`load_tda_bundle`.

        Parameters
        ----------
        path:
            Destination directory for the bundle (created if missing).

        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.

        Raises
        ------
        ValidationError
            When no TdaPlan exists on the Session.
        """
        return tda_ops.save_tda_bundle_op(self, path=path)

    def load_tda_bundle(self, path: str | Path) -> Session:
        """
        Load a TDA bundle into this Session.

        Delegates to :func:`buildml.tda.checkpoint.load_tda_bundle` and clears
        prior transform/predict/eval results.

        Parameters
        ----------
            Session instance to populate with the loaded TdaPlan.
        path:
            Path to a ``buildml.tda_bundle.v2`` directory.

        Returns
        -------
        Session
            This Session with TdaPlan attached for chaining.
        """
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
        """
        Fit a recommender on Session train interactions only.

        Delegates to :func:`buildml.recommenders.fit.fit_recommender`, stores the
        :class:`~buildml.recommenders.results.RecommenderPlan` on Session, and
        records the fit. Follow with :meth:`recommend` or
        :meth:`evaluate_recommender`.

        Parameters
        ----------
        method:
            Optional recommender method override.
        backend:
            Optional backend override (sklearn, implicit, lightfm).
        user_column:
            User identifier column.
        item_column:
            Item identifier column.
        rating_column:
            Rating or interaction strength column.
        feedback:
            Feedback mode (``explicit`` or ``implicit``).
        n_neighbors:
            Neighborhood size for kNN-style recommenders.
        n_factors:
            Latent factor dimension for matrix-factorization methods.
        min_rating:
            Optional minimum rating threshold for explicit feedback.
        item_feature_columns:
            Optional item-side content feature columns.
        user_feature_columns:
            Optional user-side content feature columns.
        cold_start:
            Cold-start policy for unseen users/items.
        random_state:
            Seed for stochastic training steps.
        n_iterations:
            Iteration count for ALS-style trainers.
        lightfm_epochs:
            Epoch count for LightFM backend.

        Notes
        -----
        **Leakage:** Requires a split. Similarities / factors / content profiles
        use train interactions only. Holdout items may appear as cold catalog
        misses (known-item protocol). Distinct from RAG and EDA Recommendation
        Findings.

        When ``feedback='implicit'`` and ``method`` is omitted, defaults to ALS
        (``implicit`` library) when ``buildml[recommenders-industry]`` is installed.


        Returns
        -------
            RecommenderFitResult
                    Fit report with method, backend, and training disclosures."""
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
        """
        Top-K recommendations for partition users or an explicit user id list.

        Delegates to :func:`buildml.recommenders.recommend.recommend` using the
        fitted plan. Defaults to the ``test`` partition when neither ``partition``
        nor ``user_ids`` is supplied.

        Parameters
        ----------
        partition:
            Optional partition whose users receive recommendations.
        user_ids:
            Optional explicit user identifiers to recommend for.
        k:
            Number of items to recommend per user.
        exclude_train_items:
            When True, exclude items seen in train interactions.

        Raises
        ------
        ValidationError
            When no recommender plan exists on the Session.


        Returns
        -------
            RecommendResult
                    Top-k recommendations per user with provenance and warnings."""
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
        """
        Evaluate ranking metrics on a holdout partition (frozen train plan).

        Delegates to :func:`buildml.recommenders.evaluate.evaluate_recommender`
        without refitting the recommender on holdout interactions.

        Parameters
        ----------
        partition:
            Holdout partition for evaluation (``test`` by default).
        k:
            Cutoff k for ranking metrics.

        Raises
        ------
        ValidationError
            When no recommender plan exists on the Session.


        Returns
        -------
            RecommenderEvalResult
                    Ranking metrics on the holdout partition."""
        return recommender_ops.evaluate_recommender_op(
            self, partition=partition, k=k
        )

    @property
    def recommender_plan(self) -> RecommenderPlan | None:
        """
        Return the recommender plan built by the most recent fit_recommender call.

        Stored on Session after :meth:`fit_recommender` or :meth:`load_recommender_bundle` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.recommenders.results.RecommenderPlan` or None
            ``None`` until :meth:`fit_recommender` or :meth:`load_recommender_bundle` has run.
        """
        return self._recommender_plan

    @property
    def recommender_fit_result(self) -> RecommenderFitResult | None:
        """
        Return the report from the most recent recommender fit.

        Stored on Session after :meth:`fit_recommender` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.recommenders.results.RecommenderFitResult` or None
            ``None`` until :meth:`fit_recommender` has run.
        """
        return self._recommender_fit_result

    @property
    def recommender_eval_result(self) -> RecommenderEvalResult | None:
        """
        Return the metrics from the most recent recommender evaluation.

        Stored on Session after :meth:`evaluate_recommender` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.recommenders.results.RecommenderEvalResult` or None
            ``None`` until :meth:`evaluate_recommender` has run.
        """
        return self._recommender_eval_result

    @property
    def recommender_recommend_result(self) -> RecommendResult | None:
        """
        Return the recommendations from the most recent recommend call.

        Stored on Session after :meth:`recommend` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.recommenders.results.RecommendResult` or None
            ``None`` until :meth:`recommend` has run.
        """
        return self._recommender_recommend_result

    def save_recommender_bundle(self, path: str | Path) -> Path:
        """
        Persist the active RecommenderPlan as ``buildml.recommender_bundle.v1``.

        Delegates to :func:`buildml.recommenders.checkpoint.save_recommender_bundle`.
        Reload with :meth:`load_recommender_bundle`.

        Parameters
        ----------
        path:
            Destination directory for the bundle (created if missing).

        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.

        Raises
        ------
        ValidationError
            When no recommender plan exists on the Session.
        """
        return recommender_ops.save_recommender_bundle_op(self, path=path)

    def load_recommender_bundle(self, path: str | Path) -> Session:
        """
        Load a recommender bundle into this Session.

        Delegates to :func:`buildml.recommenders.checkpoint.load_recommender_bundle`
        and clears prior fit/eval/recommend results.

        Parameters
        ----------
            Session instance to populate with the loaded recommender plan.
        path:
            Path to a ``buildml.recommender_bundle.v1`` directory.

        Returns
        -------
        Session
            This Session with recommender plan attached for chaining.
        """
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
        """
        Fit a tabular ranker on Session train rows only.

        Delegates to :func:`buildml.ranking.fit.fit_ranker`, stores the
        :class:`~buildml.ranking.results.RankerPlan` on Session, and records
        the fit. Follow with :meth:`rank` or :meth:`evaluate_ranker`.

        Parameters
        ----------
        backend:
            Optional ranker backend override.
        method:
            Optional method override (pointwise/pairwise/listwise identifiers).
        query_column:
            Query/group column for LTR examples.
        item_column:
            Item/document column for LTR examples.
        relevance_column:
            Relevance or grade column for supervised ranking.
        feature_columns:
            Optional explicit feature columns for ranker inputs.
        pointwise_estimator:
            Pointwise base estimator when method is pointwise.
        pairwise_estimator:
            Pairwise base estimator when method is pairwise.
        max_pairs_per_query:
            Cap on generated pairs per query for pairwise training.
        relevance_threshold:
            Threshold for binarizing graded relevance in some metrics.
        alpha:
            Regularization strength for linear rankers.
        C:
            Inverse regularization for SVM-style pairwise rankers.
        n_estimators:
            Number of trees for GBDT rankers.
        learning_rate:
            Learning rate for GBDT/torch rankers.
        hidden_dim:
            Hidden width for torch listwise ranker.
        epochs:
            Training epochs for torch backend.
        device:
            Torch device string.
        random_state:
            Seed for stochastic training steps.

        Notes
        -----
        **Leakage:** Requires a split. Prefer ``group_split`` on ``query_column``
        so test queries' labels never appear in train. Distinct from RAG and from
        recommender CF. See ``ranking_capability_matrix()`` for backends.


        Returns
        -------
            RankerFitResult
                    Fit report with backend, method, and training disclosures."""
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
        """
        Order items for queries in a partition or an explicit query id list.

        Delegates to :func:`buildml.ranking.rank.rank` using the fitted ranker
        plan. Defaults to the ``test`` partition when neither ``partition`` nor
        ``query_ids`` is supplied.

        Parameters
        ----------
        partition:
            Optional partition to rank (``train``, ``validation``, ``test``,
            or ``all``).
        query_ids:
            Optional explicit query identifiers to rank.
        k:
            Top-k items to return per query.
        backend:
            Optional backend override for ranking.

        Raises
        ------
        ValidationError
            When no ranker plan exists on the Session.


        Returns
        -------
            RankResult
                    Ranked items per query with scores and provenance."""
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
        """
        Evaluate per-query ranking metrics on a holdout partition.

        Delegates to :func:`buildml.ranking.evaluate.evaluate_ranker` using the
        frozen train ranker without refitting.

        Parameters
        ----------
        partition:
            Holdout partition for evaluation (``test`` by default).
        k:
            Cutoff k for ranking metrics (NDCG@k, etc.).
        backend:
            Optional backend override for evaluation.

        Raises
        ------
        ValidationError
            When no ranker plan exists on the Session.


        Returns
        -------
            RankerEvalResult
                    Per-query ranking metrics on the holdout partition."""
        return ranking_ops.evaluate_ranker_op(
            self, partition=partition, k=k, backend=backend
        )

    @property
    def ranker_plan(self) -> RankerPlan | None:
        """
        Return the ranker plan built by the most recent fit_ranker call.

        Stored on Session after :meth:`fit_ranker` or :meth:`load_ranker_bundle` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.ranking.results.RankerPlan` or None
            ``None`` until :meth:`fit_ranker` or :meth:`load_ranker_bundle` has run.
        """
        return self._ranker_plan

    @property
    def ranker_fit_result(self) -> RankerFitResult | None:
        """
        Return the report from the most recent ranker fit.

        Stored on Session after :meth:`fit_ranker` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.ranking.results.RankerFitResult` or None
            ``None`` until :meth:`fit_ranker` has run.
        """
        return self._ranker_fit_result

    @property
    def ranker_eval_result(self) -> RankerEvalResult | None:
        """
        Return the metrics from the most recent ranker evaluation.

        Stored on Session after :meth:`evaluate_ranker` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.ranking.results.RankerEvalResult` or None
            ``None`` until :meth:`evaluate_ranker` has run.
        """
        return self._ranker_eval_result

    @property
    def ranker_rank_result(self) -> RankResult | None:
        """
        Return the rankings from the most recent rank call.

        Stored on Session after :meth:`rank` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.ranking.results.RankResult` or None
            ``None`` until :meth:`rank` has run.
        """
        return self._ranker_rank_result

    def save_ranker_bundle(self, path: str | Path) -> Path:
        """
        Persist the active RankerPlan as ``buildml.ranker_bundle.v1``.

        Delegates to :func:`buildml.ranking.checkpoint.save_ranker_bundle`.
        Reload with :meth:`load_ranker_bundle`.

        Parameters
        ----------
        path:
            Destination directory for the bundle (created if missing).

        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.

        Raises
        ------
        ValidationError
            When no ranker plan exists on the Session.
        """
        return ranking_ops.save_ranker_bundle_op(self, path=path)

    def load_ranker_bundle(self, path: str | Path) -> Session:
        """
        Load a ranker bundle into this Session.

        Delegates to :func:`buildml.ranking.checkpoint.load_ranker_bundle` and
        clears prior fit/eval/rank results.

        Parameters
        ----------
            Session instance to populate with the loaded ranker plan.
        path:
            Path to a ``buildml.ranker_bundle.v1`` directory.

        Returns
        -------
        Session
            This Session with ranker plan attached for chaining.
        """
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
        """
        Fit a knowledge-graph embedding model on Session train triples only.

        Delegates to :func:`buildml.kg.fit.fit_kg`, stores the
        :class:`~buildml.kg.results.KgPlan` on Session, and records the fit.
        Follow with :meth:`score_triples`, :meth:`predict_links`, or
        :meth:`evaluate_kg`.

        Parameters
        ----------
        backend:
            Optional backend override (``native`` or ``pykeen``).
        method:
            Embedding method (``transe``, ``distmult``, ``rotate``, etc.).
        head_column:
            Subject/head entity column; inferred from roles when omitted.
        relation_column:
            Relation/predicate column.
        tail_column:
            Object/tail entity column.
        embedding_dim:
            Latent embedding dimensionality.
        epochs:
            Training epochs over positive triples.
        batch_size:
            Minibatch size for stochastic training.
        learning_rate:
            Optimizer learning rate.
        margin:
            Margin for ranking-loss methods like TransE.
        neg_ratio:
            Negative samples per positive triple per batch.
        norm:
            Distance norm for TransE (``l1`` or ``l2``).
        random_state:
            Seed for negative sampling and initialization.

        Returns
        -------
        KgFitResult
            Serializable fit summary including vocab sizes and disclosures.

        Notes
        -----
        **Leakage:** Requires a split. Vocabularies, embeddings, and adjacency
        use train triples only. Holdout triples never update the model.
        Distinct from Graph ML (``set_graph`` / ``fit_graph``) and from RAG.
        Honesty: Session KG learning — not a Neo4j / graph-DB product.
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
        """
        Score head-relation-tail triples with the frozen KgPlan.

        Delegates to :func:`buildml.kg.predict.score_triples` without refitting
        embeddings. Use explicit ``triples`` or a Session partition.

        Parameters
        ----------
        partition:
            Optional split partition whose triples to score.
        triples:
            Optional explicit triples as a DataFrame or sequence of tuples.

        Returns
        -------
        KgScoreResult
            Plausibility scores for each triple.

        Raises
        ------
        ValidationError
            When no KgPlan exists on the Session.
        """
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
        """
        Predict missing link components using the frozen KgPlan.

        Delegates to :func:`buildml.kg.predict.predict_links` to rank candidate
        tails, heads, or relations for given query entities.

        Parameters
        ----------
        mode:
            Which component to predict (``tail``, ``head``, or ``relation``).
        heads:
            Optional head entities to query; defaults to all known heads.
        relations:
            Optional relations to constrain predictions.
        tails:
            Optional tail entities for head/relation prediction modes.
        k:
            Number of top-ranked candidates to return per query.
        filtered:
            When True, filter out triples already present in the train graph.

        Returns
        -------
        KgPredictResult
            Ranked link predictions and scores for each query.

        Raises
        ------
        ValidationError
            When no KgPlan exists on the Session.
        """
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
        """
        Run symbolic KG queries over the train-fitted graph structure.

        Delegates to :func:`buildml.kg.query.query_kg` for neighbor lookup,
        path finding, or typed queries over the frozen KgPlan.

        Parameters
        ----------
        mode:
            Query mode (``neighbors``, ``path``, or ``typed``).
        entity:
            Anchor entity for neighbor queries.
        source:
            Path query source entity.
        target:
            Path query target entity.
        relation:
            Optional relation filter for neighbor/path queries.
        direction:
            Edge direction to traverse (``out``, ``in``, or ``both``).
        max_hops:
            Maximum path length for path queries.

        Returns
        -------
        KgQueryResult
            Query results as neighbor lists or paths.

        Raises
        ------
        ValidationError
            When no KgPlan exists on the Session.
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
        """
        Evaluate link prediction with filtered MRR and Hits@K.

        Delegates to :func:`buildml.kg.evaluate.evaluate_kg` on a holdout
        partition without updating embeddings.

        Parameters
        ----------
        partition:
            Holdout partition containing test triples (default ``test``).
        k:
            Cutoff for Hits@K metrics.

        Returns
        -------
        KgEvalResult
            Filtered ranking metrics (MRR, Hits@K) for the partition.

        Raises
        ------
        ValidationError
            When no KgPlan exists on the Session.
        """
        return kg_ops.evaluate_kg_op(self, partition=partition, k=k)

    @property
    def kg_plan(self) -> KgPlan | None:
        """
        Return the knowledge-graph plan built by the most recent fit_kg call.

        Stored on Session after :meth:`fit_kg` or :meth:`load_kg_bundle` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.kg.results.KgPlan` or None
            ``None`` until :meth:`fit_kg` or :meth:`load_kg_bundle` has run.
        """
        return self._kg_plan

    @property
    def kg_fit_result(self) -> KgFitResult | None:
        """
        Return the report from the most recent KG fit.

        Stored on Session after :meth:`fit_kg` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.kg.results.KgFitResult` or None
            ``None`` until :meth:`fit_kg` has run.
        """
        return self._kg_fit_result

    @property
    def kg_eval_result(self) -> KgEvalResult | None:
        """
        Return the metrics from the most recent KG evaluation.

        Stored on Session after :meth:`evaluate_kg` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.kg.results.KgEvalResult` or None
            ``None`` until :meth:`evaluate_kg` has run.
        """
        return self._kg_eval_result

    @property
    def kg_score_result(self) -> ScoreTriplesResult | None:
        """
        Return the triple scores from the most recent score_triples call.

        Stored on Session after :meth:`score_triples` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.kg.results.ScoreTriplesResult` or None
            ``None`` until :meth:`score_triples` has run.
        """
        return self._kg_score_result

    @property
    def kg_predict_result(self) -> PredictLinksResult | None:
        """
        Return the link predictions from the most recent predict_links call.

        Stored on Session after :meth:`predict_links` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.kg.results.PredictLinksResult` or None
            ``None`` until :meth:`predict_links` has run.
        """
        return self._kg_predict_result

    @property
    def kg_query_result(self) -> KgQueryResult | None:
        """
        Return the graph query from the most recent query_kg call.

        Stored on Session after :meth:`query_kg` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.kg.results.KgQueryResult` or None
            ``None`` until :meth:`query_kg` has run.
        """
        return self._kg_query_result

    def save_kg_bundle(self, path: str | Path) -> Path:
        """
        Persist the active KgPlan as ``buildml.kg_bundle.v1``.

        Delegates to :func:`buildml.kg.checkpoint.save_kg_bundle`.
        Reload with :meth:`load_kg_bundle`.

        Parameters
        ----------
        path:
            Destination directory for the bundle (created if missing).

        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.

        Raises
        ------
        ValidationError
            When no KgPlan exists on the Session.
        """
        return kg_ops.save_kg_bundle_op(self, path=path)

    def load_kg_bundle(self, path: str | Path) -> Session:
        """
        Load a knowledge-graph bundle into this Session.

        Delegates to :func:`buildml.kg.checkpoint.load_kg_bundle` and clears
        prior score/predict/query/eval results.

        Parameters
        ----------
            Session instance to populate with the loaded KgPlan.
        path:
            Path to a ``buildml.kg_bundle.v1`` directory.

        Returns
        -------
        Session
            This Session with KgPlan attached for chaining.
        """
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
        """
        Fit a decision policy on train or validation without refitting the model.

        Delegates to :func:`buildml.optimize.fit.fit_decision_policy`, stores the
        :class:`~buildml.optimize.results.DecisionPlan` on Session, and records
        the fit. Follow with :meth:`apply_decisions` or
        :meth:`evaluate_decisions`.

        Parameters
        ----------
        method:
            Decision strategy (``threshold``, ``knapsack``, ``lp``, etc.).
        backend:
            Optional solver backend override for MIP/LP methods.
        partition:
            Partition used for threshold/allocation tuning (default ``validation``).
        allow_test_tuning:
            When False, refuse tuning on the test partition.
        fp_cost, fn_cost:
            False-positive and false-negative costs for threshold tuning.
        tp_benefit, tn_benefit:
            True-positive and true-negative benefits for cost-sensitive tuning.
        cost_matrix:
            Optional multi-class cost matrix for threshold methods.
        class_labels:
            Class label order matching ``cost_matrix`` rows/columns.
        capacity:
            Maximum selections for knapsack-style allocation.
        budget:
            Total budget for knapsack or LP allocation methods.
        score_source:
            Where decision scores come from (model probabilities, raw scores, etc.).
        score_column:
            Explicit column for scores when ``score_source`` is column-based.
        cost_column:
            Per-row cost column for knapsack/LP methods.
        value_column:
            Per-row value column for knapsack/LP methods.
        id_column:
            Row identifier column for allocation output.
        knapsack_solver:
            Knapsack solver (``dp`` or ``pulp``).
        objective:
            Allocation objective (maximize score, minimize cost, etc.).
        min_score:
            Minimum score cutoff before allocation.
        lp_max_fraction:
            Maximum fraction of budget any single item may consume in LP mode.

        Returns
        -------
        DecisionFitResult
            Serializable fit summary including tuned threshold or allocation.
            Use :meth:`apply_decisions` to apply the frozen plan.

        Raises
        ------
        ValidationError
            When no split plan exists on the Session.

        Notes
        -----
        **Leakage:** Defaults to ``partition='validation'``. Tuning on Session
        test requires ``allow_test_tuning=True`` and emits a dangerous-opt-in
        disclosure. ``method='threshold'`` wraps the same engine as
        :meth:`Session.tune_threshold` and also updates ``last_diagnostic``.
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
        """
        Apply the frozen DecisionPlan to a partition or candidate frame.

        Delegates to :func:`buildml.optimize.apply.apply_decisions` using the
        plan from :meth:`fit_decision_policy`. Stores apply results on Session
        and records the operation.

        Parameters
        ----------
        partition:
            Split partition to apply decisions to (``train``, ``validation``,
            ``test``, or ``all``). Ignored when ``candidates`` is provided.
        candidates:
            Optional explicit candidate frame instead of a Session partition.

        Returns
        -------
        DecisionApplyResult
            Selected rows, scores, and allocation metadata for the partition.

        Raises
        ------
        ValidationError
            When no DecisionPlan exists on the Session.
        """
        return decision_ops.apply_decisions_op(
            self, partition=partition, candidates=candidates
        )

    def evaluate_decisions(
        self,
        *,
        partition: PartitionName = "test",
    ) -> DecisionEvalResult:
        """
        Evaluate the frozen DecisionPlan on a holdout partition.

        Delegates to :func:`buildml.optimize.evaluate.evaluate_decisions` and
        stores evaluation metrics on Session. Requires a prior
        :meth:`fit_decision_policy`.

        Parameters
        ----------
        partition:
            Holdout partition for evaluation (default ``test``).

        Returns
        -------
        DecisionEvalResult
            Cost, benefit, and confusion-style metrics for the frozen plan.

        Raises
        ------
        ValidationError
            When no DecisionPlan exists on the Session.
        """
        return decision_ops.evaluate_decisions_op(self, partition=partition)

    @property
    def decision_plan(self) -> DecisionPlan | None:
        """
        Return the decision policy built by the most recent fit_decision_policy call.

        Stored on Session after :meth:`fit_decision_policy` or :meth:`load_decision_bundle` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.optimize.results.DecisionPlan` or None
            ``None`` until :meth:`fit_decision_policy` or :meth:`load_decision_bundle` has run.
        """
        return self._decision_plan

    @property
    def decision_fit_result(self) -> DecisionFitResult | None:
        """
        Return the report from the most recent decision-policy fit.

        Stored on Session after :meth:`fit_decision_policy` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.optimize.results.DecisionFitResult` or None
            ``None`` until :meth:`fit_decision_policy` has run.
        """
        return self._decision_fit_result

    @property
    def decision_eval_result(self) -> DecisionEvalResult | None:
        """
        Return the metrics from the most recent decision-policy evaluation.

        Stored on Session after :meth:`evaluate_decisions` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.optimize.results.DecisionEvalResult` or None
            ``None`` until :meth:`evaluate_decisions` has run.
        """
        return self._decision_eval_result

    @property
    def decision_apply_result(self) -> ApplyDecisionsResult | None:
        """
        Return the decisions from the most recent apply_decisions call.

        Stored on Session after :meth:`apply_decisions` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.optimize.results.ApplyDecisionsResult` or None
            ``None`` until :meth:`apply_decisions` has run.
        """
        return self._decision_apply_result

    def save_decision_bundle(self, path: str | Path) -> Path:
        """
        Persist the active DecisionPlan as ``buildml.decision_bundle.v1``.

        Delegates to :func:`buildml.optimize.checkpoint.save_decision_bundle`.
        Reload with :meth:`load_decision_bundle`.

        Parameters
        ----------
        path:
            Destination directory for the bundle (created if missing).

        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.

        Raises
        ------
        ValidationError
            When no DecisionPlan exists on the Session.
        """
        return decision_ops.save_decision_bundle_op(self, path=path)

    def load_decision_bundle(self, path: str | Path) -> Session:
        """
        Load a decision bundle into this Session.

        Delegates to :func:`buildml.optimize.checkpoint.load_decision_bundle`
        and clears prior fit/eval/apply results.

        Parameters
        ----------
            Session instance to populate with the loaded DecisionPlan.
        path:
            Path to a ``buildml.decision_bundle.v1`` directory.

        Returns
        -------
        Session
            This Session with DecisionPlan attached for chaining.
        """
        return decision_ops.load_decision_bundle_op(self, path=path)

    @staticmethod
    def decision_capability_matrix() -> dict[str, Any]:
        """
        Return the decision/optimization capability matrix for this install.

        Delegates to :func:`buildml.optimize.catalog.decision_capability_matrix`.
        Use before :meth:`fit_decision_policy` to confirm ``method`` and
        ``backend`` pairs available with current extras.

        Returns
        -------
        dict
            Nested map of method identifiers to supported backends and options.
        """
        return decision_ops.decision_capability_matrix_op()

    @staticmethod
    def optimize_capability_matrix() -> dict[str, Any]:
        """
        Return the decision/optimization capability matrix for this install.

        Delegates to :func:`buildml.optimize.catalog.decision_capability_matrix`.
        Use before :meth:`fit_decision_policy` to confirm ``method`` and
        ``backend`` pairs available with current extras.

        Returns
        -------
        dict
            Nested map of method identifiers to supported backends and options.
        """
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
        """
        Fit a tabular synthesizer on Session train rows only.

        Delegates to :func:`buildml.synthetic.fit.fit_synthesizer`, stores the
        :class:`~buildml.synthetic.results.SynthesizerPlan` on Session, and records
        the fit. Follow with :meth:`sample_synthetic` to draw synthetic rows.

        Parameters
        ----------
        backend:
            Optional backend override (``native`` or ``sdv`` when installed).
        method:
            Synthesizer method key (``gaussian_copula``, ``ctgan``, etc.).
        columns:
            Optional explicit columns to model; ``None`` uses train numerics/categoricals.
        random_state:
            Seed for stochastic fitting and sampling reproducibility.
        smooth_sigma:
            Gaussian smoothing for numeric marginals in copula methods.
        correlation_ridge:
            Ridge added to correlation estimates for numerical stability.
        target_column:
            Optional target column for conditional/tabular-GAN setups.
        k_neighbors:
            Neighbors for SMOTE-like oversampling strategies when applicable.
        sampling_strategy:
            Class sampling strategy for conditional oversampling methods.
        epochs:
            Training epochs for neural synthesizer backends.
        batch_size:
            Minibatch size for neural synthesizer backends.

        Returns
        -------
        SyntheticFitResult
            Serializable fit summary including schema and method disclosures.

        Raises
        ------
        ValidationError
            When no split plan exists on the Session.

        Notes
        -----
        **Leakage:** Always fits on train. Validation/test are never used to
        estimate schema, marginals, or joints. Distinct from
        :meth:`Session.resample` (class-balance preprocess).

        **Privacy:** Not a differential-privacy product.
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
        """
        Sample synthetic rows from the frozen synthesizer plan.

        Delegates to :func:`buildml.synthetic.sample.sample_and_maybe_merge`.
        When ``merge_mode='extend_train'``, synthetic rows are appended to train
        with provenance metadata and the split plan is updated.

        Parameters
        ----------
        n:
            Number of rows to sample; defaults to train size when ``None``.
        random_state:
            Optional seed override for this sampling call.
        condition:
            Optional column-value conditions for conditional sampling.
        merge_mode:
            ``none`` returns a frame only; ``extend_train`` merges into Session train.
        provenance_column:
            Boolean column marking synthetic rows when merging into train.
        validate:
            When True, run post-sample schema/range checks and attach warnings.

        Returns
        -------
        SyntheticSampleResult
            Sampled frame and merge metadata. May update Session dataset/split.

        Raises
        ------
        ValidationError
            When no synthesizer plan or split plan exists on the Session.
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
        """
        Evaluate the frozen synthesizer for fidelity or TSTR utility.

        Delegates to :func:`buildml.synthetic.evaluate.evaluate_synthetic`.
        Holdout real data is used only for comparison — never to refit the synthesizer.

        Parameters
        ----------
        mode:
            ``fidelity`` compares marginals/joints; ``tstr`` trains on synthetic
            and tests on real holdout rows.
        eval_backend:
            Metrics backend (``auto``, ``native``, or ``sdmetrics`` when installed).
        partition:
            Real-data holdout partition for comparison or TSTR evaluation.
        n_synthetic:
            Synthetic row count for evaluation draws; defaults to holdout size.
        random_state:
            Seed for synthetic draws during evaluation.
        estimator:
            Downstream estimator for TSTR utility mode.

        Returns
        -------
        SyntheticEvalResult
            Fidelity or TSTR metrics and evaluation disclosures.

        Raises
        ------
        ValidationError
            When no synthesizer plan exists on the Session.
        """
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
        """
        Return the synthetic-data backend/method capability matrix.

        Delegates to :func:`buildml.synthetic.catalog.synthetic_capability_matrix`.
        Use before :meth:`fit_synthesizer` to see which methods require SDV extras.

        Returns
        -------
        dict
            Nested map of backend identifiers to supported synthesizer methods.
        """
        return synthetic_ops.synthetic_capability_matrix_op()

    @property
    def synthesizer_plan(self) -> SynthesizerPlan | None:
        """
        Return the synthesizer plan built by the most recent fit_synthesizer call.

        Stored on Session after :meth:`fit_synthesizer` or :meth:`load_synthetic_bundle` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.synthetic.results.SynthesizerPlan` or None
            ``None`` until :meth:`fit_synthesizer` or :meth:`load_synthetic_bundle` has run.
        """
        return self._synthesizer_plan

    @property
    def synthetic_fit_result(self) -> SynthesizerFitResult | None:
        """
        Return the report from the most recent synthesizer fit.

        Stored on Session after :meth:`fit_synthesizer` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.synthetic.results.SynthesizerFitResult` or None
            ``None`` until :meth:`fit_synthesizer` has run.
        """
        return self._synthetic_fit_result

    @property
    def synthetic_eval_result(self) -> SyntheticEvalResult | None:
        """
        Return the metrics from the most recent synthetic evaluation.

        Stored on Session after :meth:`evaluate_synthetic` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.synthetic.results.SyntheticEvalResult` or None
            ``None`` until :meth:`evaluate_synthetic` has run.
        """
        return self._synthetic_eval_result

    @property
    def synthetic_sample_result(self) -> SyntheticSampleResult | None:
        """
        Return the sample from the most recent sample_synthetic call.

        Stored on Session after :meth:`sample_synthetic` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.synthetic.results.SyntheticSampleResult` or None
            ``None`` until :meth:`sample_synthetic` has run.
        """
        return self._synthetic_sample_result

    def save_synthetic_bundle(self, path: str | Path) -> Path:
        """
        Persist the active synthesizer plan as ``buildml.synthetic_bundle.v1``.

        Delegates to :func:`buildml.synthetic.checkpoint.save_synthetic_bundle`.
        Reload with :meth:`load_synthetic_bundle`.

        Parameters
        ----------
        path:
            Destination directory for the bundle (created if missing).

        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.

        Raises
        ------
        ValidationError
            When no synthesizer plan exists on the Session.
        """
        return synthetic_ops.save_synthetic_bundle_op(self, path=path)

    def load_synthetic_bundle(self, path: str | Path) -> Session:
        """
        Load a synthetic-data bundle into this Session.

        Delegates to :func:`buildml.synthetic.checkpoint.load_synthetic_bundle`
        and clears prior fit/eval/sample results.

        Parameters
        ----------
            Session instance to populate with the loaded synthesizer plan.
        path:
            Path to a ``buildml.synthetic_bundle.v1`` directory.

        Returns
        -------
        Session
            This Session with synthesizer plan attached for chaining.
        """
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
        """Teach BuildML a preprocessing step of your own.

        The built-in transforms cover the common cases, but domain work
        routinely needs something specific — a currency conversion using rates
        learned from the training period, a geospatial encoding, a
        normalisation your field defines its own way.

        Registering it here rather than transforming the DataFrame by hand buys
        you the same guarantees the built-ins have. Your ``fit`` callable is
        shown training rows only, so the leakage rule holds. The fitted state is
        stored as a plan, so score-time replay reproduces it. And the step is
        recorded in the session history, so it appears in the walkthrough and
        the model card instead of being an invisible edit.

        Registration is on the class, not an instance: a transform registered
        once is available to every session in the process.

        Parameters
        ----------
        name:
            The identifier you will pass to :meth:`apply_custom_transform`.
        fit:
            A callable receiving the training rows for the selected columns and
            returning whatever state the transform needs — a mapping, a fitted
            object, a tuple of statistics. Only training rows are ever passed
            in, which is what makes the step leakage-safe by construction.
        transform:
            A callable receiving that fitted state along with the rows to
            transform, and returning the transformed columns. Applied to every
            partition, and later to new data at score time.
        description:
            A short account of what the transform does. It appears in
            :meth:`list_transforms` and in the model card, so write it for
            whoever inherits the model.
        output_columns:
            The names of the columns produced. ``None`` keeps the input names,
            which is right for an in-place transformation and wrong for one
            that expands or renames.
        drop_input_columns:
            Remove the source columns after transforming. Set this when the
            outputs replace the inputs rather than supplementing them.
        serializable:
            Whether the fitted state can be pickled into a saved bundle. Set
            False for state holding an open connection or an unpicklable
            object; the transform then works in-process but cannot travel in a
            pipeline bundle.
        overwrite:
            Allow replacing an existing registration under the same name.
            Without it, re-registering raises — which catches the case of two
            modules quietly claiming the same name.

        Returns
        -------
        ~buildml.preprocess.custom.CustomTransformSpec
            The registered specification, as it will appear in
            :meth:`list_transforms`.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            The name is already registered and ``overwrite`` is False, or
            ``fit`` or ``transform`` is not callable.

        Notes
        -----
        **Leakage:** The ``fit`` callable receives only train rows for the
        selected columns.

        Score-time replay needs the same name registered in the same process.
        A saved pipeline bundle stores the fitted state, not your Python code,
        so the scoring process must import whatever module performs the
        registration.

        Examples
        --------
        >>> from buildml import Session
        >>> Session.register_transform(
        ...     "log1p",
        ...     fit=lambda frame, **kwargs: {},
        ...     transform=lambda state, frame, **kwargs: frame.apply(
        ...         lambda col: (col + 1).map(float).map(__import__("math").log)
        ...     ),
        ...     description="Natural log of (x + 1), for right-skewed positives.",
        ...     overwrite=True,
        ... )  # doctest: +ELLIPSIS
        CustomTransformSpec(...)

        See Also
        --------
        Session.apply_custom_transform : Run a registered transform.
        Session.list_transforms : See what is currently registered.
        """
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
        """List the custom transforms currently registered.

        Registration is process-wide, so this shows everything available to
        :meth:`apply_custom_transform` — including transforms registered by
        modules you imported rather than wrote.

        Returns
        -------
        tuple of ~buildml.preprocess.custom.CustomTransformSpec
            Every registered specification, ordered by name, each carrying its
            description and whether its fitted state can be serialised into a
            pipeline bundle.

        Examples
        --------
        >>> from buildml import Session
        >>> [spec.name for spec in Session.list_transforms()]  # doctest: +SKIP
        ['log1p']

        See Also
        --------
        Session.register_transform : Add one.
        """
        return preprocess_ops.list_transforms(cls)

    def apply_custom_transform(
        self,
        name: str,
        *,
        columns: list[str],
        params: Mapping[str, Any] | None = None,
    ) -> Session:
        """Run a transform you registered, with the same leakage guarantees.

        Fits the named transform on the training rows and applies the result to
        every row, exactly as the built-in transforms behave. The fitted state
        is captured on :attr:`custom_plan` so score-time replay reproduces it,
        and the step is written into the session history so it shows up in the
        walkthrough and the model card.

        Parameters
        ----------
        name:
            The name given to :meth:`register_transform`.
        columns:
            Which columns to pass to the transform. These are handed to your
            ``fit`` callable as training rows, and to ``transform`` for every
            partition.
        params:
            Extra keyword arguments forwarded to the registered ``fit``
            callable, letting one registration serve several configurations.

        Returns
        -------
        Session
            ``self``, so this call chains into the next step.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            No transform is registered under that name, no split exists, or a
            named column is absent.

        Notes
        -----
        **Leakage:** Requires a split. Fit sees train rows only. Score-time
        replay requires the same name to remain registered in-process.

        Examples
        --------
        >>> _ = session.apply_custom_transform("log1p", columns=["amount"])  # doctest: +SKIP

        See Also
        --------
        Session.register_transform : Define the transform first.
        Session.custom_plan : The fitted state, for score-time replay.
        """
        return preprocess_ops.apply_custom_transform(
            self, name=name, columns=columns, params=params
        )

    @property
    def custom_plan(self) -> CustomTransformPlan | None:
        """The fitted state from the last :meth:`apply_custom_transform` call.

        A :class:`~buildml.preprocess.custom.CustomTransformPlan` holds
        whatever your ``fit`` callable returned, along with the transform name,
        the columns it covered, and the parameters it ran with — enough for
        score-time replay to reproduce the training-time transformation.

        The plan travels in a :meth:`save_pipeline` bundle only when the
        transform was registered with ``serializable=True``.

        ``None`` until :meth:`apply_custom_transform` runs.
        """
        return self._custom_plan

    def dry_run(
        self,
        operation: str | Sequence[str] | None = None,
        *,
        parameters: Mapping[str, Any] | None = None,
    ) -> DryRunReport:
        """See what an operation would do, without doing it.

        Some steps are expensive and some are hard to undo. A dry run checks
        whether an operation could run right now, what it would need, and what
        it would change — and then changes nothing. No fitting, no
        transforming, no history entry.

        It is the natural companion to :meth:`workflow`: that tells you which
        steps are available, this tells you what a particular one would
        actually do here.

        Parameters
        ----------
        operation:
            One operation name, several names to preview as a sequence, or
            ``None`` for an overview of what is currently available and what is
            blocked, with the reason for each block.
        parameters:
            The arguments you intend to pass, so the preview reflects your
            specific call rather than the defaults. Applies to a
            single-operation preview.

        Returns
        -------
        ~buildml.session.audit.DryRunReport
            What each previewed operation requires, whether those requirements
            are met, what it would change, and any warnings. Also stored on
            :attr:`last_dry_run`.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            A named operation is not one BuildML knows.

        Notes
        -----
        Dry-run does not fit, transform, or append history. Availability means
        API prerequisites pass, not that the operation is appropriate.

        That distinction matters. A dry run confirms that :meth:`split` *can*
        run; it cannot tell you that :meth:`group_split` is the one your data
        requires. Statistical judgement is still yours.

        Examples
        --------
        >>> session.dry_run("scale", parameters={"method": "minmax"})  # doctest: +SKIP
        >>> session.dry_run()  # doctest: +SKIP

        See Also
        --------
        Session.workflow : Availability across every operation.
        Session.explain : What an operation means, rather than whether it runs.
        """
        return workflow_ops.dry_run(self, operation=operation, parameters=parameters)

    @property
    def last_dry_run(self) -> DryRunReport | None:
        """The most recent :meth:`dry_run` report.

        Kept so a preview can be re-read after the fact. ``None`` until
        :meth:`dry_run` runs.
        """
        return self._last_dry_run

    def summarize_history(self) -> HistorySummary:
        """Condense what this session did, and flag what looks risky.

        The raw :attr:`history` is complete but long. This summarises it —
        which operations ran, in what order, which choices were explicit and
        which were defaults — and adds a list of unresolved risks worth a
        second look.

        The risk list is the reason to call it. Preprocessing that ran before
        the split, an evaluation on test taken more than once, a model fitted
        without a stratified split on imbalanced data: these are easy to do and
        easy to forget, and each quietly changes what your numbers mean.

        Returns
        -------
        ~buildml.session.audit.HistorySummary
            The condensed record with its risk list. Also stored on
            :attr:`last_history_summary`.

        Notes
        -----
        Read-only. Does not append history. Risks are heuristic review cues,
        not proof of leakage or invalid results.

        Treat a flagged risk as a question rather than a verdict. Some are
        deliberate — you may have every reason to preprocess before splitting
        on a dataset you are only exploring. The point is that the decision
        should be one you made rather than one that happened.

        Examples
        --------
        >>> summary = session.summarize_history()  # doctest: +SKIP
        >>> summary.risks  # doctest: +SKIP
        ['Session-global scale ran before cv_score; fold estimates may be optimistic.']

        See Also
        --------
        Session.walkthrough : The narrative version, exportable to HTML.
        Session.history : The raw records.
        """
        return workflow_ops.summarize_history(self)

    @property
    def last_history_summary(self) -> HistorySummary | None:
        """The most recent :meth:`summarize_history` result.

        Kept so the summary and its risk list can be re-read without
        recomputing. ``None`` until :meth:`summarize_history` runs.
        """
        return self._last_history_summary

    def fit(
        self,
        estimator: Any,
        *,
        task: Literal["classification", "regression", "auto"] = "auto",
    ) -> Session:
        """Train a model on the training rows.

        This is the step everything before it was preparing for. BuildML reads
        the column roles to work out what the inputs and the target are, hands
        the training rows to your estimator, and stores the fitted model on the
        session so :meth:`predict`, :meth:`evaluate`, and :meth:`save_pipeline`
        can find it.

        You supply the estimator yourself — any object with scikit-learn's
        ``fit`` and ``predict`` methods works, including XGBoost, LightGBM, and
        CatBoost models. BuildML does not maintain a private registry of model
        names, so anything installed in your environment is available and you
        configure it in the usual way.

        Before fitting, the training scope is checked: if there is no split, or
        an earlier step tried to widen the fit beyond the train rows, this
        raises rather than quietly producing an inflated score.

        Parameters
        ----------
        estimator:
            An unfitted estimator instance, already configured with whatever
            hyperparameters you want — ``RandomForestClassifier(max_depth=6)``,
            not the class itself. BuildML fits a reference to this object, so
            it is the one that ends up in the pipeline.
        task:
            Whether this is ``'classification'`` or ``'regression'``. The
            default ``'auto'`` infers it from the target column, which is
            correct nearly always; state it explicitly when the target is
            numeric but really represents classes, or when integer class labels
            would otherwise be read as a quantity to predict.

        Returns
        -------
        Session
            ``self``, so the fit chains into :meth:`evaluate`. The fitted model
            and its metadata are on :attr:`fit_result`.

        Raises
        ------
        ~buildml.core.errors.LeakageError
            No split exists. Fitting on everything leaves nothing to honestly
            measure against, so BuildML refuses rather than allowing it.
        ~buildml.core.errors.ValidationError
            No target column is assigned, features are still non-numeric or
            contain missing values, or the target does not fit the requested
            task.

        Notes
        -----
        **Leakage:** Fits on train only. Call after split and preparation.

        Only the training rows reach the estimator. Validation and test rows
        stay untouched until you ask for them by name, which is what makes the
        eventual test score meaningful.

        If a ``weight`` role column is assigned and the estimator supports
        sample weights, it is passed through, so rare-but-important rows can be
        given more influence without resampling.

        Examples
        --------
        >>> import pandas as pd
        >>> from sklearn.linear_model import LogisticRegression
        >>> from buildml import Session
        >>> frame = pd.DataFrame(
        ...     {"x": [0.1, 0.9, 0.2, 0.8], "y": [0, 1, 0, 1]}
        ... )
        >>> session = Session.ingest(frame).set_roles({"y": "target"})
        >>> _ = session.split(test_size=0.5, stratify=True)
        >>> _ = session.fit(LogisticRegression())
        >>> session.fit_result.task
        'classification'

        Any scikit-learn-compatible estimator works the same way:

        >>> from xgboost import XGBClassifier  # doctest: +SKIP
        >>> _ = session.fit(XGBClassifier(n_estimators=200))  # doctest: +SKIP

        See Also
        --------
        Session.evaluate : Measure what the fitted model actually achieves.
        Session.cv_score : A more stable estimate than a single holdout.
        Session.grid_search : Choose hyperparameters instead of guessing.
        Session.run_automl : Let a search pick the estimator for you.
        """
        return classical_ops.fit(self, estimator=estimator, task=task)

    @property
    def fit_result(self) -> FitResult | None:
        """The trained model from the last :meth:`fit` call, plus its context.

        A :class:`~buildml.model.supervised.FitResult` holds the fitted
        ``estimator`` itself, the inferred ``task``, the exact
        ``feature_columns`` in the order they were presented, the
        ``target_column``, the number of training rows, and the weight column
        if one was used.

        The column order is not incidental. A model expects features in the
        arrangement it was trained on, so recording it here is what allows
        :meth:`predict_from_pipeline` to rebuild the same design matrix from
        fresh data months later.

        ``None`` until :meth:`fit` runs. Note that the domain trainers
        (:meth:`fit_forecast`, :meth:`fit_ranker`, and the rest) store their
        results on their own properties and leave this one alone.
        """
        return self._fit_result

    def predict(
        self,
        *,
        partition: Literal["train", "validation", "test"] = "test",
        return_proba: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """Run the fitted model over one partition and return its predictions.

        Use this when you want the predictions themselves — to inspect them,
        join them back to identifiers, or compute something BuildML does not
        provide. If what you want is a score, :meth:`evaluate` computes metrics
        and diagnostics in one call instead.

        The features are rebuilt exactly as they were at fit time, using the
        column order recorded on :attr:`fit_result`, so the model receives what
        it expects.

        Parameters
        ----------
        partition:
            Which rows to score: ``'test'`` (the default) for the honest
            estimate, ``'validation'`` while tuning, or ``'train'`` to compare
            against the others. A model that scores far better on train than on
            test is overfitting, and comparing the two is how you see it.
        return_proba:
            When True, return each class's predicted probability rather than a
            single chosen label. Probabilities are what you need to move a
            decision threshold (:meth:`tune_threshold`), to rank cases by risk,
            or to check calibration. Ignored by estimators that do not expose
            ``predict_proba``.

        Returns
        -------
        pandas.Series or pandas.DataFrame
            A Series of predicted labels or values, indexed to match the
            partition's rows. With ``return_proba=True`` on a classifier, a
            DataFrame with one column per class instead.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            No model has been fitted yet, no split exists, or the named
            partition is not part of the current split.

        Notes
        -----
        Predicting on ``'train'`` tells you how well the model memorised, not
        how well it generalises. It is a useful diagnostic and a misleading
        headline number.

        Examples
        --------
        >>> import pandas as pd
        >>> from sklearn.linear_model import LogisticRegression
        >>> from buildml import Session
        >>> frame = pd.DataFrame({"x": [0.1, 0.9, 0.2, 0.8], "y": [0, 1, 0, 1]})
        >>> session = Session.ingest(frame).set_roles({"y": "target"})
        >>> _ = session.split(test_size=0.5, stratify=True)
        >>> _ = session.fit(LogisticRegression())
        >>> len(session.predict(partition="test"))
        2

        Get probabilities when you intend to choose your own cut-off:

        >>> proba = session.predict(partition="test", return_proba=True)
        >>> proba.shape[1]
        2

        See Also
        --------
        Session.evaluate : Metrics and diagnostics rather than raw output.
        Session.predict_from_pipeline : Score new data outside this session.
        Session.tune_threshold : Pick the cut-off these probabilities feed.
        """
        return classical_ops.predict(self, partition=partition, return_proba=return_proba)

    def evaluate(
        self,
        *,
        partition: Literal["train", "validation", "test"] = "test",
        export_figures: str | Path | None = None,
        export_html: str | Path | None = None,
        include_plots: bool = False,
    ) -> EvaluateResult:
        """Measure the fitted model, and explain what the measurement means.

        A single accuracy figure hides more than it reveals. 95% accuracy is
        excellent when the classes are balanced and worthless when 95% of rows
        belong to one class — the same number, opposite conclusions. So this
        returns a card rather than a score: several complementary metrics, the
        diagnostics behind them, and written recommendations about what to look
        at next.

        For classification you get accuracy and balanced accuracy, weighted
        precision and recall, macro and weighted F1, and — where the estimator
        exposes probabilities — ROC-AUC, average precision, and log loss, plus
        the confusion matrix showing which classes are being mistaken for
        which. Precision and recall matter most when errors are asymmetric:
        precision is how often a positive prediction is right, recall is how
        many of the real positives you caught, and improving one generally
        costs the other. Balanced accuracy is the one to read on imbalanced
        data, because plain accuracy is dominated by the majority class.

        For regression you get error magnitudes (MAE, RMSE) alongside R², plus
        residual diagnostics. MAE is the average miss in the target's own
        units. RMSE punishes large misses disproportionately, so a gap between
        the two means a few predictions are badly wrong. R² is the share of
        variance explained, and it can be negative — that simply means the
        model does worse than always predicting the mean.

        Parameters
        ----------
        partition:
            Which rows to score. ``'test'`` is the honest estimate and should
            be used once, at the end; ``'validation'`` is for the comparisons
            you make while deciding. Evaluating on ``'train'`` alongside test
            is the standard way to detect overfitting.
        export_figures:
            Directory to write diagnostic figures into. Implies plotting, and
            requires ``pip install 'buildml[viz]'``.
        export_html:
            Path for a self-contained HTML report of the same figures — handy
            to attach to a review or send to someone without a Python
            environment. Also implies plotting.
        include_plots:
            Build the diagnostic plot board without writing it anywhere. The
            board is stored on :attr:`last_plot_board` either way.

        Returns
        -------
        ~buildml.model.supervised.EvaluateResult
            The evaluation card: ``metrics``, ``diagnostics`` (confusion
            matrix, residual summaries, plot paths), the ``n_rows`` scored, and
            ``recommendations``. Call its ``show()`` method for a readable
            digest instead of reading the dictionaries by hand.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            No model has been fitted, no split exists, or the named partition
            is not part of the current split.
        ~buildml.core.errors.MissingExtraError
            Plots were requested without ``buildml[viz]`` installed.

        Notes
        -----
        Every glance at the test set spends a little of its independence. If
        you evaluate on test, adjust something, and evaluate again, the test
        score has quietly become a tuning signal and is no longer the unbiased
        estimate you think it is. Tune against validation or
        :meth:`cv_score`, and keep test for the end.

        Examples
        --------
        >>> import pandas as pd
        >>> from sklearn.linear_model import LogisticRegression
        >>> from buildml import Session
        >>> frame = pd.DataFrame({"x": [0.1, 0.9, 0.2, 0.8], "y": [0, 1, 0, 1]})
        >>> session = Session.ingest(frame).set_roles({"y": "target"})
        >>> _ = session.split(test_size=0.5, stratify=True)
        >>> result = session.fit(LogisticRegression()).evaluate()
        >>> result.task
        'classification'
        >>> "accuracy" in result.metrics and "balanced_accuracy" in result.metrics
        True

        Compare train against test to see whether the model is overfitting:

        >>> train_score = session.evaluate(partition="train").metrics["accuracy"]
        >>> test_score = session.evaluate(partition="test").metrics["accuracy"]

        Produce a shareable report with figures:

        >>> _ = session.evaluate(export_html="reports/eval.html")  # doctest: +SKIP

        See Also
        --------
        Session.eval_plots : Build the diagnostic board on its own.
        Session.calibration : Check whether probabilities mean what they say.
        Session.error_slices : Find the subgroups where the model fails.
        Session.compare_models : Put several candidates side by side.
        """
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
        """
        Build Torch DataLoaders from current roles and split partitions.

        Requires ``pip install 'buildml[torch]'`` (or ``buildml[dl]``). Shuffle
        applies to the train loader only. When ``normalize`` is True, mean/std
        are fit on train and frozen for validation/test.

        Classical preprocess: Session ``impute`` / ``encode`` / ``scale`` already
        mutate the attached frame with train-fitted plans. Attached plans are
        disclosed on the loader report. Pass ``apply_plans=True`` to explicitly
        re-apply fitted plans via :meth:`apply_preprocess_plans` before building
        loaders (score-time replay; does not refit).

        Parameters
        ----------
        batch_size:
            Minibatch size for all loaders.
        num_workers:
            DataLoader worker processes.
        pin_memory:
            When True, pin CPU memory for faster GPU transfer.
        shuffle_train:
            When True, shuffle the train loader each epoch.
        drop_last:
            When True, drop the final partial train batch.
        normalize:
            When True, fit normalize stats on train only.
        seed:
            Seed for shuffling and sampling.
        task:
            Supervised task (``classification``, ``regression``, or ``auto``).
        apply_plans:
            When True, re-apply Session preprocess plans before building loaders.

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
        """
        Build token-id DataLoaders for text classification (non-tabular modality).

        Vocabulary is fit on the train partition only. Requires ``buildml[torch]``.
        Delegates to :func:`buildml.dl.text.make_text_loaders`.

        Parameters
        ----------
        text_column:
            Optional text column; auto-detected when omitted.
        batch_size:
            Minibatch size for all loaders.
        max_len:
            Maximum token sequence length.
        max_vocab:
            Maximum vocabulary size fit on train.
        min_freq:
            Minimum token frequency to enter the vocabulary.
        shuffle_train:
            When True, shuffle the train loader each epoch.
        seed:
            Seed for shuffling and vocabulary sampling.

        Returns
        -------
        TorchLoaderBundle
            Text loaders plus vocabulary and text contract metadata."""
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
        """
        Train an ``nn.Module`` on the train Torch loader.

        Requires ``pip install 'buildml[torch]'``. When ``module`` is omitted, builds
        a tabular MLP, text classifier, or multimodal fusion module from the active
        loader contract so the happy path does not require a hand-rolled network.

        Does not replace classical :meth:`fit` / :attr:`fit_result`.
        Delegates to :func:`buildml.dl.train.train_supervised_module`.

        Parameters
        ----------
        module:
            Optional ``nn.Module`` to train; auto-built when omitted.
        loss_fn:
            Optional custom loss function.
        optimizer_factory:
            Optional factory returning a torch optimizer.
        epochs:
            Number of training epochs.
        learning_rate:
            Optimizer learning rate.
        device:
            Compute device (``cpu``, ``cuda``, ``mps``, or ``auto``).
        grad_clip_norm:
            Optional gradient clipping norm.
        log_every:
            Log training metrics every N epochs.
        early_stopping_patience:
            Optional validation patience for early stopping.
        early_stopping_monitor:
            Metric name monitored for early stopping.
        scheduler:
            Learning-rate scheduler kind.
        resume:
            When True, resume from the prior ``dl_train_result``.
        config:
            Optional full :class:`~buildml.dl.types.TrainConfig` override.
        hidden:
            Hidden layer sizes for auto-built tabular MLPs.
        dropout:
            Dropout rate for auto-built modules.
        mixed_precision:
            When True, enable autocast mixed precision where supported.

        Returns
        -------
        Session
            ``self`` with ``dl_train_result`` attached for chaining.

        Raises
        ------
        ValidationError
            When resume is requested without a prior trainer or multimodal
            contracts are incomplete."""
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
        """
        Build fused multimodal DataLoaders (tabular/text/image/audio mixes).

        Requires ``buildml[torch]``. Fit stats (vocab, numeric mean/std, image
        channel mean/std, audio amplitude mean/std) use the train partition only.
        Batches follow ``(numeric?, tokens?, image?, audio?, y)`` for present
        modalities. Audio fusion is a small 1D-CNN branch — not a speech foundation
        model.

        Pass ``preprocess=`` (contract / dict) to freeze restore stats, or
        ``use_saved_preprocess=True`` to reuse ``dl_train_result.multimodal_preprocess``.
        Delegates to :func:`buildml.dl.multimodal.make_multimodal_loaders`.

        Parameters
        ----------
        text_column:
            Optional text column for multimodal fusion.
        numeric_columns:
            Optional numeric columns for tabular branch.
        image_column:
            Optional image column/path column.
        audio_column:
            Optional audio column/path column.
        batch_size:
            Minibatch size for all loaders.
        max_len:
            Maximum token sequence length for text branch.
        max_vocab:
            Maximum vocabulary size fit on train.
        min_freq:
            Minimum token frequency for vocabulary.
        normalize:
            When True, normalize numeric features on train only.
        normalize_images:
            When True, normalize image channels on train only.
        normalize_audio:
            When True, normalize audio amplitude on train only.
        image_size:
            Target image height/width for image branch.
        image_channels:
            Number of image channels.
        audio_sample_rate:
            Target audio sample rate after resampling.
        audio_max_samples:
            Maximum audio samples per example.
        audio_source_sample_rate:
            Optional source sample rate before resampling.
        shuffle_train:
            When True, shuffle the train loader each epoch.
        seed:
            Seed for shuffling and preprocessing.
        task:
            Supervised task (``classification``, ``regression``, or ``auto``).
        preprocess:
            Optional frozen preprocess contract/dict to restore stats.
        use_saved_preprocess:
            When True, reuse preprocess meta from ``dl_train_result``.

        Returns
        -------
        TorchLoaderBundle
            Multimodal loaders plus contracts and preprocess disclosures.

        Raises
        ------
        ValidationError
            When both ``preprocess`` and ``use_saved_preprocess`` are supplied or
            saved preprocess meta is missing."""
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
        """
        Build image multimodal loaders (image ⊕ tabular and/or text and/or audio).

        Thin facade that requires ``image_column`` and delegates to the shared
        multimodal loader builder. Path cells need Pillow (bundled in
        ``buildml[torch]``); array/list cells work with Torch alone.

        Parameters
        ----------
        image_column:
            Required image column or path column.
        text_column:
            Optional text column for multimodal fusion.
        numeric_columns:
            Optional numeric columns for tabular branch.
        audio_column:
            Optional audio column for audio branch.
        batch_size:
            Minibatch size for all loaders.
        max_len:
            Maximum token sequence length for text branch.
        max_vocab:
            Maximum vocabulary size fit on train.
        min_freq:
            Minimum token frequency for vocabulary.
        normalize:
            When True, normalize numeric features on train only.
        normalize_images:
            When True, normalize image channels on train only.
        normalize_audio:
            When True, normalize audio amplitude on train only.
        image_size:
            Target image height/width for image branch.
        image_channels:
            Number of image channels.
        audio_sample_rate:
            Target audio sample rate after resampling.
        audio_max_samples:
            Maximum audio samples per example.
        audio_source_sample_rate:
            Optional source sample rate before resampling.
        shuffle_train:
            When True, shuffle the train loader each epoch.
        seed:
            Seed for shuffling and preprocessing.
        task:
            Supervised task (``classification``, ``regression``, or ``auto``).

        Returns
        -------
        TorchLoaderBundle
            Image-centric multimodal loaders plus contracts.

        Raises
        ------
        ValidationError
            When ``image_column`` is missing or empty."""
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
        """
        Build audio multimodal loaders (audio ⊕ tabular and/or text and/or image).

        Thin facade that requires ``audio_column`` and delegates to the shared
        multimodal loader builder. Path cells need soundfile (bundled in
        ``buildml[torch]`` / ``buildml[audio]``); waveform array cells work with
        Torch alone. Uses a small 1D-CNN fusion branch — not a speech foundation
        model.

        Parameters
        ----------
        audio_column:
            Required audio column or path column.
        text_column:
            Optional text column for multimodal fusion.
        numeric_columns:
            Optional numeric columns for tabular branch.
        image_column:
            Optional image column for image branch.
        batch_size:
            Minibatch size for all loaders.
        max_len:
            Maximum token sequence length for text branch.
        max_vocab:
            Maximum vocabulary size fit on train.
        min_freq:
            Minimum token frequency for vocabulary.
        normalize:
            When True, normalize numeric features on train only.
        normalize_images:
            When True, normalize image channels on train only.
        normalize_audio:
            When True, normalize audio amplitude on train only.
        image_size:
            Target image height/width for optional image branch.
        image_channels:
            Number of image channels.
        audio_sample_rate:
            Target audio sample rate after resampling.
        audio_max_samples:
            Maximum audio samples per example.
        audio_source_sample_rate:
            Optional source sample rate before resampling.
        shuffle_train:
            When True, shuffle the train loader each epoch.
        seed:
            Seed for shuffling and preprocessing.
        task:
            Supervised task (``classification``, ``regression``, or ``auto``).

        Returns
        -------
        TorchLoaderBundle
            Audio-centric multimodal loaders plus contracts.

        Raises
        ------
        ValidationError
            When ``audio_column`` is missing or empty."""
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
        """
        Fold-local Torch CV on the attached numeric tabular dataset.

        Normalize stats are fit per fold. Classical Session plans are disclosed as
        a limitation unless you supply a custom factory path — this helper does not
        silently refit Session-global plans inside each fold.
        Delegates to :func:`buildml.dl.cv.cross_validate_torch`.

        Parameters
        ----------
        n_folds:
            Number of cross-validation folds.
        epochs:
            Training epochs per fold.
        batch_size:
            Minibatch size per fold.
        learning_rate:
            Optimizer learning rate per fold.
        device:
            Compute device for fold-local training.
        normalize:
            When True, fit normalize stats per fold on train only.
        seed:
            Seed for fold splitting and training.
        stratify:
            When True, stratify folds for classification tasks.
        task:
            Supervised task (``classification``, ``regression``, or ``auto``).
        module_factory:
            Optional factory returning a fresh module per fold.

        Returns
        -------
        TorchCVResult
            Per-fold metrics and mean summary.

        Raises
        ------
        ValidationError
            When no dataset is attached to the Session."""
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
        """
        Inner-fold Torch hyperparameter search on the Session train universe.

        Held-out validation/test partitions are never scored. For a nested outer
        estimate after search, use :meth:`nested_cv_torch`.
        Delegates to :func:`buildml.dl.search.search_torch`.

        Parameters
        ----------
        param_grid:
            Optional grid of hyperparameter lists.
        param_distributions:
            Optional randomized search distributions.
        inner_search:
            Inner search strategy (``grid``, ``randomized``, or ``auto``).
        n_iter:
            Randomized search iterations when applicable.
        n_folds:
            Number of inner CV folds.
        epochs:
            Training epochs per candidate per fold.
        batch_size:
            Minibatch size for inner CV training.
        learning_rate:
            Optimizer learning rate for inner CV training.
        device:
            Compute device for inner CV training.
        normalize:
            When True, fit normalize stats per fold on train only.
        seed:
            Seed for fold splitting and search sampling.
        stratify:
            When True, stratify folds for classification tasks.
        task:
            Supervised task (``classification``, ``regression``, or ``auto``).
        scoring_metric:
            Optional metric name for ranking candidates.
        module_factory:
            Optional factory returning a fresh module per candidate/fold.

        Returns
        -------
        TorchSearchResult
            Best params, inner CV scores, and search disclosures.

        Raises
        ------
        ValidationError
            When no dataset is attached to the Session."""
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
        """
        Nested Torch CV: outer evaluation after fold-local inner hyperparameter search.

        Outer-eval rows never enter inner ranking. Session validation/test stay
        untouched. Normalize stats are fit fold-locally.
        Delegates to :func:`buildml.dl.search.nested_cv_torch`.

        Parameters
        ----------
        param_grid:
            Optional grid of hyperparameter lists for inner search.
        param_distributions:
            Optional randomized search distributions for inner search.
        inner_search:
            Inner search strategy (``grid``, ``randomized``, or ``auto``).
        n_iter:
            Randomized search iterations when applicable.
        outer_cv:
            Number of outer evaluation folds.
        inner_cv:
            Number of inner CV folds per outer fold.
        epochs:
            Training epochs per candidate per inner fold.
        batch_size:
            Minibatch size for nested CV training.
        learning_rate:
            Optimizer learning rate for nested CV training.
        device:
            Compute device for nested CV training.
        normalize:
            When True, fit normalize stats per fold on train only.
        seed:
            Seed for fold splitting and search sampling.
        stratify:
            When True, stratify folds for classification tasks.
        task:
            Supervised task (``classification``, ``regression``, or ``auto``).
        scoring_metric:
            Optional metric name for inner ranking and outer reporting.
        module_factory:
            Optional factory returning a fresh module per candidate/fold.

        Returns
        -------
        TorchNestedCVResult
            Outer-fold metrics, inner search summaries, and disclosures.

        Raises
        ------
        ValidationError
            When no dataset is attached to the Session."""
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
        """
        Export the last Torch trainer to TorchScript or ONNX.

        Uses train-loader example inputs when ``example_input`` is omitted.
        Alpha-quality escape hatch — see export result limitations.
        Delegates to :func:`buildml.dl.export.export_train_result`.

        Parameters
        ----------
        path:
            Destination file path for the exported artifact.
        format:
            Export format (``torchscript`` or ``onnx``).
        opset:
            ONNX opset version when ``format='onnx'``.
        dynamic_batch:
            When True, declare dynamic batch axes where supported.
        example_input:
            Optional explicit example input matching module layout.

        Returns
        -------
        TorchExportResult
            Export path, format, and limitation disclosures.

        Raises
        ------
        ValidationError
            When no torch trainer exists or non-tabular loaders/example inputs
            are missing."""
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
        """
        DDP training via a fresh ``module_factory`` per process.

        * Single-node (default): spawn local ranks. Requires
          ``torch.cuda.device_count() >= 2`` unless ``allow_cpu_ddp=True`` (gloo smoke).
        * Multi-node: ``multi_node=True`` joins a ``torchrun`` rendezvous
          (``WORLD_SIZE`` / ``RANK`` / ``LOCAL_RANK`` / ``MASTER_ADDR`` /
          ``MASTER_PORT``; ``LOCAL_RANK`` is required — global rank is not used as a
          local CUDA index). Not a Kubernetes multi-cluster orchestrator.
        Delegates to :func:`buildml.dl.ddp.train_supervised_module_ddp`.

        Parameters
        ----------
        module_factory:
            Callable returning a fresh ``nn.Module`` per DDP process.
        epochs:
            Number of training epochs.
        learning_rate:
            Optimizer learning rate.
        mixed_precision:
            When True, enable autocast mixed precision where supported.
        world_size:
            Optional explicit process/world size override.
        allow_cpu_ddp:
            When True, permit CPU gloo smoke tests with fewer GPUs.
        multi_node:
            When True, join an external torchrun rendezvous instead of spawning.
        config:
            Optional full :class:`~buildml.dl.types.TrainConfig` override.

        Returns
        -------
        DDPTrainResult
            DDP run summary and optional aggregated train result."""
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
        """
        Build speech classification loaders (finetune-lite encoder path).

        Requires ``buildml[torch]``. Amplitude stats fit on train only. This is an
        integration/finetune path — not training a foundation model from scratch.
        Delegates to :func:`buildml.dl.speech.make_speech_loaders`.

        Parameters
        ----------
        audio_column:
            Optional audio column; auto-detected when omitted.
        batch_size:
            Minibatch size for all loaders.
        sample_rate:
            Target audio sample rate after resampling.
        max_samples:
            Maximum audio samples per example.
        source_sample_rate:
            Optional source sample rate before resampling.
        normalize_audio:
            When True, normalize audio amplitude on train only.
        encoder_dim:
            Encoder embedding dimension for speech contract metadata.
        shuffle_train:
            When True, shuffle the train loader each epoch.
        seed:
            Seed for shuffling and preprocessing.

        Returns
        -------
        TorchLoaderBundle
            Speech loaders plus speech contract metadata."""
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
        """
        Fine-tune a tiny speech encoder + classifier head (finetune-lite).

        Builds speech loaders when missing. Honest alpha: not Whisper-scale FM
        training from scratch. Requires ``buildml[torch]``.
        Delegates to :func:`buildml.dl.train.train_supervised_module` after building
        a speech classifier module.

        Parameters
        ----------
        epochs:
            Number of training epochs.
        learning_rate:
            Optimizer learning rate.
        device:
            Compute device (``cpu``, ``cuda``, ``mps``, or ``auto``).
        freeze_encoder:
            When True, freeze the speech encoder during finetuning.
        audio_column:
            Optional audio column when loaders must be built.
        batch_size:
            Minibatch size when loaders must be built.
        sample_rate:
            Target sample rate when loaders must be built.
        max_samples:
            Maximum samples per example when loaders must be built.
        source_sample_rate:
            Optional source sample rate when loaders must be built.
        normalize_audio:
            When True, normalize audio amplitude on train only.
        encoder_dim:
            Encoder embedding dimension for the speech classifier.
        seed:
            Seed for shuffling and training.

        Returns
        -------
        Session
            ``self`` with ``dl_train_result`` attached for chaining."""
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
        """
        ASR transcription for an audio feature column.

        ``backend="stub"`` is CI-safe. ``backend="transformers"`` requires
        ``buildml[speech]`` and may download Whisper-class weights. Integration
        path only — not FM training from scratch.
        Delegates to :func:`buildml.dl.speech.transcribe_from_dataset`.

        Parameters
        ----------
        audio_column:
            Audio feature column to transcribe.
        backend:
            ASR backend (``stub`` or ``transformers``).
        model_id:
            Optional Hugging Face model id for transformers backend.
        sample_rate:
            Target audio sample rate for decoding.
        max_samples:
            Maximum audio samples per row.
        source_sample_rate:
            Optional source sample rate before resampling.
        partition:
            Dataset partition to transcribe (``all`` by default).

        Returns
        -------
        SpeechTranscribeResult
            Transcripts, model metadata, and row counts.

        Raises
        ------
        ValidationError
            When no dataset is attached to the Session."""
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
        """
        Launch BuildML managed serving for a pipeline or TorchScript artifact.

        Requires ``buildml[serve]``. Defaults to localhost bind. Optional
        ``api_keys`` enables Bearer / ``X-API-Key`` middleware (still not a managed
        IAM / cloud auth product). Non-loopback binds without ``api_keys`` raise
        unless ``allow_insecure_public_bind=True``. Optional ``ssl_certfile`` /
        ``ssl_keyfile`` enable local uvicorn HTTPS (library-owned; not managed
        certs). Prefer TLS at a reverse proxy for non-local exposure. When ``path``
        is omitted and ``kind="pipeline"``, uses the last saved pipeline path
        recorded on the Session if available.

        Not registered as an AI tool — CLI / Session-primary by design.
        Delegates to :func:`buildml.serving.launch.serve_bundle`.

        Parameters
        ----------
        path:
            Optional artifact path; inferred for pipelines when omitted.
        kind:
            Artifact kind (``pipeline`` or ``torchscript``).
        host:
            Bind host address.
        port:
            Bind port number.
        title:
            Service title shown in OpenAPI metadata.
        blocking:
            When True, block until the server stops.
        api_keys:
            Optional API keys enabling Bearer / header auth middleware.
        allow_insecure_public_bind:
            When True, permit non-loopback binds without API keys.
        ssl_certfile:
            Optional TLS certificate file for local HTTPS.
        ssl_keyfile:
            Optional TLS private key file for local HTTPS.

        Returns
        -------
        ServeHandle
            Running server handle with URL and lifecycle controls.

        Raises
        ------
        ValidationError
            When no resolvable artifact path is available."""
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
        """
        Load a curated pretrained vision/audio/speech backbone (integration hook).

        Delegates to :func:`buildml.dl.zoo.load_pretrained_backbone` and stores the
        backbone on the Session for downstream head attachment.

        Parameters
        ----------
        modality:
            Backbone modality (``vision``, ``audio``, or ``speech``).
        architecture:
            Optional architecture identifier within the curated zoo.
        weights:
            Weight source (``none``, ``mock``, or ``pretrained``).
        freeze:
            When True, freeze backbone parameters by default.
        seed:
            Seed for mock-weight initialization.
        model_id:
            Optional Hugging Face or zoo model identifier.

        Returns
        -------
        PretrainedBackbone
            Loaded backbone metadata and module shell."""
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
        """
        Attach a classification head to the Session pretrained backbone.

        Delegates to :func:`buildml.dl.zoo.attach_backbone_head` using the backbone
        stored by :meth:`load_pretrained_backbone`.

        Parameters
        ----------
        n_classes:
            Number of output classes for the attached head.
        freeze_backbone:
            Optional override for whether the backbone stays frozen.

        Returns
        -------
        BackboneHeadBundle
            Combined backbone+head module metadata.

        Raises
        ------
        ValidationError
            When no backbone is loaded or ``n_classes`` is invalid."""
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
        """
        Score ASR hypotheses vs references (WER/CER); reuse last transcription texts.

        Delegates to :func:`buildml.dl.speech.evaluate_asr`. When ``hypotheses`` is
        omitted, reuses texts from the prior :meth:`transcribe_speech` result.

        Parameters
        ----------
        hypotheses:
            Optional hypothesis strings; inferred from Session when omitted.
        references:
            Reference transcript strings aligned with hypotheses.
        lowercase:
            When True, lowercase strings before WER/CER scoring.

        Returns
        -------
        AsrEvalResult
            WER/CER metrics and scoring metadata.

        Raises
        ------
        ValidationError
            When hypotheses are missing and no transcription result exists."""
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
        """
        Pack a TorchScript artifact into a TorchServe-ready directory layout.

        Delegates to :func:`buildml.dl.packaging.pack_torchserve_model`. Uses the
        last TorchScript export on the Session when ``torchscript_path`` is omitted.

        Parameters
        ----------
        output_dir:
            Destination directory for the TorchServe model store layout.
        torchscript_path:
            Optional explicit TorchScript artifact path.
        model_name:
            Model name used in the TorchServe manifest.

        Returns
        -------
        PackagingResult
            Output paths and packaging disclosures.

        Raises
        ------
        ValidationError
            When no TorchScript path is available."""
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
        """
        Write a TensorRT ``trtexec`` plan next to a validated ONNX artifact.

        Delegates to :func:`buildml.dl.packaging.prepare_tensorrt_export_plan`.
        Uses the last ONNX export on the Session when ``onnx_path`` is omitted.

        Parameters
        ----------
        output_dir:
            Destination directory for the TensorRT export plan.
        onnx_path:
            Optional explicit ONNX artifact path.
        engine_name:
            Output TensorRT engine filename.
        fp16:
            When True, request FP16 optimization in the export plan.

        Returns
        -------
        PackagingResult
            Export plan paths and limitation disclosures.

        Raises
        ------
        ValidationError
            When no ONNX path is available."""
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
        """
        Emit a Kubernetes Job YAML for torchrun multi-node DDP (template only).

        Delegates to :func:`buildml.dl.k8s.write_torchrun_ddp_job`. This writes a
        starter manifest — not a managed cluster orchestrator.

        Parameters
        ----------
        path:
            Destination YAML file path.
        job_name:
            Kubernetes Job name.
        namespace:
            Kubernetes namespace.
        image:
            Container image for torchrun workers.
        nnodes:
            Number of nodes in the torchrun job.
        nproc_per_node:
            Processes launched per node.
        script_path:
            Training script path inside the container.
        cpu_request:
            CPU resource request per worker.
        memory_request:
            Memory resource request per worker.
        gpu_limit:
            GPU limit per worker.
        gpu_request:
            Optional GPU request per worker.
        service_account:
            Optional Kubernetes service account name.
        include_configmap:
            When True, include a starter ConfigMap manifest.

        Returns
        -------
        K8sManifestResult
            Written manifest paths and template limitations."""
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
        """
        Emit a Kubernetes Deployment+Service YAML for managed serve (template only).

        Delegates to :func:`buildml.dl.k8s.write_serve_deployment`. This writes a
        starter manifest — not a managed cluster orchestrator.

        Parameters
        ----------
        path:
            Destination YAML file path.
        name:
            Deployment and Service name.
        namespace:
            Kubernetes namespace.
        image:
            Container image for the serve deployment.
        replicas:
            Desired replica count.
        port:
            Service/container port for managed serve.
        cpu_request:
            CPU resource request per replica.
        memory_request:
            Memory resource request per replica.
        gpu_limit:
            Optional GPU limit per replica.
        service_account:
            Optional Kubernetes service account name.

        Returns
        -------
        K8sManifestResult
            Written manifest paths and template limitations."""
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
        """
        Domain-adapt / finetune-lite speech classify (not FM continued pretrain).

        Alias of :meth:`fit_speech_torch` with stronger domain-adapt disclosures
        recorded under the ``domain_adapt_speech_torch`` operation name.

        Parameters
        ----------
        epochs:
            Number of training epochs.
        learning_rate:
            Optimizer learning rate.
        device:
            Compute device (``cpu``, ``cuda``, ``mps``, or ``auto``).
        freeze_encoder:
            When True, freeze the speech encoder during adaptation.
        audio_column:
            Optional audio column when loaders must be built.
        batch_size:
            Minibatch size when loaders must be built.
        sample_rate:
            Target sample rate when loaders must be built.
        max_samples:
            Maximum samples per example when loaders must be built.
        source_sample_rate:
            Optional source sample rate when loaders must be built.
        normalize_audio:
            When True, normalize audio amplitude on train only.
        encoder_dim:
            Encoder embedding dimension for the speech classifier.
        seed:
            Seed for shuffling and training.

        Returns
        -------
        Session
            ``self`` with ``dl_train_result`` attached for chaining."""
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
        """Return the report from the most recent speech transcription.

        Stored on Session after :meth:`transcribe_speech` so downstream calls can
        reuse transcripts without re-running ASR.

        Returns
        -------
        SpeechTranscribeResult or None
            ``None`` until :meth:`transcribe_speech` has run."""
        return self._dl_speech_result

    @property
    def dl_backbone(self) -> Any | None:
        """Return the pretrained backbone loaded by the most recent zoo call.

        Stored on Session after :meth:`load_pretrained_backbone` for head attachment
        or finetune-lite workflows.

        Returns
        -------
        PretrainedBackbone or None
            ``None`` until :meth:`load_pretrained_backbone` has run."""
        return self._dl_backbone

    @property
    def dl_backbone_head(self) -> Any | None:
        """Return the backbone-plus-head bundle from the most recent attach.

        Stored on Session after :meth:`attach_backbone_head` for training or export.

        Returns
        -------
        BackboneHeadBundle or None
            ``None`` until :meth:`attach_backbone_head` has run."""
        return self._dl_backbone_head

    @property
    def dl_asr_eval(self) -> Any | None:
        """Return WER/CER metrics from the most recent ASR evaluation.

        Stored on Session after :meth:`evaluate_asr` for reporting and comparison.

        Returns
        -------
        AsrEvalResult or None
            ``None`` until :meth:`evaluate_asr` has run."""
        return self._dl_asr_eval

    @property
    def dl_train_result(self) -> TrainResult | None:
        """Return the last Torch training result from fit or bundle load.

        Stored on Session after :meth:`fit_torch`, :meth:`fit_speech_torch`,
        :meth:`domain_adapt_speech_torch`, or :meth:`load_torch_bundle`.

        Returns
        -------
        TrainResult or None
            ``None`` until a torch trainer has been fit or loaded."""
        return self._dl_train_result

    @property
    def dl_cv_result(self) -> TorchCVResult | None:
        """Return the last fold-local Torch cross-validation result.

        Stored on Session after :meth:`cross_validate_torch` for model selection review.

        Returns
        -------
        TorchCVResult or None
            ``None`` until :meth:`cross_validate_torch` has run."""
        return self._dl_cv_result

    @property
    def dl_search_result(self) -> Any | None:
        """Return the last inner-fold Torch hyperparameter search result.

        Stored on Session after :meth:`search_torch` for reviewing best params.

        Returns
        -------
        TorchSearchResult or None
            ``None`` until :meth:`search_torch` has run."""
        return self._dl_search_result

    @property
    def dl_nested_cv_result(self) -> Any | None:
        """Return the last nested Torch CV result with inner search.

        Stored on Session after :meth:`nested_cv_torch` for unbiased performance review.

        Returns
        -------
        TorchNestedCVResult or None
            ``None`` until :meth:`nested_cv_torch` has run."""
        return self._dl_nested_cv_result

    @property
    def dl_export_result(self) -> Any | None:
        """Return the last TorchScript or ONNX export result.

        Stored on Session after :meth:`export_torch` for serving and packaging flows.

        Returns
        -------
        TorchExportResult or None
            ``None`` until :meth:`export_torch` has run."""
        return self._dl_export_result

    @property
    def dl_ddp_result(self) -> Any | None:
        """Return the last distributed data parallel training result.

        Stored on Session after :meth:`fit_torch_ddp` for multi-GPU run review.

        Returns
        -------
        DDPTrainResult or None
            ``None`` until :meth:`fit_torch_ddp` has run."""
        return self._dl_ddp_result

    def torch_training_curve(self) -> TrainingCurveReport:
        """
        Return structured training-curve teaching data for the last Torch run.

        Requires a prior :meth:`fit_torch` / :meth:`load_torch_bundle`. Torch-free
        to read once :attr:`dl_train_result` exists.
        Delegates to :func:`buildml.dl.curves.build_training_curve`.

        Parameters
        ----------
        Returns
        -------
        TrainingCurve
            Epoch-wise loss/metric series for visualization or reporting.

        Raises
        ------
        ValidationError
            When no torch trainer exists on the Session."""
        return dl_ops.torch_training_curve(self)

    def evaluate_torch(
        self,
        *,
        partition: Literal["train", "validation", "test"] = "test",
        device: str | None = None,
    ) -> DLEvaluateResult:
        """
        Evaluate the last Torch trainer on a named partition.

        Requires ``pip install 'buildml[torch]'``. Uses loaders from
        :meth:`make_torch_loaders` (rebuilds them if missing).
        Delegates to :func:`buildml.dl.metrics.evaluate_module`.

        Parameters
        ----------
        partition:
            Partition to evaluate (``train``, ``validation``, or ``test``).
        device:
            Optional device override for evaluation.

        Returns
        -------
        TorchEvalResult
            Partition metrics for the trained module.

        Raises
        ------
        ValidationError
            When no torch trainer exists or non-tabular loaders are missing."""
        return dl_ops.evaluate_torch(self, partition=partition, device=device)

    def save_torch_bundle(self, path: str | Path) -> Path:
        """
        Persist the last Torch trainer as ``buildml.torch_bundle.v1``.

        Distinct from Session checkpoints and classical pipeline bundles.
        See :data:`buildml.dl.checkpoint.CHECKPOINT_BOUNDARY`.
        Delegates to :func:`buildml.dl.checkpoint.save_torch_bundle`.

        Parameters
        ----------
        path:
            Destination directory for the bundle (created if missing).

        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.

        Raises
        ------
        ValidationError
            When no torch trainer exists on the Session."""
        return dl_ops.save_torch_bundle(self, path=path)

    def load_torch_bundle(
        self,
        path: str | Path,
        module: Any,
        *,
        map_location: str | None = None,
    ) -> Session:
        """
        Load a Torch trainer bundle into this Session.

        Restores weights plus optional ``multimodal_preprocess`` meta (frozen
        image/audio stats and layout). Does **not** rebuild DataLoaders — remake
        multimodal/text loaders before fit/evaluate/export.
        Delegates to :func:`buildml.dl.checkpoint.load_torch_bundle`.

        Parameters
        ----------
        path:
            Bundle directory with ``meta.json`` and ``trainer.pt``.
        module:
            Compatible ``nn.Module`` shell that receives ``load_state_dict``.
        map_location:
            Optional device for ``torch.load`` (default CPU).

        Returns
        -------
        Session
            ``self`` with ``dl_train_result`` attached for chaining."""
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
        """
        Load a text corpus for the RAG path (requires ``buildml[rag]``).

        Provide a file/directory ``source``, an in-memory document sequence, or
        ``text_column`` to bridge the current Session frame. Never silently
        indexes every column. Delegates to :mod:`buildml.rag.corpus`. Distinct
        from classical ingest.

        Parameters
        ----------
        source:
            Optional path, directory, or in-memory document sequence.
        text_column:
            Optional Session frame column to ingest as documents.
        id_column:
            Optional document id column when using ``text_column``.
        glob:
            Glob pattern when ``source`` is a directory (``*.txt`` by default).
        encoding:
            Text encoding for file-based sources.
        role:
            Corpus role (``index`` or ``eval_only``).

        Returns
        -------
        Session
            ``self`` with RAG corpus attached for chaining.

        Raises
        ------
        ValidationError
            When inputs are missing or ``text_column`` is used without a dataset."""
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
        """
        Chunk the active RAG corpus (fixed or recursive strategy).

        ``strategy="recursive"`` splits on paragraph/line/sentence boundaries before
        applying size/overlap (LangChain/LlamaIndex parity). Requires ``buildml[rag]``.
        Delegates to :func:`buildml.rag.chunk.chunk_documents`.

        Parameters
        ----------
        size:
            Target chunk size in characters or tokens.
        overlap:
            Overlap between consecutive chunks.
        strategy:
            Chunking strategy (``fixed`` or ``recursive``).

        Returns
        -------
        Session
            ``self`` with chunk result attached for chaining.

        Raises
        ------
        ValidationError
            When no RAG corpus exists on the Session."""
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
        """
        Embed chunks and build the default NumPy cosine index (requires ``buildml[rag]``).

        Refuses corpora that contain ``eval_only`` documents (:class:`LeakageError`).
        Default embedder is ``auto``: sentence-transformers when ``buildml[rag]`` is
        installed, else explicit hashing fallback with disclosure.
        Pass ``embedder="hashing"`` for deterministic CI / lexical-only paths.
        ``device`` applies to sentence-transformer backends; hashing stays CPU-only.
        Delegates to :func:`buildml.rag.index.build_index`.

        Parameters
        ----------
        embedder:
            Embedder id or ``auto`` / ``hashing`` sentinel.
        chunk_size:
            Optional chunk size override before indexing.
        chunk_overlap:
            Optional chunk overlap override before indexing.
        chunk_strategy:
            Optional chunk strategy override before indexing.
        device:
            Optional device for sentence-transformer embedders.

        Returns
        -------
        Session
            ``self`` with RAG index attached for chaining.

        Raises
        ------
        ValidationError
            When no RAG corpus exists on the Session."""
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
        """
        Retrieve ranked chunks (dense / BM25 / hybrid) against the active RAG index.

        Defaults: ``mode="hybrid"`` (BM25 + dense RRF) when ``buildml[rag]`` is installed,
        else ``mode="dense"``. Metadata filters and cross-encoder rerank are opt-in.
        Delegates to :func:`buildml.rag.retrieve.retrieve`.

        Parameters
        ----------
        query:
            Natural-language query string.
        k:
            Number of chunks to retrieve.
        mode:
            Optional retrieve mode override (``dense``, ``bm25``, ``hybrid``).
        fusion:
            Optional fusion strategy for hybrid retrieval.
        filters:
            Optional metadata filters applied before ranking.
        rerank:
            Optional reranker toggle or model identifier.
        config:
            Optional full :class:`~buildml.rag.types.RetrieveConfig` override.

        Returns
        -------
        RetrieveResult
            Ranked chunks, scores, and retrieve provenance.

        Raises
        ------
        ValidationError
            When no RAG index exists on the Session."""
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
        """
        Score retrieval with gold qrels (recall@k, MRR, nDCG@k, hit-rate@k).

        ``relevance_mode="document"`` (default) scores parent ``doc_id`` hits;
        ``"chunk"`` scores ``chunk_id`` labels. Requires ``buildml[rag]``.
        Delegates to :func:`buildml.rag.evaluate.evaluate_retrieval`.

        Parameters
        ----------
        qrels:
            Gold relevance judgments mapping queries to relevant ids.
        k:
            Cutoff k for retrieval metrics.
        relevance_mode:
            Whether qrels label documents or chunks.
        mode:
            Optional retrieve mode override for evaluation queries.
        retrieve_config:
            Optional retrieve configuration override.

        Returns
        -------
        RagEvalResult
            Aggregate retrieval metrics and per-query summaries.

        Raises
        ------
        ValidationError
            When no RAG index exists on the Session."""
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
        """
        Retrieve (unless reusing the last retrieve) and generate a grounded answer.

        Requires an active RAG index and a chat provider. When ``provider`` is
        omitted, reuses ``:meth:`ai_configure```'s provider. For offline CI, pass
        :class:`buildml.rag.generate.EchoGroundedProvider` or a
        :class:`buildml.ai.provider.MockProvider`. Delegates to
        :func:`buildml.rag.generate.generate_grounded`.

        Parameters
        ----------
        query:
            Natural-language question to answer.
        k:
            Number of chunks to retrieve for grounding.
        provider:
            Optional chat provider; uses Session AI provider when omitted.
        mode:
            Optional retrieve mode passed through to retrieval.
        fusion:
            Optional fusion strategy for hybrid retrieval.
        filters:
            Optional metadata filters for retrieval.
        rerank:
            Optional reranker toggle or model identifier.
        retrieve_config:
            Optional retrieve configuration override.
        config:
            Optional :class:`~buildml.rag.types.GenerateConfig` override.
        use_last_retrieve:
            When True, reuse the prior :meth:`rag_retrieve` result.

        Returns
        -------
        GenerateResult
            Answer text, citations (source ids / chunk / doc), and retrieve provenance.

        Raises
        ------
        ValidationError
            When no RAG index or provider is available, or reuse is requested
            without a prior retrieve result."""
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
        """
        Upsert documents or chunks into the active RAG index without a full rebuild.

        Replaces existing ``chunk_id`` rows and re-embeds only new/changed text.
        Delegates to the active index object's upsert methods.

        Parameters
        ----------
        documents:
            Optional new or updated documents to upsert.
        chunks:
            Optional pre-chunked rows to upsert (mutually exclusive with
            ``documents``).
        chunk:
            When True and ``documents`` is supplied, chunk before upserting.

        Returns
        -------
        Session
            ``self`` with updated index and chunk state attached.

        Raises
        ------
        ValidationError
            When no RAG index exists or neither ``documents`` nor ``chunks`` is
            supplied."""
        return rag_ops.rag_upsert(self, documents=documents, chunks=chunks, chunk=chunk)

    def rag_delete(
        self,
        *,
        chunk_ids: Sequence[str] | None = None,
        doc_ids: Sequence[str] | None = None,
    ) -> Session:
        """
        Delete chunks by id and/or parent document id from the active RAG index.

        Removes matching rows from the in-memory index and refreshes Session chunk
        state without requiring a full rebuild.

        Parameters
        ----------
        chunk_ids:
            Optional chunk identifiers to delete.
        doc_ids:
            Optional parent document identifiers whose chunks should be deleted.

        Returns
        -------
        Session
            ``self`` with updated index and chunk state attached.

        Raises
        ------
        ValidationError
            When no RAG index exists on the Session."""
        return rag_ops.rag_delete(self, chunk_ids=chunk_ids, doc_ids=doc_ids)

    @property
    def rag_index_result(self) -> IndexResult | None:
        """Return the index metadata from the most recent embed-and-index call.

        Stored on Session after :meth:`rag_embed_and_index` or :meth:`load_rag_bundle`.

        Returns
        -------
        IndexResult or None
            ``None`` until :meth:`rag_embed_and_index` or :meth:`load_rag_bundle` has run."""
        return self._rag_index_result

    @property
    def rag_retrieve_result(self) -> RetrieveResult | None:
        """Return the ranked chunks from the most recent retrieval call.

        Stored on Session after :meth:`rag_retrieve` or a generate call that retrieved.

        Returns
        -------
        RetrieveResult or None
            ``None`` until :meth:`rag_retrieve` or :meth:`rag_generate` has run."""
        return self._rag_retrieve_result

    @property
    def rag_eval_result(self) -> RagEvalResult | None:
        """Return retrieval metrics from the most recent RAG evaluation.

        Stored on Session after :meth:`rag_evaluate` for offline retrieval QA.

        Returns
        -------
        RagEvalResult or None
            ``None`` until :meth:`rag_evaluate` has run."""
        return self._rag_eval_result

    @property
    def rag_generate_result(self) -> GenerateResult | None:
        """Return the grounded answer from the most recent RAG generate call.

        Stored on Session after :meth:`rag_generate` for audit and downstream reuse.

        Returns
        -------
        GenerateResult or None
            ``None`` until :meth:`rag_generate` has run."""
        return self._rag_generate_result

    def save_rag_bundle(self, path: str | Path) -> Path:
        """
        Persist the active RAG index as ``buildml.rag_bundle.v1``.

        Distinct from Session checkpoints and Torch trainer bundles.
        See :data:`buildml.rag.checkpoint.CHECKPOINT_BOUNDARY`.
        Delegates to :func:`buildml.rag.checkpoint.save_rag_bundle`.

        Parameters
        ----------
        path:
            Destination directory for the bundle (created if missing).

        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.

        Raises
        ------
        ValidationError
            When no RAG index exists on the Session."""
        return rag_ops.save_rag_bundle(self, path=path)

    def load_rag_bundle(self, path: str | Path) -> Session:
        """
        Load a RAG bundle into this Session (requires ``buildml[rag]``).

        Delegates to :func:`buildml.rag.checkpoint.load_rag_bundle` and restores
        index, chunk, and index-result state on the Session.

        Parameters
        ----------
        path:
            Path to a ``buildml.rag_bundle.v1`` directory.

        Returns
        -------
        Session
            ``self`` with RAG index attached for chaining."""
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
        provider:
            Provider name (currently ``"openai"`` for OpenAI-compatible APIs,
            or ``"mock"`` for CI testing without real keys).
        model:
            Model identifier for the provider.
        api_key:
            API key (if None, reads from ``api_key_env`` environment variable).
        api_key_env:
            Environment variable name for the API key.
        egress_level:
            Default egress level: ``"schema_only"``, ``"stats_only"`` (default),
            ``"redacted_sample"``, or ``"full_sample"``.
        max_iterations:
            Maximum tool iterations per AI call (default 10).
        max_tokens:
            Optional token budget limit across all AI calls.
        max_cost_usd:
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
        level:
            Override egress level for this preview (``"schema_only"``,
            ``"stats_only"``, ``"redacted_sample"``, ``"full_sample"``).
        allow_columns:
            Explicit allowlist of columns to include.
        deny_columns:
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
        question:
            The question or goal to preview.
        level:
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
        question:
            The question to ask about the workflow.
        level:
            Override egress level for this call.
        confirm:
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
        goal:
            The workflow goal to plan for.
        level:
            Override egress level for this call.
        confirm:
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
        tool:
            Name of the tool to execute (must be in the allowed registry).
        params:
            Tool arguments as a dictionary.
        confirm:
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
        plan:
            The PlanResult to execute. If None, uses the last ai_plan result.
        confirmations:
            Dict mapping step_index -> True/False for confirmation decisions.
            Steps not in the dict use default confirmation behavior.
        auto_confirm_read_only:
            If True (default), auto-confirm read-only operations.
        stop_on_error:
            If True (default), stop execution on first error.
        stop_on_unconfirmed:
            If True (default), stop at steps requiring unconfirmed confirmation.
        max_steps:
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
        """
        Explicit autonomy mode with hard caps (see :mod:`buildml.ai.autonomous`).

        Records the operation on Session history and returns the result for downstream chaining.

        Parameters
        ----------
        goal:
            Workflow goal for planning or autonomous execution.
        plan:
            Structured plan object from a prior planning call.
        confirm_autonomy:
            When True, require an explicit confirmation token before autonomous mutating AI actions.
        max_steps:
            Hard cap on advisor/planner tool-calling rounds to bound cost and loops.
        tool_allowlist:
            Allowlist of AI tool names the planner/advisor may invoke for this call.
        allow_destructive:
            When True, permit destructive Session mutations (drops/overwrites) from AI plan execution.
        provider_plan:
            Provider-side plan/config object used when executing a structured AI plan.

        Returns
        -------
        Any
            Structured domain result recorded on the Session for follow-up evaluate/explain/export steps."""
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
        """
        Save the AI transcript to a JSON file (secrets redacted by default).

        Transcripts record conversation history, tool calls, and egress

        manifests. API keys and raw data are redacted before saving.

        Parameters
        ----------
            path:
            Output file path.
            redact:
            If True (default), redact potential secrets before saving.
        path:
            Filesystem path for load or save.
        redact:
            When True, redact secrets before persisting transcripts.

        Returns
        -------
        Path
        The resolved output path."""
        return ai_ops.save_ai_transcript(self, path=path, redact=redact)

    def load_ai_transcript(self, path: str | Path) -> Session:
        """
        Load an AI transcript for resume or audit.

        Records the operation on Session history and returns the result for downstream chaining.

        Parameters
        ----------
            path:
            Input file path.
        path:
            Filesystem path for load or save.

        Returns
        -------
        Session
        Self for chaining."""
        return ai_ops.load_ai_transcript(self, path=path)

    @property
    def ai_result(
        self,
    ) -> AdvisorResult | PlanResult | ExecutorResult | PlanExecutionResult | None:
        """Return the most recent AI advisor, plan, execute, or run-plan result.

        Updated by :meth:`ai_advisor`, :meth:`ai_plan`, :meth:`ai_execute`,
        :meth:`ai_run_plan`, and :meth:`ai_run_autonomous`.

        Returns
        -------
        AdvisorResult, PlanResult, ExecutorResult, PlanExecutionResult, or None
            ``None`` until an AI operation has produced a result."""
        return self._ai_result

    @property
    def ai_transcript(self) -> TranscriptStore | None:
        """Return the active AI transcript store for this Session.

        Created by :meth:`ai_configure` and populated by AI calls; reload with
        :meth:`load_ai_transcript`.

        Returns
        -------
        TranscriptStore or None
            ``None`` until :meth:`ai_configure` or :meth:`load_ai_transcript` has run."""
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
        """Draw the standard diagnostic charts for a fitted model, in one call.

        Numbers tell you how well a model does; pictures tell you how it fails.
        This assembles the panels worth looking at for the model you actually
        fitted, skipping the ones that do not apply rather than erroring.

        Depending on the task and what the estimator supports, the board can
        include the confusion matrix or residual plots, the ROC and
        precision-recall curves, a calibration curve showing whether predicted
        probabilities match observed frequencies, the precision-recall
        trade-off across thresholds, a learning curve indicating whether more
        data would help, and permutation importance ranking the features the
        model relied on.

        Parameters
        ----------
        partition:
            Which rows the diagnostics describe. ``'test'`` shows the
            behaviour you will get in deployment; ``'train'`` next to it
            reveals overfitting.
        include_learning_curve:
            Add the learning curve. It refits the model on increasing
            subsamples, so it is the slowest panel — turn it off for a quick
            look. Read it as: converged curves mean more data will not help,
            a persistent gap means it will.
        include_importance:
            Add permutation importance, which measures how much the score drops
            when a feature's values are shuffled. Slower than reading the
            model's built-in importances, but model-agnostic and harder to
            mislead.
        n_importance_repeats:
            How many times each feature is shuffled. More repeats give a
            steadier ranking at proportional cost; the default trades a little
            noise for speed.
        learning_curve_cv:
            Fold count used at each learning-curve sample size. Kept low by
            default because the curve refits at every point.
        export_figures:
            Directory to write the individual figures into. ``None`` keeps them
            in memory only.
        export_html:
            Path for a single self-contained HTML page holding every panel —
            the artefact to attach to a review.
        show:
            Display the figures interactively, for notebook use.

        Returns
        -------
        ~buildml.model.plot_boards.PlotBoardReport
            The board: paths to any figures written, which panels were
            ``skipped`` and why, and an ``interpretation`` explaining what each
            panel shows. Also stored on :attr:`last_plot_board`.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            No model has been fitted, or no split exists.
        ~buildml.core.errors.MissingExtraError
            Plotting requires ``pip install 'buildml[viz]'``.

        Notes
        -----
        Delegates to :func:`buildml.model.plot_boards.build_eval_plot_board`.

        Panels degrade gracefully. A model without ``predict_proba`` has no
        ROC or calibration curve, and a multi-class target has no single
        precision-recall trade-off; those panels are listed in ``skipped``
        with a reason rather than raising.

        Examples
        --------
        >>> board = session.eval_plots(export_html="reports/board.html")  # doctest: +SKIP
        >>> board.skipped  # doctest: +SKIP
        ['roc_curve: estimator has no predict_proba']

        See Also
        --------
        Session.evaluate : Metrics, with these plots available inline.
        Session.calibration : The calibration panel on its own.
        Session.learning_curve : The learning-curve panel on its own.
        """
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
        """The most recent diagnostic plot board.

        Set by :meth:`eval_plots`, and also by :meth:`evaluate` when it was
        asked to produce plots. Holds the figure paths, the panels that were
        skipped along with the reason, and the written interpretation of each
        panel.

        ``None`` when no plots have been produced in this session.
        """
        return self._last_plot_board

    def compare_models(
        self,
        estimators: dict[str, Any],
        *,
        task: Literal["classification", "regression", "auto"] = "auto",
        partition: Literal["train", "validation", "test"] = "test",
        ranking_metric: str | None = None,
    ) -> ModelComparison:
        """Try several models on the same data and rank what you get.

        "Which algorithm should I use?" has no answer in the abstract — it
        depends on your data, and the reliable way to find out is to try a few.
        This fits each estimator on the training rows, evaluates them all on
        the same partition, and returns them ranked, so the comparison is
        genuinely like-for-like.

        A sensible starting set is one linear model, one tree ensemble, and one
        gradient-boosting model. That covers very different assumptions about
        the data, and the spread between them tells you a lot: if the linear
        model keeps up, your relationships are mostly additive and you should
        probably prefer it for the interpretability.

        Parameters
        ----------
        estimators:
            Label to unfitted estimator instance. The labels are yours and
            appear in the ranking, so name them for what distinguishes them
            (``"rf_depth6"``) rather than by class.
        task:
            ``'classification'``, ``'regression'``, or ``'auto'`` to infer it
            from the target. Every estimator is treated as the same task.
        partition:
            Which rows to score on. Use ``'validation'`` while choosing —
            ranking candidates on ``'test'`` and then reporting the winner's
            test score overstates it, because the winner was selected using
            that very number.
        ranking_metric:
            Which metric orders the table. ``None`` uses the task default.
            Choose deliberately when errors are asymmetric: ranking by accuracy
            on imbalanced data will happily crown a model that never predicts
            the rare class.

        Returns
        -------
        ~buildml.model.compare.ModelComparison
            The ranked comparison, holding each model's metrics, the ordering,
            and the metric used to produce it.

        Raises
        ------
        ~buildml.core.errors.LeakageError
            No split exists, so there is nothing to compare on.
        ~buildml.core.errors.ValidationError
            ``estimators`` is empty, no target is assigned, or the features are
            not yet numeric and complete.

        Notes
        -----
        Each model is scored on a single fixed partition, so small differences
        between them are within noise. When two candidates finish close
        together, confirm with :meth:`cv_score` before declaring a winner —
        a one-point gap on a few hundred rows frequently reverses on a
        different split.

        Examples
        --------
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.linear_model import LogisticRegression
        >>> comparison = session.compare_models(
        ...     {
        ...         "logistic": LogisticRegression(max_iter=500),
        ...         "forest": RandomForestClassifier(random_state=0),
        ...     },
        ...     partition="validation",
        ... )  # doctest: +SKIP

        See Also
        --------
        Session.cv_score : Confirm a close result across several folds.
        Session.run_automl : Search a space of models rather than a shortlist.
        Session.fit_voting : Combine candidates instead of picking one.
        """
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
        """Score a model across several rotating holdouts, not just one.

        A single train/test split gives you one number, and that number depends
        on which rows happened to land in test. On a few thousand rows the
        swing between two random splits is easily a couple of percentage
        points — enough to pick the wrong model.

        Cross-validation removes that luck. The training rows are divided into
        ``cv`` folds; the model is fitted ``cv`` times, each time holding out a
        different fold and scoring on it. You end up with ``cv`` scores instead
        of one, and their spread is as informative as their average: a high
        mean with a wide spread means the result is fragile, not good.

        The session's test partition takes no part in any of this. Folds are
        cut from the training rows only, so test stays untouched for the final
        measurement.

        Parameters
        ----------
        estimator:
            An unfitted estimator instance. It is cloned for each fold, so the
            object you pass is never itself fitted and can be reused.
        task:
            ``'classification'``, ``'regression'``, or ``'auto'`` to infer it
            from the target.
        cv:
            How many folds, or a scikit-learn splitter object for full control.
            Five is the usual compromise: more folds train on more data per
            fold and give a less biased estimate, at proportionally more time.
        cv_strategy:
            How rows are assigned to folds when ``cv`` is a number. ``'auto'``
            reads the column roles and picks for you. ``'stratified'``
            preserves class balance in every fold, which matters for imbalanced
            classification. ``'group'`` keeps an entity's rows in the same fold
            — the cross-validation equivalent of :meth:`group_split`.
            ``'stratified_group'`` does both. ``'time'`` only ever trains on
            folds earlier than the one being scored. Choosing wrongly here
            recreates the leakage the split was designed to prevent.
        scoring_metric:
            Which metric the summary reports. ``None`` uses the task default.
        groups:
            Group labels aligned to the training rows, for the group-aware
            strategies. ``None`` uses the ``group``-role column.
        preprocess:
            A :class:`~buildml.preprocess.fold.PreprocessRecipe` refitted
            inside every fold. This is the leakage-correct way to include
            preprocessing in a cross-validated estimate: the scaler and encoder
            are learned from that fold's training rows and applied to its
            held-out rows, exactly as they would be in production.
        allow_session_global_preprocess:
            Permit cross-validation to proceed even though session-wide
            preprocessing already ran. Off by default, and the refusal is the
            point — see the note below.

        Returns
        -------
        ~buildml.model.selection.CVScoreResult
            Per-fold scores with their mean and standard deviation, plus an
            ``interpretation``, the ``limitations`` of the estimate, and
            ``recommendations``. Also stored on :attr:`last_cv`.

        Raises
        ------
        ~buildml.core.errors.LeakageError
            Session-wide preprocessing already ran and
            ``allow_session_global_preprocess`` was not set.
        ~buildml.core.errors.ValidationError
            No split exists, the requested strategy needs a role column that is
            not assigned, or a fold would be empty.

        Notes
        -----
        **Leakage:** If Session impute/encode/scale/text/reduce already ran, CV
        refuses unless ``allow_session_global_preprocess=True``. Prefer
        re-ingesting unpoisoned data, then fold-local recipes (including
        ``text`` and ``reduce``) for selection claims that include
        preprocessing. Custom transforms and resample stay Session-global.

        The reason for that refusal is worth understanding. Calling
        :meth:`scale` fits one scaler across all the training rows. If you then
        cross-validate, each fold's held-out rows were already involved in
        computing that scaler's mean, so every fold score is slightly
        optimistic. The recipe mechanism exists to avoid this: it defers
        preprocessing until the fold boundary is known. Overriding the refusal
        does not fix the estimate, it only silences the warning about it.

        Examples
        --------
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> result = session.cv_score(
        ...     RandomForestClassifier(random_state=0), cv=5, cv_strategy="stratified"
        ... )  # doctest: +SKIP
        >>> result.mean_metrics["accuracy"], result.std_metrics["accuracy"]  # doctest: +SKIP
        (0.884, 0.021)

        With preprocessing done correctly, inside each fold:

        >>> from buildml.preprocess.fold import PreprocessRecipe
        >>> recipe = PreprocessRecipe(impute="median", scale="standard")
        >>> result = session.cv_score(
        ...     RandomForestClassifier(), preprocess=recipe
        ... )  # doctest: +SKIP

        See Also
        --------
        Session.nested_cv_score : When you are also tuning hyperparameters.
        Session.grid_search : Search a space, using this scoring underneath.
        Session.evaluate : The single-holdout estimate this replaces.
        """
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
        """Estimate how well your *tuning procedure* generalises, not just one model.

        There is a subtle trap in the usual workflow. You run :meth:`grid_search`,
        it reports the cross-validated score of the winning configuration, and
        you quote that number as your expected performance. But the winner was
        chosen *because* it scored well on those folds, so its score is
        optimistically biased — you picked the luckiest configuration and then
        reported its luck as skill. On a large search space the inflation can be
        several points.

        Nested cross-validation removes the bias by giving the search its own
        private data. The rows are split into outer folds. Within each outer
        fold's training portion, an independent inner search picks the best
        configuration; that winner is then scored once on the outer fold's
        held-out rows, which the search never saw. Averaging those outer scores
        gives an honest estimate of what "run my tuning procedure on data like
        this" is worth.

        Note what is being estimated: the procedure, not a single model. Each
        outer fold may crown a different winner, and that is fine — the spread
        across folds tells you how stable your tuning is. To get a model to
        deploy, run :meth:`grid_search` (or a sibling) once on the full training
        set afterwards, and quote the nested score as its expected performance.

        Parameters
        ----------
        estimator:
            An unfitted estimator, cloned fresh for every candidate in both
            loops.
        param_grid:
            Exhaustive estimator search space, ``{"max_depth": [3, 5, 8]}``.
            Mutually exclusive with ``param_distributions``. Optional when you
            are only searching recipe knobs.
        param_distributions:
            Sampled estimator search space for a randomized inner search.
        recipe_grid:
            Search space over preprocessing knobs — ``select_k``, ``n_bins``,
            and friends — refit inside each fold. Requires ``preprocess``.
        recipe_distributions:
            Sampled counterpart to ``recipe_grid``.
        param_space:
            Optuna search space for the estimator, used when ``inner_search``
            is ``'optuna'``. Declare-style dicts also drive the evolutionary
            search. Optuna needs ``pip install 'buildml[optuna]'``.
        recipe_space:
            Optuna or evolutionary search space for the recipe knobs.
        inner_search:
            Which search runs inside each outer fold. ``'auto'`` infers it from
            which spaces you supplied. Note the cost: the inner search runs once
            per outer fold, so an exhaustive grid multiplies quickly.
        n_iter:
            Candidates sampled per outer fold when the inner search is
            randomized.
        n_trials:
            Optuna trials per outer fold; doubles as ``max_evaluations`` for the
            evolutionary search.
        population_size:
            Candidates per generation for the evolutionary inner search.
        n_generations:
            Generations the evolutionary inner search runs for.
        random_state:
            Seed for fold assignment and candidate sampling, so the estimate
            reproduces.
        task:
            ``'classification'``, ``'regression'``, or ``'auto'`` to infer from
            the target.
        outer_cv:
            Number of outer folds, or a scikit-learn splitter. These folds
            produce the reported estimate.
        inner_cv:
            Number of inner folds, or a splitter. Kept smaller than ``outer_cv``
            by default because it runs many more times.
        cv_strategy:
            How rows are assigned to folds. ``'stratified'`` preserves class
            balance, ``'group'`` keeps related rows together, ``'time'``
            respects chronology. ``'auto'`` picks from the data and roles.
        scoring_metric:
            Metric the inner search optimises and the outer loop reports.
            Defaults to a sensible choice for the task.
        groups:
            Group labels for the group-aware strategies.
        preprocess:
            A :class:`~buildml.preprocess.fold.PreprocessRecipe` refit inside
            every fold of both loops. This is what keeps preprocessing honest:
            imputation values and scalers are learned from fold-training rows
            only.
        allow_session_global_preprocess:
            Permit running against session-wide preprocessing that was fit
            before splitting. Off by default because it leaks; the guard exists
            for deliberate exceptions.
        warm_start_studies:
            Share one Optuna study across outer folds so later folds benefit
            from earlier trials. Faster, but the folds are no longer fully
            independent searches — the outer estimate stays valid because the
            outer-eval rows are still never scored during search.

        Returns
        -------
        ~buildml.model.selection.NestedCVResult
            ``mean_metrics`` and ``std_metrics`` hold the honest estimate and
            its fold-to-fold spread. ``outer_folds`` records each fold's chosen
            ``best_params`` and ``best_recipe_knobs``, which is where you look
            to judge whether tuning is stable or thrashing.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            No split exists, no search space was supplied, mutually exclusive
            spaces were combined, or recipe knobs were given without a recipe.
        ~buildml.core.errors.MissingExtraError
            Optuna was requested without ``buildml[optuna]`` installed.

        Notes
        -----
        **Cost:** total fits are roughly ``outer_cv × inner_cv × candidates``.
        With five outer folds, three inner folds, and fifty candidates that is
        750 fits. Start with a randomized inner search and small fold counts.

        **Leakage:** the session's test and validation partitions never enter
        either loop. Both loops draw only from training rows.

        **Reading the spread:** if ``std_metrics`` is large relative to
        ``mean_metrics``, the procedure is unstable — usually too small a
        dataset for the size of the search space.

        Examples
        --------
        >>> result = session.nested_cv_score(  # doctest: +SKIP
        ...     RandomForestClassifier(),
        ...     param_distributions={"max_depth": [3, 5, 8, None]},
        ...     inner_search="randomized",
        ...     n_iter=8,
        ... )
        >>> result.mean_metrics["accuracy"]  # doctest: +SKIP

        See Also
        --------
        Session.cv_score : Honest estimate for a single fixed configuration.
        Session.grid_search : The inner search, run once, to get a deployable model.
        """
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
        """Try every combination of the settings you list, and keep the best.

        Hyperparameters are the knobs you set before training — tree depth,
        regularisation strength, learning rate — and the right values depend on
        your data. Grid search takes the values you consider plausible, builds
        every combination of them, cross-validates each one on the training
        rows, and ranks the results.

        It is exhaustive, which is both its strength and its weakness. You are
        guaranteed to find the best point *in the grid you specified*, and you
        pay for the guarantee combinatorially: three parameters with four
        values each is 64 fits, times ``cv`` folds. Use it when the space is
        small or you already know roughly where to look. Use
        :meth:`randomized_search` or :meth:`optuna_search` when it is not.

        Recipe knobs can be searched alongside estimator parameters. Whether
        five features or fifty work better is a modelling decision like any
        other, and searching it inside the folds keeps the choice honest.

        Parameters
        ----------
        estimator:
            An unfitted estimator instance supplying the defaults that the grid
            overrides.
        param_grid:
            Parameter name to the list of values to try, for example
            ``{"max_depth": [3, 6, 12]}``. Names match what the estimator's
            ``set_params`` accepts, including nested ``step__param`` forms.
        recipe_grid:
            Preprocessing knobs to search the same way, such as
            ``{"select_k": [5, 10, 20]}``. Requires ``preprocess``.
        task:
            ``'classification'``, ``'regression'``, or ``'auto'``.
        cv:
            Fold count, or a scikit-learn splitter, used to score each
            configuration.
        cv_strategy:
            How folds are formed — see :meth:`cv_score`, which describes the
            same options and the same hazards.
        ranking_metric:
            Which metric decides the winner. ``None`` uses the task default.
            This choice *is* the objective you are optimising, so pick the one
            that reflects the cost of being wrong.
        groups:
            Group labels for the group-aware strategies. ``None`` uses the
            ``group``-role column.
        preprocess:
            Fold-local recipe refit inside each fold, so the tuning estimate is
            not inflated by preprocessing that saw the held-out rows.
        allow_session_global_preprocess:
            Proceed despite session-wide preprocessing having already run. See
            the leakage note on :meth:`cv_score`.
        refit:
            When True (the default), retrain the winning configuration on the
            whole training partition and install it as :attr:`fit_result`, so
            :meth:`predict` and :meth:`evaluate` immediately use the tuned
            model. Set False to inspect the ranking before committing.

        Returns
        -------
        ~buildml.model.selection.SearchResult
            The ranked search: every trial with its score, the
            ``best_params``, ``best_score`` and ``best_std``, the winner's full
            ``best_cv`` breakdown, and the ``refit_result`` when refitting was
            requested. ``to_frame()`` renders the trials as a DataFrame. Also
            stored on :attr:`last_search`.

        Raises
        ------
        ~buildml.core.errors.LeakageError
            Session-wide preprocessing already ran and was not explicitly
            allowed.
        ~buildml.core.errors.ValidationError
            Neither a parameter grid nor a recipe grid was supplied, a
            parameter name is not one the estimator accepts, or ``recipe_grid``
            was given without ``preprocess``.

        Notes
        -----
        Folds are cut from the training partition only; test never influences
        the ranking.

        The best cross-validation score is an optimistic estimate of the
        winner's true performance. Searching many configurations and reporting
        the maximum selects for luck as well as quality. Treat the tuned
        model's honest number as the one from :meth:`evaluate` on test, or from
        :meth:`nested_cv_score` if you want that without spending the test set.

        Examples
        --------
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> search = session.grid_search(
        ...     RandomForestClassifier(random_state=0),
        ...     {"max_depth": [3, 6, None], "min_samples_leaf": [1, 5]},
        ...     cv=5,
        ... )  # doctest: +SKIP
        >>> search.best_params  # doctest: +SKIP
        {'max_depth': 6, 'min_samples_leaf': 5}
        >>> search.to_frame().head()  # doctest: +SKIP

        See Also
        --------
        Session.randomized_search : Sample the space instead of enumerating it.
        Session.optuna_search : Let earlier trials guide later ones.
        Session.nested_cv_score : An unbiased estimate of a tuned model.
        """
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
        """Sample settings at random, which usually beats an exhaustive grid.

        The result is counter-intuitive but well established: given the same
        computing budget, randomly sampling a hyperparameter space typically
        finds a better configuration than exhaustively searching a grid.

        The reason is that parameters differ enormously in how much they
        matter. A grid of four values across three parameters spends 64 fits,
        but only ever tries four distinct values of the parameter that
        actually drives performance — the other two dimensions multiply the
        cost without adding resolution where it counts. Random sampling tries
        64 *different* values of every parameter, so the important one gets
        explored properly.

        You also gain control of the budget. ``n_iter`` sets the number of
        fits directly, so adding a parameter to explore costs nothing extra.

        Parameters
        ----------
        estimator:
            An unfitted estimator instance supplying the defaults being varied.
        param_distributions:
            Parameter name to either a list to choose uniformly from, or a
            ``scipy.stats`` distribution to draw from. Distributions are the
            better choice for continuous parameters, and log-uniform is the
            right shape for learning rates and regularisation strengths, where
            the interesting variation is in orders of magnitude.
        recipe_distributions:
            Preprocessing knobs sampled the same way. Requires ``preprocess``.
        n_iter:
            How many configurations to sample — your entire budget, in fits per
            fold. Start small to gauge the cost of one fit, then raise it.
        random_state:
            Seed for the sampling, so a search can be reproduced exactly.
        task:
            ``'classification'``, ``'regression'``, or ``'auto'``.
        cv:
            Fold count or splitter used to score each sampled configuration.
        cv_strategy:
            How folds are formed — see :meth:`cv_score`.
        ranking_metric:
            Which metric decides the winner. ``None`` uses the task default.
        groups:
            Group labels for the group-aware strategies.
        preprocess:
            Fold-local recipe refit inside each fold.
        allow_session_global_preprocess:
            Proceed despite session-wide preprocessing. See :meth:`cv_score`.
        refit:
            Retrain the winner on the full training partition and install it as
            :attr:`fit_result`. On by default.

        Returns
        -------
        ~buildml.model.selection.SearchResult
            The ranked trials, ``best_params``, ``best_score``, the winner's
            ``best_cv`` breakdown, and the refit model when requested. Also
            stored on :attr:`last_search`.

        Raises
        ------
        ~buildml.core.errors.LeakageError
            Session-wide preprocessing already ran and was not explicitly
            allowed.
        ~buildml.core.errors.ValidationError
            No search space was supplied, a parameter name is not one the
            estimator accepts, or recipe distributions were given without
            ``preprocess``.

        Notes
        -----
        Same leakage contract as :meth:`grid_search`: folds stay inside train;
        the winner may be refit onto the full training partition.

        Examples
        --------
        >>> from scipy.stats import loguniform, randint
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> search = session.randomized_search(
        ...     RandomForestClassifier(random_state=0),
        ...     {"max_depth": randint(2, 20), "min_samples_leaf": randint(1, 30)},
        ...     n_iter=40,
        ... )  # doctest: +SKIP

        A learning rate should be sampled across magnitudes, not linearly:

        >>> space = {"learning_rate": loguniform(1e-3, 1e-1)}  # doctest: +SKIP

        See Also
        --------
        Session.grid_search : Exhaustive, for small well-understood spaces.
        Session.optuna_search : Adaptive, for larger budgets.
        """
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
        """Search adaptively, letting each trial learn from the ones before it.

        Grid and random search are memoryless: the hundredth configuration is
        chosen with no knowledge of the ninety-nine already scored. Optuna
        instead builds a model of which regions of the space produce good
        results and concentrates its sampling there, while still exploring
        enough to avoid getting stuck.

        That adaptivity pays off as the budget grows. For a handful of trials
        it behaves much like random search; for fifty or more it usually
        reaches a better configuration, because it stops re-testing regions it
        has already established are poor. It also handles conditional and
        mixed discrete/continuous spaces naturally, which grids do badly.

        Requires ``pip install 'buildml[optuna]'``.

        Parameters
        ----------
        estimator:
            An unfitted estimator instance supplying the defaults being tuned.
        param_space:
            The space to search, given either as a callable taking an Optuna
            ``trial`` and returning a parameter dict — full control, including
            parameters that only exist depending on others — or as a
            declarative mapping using ``float``, ``int``, and ``categorical``
            entries.
        recipe_space:
            Preprocessing knobs described the same way. Requires
            ``preprocess``.
        n_trials:
            How many configurations to evaluate. This is where Optuna earns its
            keep; below roughly twenty trials it has little history to learn
            from.
        random_state:
            Seed for the sampler, making a search reproducible.
        task:
            ``'classification'``, ``'regression'``, or ``'auto'``.
        cv:
            Fold count or splitter used to score each trial.
        cv_strategy:
            How folds are formed — see :meth:`cv_score`.
        ranking_metric:
            The metric Optuna optimises. ``None`` uses the task default.
        groups:
            Group labels for the group-aware strategies.
        preprocess:
            Fold-local recipe refit inside each fold.
        allow_session_global_preprocess:
            Proceed despite session-wide preprocessing. See :meth:`cv_score`.
        refit:
            Retrain the winner on the full training partition and install it as
            :attr:`fit_result`. On by default.

        Returns
        -------
        ~buildml.model.selection.SearchResult
            The ranked trials, ``best_params``, ``best_score``, the winner's
            ``best_cv`` breakdown, and the underlying Optuna ``study`` for
            further analysis. Also stored on :attr:`last_search`.

        Raises
        ------
        ~buildml.core.errors.MissingExtraError
            Optuna is not installed.
        ~buildml.core.errors.LeakageError
            Session-wide preprocessing already ran and was not explicitly
            allowed.
        ~buildml.core.errors.ValidationError
            No search space was supplied, or a recipe space was given without
            ``preprocess``.

        Notes
        -----
        Folds are cut from the training partition only; test never influences
        the search.

        The returned ``study`` supports Optuna's own analysis tools, including
        parameter-importance ranking — often more useful than the winning
        configuration itself, since it tells you which knobs are worth tuning
        at all next time.

        Examples
        --------
        Declarative form, which covers most cases:

        >>> space = {
        ...     "max_depth": {"type": "int", "low": 2, "high": 20},
        ...     "learning_rate": {"type": "float", "low": 1e-3, "high": 0.3, "log": True},
        ... }
        >>> search = session.optuna_search(
        ...     estimator, param_space=space, n_trials=60
        ... )  # doctest: +SKIP

        Callable form, when one parameter depends on another:

        >>> def space(trial):  # doctest: +SKIP
        ...     kind = trial.suggest_categorical("kernel", ["linear", "rbf"])
        ...     params = {"kernel": kind}
        ...     if kind == "rbf":
        ...         params["gamma"] = trial.suggest_float("gamma", 1e-4, 1.0, log=True)
        ...     return params

        See Also
        --------
        Session.randomized_search : Simpler, and adequate for small budgets.
        Session.evolutionary_search : Population-based, no extra dependency.
        Session.run_automl : Search over model families as well as settings.
        """
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
        """Evolve a population of configurations across generations.

        This borrows from natural selection. A population of random
        configurations is scored; the better ones are more likely to be chosen
        as parents; parents are combined to produce offspring; offspring are
        randomly perturbed; the best few survive untouched. Repeat for
        ``n_generations``.

        The advantage over random sampling is recombination. If one
        configuration happens to have a good tree depth and another a good
        learning rate, crossover can produce a child with both — something
        independent sampling can only stumble on. That makes evolutionary
        search well suited to spaces where parameters interact.

        Compared with :meth:`optuna_search` it needs no extra dependency (the
        algorithm is implemented here in NumPy) and it explores more broadly,
        since a whole population advances at once rather than a single
        adaptive sampler. It typically needs more total evaluations to reach
        the same quality.

        The total number of fits is roughly ``population_size *
        n_generations``, each multiplied by ``cv`` folds — worth computing
        before you start.

        Parameters
        ----------
        estimator:
            An unfitted estimator instance supplying the defaults being tuned.
        param_space:
            Declarative mapping of parameter name to a ``float``, ``int``, or
            ``categorical`` specification. Callables are not accepted here;
            the genetic operators need a described space they can recombine.
        recipe_space:
            Preprocessing knobs described the same way. Requires
            ``preprocess``.
        population_size:
            How many configurations exist in each generation. Larger
            populations explore more of the space per generation and cost
            proportionally more.
        n_generations:
            How many rounds of selection and recombination to run. More
            generations refine further, with diminishing returns once the
            population converges.
        elite_size:
            How many top performers pass into the next generation unchanged.
            This guarantees the best result never gets worse; setting it too
            high causes the population to converge prematurely on one region.
        crossover_rate:
            The probability that two parents are recombined rather than copied.
            High values mix aggressively, which is the mechanism that combines
            good traits from different configurations.
        mutation_rate:
            The probability that a parameter is randomly perturbed after
            crossover. This is the only source of genuinely new values once the
            population has converged; too low and the search stalls, too high
            and it degenerates into random sampling.
        tournament_size:
            How many random candidates compete to become a parent. Larger
            tournaments favour the strongest more strongly, converging faster
            but exploring less.
        max_evaluations:
            A hard cap on total configurations evaluated, stopping the run
            early once reached. ``None`` runs all generations.
        random_state:
            Seed for the stochastic operators, making the run reproducible.
        task:
            ``'classification'``, ``'regression'``, or ``'auto'``.
        cv:
            Fold count or splitter used to score each configuration.
        cv_strategy:
            How folds are formed — see :meth:`cv_score`.
        ranking_metric:
            The metric acting as the fitness function. ``None`` uses the task
            default.
        groups:
            Group labels for the group-aware strategies.
        preprocess:
            Fold-local recipe refit inside each fold.
        allow_session_global_preprocess:
            Proceed despite session-wide preprocessing. See :meth:`cv_score`.
        refit:
            Retrain the winner on the full training partition and install it as
            :attr:`fit_result`. On by default.

        Returns
        -------
        ~buildml.model.selection.SearchResult
            Every evaluated configuration with its score, the ``best_params``,
            ``best_score``, and the winner's ``best_cv`` breakdown. Also stored
            on :attr:`last_search`.

        Raises
        ------
        ~buildml.core.errors.LeakageError
            Session-wide preprocessing already ran and was not explicitly
            allowed.
        ~buildml.core.errors.ValidationError
            No search space was supplied, a space was given as a callable
            rather than a mapping, or ``elite_size`` is not smaller than
            ``population_size``.

        Notes
        -----
        Folds are cut from the training partition only; test never influences
        the search.

        This is a plain genetic algorithm — population, tournament selection,
        crossover, mutation, elitism. It is not neural architecture search and
        not a particle swarm, and it is not random search under another name:
        the recombination step is what makes it different.

        Examples
        --------
        >>> space = {
        ...     "max_depth": {"type": "int", "low": 2, "high": 24},
        ...     "learning_rate": {"type": "float", "low": 1e-3, "high": 0.3, "log": True},
        ...     "booster": {"type": "categorical", "choices": ["gbtree", "dart"]},
        ... }
        >>> search = session.evolutionary_search(
        ...     estimator, param_space=space, population_size=16, n_generations=8
        ... )  # doctest: +SKIP

        See Also
        --------
        Session.optuna_search : Adaptive single-sampler alternative.
        Session.randomized_search : Cheaper when parameters do not interact.
        """
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
        """The most recent :meth:`cv_score` result.

        Holds the per-fold scores, their mean and standard deviation, the fold
        strategy actually used, and the written interpretation. The standard
        deviation is the part worth reading: it tells you how much of the mean
        is signal and how much is which rows happened to land where.

        ``None`` until :meth:`cv_score` runs.
        """
        return self._last_cv

    @property
    def last_nested_cv(self) -> NestedCVResult | None:
        """The most recent :meth:`nested_cv_score` result.

        Holds each outer fold's score, the configuration its inner search
        chose, and a summary of how much those choices varied. That variation
        is the diagnostic: if every outer fold selected different
        hyperparameters, the tuning is fitting noise and the specific winning
        configuration means little.

        ``None`` until :meth:`nested_cv_score` runs.
        """
        return self._last_nested_cv

    @property
    def last_search(self) -> SearchResult | None:
        """The most recent hyperparameter search result.

        Set by whichever of :meth:`grid_search`, :meth:`randomized_search`,
        :meth:`optuna_search`, or :meth:`evolutionary_search` ran last. Holds
        every trial with its score, the winning parameters, and — when
        ``refit=True`` — the model retrained on the full training partition.

        Call ``to_frame()`` on it to see the trials as a table. Sorting that
        table is often more useful than the winner alone: if the top twenty
        configurations score within noise of each other, the parameter you
        tuned does not matter much and you should stop tuning it.

        ``None`` until a search runs.
        """
        return self._last_search

    def extract_dates(
        self,
        columns: list[str] | tuple[str, ...] | None = None,
        *,
        include_time: bool = False,
        drop_original: bool = False,
    ) -> Session:
        """Break timestamps apart into the calendar parts a model can use.

        A raw timestamp is nearly useless as a feature. As a number it counts
        seconds since 1970, which increases forever and tells a model nothing
        about the patterns that actually drive behaviour — those live in the
        parts. Retail spikes in December, support tickets arrive on weekdays,
        traffic peaks at rush hour. Splitting one datetime column into year,
        month, day, day-of-week, and optionally hour and minute makes each of
        those learnable.

        Parameters
        ----------
        columns:
            Which datetime columns to expand. ``None`` finds every datetime
            column automatically.
        include_time:
            Also produce hour, minute, and second. Leave off for daily or
            coarser data, where the clock parts would be constant noise.
        drop_original:
            Remove the source timestamp after expanding. Keep it if a later
            step still needs to order rows — :meth:`time_split` reads the
            ``time``-role column, and dropping it out from under that will
            break the split.

        Returns
        -------
        Session
            ``self``, so this call chains into the next step.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            A named column is absent or is not datetime-typed.

        Notes
        -----
        The expansion is row-wise and deterministic: nothing is learned from
        the data, so unlike the fitted transforms it carries no leakage risk
        and can be run before splitting.

        Calendar parts are numbers with a wrap-around: December (12) and
        January (1) are adjacent in reality but maximally distant as integers.
        Tree models cope, since they split on ranges. For linear models,
        consider one-hot encoding the month via :meth:`encode`, or supplying
        your own sine/cosine pair through :meth:`register_transform`.

        Examples
        --------
        >>> import pandas as pd
        >>> from buildml import Session
        >>> frame = pd.DataFrame(
        ...     {"ordered_at": pd.to_datetime(["2024-03-01", "2024-12-25"]), "y": [0, 1]}
        ... )
        >>> session = Session.ingest(frame)
        >>> _ = session.extract_dates(["ordered_at"])
        >>> "ordered_at_month" in session.dataset.columns
        True

        See Also
        --------
        Session.time_split : Split chronologically before expanding.
        Session.fit_forecast : When time is the axis, not just a feature.
        """
        return preprocess_ops.extract_dates(
            self, columns=columns, include_time=include_time, drop_original=drop_original
        )

    @property
    def date_plan(self) -> DateFeaturePlan | None:
        """Which calendar parts the last :meth:`extract_dates` call produced.

        A :class:`~buildml.preprocess.dates.DateFeaturePlan` records the source
        columns, the parts generated, and the resulting column names. Keeping
        it means new data expands into the same columns in the same order, so
        the model still recognises its inputs.

        ``None`` until :meth:`extract_dates` runs.
        """
        return self._date_plan

    def save_model(self, path: str | Path) -> Path:
        """Save the fitted estimator and the feature contract it expects.

        This writes the model itself plus the list of feature columns and their
        order — enough to load the model back and call it, provided your data
        is already in the right shape.

        It is almost never what you want. A model trained on scaled, encoded,
        imputed data will produce nonsense if handed raw data, and this bundle
        does not carry the plans needed to prepare it. Use
        :meth:`save_pipeline` instead unless you are deliberately keeping the
        preprocessing under separate control.

        Parameters
        ----------
        path:
            Destination path for the bundle.

        Returns
        -------
        pathlib.Path
            Where the bundle was written.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            No model has been fitted yet.

        See Also
        --------
        Session.save_pipeline : Save the preprocessing with the model.
        Session.checkpoint_save : Save the whole session, data included.
        """
        return classical_ops.save_model(self, path=path)

    def load_model(self, path: str | Path) -> Session:
        """Load an estimator bundle written by :meth:`save_model`.

        Restores :attr:`fit_result` — the estimator and its feature contract —
        into this session. The dataset and split are left alone, so you can
        attach a fitted model to data you loaded separately.

        Because the bundle carries no preprocessing plans, whatever data you
        attach must already be in the exact form the model was trained on.

        Parameters
        ----------
        path:
            Path to the bundle written by :meth:`save_model`.

        Returns
        -------
        Session
            ``self``, so the load chains into a predict.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            The path holds no readable bundle.

        See Also
        --------
        Session.load_pipeline : Load a model together with its preprocessing.
        Session.predict_from_pipeline : Score new data in a single call.
        """
        return classical_ops.load_model(self, path=path)

    def save_pipeline(
        self,
        path: str | Path,
        *,
        evaluate_partition: Literal["train", "validation", "test"] | None = "test",
        title: str | None = None,
    ) -> Path:
        """Save everything needed to score new data: model, prep, and card.

        This is the artefact you deploy. A model on its own is not enough,
        because raw incoming data does not look like the matrix the model was
        trained on — the categories need the same encoding, the numbers the
        same scaling, the gaps the same fill values. Saving the fitted plans
        alongside the estimator means score-time transformation reproduces
        training exactly, months later and on a different machine.

        The bundle is a directory containing ``model.joblib``, ``plans.joblib``
        (imputation, encoding, scaling, date expansion, outlier fences,
        binning, feature selection, and resampling lineage where present),
        ``meta.json``, and a model card in both JSON and Markdown.

        This is not a session checkpoint. It carries what is needed for
        inference — no data, no split membership, no operation history. To
        resume interrupted work rather than deploy a result, use
        :meth:`checkpoint_save`.

        Parameters
        ----------
        path:
            Destination directory, created if it does not exist.
        evaluate_partition:
            Which partition to score so the card records how the model
            performed. ``'test'`` by default. Pass ``None`` to skip, which is
            what you want when the session has no split attached.
        title:
            A human-readable name for the model card. Worth setting — this is
            what the person reading the card in six months sees first.

        Returns
        -------
        pathlib.Path
            The bundle directory that was written.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            No model has been fitted, or ``evaluate_partition`` names a
            partition that does not exist in the current split.

        Notes
        -----
        The model card records what the model was trained on, what it scored,
        and which preprocessing travelled with it. It is generated from the
        session's own history rather than written by hand, so it cannot drift
        from what actually happened.

        Examples
        --------
        >>> path = session.save_pipeline(
        ...     "artifacts/churn_v3", title="Churn model, Q1 refresh"
        ... )  # doctest: +SKIP

        Later, in a scoring job that has never seen this session:

        >>> from buildml import Session
        >>> scorer = Session.ingest(new_rows)  # doctest: +SKIP
        >>> result = scorer.predict_from_pipeline("artifacts/churn_v3")  # doctest: +SKIP

        See Also
        --------
        Session.load_pipeline : Restore this bundle into a session.
        Session.predict_from_pipeline : Score without restoring first.
        Session.checkpoint_save : Save work in progress rather than a result.
        Session.serve_bundle : Put a saved bundle behind an HTTP endpoint.
        """
        return classical_ops.save_pipeline(
            self, path=path, evaluate_partition=evaluate_partition, title=title
        )

    def load_pipeline(self, path: str | Path) -> Session:
        """Restore a saved model together with its preprocessing.

        Reads a bundle written by :meth:`save_pipeline` and installs its
        contents on this session: the fitted estimator lands on
        :attr:`fit_result`, the preprocessing plans on their respective
        properties (:attr:`scale_plan`, :attr:`encode_plan`, and so on), and
        the model card on :attr:`model_card`.

        Your data and split are untouched. That is deliberate — it lets you
        attach a trained model to a fresh batch of rows and score them, which
        is the usual reason to load a pipeline at all. Once loaded, run
        :meth:`apply_preprocess_plans` to transform the attached data, then
        :meth:`predict`.

        Parameters
        ----------
        path:
            The bundle directory written by :meth:`save_pipeline`.

        Returns
        -------
        Session
            ``self``, so the load chains into scoring.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            The directory is not a readable pipeline bundle, or its contents
            are incomplete.

        Notes
        -----
        For one-shot scoring, :meth:`predict_from_pipeline` does the load,
        transform, and predict in a single call and leaves the session
        unchanged. Prefer that in inference jobs; prefer this when you want the
        restored plans on the session for further work.

        See Also
        --------
        Session.save_pipeline : Create the bundle this reads.
        Session.predict_from_pipeline : Load, transform, and score in one step.
        Session.checkpoint_load : Restore data and split as well.
        """
        return classical_ops.load_pipeline(self, path=path)

    def apply_preprocess_plans(
        self,
        data: Dataset | pd.DataFrame | None = None,
        plans: dict[str, Any] | None = None,
        *,
        inplace: bool = True,
        use_session_plans: bool = True,
    ) -> ApplyPlansResult:
        """Replay fitted preprocessing on new rows, in the original order.

        Training-time preprocessing learns things — the median used to fill
        gaps, the category vocabulary, the scaler's mean and spread. New data
        must be transformed with *those* learned values, not with values
        recomputed from itself. This method replays the stored plans to do
        exactly that.

        Order matters as much as the values. Encoding before imputing, or
        scaling before encoding, produces a different matrix from the one the
        model was trained on. The sequence is therefore fixed: date expansion,
        imputation, outlier fences, encoding, binning, scaling, and finally
        feature selection.

        Nothing is fitted here. If a plan is missing, that step is skipped and
        recorded in the result rather than being quietly re-learned from the
        new data.

        Parameters
        ----------
        data:
            The rows to transform, as a Dataset or a DataFrame. ``None`` uses
            this session's own dataset — which is what you want after
            :meth:`load_pipeline` has restored plans onto a session holding
            fresh data.
        plans:
            An explicit plan mapping, such as the ``plans.joblib`` payload from
            a checkpoint or pipeline bundle. ``None`` uses the plans attached
            to the session.
        inplace:
            When True and you are transforming the session's own dataset,
            replace it with the transformed version. Split membership is
            rebuilt if an outlier plan with ``action='drop'`` removed rows.
        use_session_plans:
            Fall back to session-attached plans for any step not covered by an
            explicit ``plans`` mapping, letting you override one step while
            keeping the rest.

        Returns
        -------
        ~buildml.preprocess.apply.ApplyPlansResult
            The transformed dataset, which steps were applied, which were
            skipped and why, and any warnings. Read the skipped list: a step
            silently absent means the model is about to receive features
            shaped differently from its training data.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            A column a plan needs is missing from the incoming data, or a plan
            object is not one this method knows how to apply.

        Notes
        -----
        **Order:** dates → impute → outliers → encode → binning → scale →
        feature_select. Resample plans are lineage-only and are never
        reapplied at score time.

        **Leakage:** Plans must already be train-fitted; this method does not
        fit. Missing columns raise :class:`~buildml.core.errors.ValidationError`.

        Resampling is excluded on purpose. Rebalancing classes is a training
        trick to stop a model ignoring a rare class; applying it at score time
        would mean inventing or discarding real rows you were asked to predict.

        Examples
        --------
        >>> from buildml import Session
        >>> scorer = Session.ingest(new_rows)  # doctest: +SKIP
        >>> _ = scorer.load_pipeline("artifacts/churn_v3")  # doctest: +SKIP
        >>> applied = scorer.apply_preprocess_plans()  # doctest: +SKIP
        >>> applied.skipped  # doctest: +SKIP
        []

        See Also
        --------
        Session.predict_from_pipeline : Does this and the predict together.
        Session.load_pipeline : Restore the plans this replays.
        """
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
        """Score new rows through a saved bundle, in one call.

        This is the inference path. Point it at a directory written by
        :meth:`save_pipeline` and give it rows to score; it loads the model and
        its preprocessing plans, transforms the rows exactly as training did,
        and returns the predictions.

        Nothing on the session changes — not the dataset, not
        :attr:`fit_result`, not the plans. That makes it safe to call inside a
        batch job or a service handler, repeatedly, against different bundles,
        without one call contaminating the next.

        Parameters
        ----------
        path:
            The pipeline bundle directory to score through.
        data:
            The rows to score, as a Dataset or a plain DataFrame. ``None`` uses
            this session's dataset.
        roles:
            Column roles to apply when ``data`` is a bare DataFrame with no
            role information of its own. Needed when the bundle's
            preprocessing distinguishes features from identifiers.
        return_proba:
            Return class probabilities instead of chosen labels, where the
            estimator supports it. Use this when a downstream decision applies
            its own threshold rather than accepting the default cut-off.
        apply_plans:
            Replay the bundle's preprocessing before predicting. Leave on
            unless your incoming rows are already fully transformed — turning
            it off on raw data feeds the model inputs it cannot interpret and
            produces confident nonsense rather than an error.

        Returns
        -------
        ~buildml.pipeline.score.PipelinePredictResult
            The predictions plus the context needed to trust them: which
            preprocessing steps ran, how many rows were scored, and any
            warnings about the incoming data.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            The bundle cannot be read, or the incoming rows are missing a
            column the model or its plans require.

        Notes
        -----
        A column present at training and absent now is caught here rather than
        producing a silently wrong answer. Schema drift between training and
        serving is among the most common production failures, and this is where
        it surfaces.

        Examples
        --------
        >>> import pandas as pd
        >>> from buildml import Session
        >>> incoming = pd.DataFrame({"tenure": [4], "plan": ["basic"]})  # doctest: +SKIP
        >>> result = Session.ingest(incoming).predict_from_pipeline(
        ...     "artifacts/churn_v3", return_proba=True
        ... )  # doctest: +SKIP

        See Also
        --------
        Session.save_pipeline : Create the bundle this reads.
        Session.apply_preprocess_plans : The transform half, on its own.
        Session.serve_bundle : Expose a bundle over HTTP instead.
        """
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
        """Narrow the data in the engine before pulling it into memory.

        scikit-learn needs an in-memory matrix, which is a problem when the
        table is larger than memory. The way through is to do the reduction
        where the data already lives: let Polars or DuckDB select just the
        columns you need and, if necessary, sample the rows, so that only the
        reduced result crosses into Pandas.

        This matters only when :meth:`with_engine` has attached a native
        engine. On plain Pandas the data is already in memory and this is a
        no-op with bookkeeping.

        Parameters
        ----------
        partition:
            Which partition to prepare. Defaults to ``'train'``, the one that
            usually needs to fit in memory for a fit call.
        columns:
            Which columns to project. ``None`` selects the feature and target
            columns for the partition, which is what a fit needs and nothing
            more.
        sample_rows:
            Cap the result at this many rows, drawn at random. ``None`` keeps
            all of them. Sampling makes an oversized partition trainable, at
            the cost of learning from less of it — the sample is recorded in
            the disclosures so the compromise stays visible.
        random_state:
            Seed for the sampling, so the same subset is drawn each run.

        Returns
        -------
        ~buildml.data.engines.prep.MaterializePrepResult
            The prepared matrix together with disclosures recording which
            columns were projected and whether rows were sampled.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            No split exists, the partition is not part of it, or a named column
            is absent.

        Notes
        -----
        Projection and sampling reduce what must be materialised; they do not
        make scikit-learn out-of-core. The estimator still receives an
        in-memory matrix. If the reduced partition still does not fit, reach
        for :meth:`fit_online`, which learns incrementally from batches.

        See Also
        --------
        Session.with_engine : Attach the engine that makes this worthwhile.
        Session.fit_online : Train without holding all the data at once.
        """
        return classical_ops.prepare_design_matrix(
            self,
            partition=partition,
            columns=columns,
            sample_rows=sample_rows,
            random_state=random_state,
        )

    @property
    def model_card(self) -> ModelCard | None:
        """The documentation record for the saved or loaded pipeline.

        A :class:`~buildml.pipeline.card.ModelCard` summarises what the model
        is: what it was trained on, which preprocessing travelled with it, what
        it scored at save time, and the decisions taken along the way. It is
        generated from the session's own history rather than written by hand,
        so it cannot quietly disagree with what actually happened.

        This is the artefact to hand to a reviewer, an auditor, or the person
        who inherits the model.

        Set by :meth:`save_pipeline` and :meth:`load_pipeline`. ``None`` until
        one of those runs.
        """
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
        """Understand the data before you model it.

        Modelling before looking at the data is how people discover, three
        weeks in, that a column is 80% missing, that two features are the same
        number in different units, or that the target is nearly constant. This
        runs the checks that would have caught it.

        The screens cover data quality (missing values, constant and duplicate
        columns, suspicious cardinality), distributions and their skew,
        correlations between features and multicollinearity via VIF and PCA,
        mutual information against the target, and outlier detection. When a
        split exists it also compares train against test and reports drift —
        systematic differences between the two that would make your holdout
        estimate misleading.

        The output is narrated rather than dumped. Each finding comes with what
        it means and what to consider doing about it.

        Parameters
        ----------
        include_plots:
            Generate charts alongside the statistics. The plots are chosen to
            suit each column's type and distribution rather than drawn
            uniformly. Requires ``pip install 'buildml[viz]'``.
        show:
            Print the narrative summary to standard output, for notebook use.
        sample_rows:
            Analyse a random sample of this many rows instead of all of them.
            Worth setting on a large table, where the statistics stabilise long
            before the row count is exhausted.
        max_columns:
            How many columns the detailed analysers cover. Dataset-wide quality
            checks still see every column; this caps the expensive per-column
            work on very wide tables.
        max_plots:
            Upper bound on charts generated, so a wide table does not produce
            hundreds of figures.
        export_html:
            Path for a self-contained HTML report — the artefact to share with
            someone who will not run the code.
        export_figures:
            Directory to write individual PNG figures into.
        html_format:
            ``'studio'`` writes the interactive offline studio layout, the same
            surface :meth:`eda_app` serves. ``'research'`` writes a layered
            document with embedded matplotlib figures, better suited to reading
            top to bottom.

        Returns
        -------
        ~buildml.eda.report.EDAReport
            The findings, their interpretation, the recommendations drawn from
            them, and paths to anything exported. Also stored on
            :attr:`last_eda`.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            No dataset is attached.
        ~buildml.core.errors.MissingExtraError
            Plots were requested without ``buildml[viz]`` installed.

        Notes
        -----
        **Leakage:** Exploration is how analysts leak without noticing. Every
        pattern you find by looking at the whole dataset — including the test
        rows — informs decisions you then make about the model, so the test set
        stops being independent. Split first, and explore the training rows.
        The drift comparison is the exception: it exists precisely to compare
        partitions and reports only aggregate differences.

        **Scale:** Correlation and mutual-information analysis grows quickly
        with column count. Use ``sample_rows`` and ``max_columns`` on wide or
        tall tables.

        Examples
        --------
        >>> report = session.eda(export_html="reports/eda.html")  # doctest: +SKIP
        >>> report.recommendations[:2]  # doctest: +SKIP

        See Also
        --------
        Session.eda_app : The same analysis, served interactively.
        Session.head : A quick look rather than a full profile.
        Session.error_slices : Where the model fails, after fitting.
        """
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
        """Explore the data interactively in a browser instead of on paper.

        Starts a local web server and opens the EDA studio: the same analysis
        :meth:`eda` produces, but navigable — click into a column to see its
        distribution, sort correlations, filter findings, read the concept
        explanations behind each screen, and export what you find as PDF or
        CSV.

        The advantage over a static report is following a thread. Noticing that
        one column is skewed usually prompts a question about a second column,
        and clicking is faster than re-running an analysis with different
        arguments.

        Nothing leaves your machine: the server binds to localhost by default.
        Requires ``pip install 'buildml[dashboard]'``.

        Parameters
        ----------
        report:
            An existing :class:`~buildml.eda.report.EDAReport` to display.
            ``None`` reuses :attr:`last_eda` if present, and otherwise runs a
            fresh analysis first.
        host:
            Address to bind to. The default keeps the app on this machine;
            change it only if you intend the app to be reachable from
            elsewhere, and understand that your data becomes reachable too.
        port:
            Port to serve on. Change it if the default is already taken.
        open_browser:
            Open your browser automatically once the server is ready.
        title:
            Heading shown in the app, useful when several are running.
        sample_rows:
            Row sample size, forwarded to :meth:`eda` when a fresh report has
            to be computed.
        max_columns:
            Column cap, forwarded to :meth:`eda` on a fresh computation.
        blocking:
            Serve on the current thread until interrupted, rather than
            returning immediately. Use this in a script that would otherwise
            exit and take the server with it; leave it off in a notebook, where
            you want the cell to finish.

        Returns
        -------
        ~buildml.dashboard.launch.EDAAppHandle
            A handle exposing ``url``, ``is_running``, and ``stop()``. Call
            ``stop()`` when finished — a non-blocking server keeps running
            until you do.

        Raises
        ------
        ~buildml.core.errors.MissingExtraError
            ``buildml[dashboard]`` is not installed.
        ~buildml.core.errors.ValidationError
            No dataset is attached and no report was supplied.

        Examples
        --------
        >>> app = session.eda_app()  # doctest: +SKIP
        >>> app.url  # doctest: +SKIP
        'http://127.0.0.1:8765'
        >>> app.stop()  # doctest: +SKIP

        See Also
        --------
        Session.eda : The same analysis as a static report.
        Session.open_eda_dashboard : An alias for this method.
        """
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
        """Open the interactive EDA studio — an alias for :meth:`eda_app`.

        Identical behaviour under a more discoverable name. See
        :meth:`eda_app` for the full description of every argument.

        Parameters
        ----------
        report:
            Existing report to display, or ``None`` to reuse or compute one.
        host:
            Address to bind to.
        port:
            Port to serve on.
        open_browser:
            Open the system browser once the server is ready.
        title:
            Heading shown in the app.
        sample_rows:
            Row sample size when a fresh report must be computed.
        max_columns:
            Column cap when a fresh report must be computed.
        blocking:
            Serve on the current thread until interrupted.

        Returns
        -------
        ~buildml.dashboard.launch.EDAAppHandle
            A handle exposing ``url``, ``is_running``, and ``stop()``.

        Raises
        ------
        ~buildml.core.errors.MissingExtraError
            ``buildml[dashboard]`` is not installed.
        ~buildml.core.errors.ValidationError
            No dataset is attached and no report was supplied.

        See Also
        --------
        Session.eda_app : The method this delegates to.
        """
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
        """The most recent exploratory analysis report.

        Set by :meth:`eda`, and by :meth:`eda_app` when it had to compute a
        report to display. Kept so the findings can be re-read, re-exported, or
        handed straight back to :meth:`eda_app` without paying for the analysis
        twice.

        ``None`` until an analysis has run.
        """
        return self._last_eda

    def calibration(
        self,
        *,
        partition: Literal["train", "validation", "test"] = "test",
        export_figures: str | Path | None = None,
        export_html: str | Path | None = None,
    ) -> DiagnosticReport:
        """Check whether predicted probabilities mean what they claim.

        A classifier that outputs ``0.8`` is asserting that eight out of ten
        such cases are positive. Often that is simply false — the model ranks
        cases correctly while its probabilities are systematically too
        confident or too timid. Ranking quality (ROC-AUC) cannot detect this,
        because rescaling every probability leaves the ranking untouched.

        Calibration matters whenever the number itself is used rather than just
        the ordering: expected-value calculations, risk thresholds, or anything
        shown to a person who will read "80%" as eighty percent. This groups
        predictions into probability bands and compares each band's claimed
        rate against the rate actually observed.

        You get the Brier score (mean squared error of the probabilities),
        expected calibration error (the average gap between claimed and
        observed), and the reliability curve points behind both. A perfectly
        calibrated model traces the diagonal; sagging below it means
        overconfidence.

        Parameters
        ----------
        partition:
            Which rows to assess. Calibration must be measured on data the
            model did not learn from — on training rows almost any model looks
            well calibrated.
        export_figures:
            Directory to write the reliability diagram into. Requires
            ``pip install 'buildml[viz]'``.
        export_html:
            Path for a self-contained HTML version of the same.

        Returns
        -------
        ~buildml.model.diagnostics.DiagnosticReport
            The calibration findings: Brier score, expected calibration error,
            reliability curve points, and an interpretation of what the shape
            implies.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            No model has been fitted, the fitted model is not a classifier, or
            it does not expose ``predict_proba``.
        ~buildml.core.errors.MissingExtraError
            Figures were requested without ``buildml[viz]`` installed.

        Notes
        -----
        Poor calibration is usually fixable without retraining, by fitting a
        small correction from probability to observed rate on held-out data
        (Platt scaling or isotonic regression). Note which models tend to need
        it: naive Bayes is famously overconfident, boosted trees push
        probabilities toward the extremes, and a random forest averaging many
        votes is typically already close.

        Examples
        --------
        >>> report = session.calibration(partition="validation")  # doctest: +SKIP
        >>> report.metrics["brier_score"]  # doctest: +SKIP
        0.084

        See Also
        --------
        Session.tune_threshold : Choose the cut-off these probabilities feed.
        Session.eval_plots : The reliability curve alongside other panels.
        """
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
        """Choose the cut-off that turns a probability into a decision.

        A classifier outputs a probability, but acting on it requires a line:
        above this, treat as positive. The default line is 0.5, and 0.5 is
        almost never the right answer — it silently assumes that a false alarm
        and a miss cost the same amount.

        They rarely do. Missing a fraudulent transaction costs the value of the
        fraud; flagging a legitimate one costs an annoyed customer. Missing a
        disease costs far more than an unnecessary follow-up test. The correct
        threshold follows from those costs, not from the midpoint of the
        probability range.

        This sweeps every candidate threshold and reports how precision,
        recall, and F1 move as the line shifts. Supply ``fp_cost`` and
        ``fn_cost`` and it goes further, computing expected cost at each
        threshold and identifying the one that minimises it — turning a
        modelling choice into an arithmetic one.

        Parameters
        ----------
        partition:
            Which rows to sweep over. Use ``'validation'`` while choosing;
            selecting a threshold on ``'test'`` and then reporting that
            partition's score means the score was tuned on the data it claims
            to be independent of.
        fp_cost:
            What one false positive costs — flagging something that was fine.
            Any consistent unit works; only the ratio to ``fn_cost`` affects
            the chosen threshold.
        fn_cost:
            What one false negative costs — missing something real. Must be
            given together with ``fp_cost``.
        tp_benefit:
            What correctly catching a positive is worth, subtracted from
            expected cost. Useful when a true positive earns something concrete
            rather than merely avoiding a loss.
        tn_benefit:
            What correctly leaving a negative alone is worth.
        export_figures:
            Directory to write the threshold sweep chart into. Requires
            ``pip install 'buildml[viz]'``.
        export_html:
            Path for a self-contained HTML version of the same.

        Returns
        -------
        ~buildml.model.diagnostics.DiagnosticReport
            The sweep: metrics at every candidate threshold, the recommended
            cut-off, and — when costs were supplied — the expected cost curve
            and its minimum.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            No model has been fitted, the target is not binary, the model
            exposes no ``predict_proba``, or exactly one of ``fp_cost`` and
            ``fn_cost`` was supplied.
        ~buildml.core.errors.MissingExtraError
            Figures were requested without ``buildml[viz]`` installed.

        Notes
        -----
        The threshold you pick here is part of the model as deployed. Record it
        with the pipeline, because a bundle scored at 0.5 when it was tuned for
        0.18 will behave nothing like the version you evaluated.

        A cost-optimal threshold is only as good as the costs. If they are
        guesses, look at how flat the cost curve is around its minimum: a flat
        region means the exact number hardly matters, and a sharp one means
        your guess needs to be right.

        Examples
        --------
        >>> report = session.tune_threshold(
        ...     partition="validation", fp_cost=1.0, fn_cost=12.0
        ... )  # doctest: +SKIP
        >>> report.metrics["best_threshold"]  # doctest: +SKIP
        0.18

        See Also
        --------
        Session.calibration : Confirm the probabilities are trustworthy first.
        Session.predict : Obtain probabilities to apply the chosen cut-off to.
        """
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
        """Find out whether more data would help, before you go and get it.

        When a model underperforms there are two very different remedies, and
        pursuing the wrong one wastes weeks. Either the model is too simple to
        capture the pattern, in which case more rows change nothing and you
        need a richer model or better features; or it is complex enough but
        starved of examples, in which case more rows are exactly what is
        needed.

        A learning curve distinguishes them. The model is refitted on
        increasing fractions of the training rows and scored each time, giving
        two lines: performance on the data it trained on, and performance on
        held-out data.

        Read them by their gap and their slope. A wide gap that is still
        closing as the curves extend rightward means more data will help. Two
        curves that have converged and flattened, both mediocre, mean the model
        has learned everything it can from these features — more rows will not
        move it. Converged and both excellent means you are done.

        Parameters
        ----------
        estimator:
            An unfitted estimator to trace. Usually the same one you fitted, so
            the curve describes the model you are actually considering.
        task:
            ``'classification'``, ``'regression'``, or ``'auto'``.
        cv:
            Fold count used at each sample size, so every point on the curve is
            itself averaged rather than a single noisy measurement.
        export_figures:
            Directory to write the curve into. Requires
            ``pip install 'buildml[viz]'``.
        export_html:
            Path for a self-contained HTML version of the same.

        Returns
        -------
        ~buildml.model.diagnostics.DiagnosticReport
            The curve points at each training size, the train and validation
            scores at each, and an interpretation of what the shape implies.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            No split exists, or the training partition is too small to
            subdivide.
        ~buildml.core.errors.MissingExtraError
            Figures were requested without ``buildml[viz]`` installed.

        Notes
        -----
        The model is refitted once per sample size per fold, so this is among
        the slower diagnostics. Lower ``cv`` for a quick read.

        Examples
        --------
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> report = session.learning_curve(
        ...     RandomForestClassifier(random_state=0), cv=5
        ... )  # doctest: +SKIP

        See Also
        --------
        Session.eval_plots : The learning curve alongside other diagnostics.
        Session.cv_score : Score at full training size only.
        """
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
        """Measure which features the model genuinely depends on.

        The method is simple and that is its strength: take one feature, shuffle
        its values across rows so it keeps its distribution but loses its
        relationship to the target, and re-score. However far the score falls
        is how much the model was relying on that feature. Repeat for each
        feature.

        This works for any model — the internals are never inspected, only the
        predictions — so a neural network, a boosted ensemble, and a linear
        model can be compared on the same footing. It also avoids the known
        distortions of tree-based built-in importances, which systematically
        favour high-cardinality and continuous features regardless of whether
        they carry signal.

        Run it on held-out rows. Importance measured on training data tells you
        what the model memorised; importance on a holdout tells you what
        actually generalises, which is the question worth asking.

        Parameters
        ----------
        partition:
            Which rows to measure on. Default ``'test'``; ``'validation'`` is
            the better choice if you intend to act on the result and want to
            keep test clean.
        n_repeats:
            How many times each feature is shuffled. Shuffling is random, so a
            single pass is noisy; more repeats give a steadier ranking at
            proportionally more time.
        export_figures:
            Directory to write the importance chart into. Requires
            ``pip install 'buildml[viz]'``.
        export_html:
            Path for a self-contained HTML version of the same.

        Returns
        -------
        ~buildml.model.diagnostics.DiagnosticReport
            Per-feature importance with the spread across repeats, ranked, plus
            an interpretation. The spread matters: a feature whose importance
            varies wildly between repeats has not been shown to matter.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            No model has been fitted, or no split exists.
        ~buildml.core.errors.MissingExtraError
            Figures were requested without ``buildml[viz]`` installed.

        Notes
        -----
        Correlated features mislead this method, and it is worth knowing how.
        If two columns carry the same information, shuffling either one leaves
        the model able to recover the signal from the other, so both look
        unimportant — even though together they are essential. When you see a
        feature you expect to matter scoring near zero, check what it is
        correlated with before concluding it is useless.

        Importance is not causation. A feature the model leans on may be a
        symptom of the outcome rather than a cause of it, and intervening on it
        will change nothing. Use :meth:`fit_causal` when the question is what
        to act on.

        Examples
        --------
        >>> report = session.feature_importance(
        ...     partition="validation", n_repeats=20
        ... )  # doctest: +SKIP

        See Also
        --------
        Session.select_features : Act on importance by dropping columns.
        Session.error_slices : Where the model fails, rather than on what.
        Session.fit_causal : What to change, rather than what predicts.
        """
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
        """Break performance down by subgroup, to find where the model fails.

        An overall score is an average, and averages conceal. A model at 92%
        accuracy might be at 97% for the large customer segment and 61% for the
        small one — a difference invisible in the headline number and highly
        visible to the people in that second group.

        This splits the scored rows by the columns you name and reports metrics
        for each segment alongside the overall figure, so the gaps become
        explicit. Slice by region, product line, customer tier, or any
        categorical column whose subgroups you would be unhappy to serve badly.

        Parameters
        ----------
        by:
            One column name, or several. Passing several slices by their
            combination, which finds interaction failures — a model can be fine
            on each of two dimensions separately and poor on a particular
            intersection.
        partition:
            Which rows to slice. Use ``'validation'`` while exploring so test
            stays reserved.
        max_segments:
            Cap on how many segments to report, keeping a high-cardinality
            column from producing an unreadable table. The largest segments are
            kept.
        min_segment_n:
            Minimum rows for a segment to be reported as a finding. Below this,
            a metric is mostly noise — three rows and two errors is not a 67%
            error rate, it is three rows. Smaller segments are listed
            separately rather than discarded.
        export_html:
            Optional path to write an HTML report of segment findings.

        Returns
        -------
        ~buildml.model.diagnostics.DiagnosticReport
            Per-segment metrics and sizes, the segments that fell below
            ``min_segment_n`` under ``small_segments``, and an interpretation
            highlighting the largest gaps.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            No model has been fitted, no split exists, or a named column is
            absent from the partition.

        Notes
        -----
        Observational only: segment gaps are not fairness proof. Prefer
        validation for exploration and keep test for a final estimate.
        Segments with ``n < min_segment_n`` are listed under ``small_segments``.

        A gap tells you where the model is worse, not why. Small segments have
        fewer training examples, may genuinely be harder to predict, and may
        have been measured differently. Any of those explains a gap without
        implicating the model. Treat the output as a list of places to
        investigate, not a verdict.

        Examples
        --------
        >>> report = session.error_slices(
        ...     by="region", partition="validation"
        ... )  # doctest: +SKIP

        Look for an interaction the single-column view would hide:

        >>> report = session.error_slices(by=["region", "product_tier"])  # doctest: +SKIP

        See Also
        --------
        Session.feature_importance : What the model uses, rather than where it
            fails.
        Session.evaluate : The aggregate this decomposes.
        """
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
        """Rebalance the training classes so the rare one is not ignored.

        When 2% of rows are fraud, a model can reach 98% accuracy by predicting
        "not fraud" every time. It has learned nothing, and the metric
        congratulates it. Resampling changes the training distribution so the
        rare class carries enough weight for the model to take it seriously.

        The oversampling methods add minority rows. ``'random_oversample'``
        duplicates existing ones, which is safe but gives the model repeated
        copies to overfit. ``'smote'`` instead synthesises new rows by
        interpolating between nearby minority examples, producing variety
        rather than duplicates — usually the better default.
        ``'borderline_smote'`` concentrates that synthesis near the decision
        boundary, where the difficult cases are. ``'adasyn'`` puts more
        synthetic rows around minority examples the model currently gets wrong.

        ``'random_undersample'`` goes the other way and discards majority rows.
        Fast, and it throws away real data — reasonable only when the majority
        class is enormous and largely redundant.

        Only the training partition is altered. Validation and test keep the
        real class balance, because those are meant to reflect the world you
        will deploy into.

        Requires ``pip install 'buildml[imbalanced]'``.

        Parameters
        ----------
        sampler:
            Which method to apply, from those described above.
        random_state:
            Seed for the sampling and synthesis, making the result
            reproducible.
        sampling_strategy:
            How far to rebalance. ``'auto'`` levels the classes fully. A float
            sets the target minority-to-majority ratio, so ``0.5`` brings the
            minority to half the majority rather than all the way — often a
            better trade, since full balancing can push a model into
            over-predicting the rare class. A dict names target counts per
            class.

        Returns
        -------
        Session
            ``self``, so this call chains into the fit.

        Raises
        ------
        ~buildml.core.errors.MissingExtraError
            ``buildml[imbalanced]`` is not installed.
        ~buildml.core.errors.ValidationError
            No split exists, no target is assigned, the features are not yet
            numeric, or a minority class has too few rows for the chosen
            synthesis method.

        Notes
        -----
        SMOTE interpolates between neighbours, so it needs numeric features:
        encode and impute first. It also assumes the space between two
        minority rows is itself plausible, which is false when features are
        categorical or constrained — a synthetic point can be an impossible
        record.

        Resampling is not the only answer to imbalance, and often not the best.
        Class weights in the estimator, or moving the decision threshold with
        :meth:`tune_threshold`, address the same problem without inventing
        rows. Try those first.

        Examples
        --------
        >>> _ = session.resample(sampler="smote", sampling_strategy=0.5)  # doctest: +SKIP

        See Also
        --------
        Session.resample_strategies : Guidance on choosing among these.
        Session.tune_threshold : Handle imbalance at decision time instead.
        """
        return preprocess_ops.resample(
            self, sampler=sampler, random_state=random_state, sampling_strategy=sampling_strategy
        )

    def resample_strategies(self) -> list[dict[str, Any]]:
        """List the available resampling methods and when each one fits.

        A reference you can read at runtime rather than looking up: each entry
        names a strategy, describes what it does to the data, and says when it
        is the appropriate choice.

        Returns
        -------
        list of dict
            One entry per strategy accepted by :meth:`resample`, with its name,
            description, and guidance on when to use it.

        Examples
        --------
        >>> [s["name"] for s in session.resample_strategies()]  # doctest: +SKIP
        ['smote', 'random_oversample', 'random_undersample', ...]

        See Also
        --------
        Session.resample : Apply one of these strategies.
        """
        return preprocess_ops.resample_strategies(self)

    @property
    def resample_plan(self) -> ResamplePlan | None:
        """What the last :meth:`resample` call did to the training rows.

        A :class:`~buildml.preprocess.imbalance.ResamplePlan` records the
        strategy, the class counts before and after, and how many rows were
        added or removed.

        This is lineage, not a reusable transform. Resampling is never replayed
        at score time — :meth:`apply_preprocess_plans` deliberately skips it,
        because inventing or discarding rows you were asked to predict would be
        nonsense. The record exists so the model card can state that the
        training distribution was altered, which anyone interpreting the
        model's predicted rates needs to know.

        ``None`` until :meth:`resample` runs.
        """
        return self._resample_plan

    def to_engine(self, engine: EngineName | str | None = None) -> Any:
        """Hand back the data as the chosen engine's own object.

        An escape hatch. When you need a real ``polars.DataFrame`` or a DuckDB
        relation to run something BuildML does not expose, this converts the
        current data and returns the native object directly.

        Unlike :meth:`with_engine`, this does not change what the session uses;
        it produces a value for you to work with.

        Parameters
        ----------
        engine:
            Which engine's type to produce: ``'pandas'``, ``'polars'``, or
            ``'duckdb'``. ``None`` uses the engine the dataset is already set
            to.

        Returns
        -------
        object
            A ``pandas.DataFrame``, ``polars.DataFrame``, or DuckDB relation,
            depending on the engine.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            No dataset is attached.
        ~buildml.core.errors.MissingExtraError
            The requested engine's package is not installed.

        Notes
        -----
        The returned object is detached from the session. Changes you make to
        it do not flow back — call :meth:`sync_native` if you have mutated the
        session's own frame and need the engine table rebuilt.

        See Also
        --------
        Session.with_engine : Change the engine the session itself uses.
        Session.to_pandas : The common case, spelled directly.
        """
        return data_ops.to_engine(self, engine=engine)

    def checkpoint_save(
        self,
        path: str | Path,
        *,
        sidecar_partition_rows: int | None = None,
        sidecar_compression: str | None = None,
        sidecar_layout: str | None = None,
    ) -> Path:
        """Save the whole session so you can stop and pick up where you left off.

        Long workflows get interrupted — a laptop closes, a job hits its time
        limit, a notebook kernel dies three hours into feature engineering.
        A checkpoint writes the current data, the split membership, the fitted
        preprocessing plans, and the full operation history to disk so
        :meth:`checkpoint_load` can restore all of it later.

        This is for work in progress, not for deployment. It deliberately does
        not embed a fitted estimator; models are inference artefacts and belong
        in a :meth:`save_pipeline` bundle. Think of a checkpoint as saving your
        place, and a pipeline as shipping the result.

        Parameters
        ----------
        path:
            Destination directory, created if needed.
        sidecar_partition_rows:
            How many rows go in each Parquet file when the data is split across
            several. Defaults to 25,000. Ignored under
            ``sidecar_layout='single'``.
        sidecar_compression:
            Parquet compression codec. Defaults to ``zstd``, which compresses
            well without being slow to read back.
        sidecar_layout:
            How the data files are arranged. ``'auto'`` (the default) writes a
            single file for small data and partitions at 50,000 rows or more.
            ``'single'`` always writes one file, simpler to move around.
            ``'partitioned'`` always splits, which reads back faster for large
            data and lets a reader skip parts of it.

        Returns
        -------
        pathlib.Path
            The checkpoint directory that was written.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            No dataset is attached, or the destination cannot be written.

        Notes
        -----
        The checkpoint records a fingerprint of the data. When it is loaded
        back, that fingerprint is re-checked and the outcome lands on
        :attr:`reattach_result`, so a checkpoint whose underlying data has
        shifted announces itself instead of quietly resuming against something
        different.

        Examples
        --------
        >>> path = session.checkpoint_save("checkpoints/step_3")  # doctest: +SKIP

        Later, in a new process:

        >>> from buildml import Session
        >>> session = Session.checkpoint_load("checkpoints/step_3")  # doctest: +SKIP
        >>> session.reattach_result.status  # doctest: +SKIP
        'clean'

        See Also
        --------
        Session.checkpoint_load : Restore what this writes.
        Session.save_pipeline : Save a deployable model instead.
        """
        return data_ops.checkpoint_save(
            self,
            path=path,
            sidecar_partition_rows=sidecar_partition_rows,
            sidecar_compression=sidecar_compression,
            sidecar_layout=sidecar_layout,
        )

    @classmethod
    def checkpoint_load(cls, path: str | Path, *, data_only: bool = False) -> Session:
        """Restore a saved session and check the data still matches.

        Rebuilds a session from a :meth:`checkpoint_save` bundle: the data, the
        split membership, the fitted preprocessing plans, and the operation
        history all come back, so the audit trail spans the interruption rather
        than restarting at it.

        Restoration is verified, not assumed. The data is re-checked against
        the fingerprint recorded when the checkpoint was written, and the
        outcome lands on :attr:`reattach_result`. Read it before continuing —
        plans fitted against data that has since changed are no longer the
        right plans.

        Parameters
        ----------
        path:
            The checkpoint directory to restore from.
        data_only:
            Load only the rows and discard the rest — no split, no plans, no
            history. Use this when you want the stored data as a starting point
            for something new, and the previous session's decisions would only
            get in the way.

        Returns
        -------
        Session
            A new session holding the restored state.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            The directory is not a readable checkpoint, or its contents are
            incomplete.

        Notes
        -----
        Checkpoints do not embed a fitted estimator. Use :meth:`load_pipeline`
        for inference artefacts; the two are complementary, and a workflow that
        both resumes and ships will use each for its own purpose.

        Examples
        --------
        >>> from buildml import Session
        >>> session = Session.checkpoint_load("checkpoints/step_3")  # doctest: +SKIP
        >>> session.reattach_result.status  # doctest: +SKIP
        'clean'
        >>> len(session.history)  # doctest: +SKIP
        14

        See Also
        --------
        Session.checkpoint_save : Write the bundle this reads.
        Session.reattach : Restore into an existing session instead.
        Session.reattach_result : The verification outcome to check.
        """
        return data_ops.checkpoint_load_session(cls, path=path, data_only=data_only)

    def reattach(self, path: str | Path, *, data_only: bool = False) -> Session:
        """Replace this session's state from a checkpoint, in place.

        Does what :meth:`checkpoint_load` does, except it overwrites the
        current session rather than returning a new one. Useful in a loop or a
        long-lived process where you want to keep the same session object
        while swapping what it holds.

        Any native engine connection this session owns is closed first, so
        resources are not leaked across the swap.

        Parameters
        ----------
        path:
            The checkpoint directory to restore from.
        data_only:
            Restore only the rows, clearing the split, plans, and history.

        Returns
        -------
        Session
            ``self``, now holding the restored state.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            The directory is not a readable checkpoint.

        Notes
        -----
        Everything the session currently holds is discarded. Save first if the
        current state matters.

        See Also
        --------
        Session.checkpoint_load : The classmethod form, returning a new session.
        """
        return data_ops.reattach(self, path=path, data_only=data_only)

    def to_pandas(self) -> pd.DataFrame:
        """Take the data out as a plain Pandas DataFrame.

        The escape hatch. When you need to do something BuildML does not cover,
        this hands you an ordinary DataFrame to work with. If the data is
        currently held by Polars or DuckDB, it is materialised into memory
        here.

        Returns
        -------
        pandas.DataFrame
            A copy of the current data with every transform applied so far.
            Because it is a copy, editing it does not affect the session.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            No dataset is attached.

        Notes
        -----
        **Scale:** On a lazy or engine-backed dataset this forces full
        materialisation. That is precisely what the engine paths exist to
        avoid, so on a large table it may be slow or exhaust memory. Reach for
        :meth:`head` to look, or :meth:`prepare_design_matrix` to narrow first.

        See Also
        --------
        Session.head : A small preview instead of the whole table.
        Session.partition : One partition rather than everything.
        """
        return data_ops.to_pandas(self)

    def to_parquet(self, path: str | Path) -> Path:
        """Write the current data to a Parquet file.

        Parquet stores columns rather than rows, which makes it much smaller
        than CSV and much faster to read back — and unlike CSV it preserves
        dtypes, so a datetime column returns as a datetime rather than as text
        you have to re-parse.

        Use this to hand transformed data to another tool, or to save an
        intermediate result you do not want to recompute.

        Parameters
        ----------
        path:
            Destination file path.

        Returns
        -------
        pathlib.Path
            Where the file was written.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            No dataset is attached, or the destination cannot be written.

        Notes
        -----
        This writes the data only. Roles, split membership, and fitted plans
        are not included — :meth:`checkpoint_save` is the option that preserves
        those.

        See Also
        --------
        Session.checkpoint_save : Save the session, not just the table.
        Session.ingest : Read a Parquet file back in.
        """
        return data_ops.to_parquet(self, path=path)

    def head(self, n: int = 5) -> pd.DataFrame:
        """Look at the first few rows.

        The quickest way to see what you are working with, and worth doing
        after every transform — a column that has become all zeros or all
        ``NaN`` shows up immediately here and can otherwise go unnoticed until
        the model underperforms for no visible reason.

        Parameters
        ----------
        n:
            How many rows to return.

        Returns
        -------
        pandas.DataFrame
            The first ``n`` rows with all current columns.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            No dataset is attached.

        Notes
        -----
        Only the requested rows are materialised, so this stays cheap on
        engine-backed data where :meth:`to_pandas` would not be.

        These are the first rows in storage order, not a random sample. If the
        file is sorted, they are not representative — use :meth:`eda` for a
        picture of the whole table.

        Examples
        --------
        >>> import pandas as pd
        >>> from buildml import Session
        >>> session = Session.ingest(pd.DataFrame({"a": range(100)}))
        >>> session.head(3).shape
        (3, 1)

        See Also
        --------
        Session.eda : A full profile rather than a glance.
        """
        return data_ops.head(self, n=n)

    def with_mode(self, mode: DataMode | str) -> Session:
        """Set whether data is held in memory or kept lazy.

        ``'memory'`` means the rows are fully materialised and every operation
        works on them directly. ``'lazy'`` means the dataset keeps an engine
        handle and defers materialising until something genuinely requires it —
        which is how a table larger than memory stays workable.

        This records the intent on the dataset. Whether laziness actually
        happens depends on the engine: it is real for Polars and DuckDB and
        cannot apply to a Pandas-backed frame, which is already in memory by
        definition.

        Parameters
        ----------
        mode:
            ``'memory'`` or ``'lazy'``. The historical value ``'out_of_core'``
            is accepted and coerced to ``'lazy'``; there is no separate
            out-of-core fit path.

        Returns
        -------
        Session
            ``self``, so this call chains.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            No dataset is attached, or the mode is not a recognised value.

        Notes
        -----
        Lazy mode defers work; it does not make scikit-learn out-of-core. The
        estimator still needs an in-memory matrix at fit time. What laziness
        buys you is the chance to filter and project first, so that only the
        reduced result has to fit.

        See Also
        --------
        Session.with_engine : Choose the engine that makes lazy meaningful.
        Session.prepare_design_matrix : Narrow the data before materialising.
        """
        return data_ops.with_mode(self, mode=mode)

    def with_engine(self, engine: EngineName | str) -> Session:
        """Switch the compute engine backing the data.

        Pandas is the default and the right choice for anything that
        comfortably fits in memory. Polars and DuckDB exist for when it does
        not: both hold the data in their own columnar format and can filter,
        project, and aggregate over it far faster and with less memory than
        Pandas.

        Choosing between them is mostly about how you like to express things.
        Polars offers a DataFrame API with strong lazy evaluation; DuckDB lets
        you write SQL against the table. Either way, BuildML attaches a native
        handle that :meth:`prepare_design_matrix` and the filter and sample
        helpers use to reduce the data before anything crosses into Pandas.

        Parameters
        ----------
        engine:
            ``'pandas'``, ``'polars'``, or ``'duckdb'``. The latter two require
            ``pip install 'buildml[engines]'``.

        Returns
        -------
        Session
            ``self``, so this call chains.

        Raises
        ------
        ~buildml.core.errors.MissingExtraError
            The requested engine's package is not installed.
        ~buildml.core.errors.ValidationError
            No dataset is attached, or the engine name is not recognised.

        Notes
        -----
        Switching to Pandas releases any native handle; switching to Polars or
        DuckDB builds one. DuckDB's handle holds a connection, so close it with
        :meth:`close_native` or use ``with session:``.

        The estimator boundary is unchanged. scikit-learn still requires an
        in-memory matrix, so the engine's value lies in everything that happens
        before the fit.

        Examples
        --------
        >>> session = Session.ingest("events.parquet")  # doctest: +SKIP
        >>> with session.with_engine("duckdb") as s:  # doctest: +SKIP
        ...     prepared = s.prepare_design_matrix(sample_rows=500_000)

        See Also
        --------
        Session.to_engine : Get a native object without switching.
        Session.close_native : Release a DuckDB connection.
        """
        return data_ops.with_engine(self, engine=engine)

    def sync_native(self) -> Session:
        """Rebuild the engine's table from the current Pandas frame.

        With a Polars or DuckDB engine attached, the data exists in two places:
        the engine's native table and a Pandas cache. BuildML's own transforms
        keep them in step. Code outside BuildML that reaches in and edits
        ``dataset.frame`` directly does not, leaving the engine table stale.

        This resynchronises them, converting the current frame into a fresh
        engine table. On a Pandas-backed dataset there is nothing to sync and
        the call simply records that.

        Returns
        -------
        Session
            ``self``, so this call chains.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            No dataset is attached.

        Notes
        -----
        This is eager and total: the whole current frame is converted. It does
        not replay earlier steps as a lazy plan, so on a large table it costs
        what a full conversion costs.

        See Also
        --------
        Session.with_engine : Attach the engine this keeps in step.
        """
        return data_ops.sync_native(self)

    def metadata(self) -> dict[str, Any]:
        """Take a serialisable snapshot of everything the session knows.

        Returns the session's state as plain dictionaries and lists — no
        BuildML objects — so it can be written to JSON, logged, compared
        between runs, or attached to an experiment tracker.

        Returns
        -------
        dict
            Whether a dataset is attached, the ingest report, the split plan,
            the full operation history, the checkpoint reattach outcome, and
            the dataset's own metadata (schema, roles, row count, engine).
            Contains no row data, so it is safe to log.

        Notes
        -----
        Useful as a run fingerprint. Diffing two runs' metadata is a fast way
        to find why yesterday's numbers and today's disagree.

        See Also
        --------
        Session.history : The operation record on its own.
        Session.summarize_history : A readable summary rather than raw state.
        """
        return data_ops.metadata(self)

    def workflow(self) -> tuple[WorkflowStep, ...]:
        """List every operation, with what it needs and whether it can run now.

        A session exposes several hundred methods, and which of them make sense
        depends entirely on where you are: you cannot fit before splitting, or
        evaluate before fitting. This resolves the whole surface against the
        session's current state and reports the status of each step.

        It answers "what can I do next?" without reading the documentation
        first, which is also why it backs the AI tooling — an agent needs the
        same answer, in the same machine-readable form.

        Returns
        -------
        tuple of ~buildml.explain.schemas.WorkflowStep
            One entry per public operation, with its identifier, what it
            requires, whether those requirements are currently met, and whether
            it has already run.

        Examples
        --------
        >>> ready = [s for s in session.workflow() if s.available]  # doctest: +SKIP

        See Also
        --------
        Session.explain : What one specific operation will do.
        Session.walkthrough : A narrative of what has already happened.
        Session.dry_run : Preview a step's effect without running it.
        """
        return workflow_ops.workflow(self)

    def walkthrough(
        self,
        *,
        export_html: str | Path | None = None,
    ) -> WorkflowWalkthroughReport:
        """Narrate everything this session did, and why.

        Turns the operation history into a readable account: which steps ran,
        what they were given, which choices were yours and which were BuildML's
        defaults, and what each one changed. It is the report you produce when
        someone asks how a number was arrived at — a colleague reviewing the
        work, an auditor, or yourself in three months.

        Because it is generated from the recorded history rather than written
        by hand, it cannot drift away from what actually happened.

        Parameters
        ----------
        export_html:
            Path to write a self-contained HTML version to. ``None`` returns
            the report without writing anything.

        Returns
        -------
        ~buildml.session.walkthrough.WorkflowWalkthroughReport
            The narrated report: the ordered steps, the reasoning behind each,
            and any warnings raised along the way. Also stored on
            :attr:`last_walkthrough`.

        Examples
        --------
        >>> report = session.walkthrough(export_html="reports/run.html")  # doctest: +SKIP

        See Also
        --------
        Session.summarize_history : A shorter, structured summary.
        Session.model_card : The equivalent artefact for a saved pipeline.
        Session.history : The raw records underneath.
        """
        return workflow_ops.walkthrough(self, export_html=export_html)

    @property
    def last_walkthrough(self) -> WorkflowWalkthroughReport | None:
        """The most recently generated walkthrough report.

        Set by :meth:`walkthrough`. Kept on the session so a report built
        earlier can be re-read or re-exported without regenerating it.

        ``None`` until :meth:`walkthrough` runs.
        """
        return self._last_walkthrough

    def explain(
        self,
        operation: str | None = None,
        *,
        moment: Literal["before", "after"] = "before",
        level: str = "beginner",
    ) -> Any:
        """Ask what an operation does, in plain language, at any point.

        BuildML's explanations are part of the library rather than a separate
        manual, so you can ask from inside your code. Name an operation and you
        get an account of what it does, what it needs, what it will change, and
        the traps worth knowing about — written for someone meeting the concept
        for the first time.

        The ``moment`` argument changes the tense and therefore the usefulness.
        Before running a step, you get what it is about to do and what to watch
        for. After running it, you get what it actually did to *this* session,
        with the real numbers.

        Parameters
        ----------
        operation:
            The operation to explain, named as the method is
            (``'split'``, ``'encode'``, ``'cv_score'``). ``None`` returns the
            whole workflow view, the same as :meth:`workflow`.
        moment:
            ``'before'`` for what the step will do and what it requires;
            ``'after'`` for what it did here, grounded in this session's state.
        level:
            How much depth to render: ``'beginner'`` (the default) leads with a
            plain-language primer, an analogy, the steps in order, and a
            glossary of the terms it uses; ``'intermediate'`` trims the
            introductory material; ``'advanced'`` assumes the vocabulary and
            keeps the full risk and assumption lists.

        Returns
        -------
        object
            An explanation record for the named operation, or the full workflow
            tuple when ``operation`` is ``None``. Operation explanations carry a
            ``beginner`` primer alongside the expert sections.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            The named operation is not one BuildML knows.
        ValueError
            ``level`` is not one of the three reading levels.

        Notes
        -----
        The conceptual material lives in :mod:`buildml.explain`, which is also
        where the guides and the AI tooling read from — so what you are told
        here is the same thing every other surface is told. The level changes
        how much is shown, never what is true.

        Examples
        --------
        >>> session.explain("group_split")  # doctest: +SKIP
        >>> session.explain("split").beginner.analogy  # doctest: +SKIP
        >>> session.explain("encode", moment="after", level="advanced")  # doctest: +SKIP

        See Also
        --------
        Session.learn : Teach a concept, operation, or term from first principles.
        Session.workflow : Every operation and its current availability.
        Session.walkthrough : What this session has already done.
        Session.dry_run : Preview an operation's effect on real data.
        """
        return workflow_ops.explain(self, operation=operation, moment=moment, level=level)

    def learn(self, topic: str | None = None, *, level: str = "beginner") -> Any:
        """Teach a concept, an operation, or a term — and say what to read first.

        :meth:`explain` answers "what will this call do here, now?".
        :meth:`learn` answers the prior question: "what is this, and what do I
        need to understand before it makes sense?". You can name either side of
        the vocabulary — the operation (``'split'``), the concept behind it
        (``'data-splitting'``), or the word you tripped over (``'leakage'``) —
        and BuildML works out which you meant.

        Called with no topic it returns the foundation concepts, which is the
        sensible place to start if you are new to this.

        Parameters
        ----------
        topic:
            A concept key, an operation name, or a glossary term. ``None``
            returns the foundation reading list.
        level:
            ``'beginner'`` (the default), ``'intermediate'``, or ``'advanced'``.

        Returns
        -------
        ~buildml.explain.academy.LearningBrief
            The material for the topic, plus ``read_first`` and ``read_next``
            concept notes giving a reading order rather than an index.

        Raises
        ------
        KeyError
            No concept, operation, or term matches; close matches are suggested
            in the message.
        ValueError
            ``level`` is not one of the three reading levels.

        Examples
        --------
        >>> session.learn()                        # doctest: +SKIP
        >>> session.learn("leakage-boundary")      # doctest: +SKIP
        >>> session.learn("fit", level="advanced") # doctest: +SKIP

        See Also
        --------
        Session.explain : What an operation does at this point in this session.
        Session.workflow : Which operations can run right now.
        """
        return workflow_ops.learn(self, topic=topic, level=level)

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
