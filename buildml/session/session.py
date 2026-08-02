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
    classical_ops,
    data_ops,
    dl_ops,
    eda_ops,
    preprocess_ops,
    rag_ops,
    state,
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
        self._ai_autonomy_result: Any | None = None
        self._rag_corpus: Any | None = None
        self._rag_chunks: Any | None = None
        self._rag_index: Any | None = None
        self._rag_index_result: IndexResult | None = None
        self._rag_retrieve_result: RetrieveResult | None = None
        self._rag_eval_result: RagEvalResult | None = None
        self._rag_generate_result: GenerateResult | None = None
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
            Columns to impute. Defaults to numeric non-target columns.
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
        method: Literal["pca"] = "pca",
        n_components: int | float | None = None,
        drop_input_columns: bool = True,
        prefix: str = "pc",
    ) -> Session:
        """Fit dimensionality reduction on train and replace numeric columns.

        Parameters
        ----------
        method:
            Currently ``pca`` only.
        n_components:
            Integer count, float variance target in (0, 1], or ``None`` for the
            maximum feasible components.
        prefix:
            Output column prefix (``pc_1``, ``pc_2``, …).

        Notes
        -----
        **Leakage:** Requires a split. The rotation is learned on train only.
        Explained variance is unsupervised and is not predictive utility.
        Scale numeric inputs first when magnitudes differ."""
        return preprocess_ops.reduce_dimensions(
            self,
            columns=columns,
            method=method,
            n_components=n_components,
            drop_input_columns=drop_input_columns,
            prefix=prefix,
        )

    @property
    def reduce_plan(self) -> ReducePlan | None:
        """Last fitted dimensionality-reduction plan, if any."""
        return self._reduce_plan

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
        batch_size: int = 16,
        max_len: int = 64,
        max_vocab: int = 5000,
        min_freq: int = 1,
        normalize: bool = True,
        shuffle_train: bool = True,
        seed: int = 0,
        task: Literal["classification", "regression", "auto"] = "auto",
    ) -> TorchLoaderBundle:
        """Build fused tabular+text DataLoaders (train-only vocab + normalize)."""
        return dl_ops.make_multimodal_torch_loaders(
            self,
            text_column=text_column,
            numeric_columns=numeric_columns,
            batch_size=batch_size,
            max_len=max_len,
            max_vocab=max_vocab,
            min_freq=min_freq,
            normalize=normalize,
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
        config: TrainConfig | None = None,
    ) -> Any:
        """Single-node DDP training (requires multi-GPU unless allow_cpu_ddp)."""
        return dl_ops.fit_torch_ddp(
            self,
            module_factory,
            epochs=epochs,
            learning_rate=learning_rate,
            mixed_precision=mixed_precision,
            world_size=world_size,
            allow_cpu_ddp=allow_cpu_ddp,
            config=config,
        )

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
    ) -> Session:
        """Chunk the active RAG corpus with size + overlap (requires ``buildml[rag]``)."""
        return rag_ops.rag_chunk(self, size=size, overlap=overlap)

    def rag_embed_and_index(
        self,
        *,
        embedder: Any | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        device: str | None = None,
    ) -> Session:
        """Embed chunks and build the default NumPy cosine index (requires ``buildml[rag]``).

        Refuses corpora that contain ``eval_only`` documents (:class:`LeakageError`).
        Default embedder is ``buildml.hashing_embed.v1`` (lexical/hashed, not semantic).
        ``device`` applies to sentence-transformer backends; hashing stays CPU-only."""
        return rag_ops.rag_embed_and_index(
            self,
            embedder=embedder,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
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

        Defaults: ``mode="dense"``, no metadata filters, ``rerank=False``. Hybrid
        defaults to RRF (``rrf_k=60``). Cross-encoder rerank requires ``buildml[rag]``."""
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
        inner_search: Literal["auto", "grid", "randomized", "optuna"] = "auto",
        n_iter: int = 10,
        n_trials: int = 20,
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
            args). Requires ``pip install 'buildml[optuna]'``.
        inner_search:
            ``auto``, ``grid``, ``randomized``, or ``optuna``.
        n_trials:
            Optuna inner trials per outer fold.
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
        configs (including Optuna winners)."""
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
        """Most recent grid/randomized/optuna search result, if any."""
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
