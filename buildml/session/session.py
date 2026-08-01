"""BuildML Session — OOP facade that delegates to domain packages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from buildml.checkpoint.bundle import load_checkpoint, save_checkpoint
from buildml.checkpoint.validate import ReattachResult
from buildml.core.errors import ValidationError
from buildml.core.results import IngestReport
from buildml.core.types import ColumnRole, DataMode, EngineName
from buildml.data.dataset import Dataset
from buildml.data.engines.prep import MaterializePrepResult, prepare_design_frame
from buildml.data.splits import (
    PartitionName,
    SplitPlan,
    assert_fit_partition,
    create_group_split,
    create_split,
    create_time_split,
    frame_for_partition,
    inject_partitions,
)
from buildml.eda.profile import explore_dataset
from buildml.eda.report import EDAReport
from buildml.explain.history import (
    make_operation_record,
    normalize_history,
    prior_state,
    session_state,
)
from buildml.explain.resolver import explain as explain_session
from buildml.explain.resolver import resolve_workflow
from buildml.explain.schemas import WorkflowStep
from buildml.ingest.pipeline import ingest as ingest_source
from buildml.model.compare import ModelComparison, compare_estimators
from buildml.model.diagnostics import (
    DiagnosticReport,
    calibration_report,
    learning_curve_report,
    permutation_importance_report,
    segment_error_report,
    threshold_report,
)
from buildml.model.plot_boards import PlotBoardReport, build_eval_plot_board
from buildml.model.selection import (
    CVScoreResult,
    NestedCVResult,
    SearchResult,
)
from buildml.model.selection import (
    cv_score as run_cv_score,
)
from buildml.model.selection import (
    grid_search as run_grid_search,
)
from buildml.model.selection import (
    nested_cv_score as run_nested_cv_score,
)
from buildml.model.selection import (
    optuna_search as run_optuna_search,
)
from buildml.model.selection import (
    randomized_search as run_randomized_search,
)
from buildml.model.supervised import (
    EvaluateResult,
    FitResult,
    evaluate_estimator,
    fit_estimator,
    materialize_partition_design,
    predict_estimator,
)
from buildml.pipeline.bundle import load_pipeline_bundle, save_pipeline_bundle
from buildml.pipeline.card import ModelCard
from buildml.pipeline.persist import load_fit_result, save_fit_result
from buildml.pipeline.score import PipelinePredictResult
from buildml.pipeline.score import predict_from_pipeline as run_predict_from_pipeline
from buildml.preprocess.apply import ApplyPlansResult, apply_preprocess_plans
from buildml.preprocess.binning import BinningPlan, fit_binning, transform_binning
from buildml.preprocess.columns import drop_columns as drop_columns_transform
from buildml.preprocess.custom import (
    CustomTransformPlan,
    CustomTransformSpec,
    fit_custom_transform,
    transform_custom,
)
from buildml.preprocess.custom import (
    list_transforms as list_registered_transforms,
)
from buildml.preprocess.custom import (
    register_transform as register_custom_transform,
)
from buildml.preprocess.dates import DateFeaturePlan, extract_date_features
from buildml.preprocess.encode import EncodePlan, fit_encoder, transform_encoder
from buildml.preprocess.fold import PreprocessRecipe
from buildml.preprocess.imbalance import (
    ResamplePlan,
    list_resample_strategies,
    resample_train,
)
from buildml.preprocess.impute import (
    SimpleImputePlan,
    fit_simple_imputer,
    transform_simple_imputer,
)
from buildml.preprocess.outliers import OutlierPlan, apply_outlier_plan, fit_outlier_plan
from buildml.preprocess.reduce import ReducePlan, fit_reducer, transform_reducer
from buildml.preprocess.result import PreprocessResult
from buildml.preprocess.scale import ScalePlan, fit_scaler, transform_scaler
from buildml.preprocess.select import (
    FeatureSelectPlan,
    fit_feature_selector,
    transform_feature_selector,
)
from buildml.preprocess.text import TextFeaturePlan, fit_text_features, transform_text_features
from buildml.session.audit import DryRunReport, HistorySummary
from buildml.session.audit import dry_run_session as run_dry_run
from buildml.session.audit import summarize_history as build_history_summary
from buildml.session.walkthrough import WorkflowWalkthroughReport, build_walkthrough


class Session:
    """Primary user-facing object for BuildML 2.x workflows.

    A session owns ingested data, roles, splits, history, and checkpoint
    reattach state. Methods delegate to domain packages and do not
    reimplement transform logic.

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
        self._eda_app_handle: Any | None = None
        self._last_cv: CVScoreResult | None = None
        self._last_nested_cv: NestedCVResult | None = None
        self._last_search: SearchResult | None = None
        self._model_card: ModelCard | None = None
        self._torch_loaders: Any | None = None
        self._dl_train_result: Any | None = None
        self._rag_corpus: Any | None = None
        self._rag_chunks: Any | None = None
        self._rag_index: Any | None = None
        self._rag_index_result: Any | None = None
        self._rag_retrieve_result: Any | None = None
        self._rag_eval_result: Any | None = None

    def close_native(self) -> None:
        """Close an owned DuckDB connection on the session dataset, if any.

        Safe to call when no dataset is attached or the engine is not DuckDB.
        Derived Datasets that share a connection are not owners; only the root
        handle closes the connection.
        """
        dataset = self._dataset
        if dataset is None:
            return
        closer = getattr(dataset, "close_native", None)
        if callable(closer):
            closer()

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
        :meth:`assert_can_fit` to enforce train-only fit scope.
        """
        dataset, report = ingest_source(
            source,
            mode=mode,
            engine=engine,
            dry_run=dry_run,
            mock_byte_estimate=mock_byte_estimate,
            read_nrows=read_nrows,
        )
        session = cls(dataset=dataset, ingest_report=report)
        session._record(
            "ingest",
            {
                "source_type": report.source_type,
                "format": report.format_name,
                "mode": report.recommended_mode.value if mode is None else str(mode),
                "engine": report.recommended_engine.value if engine is None else str(engine),
                "dry_run": dry_run,
                "read_nrows": read_nrows,
            },
            decision_origin="automatic" if mode is None and engine is None else "explicit",
            warnings=report.warnings,
        )
        return session

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
            ``self`` for fluent chaining.
        """
        self.dataset.set_roles(mapping)
        self._record(
            "set_roles",
            {
                "mapping": {
                    name: role.value if isinstance(role, ColumnRole) else str(role)
                    for name, role in mapping.items()
                }
            },
        )
        return self

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
        partition only (:meth:`assert_can_fit`).
        """
        self._split_plan = create_split(
            self.dataset,
            test_size=test_size,
            validation_size=validation_size,
            random_state=random_state,
            stratify=stratify,
        )
        self._record(
            "split",
            {
                "kind": self._split_plan.kind,
                "test_size": test_size,
                "validation_size": validation_size,
                "stratify": stratify,
            },
        )
        return self

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
            Positional indices into the current dataset.
        """
        self._split_plan = inject_partitions(
            self.dataset,
            train_indices=train_indices,
            test_indices=test_indices,
            validation_indices=validation_indices,
        )
        self._record(
            "inject_split",
            {
                "train_indices": list(train_indices),
                "test_indices": list(test_indices),
                "validation_indices": None
                if validation_indices is None
                else list(validation_indices),
                "kind": "injected",
            },
        )
        return self

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
        (customers, sites, documents). Random row splits leak across groups.
        """
        self._split_plan = create_group_split(
            self.dataset,
            test_size=test_size,
            validation_size=validation_size,
            random_state=random_state,
            group_column=group_column,
        )
        self._record(
            "group_split",
            {
                "kind": self._split_plan.kind,
                "test_size": test_size,
                "validation_size": validation_size,
                "group_column": self._split_plan.stratify_column,
            },
        )
        return self

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
        The splitter does not add a calendar embargo beyond strict ordering.
        """
        self._split_plan = create_time_split(
            self.dataset,
            test_size=test_size,
            validation_size=validation_size,
            time_column=time_column,
        )
        self._record(
            "time_split",
            {
                "kind": self._split_plan.kind,
                "test_size": test_size,
                "validation_size": validation_size,
                "time_column": self._split_plan.stratify_column,
            },
        )
        return self

    def partition(
        self,
        name: PartitionName | Literal["train", "validation", "test"],
    ) -> pd.DataFrame:
        """Return a copy of rows for a named partition.

        Raises
        ------
        ValidationError
            If no split exists.
        """
        if self._split_plan is None:
            raise ValidationError("No split defined. Call split(...) or inject_split(...) first.")
        frame = frame_for_partition(self.dataset, self._split_plan, name)  # type: ignore[arg-type]
        self._record(
            "partition",
            {"name": str(name)},
            result_summary={"name": str(name), "rows": int(len(frame))},
        )
        return frame

    def assert_can_fit(self, partition: PartitionName = "train") -> Session:
        """Enforce leakage-safe fit scope.

        Parameters
        ----------
        partition:
            Partition the caller intends to fit on (must be ``train``).

        Raises
        ------
        LeakageError
            If no split exists or partition is not train.
        """
        assert_fit_partition(self._split_plan, partition)
        return self

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
        columns are removed.
        """
        self._dataset = drop_columns_transform(self.dataset, columns)
        self._record("drop_columns", {"columns": list(columns)})
        return self

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
        the train partition only, then applied to all rows.
        """
        self.assert_can_fit("train")
        plan = fit_simple_imputer(
            self.dataset,
            self._split_plan,
            columns=columns,
            strategy=strategy,
            fill_value=fill_value,
        )
        self._dataset = transform_simple_imputer(self.dataset, plan)
        self._impute_plan = plan
        self._record("impute", plan.to_dict())
        return self

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
        full-train means on holdouts.
        """
        self.assert_can_fit("train")
        plan = fit_encoder(
            self.dataset,
            self._split_plan,
            columns=columns,
            method=method,
            min_frequency=min_frequency,
            n_folds=n_folds,
            random_state=random_state,
            smoothing=smoothing,
        )
        self._dataset, result = transform_encoder(
            self.dataset,
            plan,
            split_plan=self._split_plan,
        )
        self._encode_plan = plan
        self._last_preprocess = result
        self._record(
            "encode",
            plan.to_dict(),
            warnings=result.warnings,
            result_summary=result.to_dict(),
        )
        return self

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
        with the frozen bounds. Heuristic screens are not proof of error.
        """
        self.assert_can_fit("train")
        assert self._split_plan is not None
        plan = fit_outlier_plan(
            self.dataset,
            self._split_plan,
            columns=columns,
            method=method,
            action=action,
            iqr_multiplier=iqr_multiplier,
            zscore_threshold=zscore_threshold,
        )
        dataset, split_plan, plan, result = apply_outlier_plan(
            self.dataset,
            self._split_plan,
            plan,
        )
        self._dataset = dataset
        self._split_plan = split_plan
        self._outlier_plan = plan
        self._last_preprocess = result
        self._record(
            "handle_outliers",
            plan.to_dict(),
            warnings=result.warnings,
            result_summary=result.to_dict(),
        )
        return self

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
        ``±inf`` edges so score-time extremes remain defined.
        """
        self.assert_can_fit("train")
        plan = fit_binning(
            self.dataset,
            self._split_plan,
            columns=columns,
            strategy=strategy,
            n_bins=n_bins,
            encode_as=encode_as,
        )
        self._dataset, result = transform_binning(self.dataset, plan)
        self._binning_plan = plan
        self._last_preprocess = result
        self._record(
            "bin",
            plan.to_dict(),
            warnings=result.warnings,
            result_summary=result.to_dict(),
        )
        return self

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
        impute before calling when features are non-numeric or contain nulls.
        """
        self.assert_can_fit("train")
        plan = fit_feature_selector(
            self.dataset,
            self._split_plan,
            strategy=strategy,
            columns=columns,
            threshold=threshold,
            k=k,
            score_func=score_func,
            estimator=estimator,
        )
        self._dataset, result = transform_feature_selector(self.dataset, plan)
        self._feature_select_plan = plan
        self._last_preprocess = result
        self._record(
            "select_features",
            plan.to_dict(),
            warnings=result.warnings,
            result_summary=result.to_dict(),
        )
        return self

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
        **Leakage:** Requires a split. Scaler is fit on train only.
        """
        self.assert_can_fit("train")
        plan = fit_scaler(
            self.dataset,
            self._split_plan,
            columns=columns,
            method=method,
        )
        self._dataset = transform_scaler(self.dataset, plan)
        self._scale_plan = plan
        self._record("scale", plan.to_dict())
        return self

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
        from train documents only. Missing text becomes empty strings.
        """
        self.assert_can_fit("train")
        plan = fit_text_features(
            self.dataset,
            self._split_plan,
            columns=columns,
            method=method,
            max_features=max_features,
            ngram_range=ngram_range,
            drop_input_columns=drop_input_columns,
        )
        self._dataset, result = transform_text_features(self.dataset, plan)
        self._text_plan = plan
        self._last_preprocess = result
        self._record(
            "text_features",
            plan.to_dict(),
            warnings=result.warnings,
            result_summary=result.to_dict(),
        )
        return self

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
        Scale numeric inputs first when magnitudes differ.
        """
        self.assert_can_fit("train")
        plan = fit_reducer(
            self.dataset,
            self._split_plan,
            columns=columns,
            method=method,
            n_components=n_components,
            drop_input_columns=drop_input_columns,
            prefix=prefix,
        )
        self._dataset, result = transform_reducer(self.dataset, plan)
        self._reduce_plan = plan
        self._last_preprocess = result
        self._record(
            "reduce_dimensions",
            plan.to_dict(),
            warnings=result.warnings,
            result_summary=result.to_dict(),
        )
        return self

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
        See :func:`buildml.preprocess.register_transform` for the full contract.
        """
        return register_custom_transform(
            name,
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
        return list_registered_transforms()

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
        replay requires the same name to remain registered in-process.
        """
        self.assert_can_fit("train")
        plan = fit_custom_transform(
            self.dataset,
            self._split_plan,
            name=name,
            columns=columns,
            params=params,
        )
        self._dataset, result = transform_custom(self.dataset, plan)
        self._custom_plan = plan
        self._last_preprocess = result
        self._record(
            "apply_custom_transform",
            plan.to_dict(),
            warnings=result.warnings,
            result_summary=result.to_dict(),
        )
        return self

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
        API prerequisites pass, not that the operation is appropriate.
        """
        report = run_dry_run(self, operation, parameters=parameters)
        self._last_dry_run = report
        return report

    @property
    def last_dry_run(self) -> DryRunReport | None:
        """Most recent dry-run report, if any."""
        return self._last_dry_run

    def summarize_history(self) -> HistorySummary:
        """Summarize operation history and list unresolved risks.

        Notes
        -----
        Read-only. Does not append history. Risks are heuristic review cues,
        not proof of leakage or invalid results.
        """
        summary = build_history_summary(self)
        self._last_history_summary = summary
        return summary

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
        **Leakage:** Fits on train only. Call after split and preparation.
        """
        self.assert_can_fit("train")
        self._fit_result = fit_estimator(
            self.dataset,
            self._split_plan,
            estimator,
            task=task,
        )
        self._record(
            "fit",
            {
                "estimator": type(estimator).__name__,
                "task": task,
            },
            result_summary=self._fit_result.to_dict(),
        )
        return self

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
            If True and supported, return class probabilities.
        """
        if self._fit_result is None:
            raise ValidationError("No fitted estimator. Call fit(...) first.")
        preds = predict_estimator(
            self.dataset,
            self._split_plan,
            self._fit_result,
            partition=partition,
            return_proba=return_proba,
        )
        self._record(
            "predict",
            {"partition": partition, "n_rows": int(len(preds)), "proba": return_proba},
        )
        return preds

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
            :attr:`last_plot_board`.
        """
        if self._fit_result is None:
            raise ValidationError("No fitted estimator. Call fit(...) first.")
        result = evaluate_estimator(
            self.dataset,
            self._split_plan,
            self._fit_result,
            partition=partition,
        )
        if include_plots or export_figures is not None or export_html is not None:
            board = build_eval_plot_board(
                self.dataset,
                self._split_plan,
                self._fit_result,
                partition=partition,
                export_figures=export_figures,
                export_html=export_html,
            )
            self._last_plot_board = board
            result.diagnostics["plot_board"] = {
                "figure_dir": board.figure_dir,
                "html_path": board.html_path,
                "figure_paths": dict(board.figure_paths),
                "skipped": list(board.skipped),
                "interpretation": list(board.interpretation),
            }
        self._record(
            "evaluate",
            {
                "partition": partition,
                "include_plots": include_plots,
                "export_figures": export_figures,
                "export_html": export_html,
            },
            result_summary=result.to_dict(),
        )
        return result

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
    ) -> Any:
        """Build Torch DataLoaders from current roles and split partitions.

        Requires ``pip install 'buildml[torch]'`` (or ``buildml[dl]``). Shuffle
        applies to the train loader only. When ``normalize`` is True, mean/std
        are fit on train and frozen for validation/test. Classical preprocess
        plans are not auto-applied; call them first if needed.

        Returns
        -------
        TorchLoaderBundle
            Loaders keyed by partition plus the feature contract.
        """
        from buildml.dl.loaders import make_loaders
        from buildml.dl.types import LoaderConfig

        self.assert_can_fit("train")
        bundle = make_loaders(
            self.dataset,
            self._split_plan,
            config=LoaderConfig(
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                shuffle_train=shuffle_train,
                drop_last=drop_last,
                normalize=normalize,
                seed=seed,
            ),
            task=task,
        )
        self._torch_loaders = bundle
        self._record(
            "make_torch_loaders",
            {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "shuffle_train": shuffle_train,
                "drop_last": drop_last,
                "normalize": normalize,
                "seed": seed,
                "task": task,
            },
            result_summary=bundle.report.to_dict(),
            warnings=tuple(bundle.report.warnings),
        )
        return bundle

    def fit_torch(
        self,
        module: Any,
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
        config: Any | None = None,
    ) -> Session:
        """Train a caller-supplied ``nn.Module`` on the train Torch loader.

        Requires ``pip install 'buildml[torch]'``. Delegates to
        :func:`buildml.dl.train.train_supervised_module`. Does not replace
        classical :meth:`fit` / :attr:`fit_result`.

        Parameters
        ----------
        module:
            Unfitted (or warm) ``torch.nn.Module``. When ``resume=True``, weights
            are restored from :attr:`dl_train_result` before continuing.
        loss_fn:
            Optional ``(module, xb, yb) -> loss``. Defaults to CrossEntropy
            (classification) or MSE (regression).
        optimizer_factory:
            Optional ``callable(params) -> optimizer``. Defaults to Adam.
        epochs / learning_rate / device / grad_clip_norm / log_every:
            Train-loop knobs used when ``config`` is omitted. With ``resume=True``,
            ``epochs`` are **additional** epochs.
        early_stopping_patience / early_stopping_monitor / scheduler:
            M2 knobs when ``config`` is omitted. Patience requires a validation
            loader. Scheduler defaults to ``none`` (see :class:`~buildml.dl.types.TrainConfig`).
        resume:
            When True, continue from :attr:`dl_train_result` (e.g. after
            :meth:`load_torch_bundle`), restoring optimizer/scheduler state.
        config:
            Optional :class:`~buildml.dl.types.TrainConfig` overriding the
            scalar knobs above.
        """
        from buildml.dl.train import train_supervised_module
        from buildml.dl.types import TrainConfig

        self.assert_can_fit("train")
        if self._torch_loaders is None:
            self.make_torch_loaders()
        assert self._torch_loaders is not None
        if config is None:
            config = TrainConfig(
                epochs=epochs,
                learning_rate=learning_rate,
                device=device,
                grad_clip_norm=grad_clip_norm,
                log_every=log_every,
                early_stopping_patience=early_stopping_patience,
                early_stopping_monitor=early_stopping_monitor,
                scheduler=scheduler,
                batch_size=getattr(self._torch_loaders.report, "batch_size", 32),
                normalize=getattr(self._torch_loaders.report, "normalize", True),
            )
        prior = None
        if resume:
            if self._dl_train_result is None:
                raise ValidationError(
                    "resume=True requires dl_train_result. "
                    "Call load_torch_bundle(...) or fit_torch(...) first."
                )
            prior = self._dl_train_result
        result = train_supervised_module(
            module,
            self._torch_loaders,
            config=config,
            loss_fn=loss_fn,
            optimizer_factory=optimizer_factory,
            resume_from=prior,
        )
        self._dl_train_result = result
        self._record(
            "fit_torch",
            {
                "module": type(module).__name__,
                "epochs": result.n_epochs_ran,
                "device": result.device.to_dict(),
                "task": result.task,
                "resume": resume,
                "scheduler": result.scheduler_name,
                "early_stopping_patience": result.config.early_stopping_patience,
            },
            result_summary=result.to_dict(),
            warnings=tuple(result.warnings),
        )
        return self

    @property
    def dl_train_result(self) -> Any | None:
        """Last Torch :class:`~buildml.dl.results.TrainResult`, if any."""
        return self._dl_train_result

    def torch_training_curve(self) -> Any:
        """Return structured training-curve teaching data for the last Torch run.

        Requires a prior :meth:`fit_torch` / :meth:`load_torch_bundle`. Torch-free
        to read once :attr:`dl_train_result` exists.
        """
        from buildml.dl.curves import build_training_curve

        if self._dl_train_result is None:
            raise ValidationError(
                "No Torch trainer. Call fit_torch(...) or load_torch_bundle(...) first."
            )
        curve = self._dl_train_result.training_curve
        if curve is None:
            curve = build_training_curve(self._dl_train_result)
            self._dl_train_result.training_curve = curve
        self._record(
            "torch_training_curve",
            {},
            result_summary=curve.to_dict(),
        )
        return curve

    def evaluate_torch(
        self,
        *,
        partition: Literal["train", "validation", "test"] = "test",
        device: str | None = None,
    ) -> Any:
        """Evaluate the last Torch trainer on a named partition.

        Requires ``pip install 'buildml[torch]'``. Uses loaders from
        :meth:`make_torch_loaders` (rebuilds them if missing).
        """
        from buildml.dl.metrics import evaluate_module

        if self._dl_train_result is None:
            raise ValidationError(
                "No Torch trainer. Call fit_torch(...) or load_torch_bundle(...) first."
            )
        if self._torch_loaders is None:
            self.make_torch_loaders(
                normalize=self._dl_train_result.contract.normalize_mean is not None,
                task=self._dl_train_result.task,
            )
        assert self._torch_loaders is not None
        result = evaluate_module(
            self._dl_train_result,
            self._torch_loaders,
            partition=partition,
            device=device,
        )
        self._record(
            "evaluate_torch",
            {"partition": partition, "device": device},
            result_summary=result.to_dict(),
        )
        return result

    def save_torch_bundle(self, path: str | Path) -> Path:
        """Persist the last Torch trainer as ``buildml.torch_bundle.v1``.

        Distinct from Session checkpoints and classical pipeline bundles.
        See :data:`buildml.dl.checkpoint.CHECKPOINT_BOUNDARY`.
        """
        from buildml.dl.checkpoint import save_torch_bundle

        if self._dl_train_result is None:
            raise ValidationError("No Torch trainer. Call fit_torch(...) first.")
        destination = save_torch_bundle(path, self._dl_train_result)
        self._record("save_torch_bundle", {"path": str(destination)})
        return destination

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
            Optional device for ``torch.load`` (default CPU).
        """
        from buildml.dl.checkpoint import load_torch_bundle

        self._dl_train_result = load_torch_bundle(path, module, map_location=map_location)
        self._record(
            "load_torch_bundle",
            {"path": str(path), "module": type(module).__name__, "map_location": map_location},
            result_summary=self._dl_train_result.to_dict(),
        )
        return self

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

        Delegates to :mod:`buildml.rag.corpus`. Distinct from classical ingest.
        """
        from buildml.rag.corpus import (
            corpus_from_documents,
            corpus_from_frame,
            load_text_corpus,
        )
        from buildml.rag.extras import require_rag_stack

        require_rag_stack(feature="RAG corpus ingest")
        if text_column is not None:
            if self._dataset is None:
                raise ValidationError(
                    "text_column requires an attached dataset. "
                    "Call Session.ingest(...) first or pass source= documents/path."
                )
            corpus = corpus_from_frame(
                self._dataset.frame,
                text_column=text_column,
                id_column=id_column,
                role=role,
                source=f"session[{text_column}]",
            )
        elif source is None:
            raise ValidationError(
                "rag_ingest_corpus requires source= (path or documents) or text_column=."
            )
        elif isinstance(source, (str, Path)):
            corpus = load_text_corpus(source, glob=glob, encoding=encoding, role=role)
        else:
            corpus = corpus_from_documents(source, source="memory", default_role=role)

        self._rag_corpus = corpus
        self._rag_chunks = None
        self._rag_index = None
        self._rag_index_result = None
        self._rag_retrieve_result = None
        self._rag_eval_result = None
        self._record(
            "rag_ingest_corpus",
            {
                "source": corpus.source,
                "role": role,
                "text_column": text_column,
                "id_column": id_column,
            },
            result_summary=corpus.to_dict(),
        )
        return self

    def rag_chunk(
        self,
        *,
        size: int = 512,
        overlap: int = 64,
    ) -> Session:
        """Chunk the active RAG corpus with size + overlap (requires ``buildml[rag]``)."""
        from buildml.rag.chunk import chunk_documents
        from buildml.rag.extras import require_rag_stack
        from buildml.rag.types import ChunkConfig

        require_rag_stack(feature="RAG chunking")
        if self._rag_corpus is None:
            raise ValidationError("No RAG corpus. Call rag_ingest_corpus(...) first.")
        result = chunk_documents(
            self._rag_corpus,
            config=ChunkConfig(size=size, overlap=overlap),
        )
        self._rag_chunks = result
        self._record(
            "rag_chunk",
            {"size": size, "overlap": overlap},
            result_summary=result.to_dict(),
        )
        return self

    def rag_embed_and_index(
        self,
        *,
        embedder: Any | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> Session:
        """Embed chunks and build the default NumPy cosine index (requires ``buildml[rag]``).

        Refuses corpora that contain ``eval_only`` documents (:class:`LeakageError`).
        Default embedder is ``buildml.hashing_embed.v1`` (lexical/hashed, not semantic).
        """
        from buildml.rag.extras import require_rag_stack
        from buildml.rag.index import build_index
        from buildml.rag.results import ChunkResult

        require_rag_stack(feature="RAG embed and index")
        if self._rag_corpus is None:
            raise ValidationError("No RAG corpus. Call rag_ingest_corpus(...) first.")
        index = build_index(
            self._rag_corpus,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedder=embedder,
            chunks=self._rag_chunks,
        )
        self._rag_index = index
        self._rag_index_result = index.to_index_result()
        self._rag_chunks = ChunkResult(
            chunks=index.chunks,
            config=index.chunk_config.to_dict(),
            n_documents=index.n_documents,
        )
        self._record(
            "rag_embed_and_index",
            {
                "embedder_id": index.embed_config.embedder_id,
                "dim": index.embed_config.dim,
                "store_backend": index.index_config.store_backend,
            },
            result_summary=self._rag_index_result.to_dict(),
            warnings=tuple(index.warnings),
        )
        return self

    def rag_retrieve(self, query: str, *, k: int = 5) -> Any:
        """Dense top-k retrieve against the active RAG index (requires ``buildml[rag]``)."""
        from buildml.rag.extras import require_rag_stack
        from buildml.rag.retrieve import retrieve

        require_rag_stack(feature="RAG retrieve")
        if self._rag_index is None:
            raise ValidationError("No RAG index. Call rag_embed_and_index(...) first.")
        result = retrieve(self._rag_index, query, k=k)
        self._rag_retrieve_result = result
        self._record(
            "rag_retrieve",
            {"query": query, "k": k},
            result_summary=result.to_dict(),
        )
        return result

    def rag_evaluate(
        self,
        qrels: Any,
        *,
        k: int = 5,
    ) -> Any:
        """Score retrieval with gold qrels (recall@k, MRR). Requires ``buildml[rag]``.

        Document-level relevance: a chunk hit counts via its parent ``doc_id``.
        """
        from buildml.rag.evaluate import evaluate_retrieval
        from buildml.rag.extras import require_rag_stack

        require_rag_stack(feature="RAG evaluate")
        if self._rag_index is None:
            raise ValidationError("No RAG index. Call rag_embed_and_index(...) first.")
        result = evaluate_retrieval(self._rag_index, qrels, k=k)
        self._rag_eval_result = result
        self._record(
            "rag_evaluate",
            {"k": k, "n_queries": result.n_queries},
            result_summary=result.to_dict(),
            warnings=tuple(result.warnings),
        )
        return result

    @property
    def rag_index_result(self) -> Any | None:
        """Last :class:`~buildml.rag.results.IndexResult`, if any."""
        return self._rag_index_result

    @property
    def rag_retrieve_result(self) -> Any | None:
        """Last :class:`~buildml.rag.results.RetrieveResult`, if any."""
        return self._rag_retrieve_result

    @property
    def rag_eval_result(self) -> Any | None:
        """Last :class:`~buildml.rag.results.RagEvalResult`, if any."""
        return self._rag_eval_result

    def save_rag_bundle(self, path: str | Path) -> Path:
        """Persist the active RAG index as ``buildml.rag_bundle.v1``.

        Distinct from Session checkpoints and Torch trainer bundles.
        See :data:`buildml.rag.checkpoint.CHECKPOINT_BOUNDARY`.
        """
        from buildml.rag.checkpoint import save_rag_bundle
        from buildml.rag.extras import require_rag_stack

        require_rag_stack(feature="RAG bundle save")
        if self._rag_index is None:
            raise ValidationError("No RAG index. Call rag_embed_and_index(...) first.")
        destination = save_rag_bundle(
            path,
            self._rag_index,
            eval_result=self._rag_eval_result,
        )
        self._record("save_rag_bundle", {"path": str(destination)})
        return destination

    def load_rag_bundle(self, path: str | Path) -> Session:
        """Load a RAG bundle into this Session (requires ``buildml[rag]``)."""
        from buildml.rag.checkpoint import load_rag_bundle
        from buildml.rag.extras import require_rag_stack
        from buildml.rag.results import ChunkResult

        require_rag_stack(feature="RAG bundle load")
        index = load_rag_bundle(path)
        self._rag_index = index
        self._rag_index_result = index.to_index_result()
        self._rag_chunks = ChunkResult(
            chunks=index.chunks,
            config=index.chunk_config.to_dict(),
            n_documents=index.n_documents,
        )
        self._record(
            "load_rag_bundle",
            {"path": str(path)},
            result_summary=self._rag_index_result.to_dict(),
        )
        return self

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
        :func:`buildml.model.plot_boards.build_eval_plot_board`.
        """
        if self._fit_result is None:
            raise ValidationError("No fitted estimator. Call fit(...) first.")
        board = build_eval_plot_board(
            self.dataset,
            self._split_plan,
            self._fit_result,
            partition=partition,
            include_learning_curve=include_learning_curve,
            include_importance=include_importance,
            n_importance_repeats=n_importance_repeats,
            learning_curve_cv=learning_curve_cv,
            export_figures=export_figures,
            export_html=export_html,
            show=show,
        )
        self._last_plot_board = board
        self._record(
            "eval_plots",
            {
                "partition": partition,
                "include_learning_curve": include_learning_curve,
                "include_importance": include_importance,
                "n_importance_repeats": n_importance_repeats,
                "learning_curve_cv": learning_curve_cv,
                "n_figures": len(board.figure_paths) or len(board.figures),
                "n_skipped": len(board.skipped),
                "figure_dir": board.figure_dir,
                "html_path": board.html_path,
            },
        )
        return board

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
        self.assert_can_fit("train")
        comparison = compare_estimators(
            self.dataset,
            self._split_plan,
            estimators,
            task=task,
            partition=partition,
            ranking_metric=ranking_metric,
        )
        self._last_comparison = comparison
        # Keep the top-ranked model as the active fit for convenience.
        winner = comparison.rows[0]["model"]
        self._fit_result = comparison.fits[winner]
        self._record(
            "compare_models",
            {
                "estimators": list(estimators),
                "task": task,
                "partition": partition,
                "ranking_metric": ranking_metric,
            },
            result_summary={"winner": winner, "ranking_metric": comparison.ranking_metric},
        )
        return comparison

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

        Notes
        -----
        **Leakage:** If Session impute/encode/scale/text/reduce already ran,
        that train-global preprocess is recorded as a limitation unless
        ``preprocess`` refits fold-locally. Prefer fold-local recipes
        (including ``text`` and ``reduce``) for selection claims that include
        preprocessing. Custom transforms and resample stay Session-global.
        """
        self.assert_can_fit("train")
        result = run_cv_score(
            self.dataset,
            self._split_plan,
            estimator,
            task=task,
            cv=cv,
            cv_strategy=cv_strategy,
            scoring_metric=scoring_metric,
            groups=groups,
            preprocess=preprocess,
            session_preprocess_applied=self._session_preprocess_applied(),
        )
        self._last_cv = result
        self._record(
            "cv_score",
            {
                "estimator": type(estimator).__name__,
                "task": task,
                "cv": cv if isinstance(cv, int) else type(cv).__name__,
                "cv_strategy": cv_strategy,
                "scoring_metric": scoring_metric,
                "fold_preprocess": None if preprocess is None else preprocess.to_dict(),
            },
            result_summary={
                "scoring_metric": result.scoring_metric,
                "mean": result.mean_metrics.get(result.scoring_metric),
                "std": result.std_metrics.get(result.scoring_metric),
                "n_splits": result.n_splits,
                "cv_strategy": result.cv_strategy,
            },
        )
        return result

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
        configs (including Optuna winners).
        """
        self.assert_can_fit("train")
        result = run_nested_cv_score(
            self.dataset,
            self._split_plan,
            estimator,
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
            session_preprocess_applied=self._session_preprocess_applied(),
            warm_start_studies=warm_start_studies,
        )
        self._last_nested_cv = result
        self._record(
            "nested_cv_score",
            {
                "estimator": type(estimator).__name__,
                "task": task,
                "outer_cv": outer_cv if isinstance(outer_cv, int) else type(outer_cv).__name__,
                "inner_cv": inner_cv if isinstance(inner_cv, int) else type(inner_cv).__name__,
                "cv_strategy": cv_strategy,
                "scoring_metric": scoring_metric,
                "search_method": result.search_method,
                "inner_search": inner_search,
                "n_trials": n_trials if result.search_method == "optuna" else None,
                "warm_start_studies": bool(warm_start_studies),
                "recipe_grid": (
                    None
                    if recipe_grid is None
                    else {k: list(v) for k, v in recipe_grid.items()}
                ),
                "fold_preprocess": None if preprocess is None else preprocess.to_dict(),
            },
            result_summary={
                "scoring_metric": result.scoring_metric,
                "mean": result.mean_metrics.get(result.scoring_metric),
                "std": result.std_metrics.get(result.scoring_metric),
                "n_outer_splits": result.n_outer_splits,
                "n_inner_splits": result.n_inner_splits,
                "param_stability": result.inner_selection_summary.get("param_stability"),
                "search_method": result.search_method,
                "warm_start_studies": result.warm_start_studies,
            },
        )
        return result

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
        refit: bool = True,
    ) -> SearchResult:
        """Grid-search estimator params and/or fold-local recipe knobs.

        Ranks configurations by mean CV score, never peeking at test. When
        ``refit=True`` (default), the winning params/knobs are refit on full
        train and become the active :attr:`fit_result`.
        """
        self.assert_can_fit("train")
        result = run_grid_search(
            self.dataset,
            self._split_plan,
            estimator,
            param_grid,
            recipe_grid=recipe_grid,
            task=task,
            cv=cv,
            cv_strategy=cv_strategy,
            ranking_metric=ranking_metric,
            groups=groups,
            preprocess=preprocess,
            session_preprocess_applied=self._session_preprocess_applied(),
            refit=refit,
        )
        self._last_search = result
        if refit and result.refit_result is not None:
            self._fit_result = result.refit_result
        self._record(
            "grid_search",
            {
                "estimator": type(estimator).__name__,
                "param_grid": (
                    None
                    if param_grid is None
                    else {k: list(v) for k, v in param_grid.items()}
                ),
                "recipe_grid": (
                    None
                    if recipe_grid is None
                    else {k: list(v) for k, v in recipe_grid.items()}
                ),
                "task": task,
                "cv": cv if isinstance(cv, int) else type(cv).__name__,
                "cv_strategy": cv_strategy,
                "ranking_metric": ranking_metric,
                "refit": refit,
            },
            result_summary={
                "best_params": result.best_params,
                "best_recipe_knobs": result.best_recipe_knobs,
                "best_score": result.best_score,
                "best_std": result.best_std,
                "ranking_metric": result.ranking_metric,
                "n_trials": len(result.trials),
            },
        )
        return result

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
        refit: bool = True,
    ) -> SearchResult:
        """Randomized search over estimator params and/or recipe knobs.

        Same leakage contract as :meth:`grid_search`: folds stay inside train;
        the winner may be refit onto the full training partition.
        """
        self.assert_can_fit("train")
        result = run_randomized_search(
            self.dataset,
            self._split_plan,
            estimator,
            param_distributions,
            recipe_distributions=recipe_distributions,
            n_iter=n_iter,
            random_state=random_state,
            task=task,
            cv=cv,
            cv_strategy=cv_strategy,
            ranking_metric=ranking_metric,
            groups=groups,
            preprocess=preprocess,
            session_preprocess_applied=self._session_preprocess_applied(),
            refit=refit,
        )
        self._last_search = result
        if refit and result.refit_result is not None:
            self._fit_result = result.refit_result
        self._record(
            "randomized_search",
            {
                "estimator": type(estimator).__name__,
                "param_distributions": (
                    None
                    if param_distributions is None
                    else {k: str(v) for k, v in param_distributions.items()}
                ),
                "recipe_distributions": (
                    None
                    if recipe_distributions is None
                    else {k: str(v) for k, v in recipe_distributions.items()}
                ),
                "n_iter": n_iter,
                "random_state": random_state,
                "task": task,
                "cv": cv if isinstance(cv, int) else type(cv).__name__,
                "cv_strategy": cv_strategy,
                "ranking_metric": ranking_metric,
                "refit": refit,
            },
            result_summary={
                "best_params": result.best_params,
                "best_recipe_knobs": result.best_recipe_knobs,
                "best_score": result.best_score,
                "best_std": result.best_std,
                "ranking_metric": result.ranking_metric,
                "n_trials": len(result.trials),
            },
        )
        return result

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
        refit: bool = True,
    ) -> SearchResult:
        """Optuna TPE search with leakage-safe train-fold CV.

        Requires ``pip install 'buildml[optuna]'``. ``param_space`` may be a
        ``trial -> dict`` callable or a declare-style mapping
        (``float`` / ``int`` / ``categorical``). ``recipe_space`` sweeps
        fold-local recipe knobs and requires ``preprocess``.
        """
        self.assert_can_fit("train")
        result = run_optuna_search(
            self.dataset,
            self._split_plan,
            estimator,
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
            session_preprocess_applied=self._session_preprocess_applied(),
            refit=refit,
        )
        self._last_search = result
        if refit and result.refit_result is not None:
            self._fit_result = result.refit_result
        self._record(
            "optuna_search",
            {
                "estimator": type(estimator).__name__,
                "n_trials": n_trials,
                "random_state": random_state,
                "task": task,
                "cv": cv if isinstance(cv, int) else type(cv).__name__,
                "cv_strategy": cv_strategy,
                "ranking_metric": ranking_metric,
                "refit": refit,
                "has_param_space": param_space is not None,
                "has_recipe_space": recipe_space is not None,
            },
            result_summary={
                "best_params": result.best_params,
                "best_recipe_knobs": result.best_recipe_knobs,
                "best_score": result.best_score,
                "best_std": result.best_std,
                "ranking_metric": result.ranking_metric,
                "n_trials": len(result.trials),
            },
        )
        return result

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
        self._dataset, plan = extract_date_features(
            self.dataset,
            columns=columns,
            include_time=include_time,
            drop_original=drop_original,
        )
        self._date_plan = plan
        self._record("extract_dates", plan.to_dict())
        return self

    @property
    def date_plan(self) -> DateFeaturePlan | None:
        """Last date-feature plan, if any."""
        return self._date_plan

    def save_model(self, path: str | Path) -> Path:
        """Persist the last fitted estimator bundle.

        This stores the estimator and feature contract only. Prefer
        :meth:`save_pipeline` when impute/encode/scale plans must travel with
        the model.
        """
        if self._fit_result is None:
            raise ValidationError("No fitted estimator. Call fit(...) first.")
        destination = save_fit_result(path, self._fit_result)
        self._record("save_model", {"path": str(destination)})
        return destination

    def load_model(self, path: str | Path) -> Session:
        """Load a previously saved fitted estimator bundle into this session."""
        self._fit_result = load_fit_result(path)
        self._record("load_model", {"path": str(path)})
        return self

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
            Optional model-card title.
        """
        if self._fit_result is None:
            raise ValidationError("No fitted estimator. Call fit(...) first.")
        metrics: dict[str, dict[str, float]] = {}
        notes = [
            "Pipeline bundle stores fitted preprocess plans and the estimator feature contract.",
            "It does not embed a Session checkpoint or the raw training frame.",
            "Resample plans are lineage metadata only and are not reapplied at inference.",
        ]
        if evaluate_partition is not None and self._split_plan is not None:
            try:
                evaluation = evaluate_estimator(
                    self.dataset,
                    self._split_plan,
                    self._fit_result,
                    partition=evaluate_partition,
                )
                metrics[evaluate_partition] = dict(evaluation.metrics)
            except (ValidationError, ValueError, TypeError) as exc:
                notes.append(f"Evaluation at save time was skipped: {exc}")

        from buildml.pipeline.bundle import CHECKPOINT_COMPATIBILITY
        from buildml.pipeline.card import build_model_card, load_model_card

        preprocess_summary = self._preprocess_summary()
        card = build_model_card(
            fit_result=self._fit_result,
            dataset_schema=self.dataset.schema.to_dict(),
            preprocess_summary=preprocess_summary,
            history=self._history,
            metrics=metrics,
            title=title,
            notes=notes,
            lineage={
                "artifact": "pipeline_bundle",
                "contains_checkpoint": False,
                "contains_raw_dataset": False,
                "checkpoint_compatibility": CHECKPOINT_COMPATIBILITY,
                "plans_present": sorted(
                    key for key, value in preprocess_summary.items() if value is not None
                ),
            },
        )
        destination = save_pipeline_bundle(
            path,
            fit_result=self._fit_result,
            impute_plan=self._impute_plan,
            encode_plan=self._encode_plan,
            scale_plan=self._scale_plan,
            date_plan=self._date_plan,
            outlier_plan=self._outlier_plan,
            binning_plan=self._binning_plan,
            feature_select_plan=self._feature_select_plan,
            text_plan=self._text_plan,
            reduce_plan=self._reduce_plan,
            custom_plan=self._custom_plan,
            resample_plan=self._resample_plan,
            model_card=card,
            dataset_schema=self.dataset.schema.to_dict(),
            roles={k: v.value for k, v in self.dataset.roles.items()},
            history=self._history,
            metrics=metrics,
            title=title,
        )
        self._model_card = load_model_card(destination)
        self._record(
            "save_pipeline",
            {
                "path": str(destination),
                "evaluate_partition": evaluate_partition,
                "has_impute": self._impute_plan is not None,
                "has_encode": self._encode_plan is not None,
                "has_scale": self._scale_plan is not None,
                "has_dates": self._date_plan is not None,
                "has_outliers": self._outlier_plan is not None,
                "has_binning": self._binning_plan is not None,
                "has_feature_select": self._feature_select_plan is not None,
                "has_text": self._text_plan is not None,
                "has_reduce": self._reduce_plan is not None,
                "has_custom": self._custom_plan is not None,
                "has_resample": self._resample_plan is not None,
            },
            result_summary={"path": str(destination), "metrics_partitions": list(metrics)},
        )
        return destination

    def load_pipeline(self, path: str | Path) -> Session:
        """Load a pipeline bundle (estimator + preprocess plans + model card).

        Restores :attr:`fit_result`, preprocess plan attributes, and
        :attr:`model_card`. Does not replace the dataset or split; attach
        compatible data separately (or via :meth:`checkpoint_load`).
        """
        bundle = load_pipeline_bundle(path)
        self._fit_result = bundle.fit_result
        self._impute_plan = bundle.impute_plan
        self._encode_plan = bundle.encode_plan
        self._scale_plan = bundle.scale_plan
        self._date_plan = bundle.date_plan
        self._outlier_plan = bundle.outlier_plan
        self._binning_plan = bundle.binning_plan
        self._feature_select_plan = bundle.feature_select_plan
        self._text_plan = bundle.text_plan
        self._reduce_plan = bundle.reduce_plan
        self._custom_plan = bundle.custom_plan
        self._resample_plan = bundle.resample_plan
        self._model_card = bundle.model_card
        self._record(
            "load_pipeline",
            {
                "path": str(path),
                "estimator": bundle.fit_result.to_dict().get("estimator"),
                "has_model_card": bundle.model_card is not None,
                "bundle_format": bundle.bundle_format,
                "plans_format": bundle.plans_format,
                "plans_present": [
                    name
                    for name, plan in (
                        ("impute", bundle.impute_plan),
                        ("encode", bundle.encode_plan),
                        ("scale", bundle.scale_plan),
                        ("dates", bundle.date_plan),
                        ("outliers", bundle.outlier_plan),
                        ("binning", bundle.binning_plan),
                        ("feature_select", bundle.feature_select_plan),
                        ("text", bundle.text_plan),
                        ("reduce", bundle.reduce_plan),
                        ("custom", bundle.custom_plan),
                        ("resample", bundle.resample_plan),
                    )
                    if plan is not None
                ],
            },
        )
        return self

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
        fit. Missing columns raise :class:`~buildml.core.errors.ValidationError`.
        """
        if self._dataset is None and data is None:
            raise ValidationError("No dataset attached. Ingest data or pass data=...")
        resolved_plans = dict(plans or {})
        if use_session_plans:
            for key, value in self._plan_objects().items():
                resolved_plans.setdefault(key, value)
        target = self.dataset if data is None else data
        result = apply_preprocess_plans(
            target,
            resolved_plans,
            split_plan=self._split_plan,
        )
        mutating = inplace and (
            data is None or (isinstance(data, Dataset) and data is self._dataset)
        )
        if mutating:
            self._dataset = result.dataset
            if result.split_plan is not None:
                self._split_plan = result.split_plan
        self._record(
            "apply_preprocess_plans",
            {
                "inplace": mutating,
                "applied": list(result.applied),
                "skipped": list(result.skipped),
            },
            warnings=result.warnings,
            result_summary=result.to_dict(),
        )
        return result

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
        inference-only scoring of new frames.
        """
        if data is None:
            if self._dataset is None:
                raise ValidationError("No dataset attached. Ingest data or pass data=...")
            score_data: Dataset | pd.DataFrame = self.dataset
        else:
            score_data = data
        result = run_predict_from_pipeline(
            path,
            score_data,
            roles=roles,
            return_proba=return_proba,
            apply_plans=apply_plans,
        )
        self._record(
            "predict_from_pipeline",
            {
                "path": str(path),
                "return_proba": return_proba,
                "apply_plans": apply_plans,
                "n_rows": result.n_rows,
            },
            warnings=result.warnings,
            result_summary=result.to_dict(),
        )
        return result

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
        sampling; sklearn still requires an in-memory matrix.
        """
        if columns is not None:
            result = prepare_design_frame(
                self.dataset,
                columns,
                sample_rows=sample_rows,
                random_state=random_state,
                context=f"session prepare_design_matrix ({partition})",
            )
        else:
            self.assert_can_fit("train")
            assert self._split_plan is not None
            result = materialize_partition_design(
                self.dataset,
                self._split_plan,
                partition,
                sample_rows=sample_rows,
                random_state=random_state,
            )
        self._record(
            "prepare_design_matrix",
            {
                "partition": partition,
                "sample_rows": sample_rows,
                "engine": result.engine,
                "sampled": result.sampled,
            },
            result_summary=result.to_dict(),
        )
        return result

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
            writes the layered research HTML shell with matplotlib embeds.
        """
        report = explore_dataset(
            self.dataset,
            split_plan=self._split_plan,
            sample_rows=sample_rows,
            max_columns=max_columns,
            max_plots=max_plots,
            include_plots=include_plots,
            show=show,
            export_html=export_html,
            export_figures=export_figures,
            html_format=html_format,
        )
        from buildml.session.walkthrough import (
            preprocess_scope_status,
            torch_training_status_for_walkthrough,
            warm_start_studies_status,
        )

        warm = warm_start_studies_status(
            self._history,
            last_nested_cv=self._last_nested_cv,
        )
        report.overview["warm_start_status"] = warm
        report.overview["preprocess_scope_status"] = preprocess_scope_status(
            self._history,
            session=self,
            last_cv=self._last_cv,
            last_nested_cv=self._last_nested_cv,
        )
        report.overview["torch_training_status"] = torch_training_status_for_walkthrough(self)
        self._last_eda = report
        self._record(
            "eda",
            {
                "include_plots": include_plots,
                "show": show,
                "sample_rows": sample_rows,
                "max_columns": max_columns,
                "max_plots": max_plots,
                "export_html": export_html,
                "export_figures": export_figures,
                "html_format": html_format,
            },
            result_summary={
                "n_rows": report.overview.get("n_rows"),
                "n_columns": report.overview.get("n_columns"),
                "recommendations": len(report.recommendations),
                "narrative": len(report.narrative),
                "plots": len(report.figures),
                "html_path": report.html_path,
                "html_format": html_format if export_html is not None else None,
                "warm_start_studies": bool(warm.get("enabled")),
            },
        )
        return report

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
    ) -> Any:
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
            Handle with ``url``, ``stop()``, and ``is_running``.
        """
        from buildml.dashboard.launch import launch_eda_app

        eda_report = report or self._last_eda
        if eda_report is None:
            eda_report = self.eda(
                include_plots=False,
                show=False,
                sample_rows=sample_rows,
                max_columns=max_columns,
            )
        roles = {}
        if self._dataset is not None:
            roles = {
                str(column): getattr(role, "value", str(role))
                for column, role in self.dataset.roles.items()
            }
        meta = {
            "has_split": self._split_plan is not None,
            "history_len": len(self._history),
            "roles": roles,
        }
        handle = launch_eda_app(
            eda_report,
            host=host,
            port=port,
            open_browser=open_browser,
            title=title,
            session_meta=meta,
            blocking=blocking,
        )
        self._eda_app_handle = handle
        self._record(
            "eda_app",
            {"host": host, "port": port, "title": title, "url": handle.url},
            result_summary={"url": handle.url},
        )
        return handle

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
    ) -> Any:
        """Alias for :meth:`eda_app`."""
        return self.eda_app(
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
        Optional figure/HTML export uses the viz extra.
        """
        if self._fit_result is None:
            raise ValidationError("No fitted estimator. Call fit(...) first.")
        report = calibration_report(
            self.dataset,
            self._split_plan,
            self._fit_result,
            partition=partition,
            export_figures=export_figures,
            export_html=export_html,
        )
        self._last_diagnostic = report
        self._record(
            "calibration",
            {
                "partition": partition,
                "export_figures": export_figures,
                "export_html": export_html,
            },
            result_summary=report.to_dict(),
        )
        return report

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
            Optional benefits subtracted from cost for true positives / negatives.
        """
        if self._fit_result is None:
            raise ValidationError("No fitted estimator. Call fit(...) first.")
        report = threshold_report(
            self.dataset,
            self._split_plan,
            self._fit_result,
            partition=partition,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
            tp_benefit=tp_benefit,
            tn_benefit=tn_benefit,
            export_figures=export_figures,
            export_html=export_html,
        )
        self._last_diagnostic = report
        self._record(
            "tune_threshold",
            {
                "partition": partition,
                "fp_cost": fp_cost,
                "fn_cost": fn_cost,
                "tp_benefit": tp_benefit,
                "tn_benefit": tn_benefit,
                "export_figures": export_figures,
                "export_html": export_html,
            },
            result_summary=report.to_dict(),
        )
        return report

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
        report = learning_curve_report(
            self.dataset,
            self._split_plan,
            estimator,
            task=task,
            cv=cv,
            export_figures=export_figures,
            export_html=export_html,
        )
        self._last_diagnostic = report
        self._record(
            "learning_curve",
            {
                "estimator": type(estimator).__name__,
                "task": task,
                "cv": cv,
                "export_figures": export_figures,
                "export_html": export_html,
            },
            result_summary=report.to_dict(),
        )
        return report

    def feature_importance(
        self,
        *,
        partition: Literal["train", "validation", "test"] = "test",
        n_repeats: int = 8,
        export_figures: str | Path | None = None,
        export_html: str | Path | None = None,
    ) -> DiagnosticReport:
        """Permutation feature importance on a holdout partition."""
        if self._fit_result is None:
            raise ValidationError("No fitted estimator. Call fit(...) first.")
        report = permutation_importance_report(
            self.dataset,
            self._split_plan,
            self._fit_result,
            partition=partition,
            n_repeats=n_repeats,
            export_figures=export_figures,
            export_html=export_html,
        )
        self._last_diagnostic = report
        self._record(
            "feature_importance",
            {
                "partition": partition,
                "n_repeats": n_repeats,
                "export_figures": export_figures,
                "export_html": export_html,
            },
            result_summary=report.to_dict(),
        )
        return report

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
        Segments with ``n < min_segment_n`` are listed under ``small_segments``.
        """
        if self._fit_result is None:
            raise ValidationError("No fitted estimator. Call fit(...) first.")
        report = segment_error_report(
            self.dataset,
            self._split_plan,
            self._fit_result,
            by=by,
            partition=partition,
            max_segments=max_segments,
            min_segment_n=min_segment_n,
            export_html=export_html,
        )
        self._last_diagnostic = report
        self._record(
            "error_slices",
            {
                "by": by if isinstance(by, str) else list(by),
                "partition": partition,
                "max_segments": max_segments,
                "min_segment_n": min_segment_n,
                "export_html": export_html,
            },
            result_summary=report.to_dict(),
        )
        return report

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
        :meth:`resample_strategies` for strategy guidance.
        """
        dataset, plan, resample_plan = resample_train(
            self.dataset,
            self._split_plan,
            sampler=sampler,
            random_state=random_state,
            sampling_strategy=sampling_strategy,
        )
        self._dataset = dataset
        self._split_plan = plan
        self._resample_plan = resample_plan
        self._record("resample", resample_plan.to_dict())
        return self

    def resample_strategies(self) -> list[dict[str, Any]]:
        """List imbalance resampling strategies and when to use them."""
        return list_resample_strategies()

    @property
    def resample_plan(self) -> ResamplePlan | None:
        """Last train-only resample plan, if any."""
        return self._resample_plan

    def to_engine(self, engine: EngineName | str | None = None) -> Any:
        """Materialize the current dataset in a selected engine's native type.

        Parameters
        ----------
        engine:
            Target engine. Defaults to the dataset's current engine setting.
        """
        native = self.dataset.to_engine(engine)
        selected = self.dataset.engine if engine is None else EngineName(engine)
        self._record(
            "to_engine",
            {"engine": selected.value},
            result_summary={"engine": selected.value, "native_type": type(native).__name__},
        )
        return native

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
            ``'partitioned'``.
        """
        before = prior_state(self._history)
        sidecar_params = {
            "sidecar_partition_rows": sidecar_partition_rows,
            "sidecar_compression": sidecar_compression,
            "sidecar_layout": sidecar_layout,
        }
        record = make_operation_record(
            sequence=len(self._history) + 1,
            operation_id="checkpoint_save",
            parameters={"path": str(path), **sidecar_params},
            decision_origin="explicit",
            before=before,
            after=session_state(self),
            result_summary={"path": str(Path(path))},
        )
        destination = save_checkpoint(
            path,
            dataset=self.dataset,
            split_plan=self._split_plan,
            history=[*self._history, record],
            plans=self._plan_objects(),
            sidecar_partition_rows=sidecar_partition_rows,
            sidecar_compression=sidecar_compression,
            sidecar_layout=sidecar_layout,
        )
        record["parameters"] = {"path": str(destination), **sidecar_params}
        record["details"] = {"path": str(destination)}
        record["result_summary"] = {
            "path": str(destination),
            "plans_present": [
                key for key, value in self._plan_objects().items() if value is not None
            ],
        }
        self._history.append(record)
        return destination

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
        use :meth:`load_pipeline` for inference artifacts.
        """
        loaded = load_checkpoint(path, data_only=data_only)
        session = cls(
            dataset=loaded.dataset,
            split_plan=loaded.split_plan,
            history=loaded.history,
            reattach_result=loaded.reattach,
        )
        if not data_only:
            session._restore_plans(loaded.plans)
        session._record(
            "checkpoint_load",
            {
                "path": str(path),
                "status": loaded.reattach.status,
                "data_only": data_only,
                "plans_restored": sorted(
                    key for key, value in loaded.plans.items() if value is not None
                ),
            },
        )
        return session

    def reattach(self, path: str | Path, *, data_only: bool = False) -> Session:
        """Replace this session state from a checkpoint path (instance helper)."""
        loaded = load_checkpoint(path, data_only=data_only)
        # Release any owned DuckDB connection before replacing the Dataset.
        self.close_native()
        self._dataset = loaded.dataset
        self._split_plan = loaded.split_plan
        self._history = list(loaded.history)
        self._reattach_result = loaded.reattach
        self._ingest_report = None
        if data_only:
            self._clear_plans()
        else:
            self._restore_plans(loaded.plans)
        self._record(
            "reattach",
            {
                "path": str(path),
                "status": loaded.reattach.status,
                "data_only": data_only,
                "plans_restored": sorted(
                    key for key, value in loaded.plans.items() if value is not None
                ),
            },
        )
        return self

    def to_pandas(self) -> pd.DataFrame:
        """Escape hatch: copy the current dataset as a Pandas DataFrame."""
        frame = self.dataset.to_pandas()
        self._record(
            "to_pandas",
            result_summary={"rows": int(len(frame)), "columns": int(frame.shape[1])},
        )
        return frame

    def to_parquet(self, path: str | Path) -> Path:
        """Write the current dataset to Parquet."""
        destination = self.dataset.to_parquet(path)
        self._record(
            "to_parquet",
            {"path": str(destination)},
            result_summary={"path": str(destination)},
        )
        return destination

    def head(self, n: int = 5) -> pd.DataFrame:
        """Preview the first rows."""
        frame = self.dataset.head(n)
        self._record(
            "head",
            {"n": n},
            result_summary={"rows": int(len(frame)), "columns": int(frame.shape[1])},
        )
        return frame

    def with_mode(self, mode: DataMode | str) -> Session:
        """Record a mode override on the dataset metadata (Phase-1 marker)."""
        self.dataset.mode = DataMode(mode)
        self._record("with_mode", {"mode": self.dataset.mode.value})
        return self

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
        :class:`~buildml.core.errors.MissingExtraError`.
        """
        from buildml.data.engines import get_engine

        chosen = EngineName(engine)
        get_engine(chosen)  # validates install / raises MissingExtraError
        self.dataset.engine = chosen
        if chosen == EngineName.PANDAS:
            self.dataset.clear_native()
        else:
            self.dataset.attach_native(rebuild=True)
        self._record(
            "with_engine",
            {"engine": chosen.value, "has_native": self.dataset.has_native},
        )
        return self

    def sync_native(self) -> Session:
        """Rebuild ``Dataset.native`` from the current Pandas frame (eager).

        Session preprocess transforms already sync when ``engine`` is Polars or
        DuckDB. Call this after external Pandas mutation of ``dataset.frame``,
        or after a transform that opted out of sync. This is not a lazy plan
        of prior steps — it converts the full current frame into the engine
        table.
        """
        has_native = False
        if self.dataset.engine != EngineName.PANDAS:
            self.dataset.sync_native()
            has_native = self.dataset.has_native
        self._record(
            "sync_native",
            {"engine": self.dataset.engine.value, "has_native": has_native},
        )
        return self

    def metadata(self) -> dict[str, Any]:
        """Session/dataset metadata snapshot."""
        payload: dict[str, Any] = {
            "has_dataset": self._dataset is not None,
            "ingest_report": None if self._ingest_report is None else self._ingest_report.to_dict(),
            "split": None if self._split_plan is None else self._split_plan.to_dict(),
            "history": self.history,
            "reattach": None
            if self._reattach_result is None
            else {
                "status": self._reattach_result.status,
                "messages": list(self._reattach_result.messages),
            },
        }
        if self._dataset is not None:
            payload["dataset"] = self._dataset.metadata()
        return payload

    def workflow(self) -> tuple[WorkflowStep, ...]:
        """Resolve every public operation against current workflow state."""
        return resolve_workflow(self)

    def walkthrough(
        self,
        *,
        export_html: str | Path | None = None,
    ) -> WorkflowWalkthroughReport:
        """Build a workflow walkthrough from resolver state and history."""
        report = build_walkthrough(self)
        if export_html is not None:
            report.export_html(export_html)
        self._last_walkthrough = report
        return report

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
        return explain_session(self, operation, moment=moment)

    def _session_preprocess_applied(self) -> bool:
        """True when Session-level train-global preprocess plans exist."""
        return any(plan is not None for plan in self._plan_objects().values())

    def _plan_objects(self) -> dict[str, Any]:
        return {
            "impute_plan": self._impute_plan,
            "encode_plan": self._encode_plan,
            "scale_plan": self._scale_plan,
            "date_plan": self._date_plan,
            "outlier_plan": self._outlier_plan,
            "binning_plan": self._binning_plan,
            "feature_select_plan": self._feature_select_plan,
            "text_plan": self._text_plan,
            "reduce_plan": self._reduce_plan,
            "custom_plan": self._custom_plan,
            "resample_plan": self._resample_plan,
        }

    def _preprocess_summary(self) -> dict[str, Any]:
        return {
            "impute": None if self._impute_plan is None else self._impute_plan.to_dict(),
            "encode": None if self._encode_plan is None else self._encode_plan.to_dict(),
            "scale": None if self._scale_plan is None else self._scale_plan.to_dict(),
            "dates": None if self._date_plan is None else self._date_plan.to_dict(),
            "outliers": None if self._outlier_plan is None else self._outlier_plan.to_dict(),
            "binning": None if self._binning_plan is None else self._binning_plan.to_dict(),
            "feature_select": (
                None if self._feature_select_plan is None else self._feature_select_plan.to_dict()
            ),
            "text": None if self._text_plan is None else self._text_plan.to_dict(),
            "reduce": None if self._reduce_plan is None else self._reduce_plan.to_dict(),
            "custom": None if self._custom_plan is None else self._custom_plan.to_dict(),
            "resample": None if self._resample_plan is None else self._resample_plan.to_dict(),
        }

    def _restore_plans(self, plans: dict[str, Any] | None) -> None:
        payload = plans or {}
        self._impute_plan = payload.get("impute_plan")
        self._encode_plan = payload.get("encode_plan")
        self._scale_plan = payload.get("scale_plan")
        self._date_plan = payload.get("date_plan")
        self._outlier_plan = payload.get("outlier_plan")
        self._binning_plan = payload.get("binning_plan")
        self._feature_select_plan = payload.get("feature_select_plan")
        self._text_plan = payload.get("text_plan")
        self._reduce_plan = payload.get("reduce_plan")
        self._custom_plan = payload.get("custom_plan")
        self._resample_plan = payload.get("resample_plan")

    def _clear_plans(self) -> None:
        self._impute_plan = None
        self._encode_plan = None
        self._scale_plan = None
        self._date_plan = None
        self._outlier_plan = None
        self._binning_plan = None
        self._feature_select_plan = None
        self._text_plan = None
        self._reduce_plan = None
        self._custom_plan = None
        self._resample_plan = None

    def _record(
        self,
        action: str,
        details: dict[str, Any] | None = None,
        *,
        decision_origin: Literal["automatic", "recommended", "explicit"] = "explicit",
        warnings: list[str] | tuple[str, ...] = (),
        result_summary: dict[str, Any] | None = None,
    ) -> None:
        before = prior_state(self._history)
        after = session_state(self)
        self._history.append(
            make_operation_record(
                sequence=len(self._history) + 1,
                operation_id=action,
                parameters=details,
                decision_origin=decision_origin,
                before=before,
                after=after,
                warnings=warnings,
                result_summary=result_summary,
            )
        )
