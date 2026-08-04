"""Session mixin: classical domain public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from buildml.session import classical_ops
from buildml.session.mixins._shared import *  # noqa: F403


class ClassicalSessionMixin:
    """Public Session methods for the classical domain.

    Preferred namespaced API: ``session.classical.*`` (classical/core dual: flat methods remain first-class without warnings).
    """
    # mypy: session private attrs (owned by Session.__init__)
    if TYPE_CHECKING:
        _fit_result: Any
        _last_cv: Any
        _last_nested_cv: Any
        _last_plot_board: Any
        _last_search: Any
        _model_card: Any

    def fit(
        self,
        estimator: Any,
        *,
        task: Literal["classification", "regression", "auto"] = "auto",
    ) -> Session:
        """Train a model on the training rows.

        Session facade over :func:`buildml.session.classical_ops.fit`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        Session
            ``self``, so the fit chains into :meth:`evaluate`. The fitted model

        See Also
        --------
        :func:`buildml.session.classical_ops.fit`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", classical_ops.fit(self, estimator=estimator, task=task))

    @property
    def fit_result(self) -> FitResult | None:
        """The trained model from the last :meth:`fit` call, plus its context.

        Session-held result for ``fit_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("FitResult | None", self._fit_result)

    def predict(
        self,
        *,
        partition: Literal["train", "validation", "test"] = "test",
        return_proba: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """Run the fitted model over one partition and return its predictions.

        Session facade over :func:`buildml.session.classical_ops.predict`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        pandas.Series or pandas.DataFrame
            A Series of predicted labels or values, indexed to match the

        See Also
        --------
        :func:`buildml.session.classical_ops.predict`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("pd.Series | pd.DataFrame", classical_ops.predict(self, partition=partition, return_proba=return_proba))

    def evaluate(
        self,
        *,
        partition: Literal["train", "validation", "test"] = "test",
        export_figures: str | Path | None = None,
        export_html: str | Path | None = None,
        include_plots: bool = False,
    ) -> EvaluateResult:
        """Measure the fitted model, and explain what the measurement means.

        Session facade over :func:`buildml.session.classical_ops.evaluate`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        ~buildml.model.supervised.EvaluateResult
            The evaluation card: ``metrics``, ``diagnostics`` (confusion

        See Also
        --------
        :func:`buildml.session.classical_ops.evaluate`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("EvaluateResult", classical_ops.evaluate(
            self,
            partition=partition,
            export_figures=export_figures,
            export_html=export_html,
            include_plots=include_plots,
        ))

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

        Session facade over :func:`buildml.session.classical_ops.eval_plots`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        ~buildml.model.plot_boards.PlotBoardReport
            The board: paths to any figures written, which panels were

        See Also
        --------
        :func:`buildml.session.classical_ops.eval_plots`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("PlotBoardReport", classical_ops.eval_plots(
            self,
            partition=partition,
            include_learning_curve=include_learning_curve,
            include_importance=include_importance,
            n_importance_repeats=n_importance_repeats,
            learning_curve_cv=learning_curve_cv,
            export_figures=export_figures,
            export_html=export_html,
            show=show,
        ))

    @property
    def last_plot_board(self) -> PlotBoardReport | None:
        """The most recent diagnostic plot board.

        Session-held result for ``last_plot_board``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("PlotBoardReport | None", self._last_plot_board)

    def compare_models(
        self,
        estimators: dict[str, Any],
        *,
        task: Literal["classification", "regression", "auto"] = "auto",
        partition: Literal["train", "validation", "test"] = "test",
        ranking_metric: str | None = None,
    ) -> ModelComparison:
        """Try several models on the same data and rank what you get.

        Session facade over :func:`buildml.session.classical_ops.compare_models`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        ~buildml.model.compare.ModelComparison
            The ranked comparison, holding each model's metrics, the ordering,

        See Also
        --------
        :func:`buildml.session.classical_ops.compare_models`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("ModelComparison", classical_ops.compare_models(
            self,
            estimators=estimators,
            task=task,
            partition=partition,
            ranking_metric=ranking_metric,
        ))

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

        Session facade over :func:`buildml.session.classical_ops.cv_score`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        ~buildml.model.selection.CVScoreResult
            Per-fold scores with their mean and standard deviation, plus an

        See Also
        --------
        :func:`buildml.session.classical_ops.cv_score`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("CVScoreResult", classical_ops.cv_score(
            self,
            estimator=estimator,
            task=task,
            cv=cv,
            cv_strategy=cv_strategy,
            scoring_metric=scoring_metric,
            groups=groups,
            preprocess=preprocess,
            allow_session_global_preprocess=allow_session_global_preprocess,
        ))

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

        Session facade over :func:`buildml.session.classical_ops.nested_cv_score`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        ~buildml.model.selection.NestedCVResult
            ``mean_metrics`` and ``std_metrics`` hold the honest estimate and

        See Also
        --------
        :func:`buildml.session.classical_ops.nested_cv_score`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("NestedCVResult", classical_ops.nested_cv_score(
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
        ))

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

        Session facade over :func:`buildml.session.classical_ops.grid_search`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        ~buildml.model.selection.SearchResult
            The ranked search: every trial with its score, the

        See Also
        --------
        :func:`buildml.session.classical_ops.grid_search`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("SearchResult", classical_ops.grid_search(
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
        ))

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

        Session facade over :func:`buildml.session.classical_ops.randomized_search`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        ~buildml.model.selection.SearchResult
            The ranked trials, ``best_params``, ``best_score``, the winner's

        See Also
        --------
        :func:`buildml.session.classical_ops.randomized_search`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("SearchResult", classical_ops.randomized_search(
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
        ))

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

        Session facade over :func:`buildml.session.classical_ops.optuna_search`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        ~buildml.model.selection.SearchResult
            The ranked trials, ``best_params``, ``best_score``, the winner's

        See Also
        --------
        :func:`buildml.session.classical_ops.optuna_search`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("SearchResult", classical_ops.optuna_search(
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
        ))

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

        Session facade over :func:`buildml.session.classical_ops.evolutionary_search`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        ~buildml.model.selection.SearchResult
            Every evaluated configuration with its score, the ``best_params``,

        See Also
        --------
        :func:`buildml.session.classical_ops.evolutionary_search`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("SearchResult", classical_ops.evolutionary_search(
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
        ))

    @property
    def last_cv(self) -> CVScoreResult | None:
        """The most recent :meth:`cv_score` result.

        Session-held result for ``last_cv``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("CVScoreResult | None", self._last_cv)

    @property
    def last_nested_cv(self) -> NestedCVResult | None:
        """The most recent :meth:`nested_cv_score` result.

        Session-held result for ``last_nested_cv``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("NestedCVResult | None", self._last_nested_cv)

    @property
    def last_search(self) -> SearchResult | None:
        """The most recent hyperparameter search result.

        Session-held result for ``last_search``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("SearchResult | None", self._last_search)

    def save_model(self, path: str | Path) -> Path:
        """Save the fitted estimator and the feature contract it expects.

        Session facade over :func:`buildml.session.classical_ops.save_model`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        pathlib.Path
            Where the bundle was written.

        See Also
        --------
        :func:`buildml.session.classical_ops.save_model`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Path", classical_ops.save_model(self, path=path))

    def load_model(self, path: str | Path, *, trusted: bool = False) -> Session:
        """Load an estimator bundle written by :meth:`save_model`.

        Session facade over :func:`buildml.session.classical_ops.load_model`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        Session
            ``self``, so the load chains into a predict.

        See Also
        --------
        :func:`buildml.session.classical_ops.load_model`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", classical_ops.load_model(self, path=path, trusted=trusted))

    def save_pipeline(
        self,
        path: str | Path,
        *,
        evaluate_partition: Literal["train", "validation", "test"] | None = "test",
        title: str | None = None,
    ) -> Path:
        """Save everything needed to score new data: model, prep, and card.

        Session facade over :func:`buildml.session.classical_ops.save_pipeline`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        pathlib.Path
            The bundle directory that was written.

        See Also
        --------
        :func:`buildml.session.classical_ops.save_pipeline`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Path", classical_ops.save_pipeline(
            self, path=path, evaluate_partition=evaluate_partition, title=title
        ))

    def load_pipeline(self, path: str | Path, *, trusted: bool = False) -> Session:
        """Restore a saved model together with its preprocessing.

        Session facade over :func:`buildml.session.classical_ops.load_pipeline`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        Session
            ``self``, so the load chains into scoring.

        See Also
        --------
        :func:`buildml.session.classical_ops.load_pipeline`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", classical_ops.load_pipeline(self, path=path, trusted=trusted))

    def predict_from_pipeline(
        self,
        path: str | Path,
        data: Dataset | pd.DataFrame | None = None,
        *,
        roles: dict[str, ColumnRole | str] | None = None,
        return_proba: bool = False,
        apply_plans: bool = True,
        trusted: bool = False,
    ) -> PipelinePredictResult:
        """Score new rows through a saved bundle, in one call.

        Session facade over :func:`buildml.session.classical_ops.predict_from_pipeline`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        ~buildml.pipeline.score.PipelinePredictResult
            The predictions plus the context needed to trust them: which

        See Also
        --------
        :func:`buildml.session.classical_ops.predict_from_pipeline`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("PipelinePredictResult", classical_ops.predict_from_pipeline(
            self,
            path=path,
            data=data,
            roles=roles,
            return_proba=return_proba,
            apply_plans=apply_plans,
            trusted=trusted,
        ))

    def prepare_design_matrix(
        self,
        *,
        partition: Literal["train", "validation", "test"] = "train",
        columns: list[str] | tuple[str, ...] | None = None,
        sample_rows: int | None = None,
        random_state: int | None = 0,
    ) -> MaterializePrepResult:
        """Narrow the data in the engine before pulling it into memory.

        Session facade over :func:`buildml.session.classical_ops.prepare_design_matrix`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        ~buildml.data.engines.prep.MaterializePrepResult
            The prepared matrix together with disclosures recording which

        See Also
        --------
        :func:`buildml.session.classical_ops.prepare_design_matrix`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("MaterializePrepResult", classical_ops.prepare_design_matrix(
            self,
            partition=partition,
            columns=columns,
            sample_rows=sample_rows,
            random_state=random_state,
        ))

    @property
    def model_card(self) -> ModelCard | None:
        """The documentation record for the saved or loaded pipeline.

        Session-held result for ``model_card``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("ModelCard | None", self._model_card)

    def calibration(
        self,
        *,
        partition: Literal["train", "validation", "test"] = "test",
        export_figures: str | Path | None = None,
        export_html: str | Path | None = None,
    ) -> DiagnosticReport:
        """Check whether predicted probabilities mean what they claim.

        Session facade over :func:`buildml.session.classical_ops.calibration`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        ~buildml.model.diagnostics.DiagnosticReport
            The calibration findings: Brier score, expected calibration error,

        See Also
        --------
        :func:`buildml.session.classical_ops.calibration`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("DiagnosticReport", classical_ops.calibration(
            self, partition=partition, export_figures=export_figures, export_html=export_html
        ))

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

        Session facade over :func:`buildml.session.classical_ops.tune_threshold`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        ~buildml.model.diagnostics.DiagnosticReport
            The sweep: metrics at every candidate threshold, the recommended

        See Also
        --------
        :func:`buildml.session.classical_ops.tune_threshold`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("DiagnosticReport", classical_ops.tune_threshold(
            self,
            partition=partition,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
            tp_benefit=tp_benefit,
            tn_benefit=tn_benefit,
            export_figures=export_figures,
            export_html=export_html,
        ))

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

        Session facade over :func:`buildml.session.classical_ops.learning_curve`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        ~buildml.model.diagnostics.DiagnosticReport
            The curve points at each training size, the train and validation

        See Also
        --------
        :func:`buildml.session.classical_ops.learning_curve`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("DiagnosticReport", classical_ops.learning_curve(
            self,
            estimator=estimator,
            task=task,
            cv=cv,
            export_figures=export_figures,
            export_html=export_html,
        ))

    def explain_shap(
        self,
        *,
        partition: Literal["train", "validation", "test"] = "test",
        max_samples: int = 100,
        random_state: int | None = 0,
    ) -> Any:
        """Attribute predictions with SHAP (requires ``buildml[shap]``).

        Session facade over :func:`buildml.session.classical_ops.explain_shap`.

        Returns
        -------
        Any
            SHAP attribution payload (``ShapExplainResult`` when shap is installed).
        """
        return classical_ops.explain_shap(
            self,
            partition=partition,
            max_samples=max_samples,
            random_state=random_state,
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

        Session facade over :func:`buildml.session.classical_ops.feature_importance`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        ~buildml.model.diagnostics.DiagnosticReport
            Per-feature importance with the spread across repeats, ranked, plus

        See Also
        --------
        :func:`buildml.session.classical_ops.feature_importance`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("DiagnosticReport", classical_ops.feature_importance(
            self,
            partition=partition,
            n_repeats=n_repeats,
            export_figures=export_figures,
            export_html=export_html,
        ))

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

        Session facade over :func:`buildml.session.classical_ops.error_slices`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        ~buildml.model.diagnostics.DiagnosticReport
            Per-segment metrics and sizes, the segments that fell below

        See Also
        --------
        :func:`buildml.session.classical_ops.error_slices`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("DiagnosticReport", classical_ops.error_slices(
            self,
            by=by,
            partition=partition,
            max_segments=max_segments,
            min_segment_n=min_segment_n,
            export_html=export_html,
        ))
