"""Session mixin: automl domain public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from buildml.session import automl_ops
from buildml.session.mixins._shared import *  # noqa: F403


class AutomlSessionMixin:
    """Public Session methods for the automl domain."""
    # mypy: session private attrs (owned by Session.__init__)
    if TYPE_CHECKING:
        _automl_plan: Any
        _automl_result: Any

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

        Session facade over :func:`buildml.session.automl_ops.run_automl_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        AutoMLResult
            Ranked trial table, winner metadata, and search disclosures.

        See Also
        --------
        :func:`buildml.session.automl_ops.run_automl_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("AutoMLResult", automl_ops.run_automl_op(
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
        ))

    def evaluate_automl(
        self,
        *,
        partition: Literal["train", "validation", "test"] = "test",
    ) -> EvaluateResult:
        """Evaluate the last AutoML winner with classical supervised metrics.

        Session facade over :func:`buildml.session.automl_ops.evaluate_automl`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        EvaluateResult
            Metrics, diagnostics, and recommendations for the winning estimator.

        See Also
        --------
        :func:`buildml.session.automl_ops.evaluate_automl`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("EvaluateResult", automl_ops.evaluate_automl(self, partition=partition))

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
        return cast("AutoMLPlan | None", self._automl_plan)

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
        return cast("AutoMLResult | None", self._automl_result)

    def save_automl_bundle(self, path: str | Path) -> Path:
        """Persist the active AutoML plan as ``buildml.automl_bundle.v1``.

        Session facade over :func:`buildml.session.automl_ops.save_automl_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.

        See Also
        --------
        :func:`buildml.session.automl_ops.save_automl_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Path", automl_ops.save_automl_bundle_op(self, path=path))

    def load_automl_bundle(self, path: str | Path, *, trusted: bool = False) -> Session:
        """Load an AutoML bundle into this Session.

        Session facade over :func:`buildml.session.automl_ops.load_automl_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        Session
            this Session with AutoML plan and fit result attached for chaining.

        See Also
        --------
        :func:`buildml.session.automl_ops.load_automl_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", automl_ops.load_automl_bundle_op(self, path=path, trusted=trusted))

    @staticmethod
    def automl_capability_matrix() -> dict[str, Any]:
        """
        Report which AutoML search backends and model families are available here.

        Call before :meth:`run_automl` to confirm FLAML, Optuna HPO, or native
        sklearn search paths on this install. Read-only: no dataset required.

        Returns
        -------
        dict[str, Any]
            AutoML backends, search spaces, and install hints from
            :func:`buildml.automl.catalog.automl_capability_matrix`.
        """
        from buildml.automl.catalog import automl_capability_matrix

        return cast("dict[str, Any]", automl_capability_matrix())
