"""Session mixin: semisupervised domain public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from buildml.session import semisupervised_ops
from buildml.session.mixins._shared import *  # noqa: F403


class SemisupervisedSessionMixin:
    """Public Session methods for the semisupervised domain."""
    # mypy: session private attrs (owned by Session.__init__)
    if TYPE_CHECKING:
        _semisupervised_eval_result: Any
        _semisupervised_fit_result: Any
        _semisupervised_plan: Any
        _semisupervised_predict_result: Any

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

        Session facade over :func:`buildml.session.semisupervised_ops.fit_semisupervised_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        SemiSupervisedFitResult
            Serializable fit summary including labeled/unlabeled train counts.

        See Also
        --------
        :func:`buildml.session.semisupervised_ops.fit_semisupervised_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("SemiSupervisedFitResult", semisupervised_ops.fit_semisupervised_op(
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
        ))

    def predict_semisupervised(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        attach: bool = False,
        prediction_column: str = "semisupervised_prediction",
    ) -> SemiSupervisedPredictResult:
        """Predict with the train-fitted semi-supervised plan without refitting.

        Session facade over :func:`buildml.session.semisupervised_ops.predict_semisupervised_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        SemiSupervisedPredictResult
            Predictions and optional probabilities for the requested partition.

        See Also
        --------
        :func:`buildml.session.semisupervised_ops.predict_semisupervised_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("SemiSupervisedPredictResult", semisupervised_ops.predict_semisupervised_op(
            self,
            partition=partition,
            attach=attach,
            prediction_column=prediction_column,
        ))

    def evaluate_semisupervised(
        self,
        *,
        partition: PartitionName | Literal["all"] = "validation",
        unlabeled_marker: Any = None,
    ) -> SemiSupervisedEvalResult:
        """Evaluate the semi-supervised plan on labeled rows of a holdout partition.

        Session facade over :func:`buildml.session.semisupervised_ops.evaluate_semisupervised_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        SemiSupervisedEvalResult
            Holdout metrics computed on labeled rows only.

        See Also
        --------
        :func:`buildml.session.semisupervised_ops.evaluate_semisupervised_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("SemiSupervisedEvalResult", semisupervised_ops.evaluate_semisupervised_op(
            self,
            partition=partition,
            unlabeled_marker=unlabeled_marker,
        ))

    @property
    def semisupervised_plan(self) -> SemiSupervisedPlan | None:
        """Return the last semi-supervised plan, if any.

        Session-held result for ``semisupervised_plan``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("SemiSupervisedPlan | None", self._semisupervised_plan)

    @property
    def semisupervised_fit_result(self) -> SemiSupervisedFitResult | None:
        """Return the last semi-supervised fit result, if any.

        Session-held result for ``semisupervised_fit_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("SemiSupervisedFitResult | None", self._semisupervised_fit_result)

    @property
    def semisupervised_predict_result(self) -> SemiSupervisedPredictResult | None:
        """Return the last semi-supervised prediction result, if any.

        Session-held result for ``semisupervised_predict_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("SemiSupervisedPredictResult | None", self._semisupervised_predict_result)

    @property
    def semisupervised_eval_result(self) -> SemiSupervisedEvalResult | None:
        """Return the last semi-supervised evaluation result, if any.

        Session-held result for ``semisupervised_eval_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("SemiSupervisedEvalResult | None", self._semisupervised_eval_result)

    def save_semisupervised_bundle(self, path: str | Path) -> Path:
        """Persist the semi-supervised plan as ``buildml.semisupervised_bundle.v1``.

        Session facade over :func:`buildml.session.semisupervised_ops.save_semisupervised_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.

        See Also
        --------
        :func:`buildml.session.semisupervised_ops.save_semisupervised_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Path", semisupervised_ops.save_semisupervised_bundle_op(self, path=path))

    def load_semisupervised_bundle(self, path: str | Path, *, trusted: bool = False) -> Session:
        """Load a semi-supervised bundle into this Session.

        Session facade over :func:`buildml.session.semisupervised_ops.load_semisupervised_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        Session
            this Session with semi-supervised plan attached for chaining.

        See Also
        --------
        :func:`buildml.session.semisupervised_ops.load_semisupervised_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", semisupervised_ops.load_semisupervised_bundle_op(self, path=path, trusted=trusted))

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

        return cast("dict[str, Any]", semisupervised_capability_matrix())
