"""Session mixin: anomaly domain public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from buildml.session import anomaly_ops
from buildml.session.mixins._shared import *  # noqa: F403


class AnomalySessionMixin:
    """Public Session methods for the anomaly domain.

    Preferred namespaced API: ``session.anomaly.*`` (domain flat actions emit DeprecationWarning until BuildML 3.0).
    """
    # mypy: session private attrs (owned by Session.__init__)
    if TYPE_CHECKING:
        _anomaly_eval_result: Any
        _anomaly_fit_result: Any
        _anomaly_plan: Any
        _anomaly_score_result: Any

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

        Session facade over :func:`buildml.session.anomaly_ops.fit_anomaly`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        AnomalyFitResult
            Serializable fit summary including threshold and alert-rate disclosures.

        See Also
        --------
        :func:`buildml.session.anomaly_ops.fit_anomaly`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("AnomalyFitResult", anomaly_ops.fit_anomaly(
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
        ))

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

        Session facade over :func:`buildml.session.anomaly_ops.tune_anomaly_threshold_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        AnomalyThresholdTuneResult
            Tuned threshold, metric value, and partition used for search.

        See Also
        --------
        :func:`buildml.session.anomaly_ops.tune_anomaly_threshold_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("AnomalyThresholdTuneResult", anomaly_ops.tune_anomaly_threshold_op(
            self,
            partition=partition,
            label_column=label_column,
            positive_label=positive_label,
            metric=metric,
            fbeta=fbeta,
            allow_test_tuning=allow_test_tuning,
            update_plan=update_plan,
        ))

    @staticmethod
    def anomaly_capability_matrix() -> dict[str, Any]:
        """Return the anomaly backend/method capability matrix for this install.

        Session facade over :func:`buildml.session.anomaly_ops.anomaly_capability_matrix_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        dict
            Nested map of backend identifiers to supported methods and modes.

        See Also
        --------
        :func:`buildml.session.anomaly_ops.anomaly_capability_matrix_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("dict[str, Any]", anomaly_ops.anomaly_capability_matrix_op())

    def score_anomalies(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        attach: bool = False,
        override_threshold: float | None = None,
    ) -> AnomalyScoreResult:
        """Score and flag rows with the train-fitted anomaly plan without refitting.

        Session facade over :func:`buildml.session.anomaly_ops.score_anomalies_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        AnomalyScoreResult
            Scores, flags, and optional alert-rate summary for the partition.

        See Also
        --------
        :func:`buildml.session.anomaly_ops.score_anomalies_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("AnomalyScoreResult", anomaly_ops.score_anomalies_op(
            self,
            partition=partition,
            attach=attach,
            override_threshold=override_threshold,
        ))

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

        Session facade over :func:`buildml.session.anomaly_ops.evaluate_anomaly_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        AnomalyEvalResult
            Holdout classification metrics and ranking diagnostics when labels exist.

        See Also
        --------
        :func:`buildml.session.anomaly_ops.evaluate_anomaly_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("AnomalyEvalResult", anomaly_ops.evaluate_anomaly_op(
            self,
            partition=partition,
            label_column=label_column,
            positive_label=positive_label,
            k=k,
            override_threshold=override_threshold,
        ))

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
        return cast("AnomalyPlan | None", self._anomaly_plan)

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
        return cast("AnomalyFitResult | None", self._anomaly_fit_result)

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
        return cast("AnomalyScoreResult | None", self._anomaly_score_result)

    @property
    def anomaly_eval_result(self) -> AnomalyEvalResult | None:
        """Return the last anomaly evaluation result, if any.

        Session-held result for ``anomaly_eval_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("AnomalyEvalResult | None", self._anomaly_eval_result)

    def save_anomaly_bundle(self, path: str | Path) -> Path:
        """Persist the active anomaly plan as ``buildml.anomaly_bundle.v1``.

        Session facade over :func:`buildml.session.anomaly_ops.save_anomaly_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.

        See Also
        --------
        :func:`buildml.session.anomaly_ops.save_anomaly_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Path", anomaly_ops.save_anomaly_bundle_op(self, path=path))

    def load_anomaly_bundle(self, path: str | Path, *, trusted: bool = False) -> Session:
        """Load an anomaly bundle into this Session.

        Session facade over :func:`buildml.session.anomaly_ops.load_anomaly_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        Session
            this Session with anomaly plan attached for chaining.

        See Also
        --------
        :func:`buildml.session.anomaly_ops.load_anomaly_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", anomaly_ops.load_anomaly_bundle_op(self, path=path, trusted=trusted))
