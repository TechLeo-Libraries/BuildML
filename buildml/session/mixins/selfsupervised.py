"""Session mixin: selfsupervised domain public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from buildml.session import selfsupervised_ops
from buildml.session.mixins._shared import *  # noqa: F403


class SelfsupervisedSessionMixin:
    """Public Session methods for the selfsupervised domain."""
    # mypy: session private attrs (owned by Session.__init__)
    if TYPE_CHECKING:
        _ssl_eval_result: Any
        _ssl_fit_result: Any
        _ssl_head_fit_result: Any
        _ssl_head_plan: Any
        _ssl_plan: Any
        _ssl_transform_result: Any

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

        Session facade over :func:`buildml.session.selfsupervised_ops.fit_ssl_pretext_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        SSLFitResult
            Serializable fit summary including method, modality, and disclosures.

        See Also
        --------
        :func:`buildml.session.selfsupervised_ops.fit_ssl_pretext_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("SelfSupervisedFitResult", selfsupervised_ops.fit_ssl_pretext_op(
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
        ))

    def transform_ssl(
        self,
        *,
        partition: PartitionName | Literal["all"] = "train",
        attach: bool = False,
    ) -> SelfSupervisedTransformResult:
        """Export SSL representations with the train-fitted pretext encoder.

        Session facade over :func:`buildml.session.selfsupervised_ops.transform_ssl_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        SSLTransformResult
            Embedding matrix metadata and optional attached column names.

        See Also
        --------
        :func:`buildml.session.selfsupervised_ops.transform_ssl_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("SelfSupervisedTransformResult", selfsupervised_ops.transform_ssl_op(
            self,
            partition=partition,
            attach=attach,
        ))

    def finetune_ssl_head(
        self,
        *,
        estimator: SSLHeadEstimator = "logistic_regression",
        random_state: int | None = 0,
        unlabeled_marker: Any = None,
    ) -> SSLHeadFitResult:
        """Fit a supervised head on frozen SSL embeddings using labeled train rows.

        Session facade over :func:`buildml.session.selfsupervised_ops.finetune_ssl_head_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        SSLHeadFitResult
            Head fit summary including labeled row counts and disclosures.

        See Also
        --------
        :func:`buildml.session.selfsupervised_ops.finetune_ssl_head_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("SSLHeadFitResult", selfsupervised_ops.finetune_ssl_head_op(
            self,
            estimator=estimator,
            random_state=random_state,
            unlabeled_marker=unlabeled_marker,
        ))

    def evaluate_ssl(
        self,
        *,
        partition: PartitionName | Literal["all"] = "validation",
        unlabeled_marker: Any = None,
    ) -> SelfSupervisedEvalResult:
        """Evaluate frozen SSL pretext encoder and head on a labeled partition.

        Session facade over :func:`buildml.session.selfsupervised_ops.evaluate_ssl_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        SSLEvalResult
            Holdout metrics for the frozen pretext + head pipeline.

        See Also
        --------
        :func:`buildml.session.selfsupervised_ops.evaluate_ssl_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("SelfSupervisedEvalResult", selfsupervised_ops.evaluate_ssl_op(
            self,
            partition=partition,
            unlabeled_marker=unlabeled_marker,
        ))

    @property
    def ssl_plan(self) -> SelfSupervisedPlan | None:
        """Return the last self-supervised pretext plan, if any.

        Session-held result for ``ssl_plan``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("SelfSupervisedPlan | None", self._ssl_plan)

    @property
    def ssl_fit_result(self) -> SelfSupervisedFitResult | None:
        """Return the last self-supervised fit result, if any.

        Session-held result for ``ssl_fit_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("SelfSupervisedFitResult | None", self._ssl_fit_result)

    @property
    def ssl_transform_result(self) -> SelfSupervisedTransformResult | None:
        """Return the last SSL transform result, if any.

        Session-held result for ``ssl_transform_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("SelfSupervisedTransformResult | None", self._ssl_transform_result)

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
        return cast("SSLHeadPlan | None", self._ssl_head_plan)

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
        return cast("SSLHeadFitResult | None", self._ssl_head_fit_result)

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
        return cast("SelfSupervisedEvalResult | None", self._ssl_eval_result)

    def save_ssl_bundle(self, path: str | Path) -> Path:
        """Persist the active SSL plan as ``buildml.ssl_bundle.v2``.

        Session facade over :func:`buildml.session.selfsupervised_ops.save_ssl_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.

        See Also
        --------
        :func:`buildml.session.selfsupervised_ops.save_ssl_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Path", selfsupervised_ops.save_ssl_bundle_op(self, path=path))

    def load_ssl_bundle(self, path: str | Path, *, trusted: bool = False) -> Session:
        """Load a self-supervised bundle into this Session.

        Session facade over :func:`buildml.session.selfsupervised_ops.load_ssl_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        Session
            this Session with SSLPlan and optional head plan attached.

        See Also
        --------
        :func:`buildml.session.selfsupervised_ops.load_ssl_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", selfsupervised_ops.load_ssl_bundle_op(self, path=path, trusted=trusted))

    @staticmethod
    def ssl_capability_matrix() -> dict[str, Any]:
        """
        Report which self-supervised learning backends are available on this machine.

        Call before contrastive or masked-model fit methods to confirm torch and
        industry SSL extras. Read-only: no Session state is changed.

        Returns
        -------
        dict[str, Any]
            SSL backends, pretext tasks, and install hints from
            :func:`buildml.selfsupervised.torch.catalog.ssl_capability_matrix`.
        """
        from buildml.selfsupervised.torch.catalog import ssl_capability_matrix

        return cast("dict[str, Any]", ssl_capability_matrix())
