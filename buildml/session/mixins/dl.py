"""Session mixin: dl domain public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from buildml.session import dl_ops
from buildml.session.mixins._shared import *  # noqa: F403


class DlSessionMixin:
    """Public Session methods for the dl domain."""
    # mypy: session private attrs (owned by Session.__init__)
    if TYPE_CHECKING:
        _dl_asr_eval: Any
        _dl_backbone: Any
        _dl_backbone_head: Any
        _dl_cv_result: Any
        _dl_ddp_result: Any
        _dl_export_result: Any
        _dl_nested_cv_result: Any
        _dl_search_result: Any
        _dl_speech_result: Any
        _dl_train_result: Any

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

        return cast("dict[str, Any]", dl_capability_matrix())

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

        Session facade over :func:`buildml.session.dl_ops.make_torch_loaders`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        TorchLoaderBundle
            Loaders keyed by partition plus the feature contract.

        See Also
        --------
        :func:`buildml.session.dl_ops.make_torch_loaders`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("TorchLoaderBundle", dl_ops.make_torch_loaders(
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
        ))

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
        """Build token-id DataLoaders for text classification (non-tabular modality).

        Session facade over :func:`buildml.session.dl_ops.make_text_torch_loaders`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        TorchLoaderBundle
            Text loaders plus vocabulary and text contract metadata.

        See Also
        --------
        :func:`buildml.session.dl_ops.make_text_torch_loaders`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("TorchLoaderBundle", dl_ops.make_text_torch_loaders(
            self,
            text_column=text_column,
            batch_size=batch_size,
            max_len=max_len,
            max_vocab=max_vocab,
            min_freq=min_freq,
            shuffle_train=shuffle_train,
            seed=seed,
        ))

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

        Session facade over :func:`buildml.session.dl_ops.fit_torch`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        Session
            ``self`` with ``dl_train_result`` attached for chaining.

        See Also
        --------
        :func:`buildml.session.dl_ops.fit_torch`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", dl_ops.fit_torch(
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
        ))

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
        """Build fused multimodal DataLoaders (tabular/text/image/audio mixes).

        Session facade over :func:`buildml.session.dl_ops.make_multimodal_torch_loaders`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        TorchLoaderBundle
            Multimodal loaders plus contracts and preprocess disclosures.

        See Also
        --------
        :func:`buildml.session.dl_ops.make_multimodal_torch_loaders`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("TorchLoaderBundle", dl_ops.make_multimodal_torch_loaders(
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
        ))

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

        Session facade over :func:`buildml.session.dl_ops.make_image_multimodal_torch_loaders`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        TorchLoaderBundle
            Image-centric multimodal loaders plus contracts.

        See Also
        --------
        :func:`buildml.session.dl_ops.make_image_multimodal_torch_loaders`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("TorchLoaderBundle", dl_ops.make_image_multimodal_torch_loaders(
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
        ))

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

        Session facade over :func:`buildml.session.dl_ops.make_audio_multimodal_torch_loaders`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        TorchLoaderBundle
            Audio-centric multimodal loaders plus contracts.

        See Also
        --------
        :func:`buildml.session.dl_ops.make_audio_multimodal_torch_loaders`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("TorchLoaderBundle", dl_ops.make_audio_multimodal_torch_loaders(
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
        ))

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
        """Fold-local Torch CV on the attached numeric tabular dataset.

        Session facade over :func:`buildml.session.dl_ops.cross_validate_torch`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        TorchCVResult
            Per-fold metrics and mean summary.

        See Also
        --------
        :func:`buildml.session.dl_ops.cross_validate_torch`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("TorchCVResult", dl_ops.cross_validate_torch(
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
        ))

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
        """Inner-fold Torch hyperparameter search on the Session train universe.

        Session facade over :func:`buildml.session.dl_ops.search_torch`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        TorchSearchResult
            Best params, inner CV scores, and search disclosures.

        See Also
        --------
        :func:`buildml.session.dl_ops.search_torch`
            Canonical documentation for parameters, raises, and examples.
        """
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
        """Nested Torch CV: outer evaluation after fold-local inner hyperparameter search.

        Session facade over :func:`buildml.session.dl_ops.nested_cv_torch`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        TorchNestedCVResult
            Outer-fold metrics, inner search summaries, and disclosures.

        See Also
        --------
        :func:`buildml.session.dl_ops.nested_cv_torch`
            Canonical documentation for parameters, raises, and examples.
        """
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
        """Export the last Torch trainer to TorchScript or ONNX.

        Session facade over :func:`buildml.session.dl_ops.export_torch`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        TorchExportResult
            Export path, format, and limitation disclosures.

        See Also
        --------
        :func:`buildml.session.dl_ops.export_torch`
            Canonical documentation for parameters, raises, and examples.
        """
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
        """DDP training via a fresh ``module_factory`` per process.

        Session facade over :func:`buildml.session.dl_ops.fit_torch_ddp`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        DDPTrainResult
            DDP run summary and optional aggregated train result.

        See Also
        --------
        :func:`buildml.session.dl_ops.fit_torch_ddp`
            Canonical documentation for parameters, raises, and examples.
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
        """Build speech classification loaders (finetune-lite encoder path).

        Session facade over :func:`buildml.session.dl_ops.make_speech_torch_loaders`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        TorchLoaderBundle
            Speech loaders plus speech contract metadata.

        See Also
        --------
        :func:`buildml.session.dl_ops.make_speech_torch_loaders`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("TorchLoaderBundle", dl_ops.make_speech_torch_loaders(
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
        ))

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
        """Fine-tune a tiny speech encoder + classifier head (finetune-lite).

        Session facade over :func:`buildml.session.dl_ops.fit_speech_torch`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        Session
            ``self`` with ``dl_train_result`` attached for chaining.

        See Also
        --------
        :func:`buildml.session.dl_ops.fit_speech_torch`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", dl_ops.fit_speech_torch(
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
        ))

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
        """ASR transcription for an audio feature column.

        Session facade over :func:`buildml.session.dl_ops.transcribe_speech`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        SpeechTranscribeResult
            Transcripts, model metadata, and row counts.

        See Also
        --------
        :func:`buildml.session.dl_ops.transcribe_speech`
            Canonical documentation for parameters, raises, and examples.
        """
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
        trusted: bool = False,
    ) -> Any:
        """Launch BuildML managed serving for a pipeline or TorchScript artifact.

        Session facade over :func:`buildml.session.dl_ops.serve_bundle`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        ServeHandle
            Running server handle with URL and lifecycle controls.

        See Also
        --------
        :func:`buildml.session.dl_ops.serve_bundle`
            Canonical documentation for parameters, raises, and examples.
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
            trusted=trusted,
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
        """Load a curated pretrained vision/audio/speech backbone (integration hook).

        Session facade over :func:`buildml.session.dl_ops.load_pretrained_backbone`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        PretrainedBackbone
            Loaded backbone metadata and module shell.

        See Also
        --------
        :func:`buildml.session.dl_ops.load_pretrained_backbone`
            Canonical documentation for parameters, raises, and examples.
        """
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
        """Attach a classification head to the Session pretrained backbone.

        Session facade over :func:`buildml.session.dl_ops.attach_backbone_head`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        BackboneHeadBundle
            Combined backbone+head module metadata.

        See Also
        --------
        :func:`buildml.session.dl_ops.attach_backbone_head`
            Canonical documentation for parameters, raises, and examples.
        """
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
        """Score ASR hypotheses vs references (WER/CER); reuse last transcription texts.

        Session facade over :func:`buildml.session.dl_ops.evaluate_asr`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        AsrEvalResult
            WER/CER metrics and scoring metadata.

        See Also
        --------
        :func:`buildml.session.dl_ops.evaluate_asr`
            Canonical documentation for parameters, raises, and examples.
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
        """Pack a TorchScript artifact into a TorchServe-ready directory layout.

        Session facade over :func:`buildml.session.dl_ops.pack_torchserve`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        PackagingResult
            Output paths and packaging disclosures.

        See Also
        --------
        :func:`buildml.session.dl_ops.pack_torchserve`
            Canonical documentation for parameters, raises, and examples.
        """
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
        """Write a TensorRT ``trtexec`` plan next to a validated ONNX artifact.

        Session facade over :func:`buildml.session.dl_ops.prepare_tensorrt_export`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        PackagingResult
            Export plan paths and limitation disclosures.

        See Also
        --------
        :func:`buildml.session.dl_ops.prepare_tensorrt_export`
            Canonical documentation for parameters, raises, and examples.
        """
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
        """Emit a Kubernetes Job YAML for torchrun multi-node DDP (template only).

        Session facade over :func:`buildml.session.dl_ops.emit_k8s_ddp_job`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        K8sManifestResult
            Written manifest paths and template limitations.

        See Also
        --------
        :func:`buildml.session.dl_ops.emit_k8s_ddp_job`
            Canonical documentation for parameters, raises, and examples.
        """
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
        """Emit a Kubernetes Deployment+Service YAML for managed serve (template only).

        Session facade over :func:`buildml.session.dl_ops.emit_k8s_serve_deployment`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        K8sManifestResult
            Written manifest paths and template limitations.

        See Also
        --------
        :func:`buildml.session.dl_ops.emit_k8s_serve_deployment`
            Canonical documentation for parameters, raises, and examples.
        """
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
        """Domain-adapt / finetune-lite speech classify (not FM continued pretrain).

        Session facade over :func:`buildml.session.dl_ops.domain_adapt_speech_torch`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        Session
            ``self`` with ``dl_train_result`` attached for chaining.

        See Also
        --------
        :func:`buildml.session.dl_ops.domain_adapt_speech_torch`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", dl_ops.domain_adapt_speech_torch(
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
        ))

    def refuse_speech_foundation_pretrain(self) -> None:
        """Refuse FM-from-scratch / large continued-pretrain with an explicit error.

        Session facade over :func:`buildml.session.dl_ops.refuse_speech_foundation_pretrain`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        See Also
        --------
        :func:`buildml.session.dl_ops.refuse_speech_foundation_pretrain`
            Canonical documentation for parameters, raises, and examples.
        """
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
        return cast("Any | None", self._dl_speech_result)

    @property
    def dl_backbone(self) -> Any | None:
        """Return the pretrained backbone loaded by the most recent zoo call.

        Stored on Session after :meth:`load_pretrained_backbone` for head attachment
        or finetune-lite workflows.

        Returns
        -------
        PretrainedBackbone or None
            ``None`` until :meth:`load_pretrained_backbone` has run."""
        return cast("Any | None", self._dl_backbone)

    @property
    def dl_backbone_head(self) -> Any | None:
        """Return the backbone-plus-head bundle from the most recent attach.

        Stored on Session after :meth:`attach_backbone_head` for training or export.

        Returns
        -------
        BackboneHeadBundle or None
            ``None`` until :meth:`attach_backbone_head` has run."""
        return cast("Any | None", self._dl_backbone_head)

    @property
    def dl_asr_eval(self) -> Any | None:
        """Return WER/CER metrics from the most recent ASR evaluation.

        Stored on Session after :meth:`evaluate_asr` for reporting and comparison.

        Returns
        -------
        AsrEvalResult or None
            ``None`` until :meth:`evaluate_asr` has run."""
        return cast("Any | None", self._dl_asr_eval)

    @property
    def dl_train_result(self) -> TrainResult | None:
        """Return the last Torch training result from fit or bundle load.

        Session-held result for ``dl_train_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("TrainResult | None", self._dl_train_result)

    @property
    def dl_cv_result(self) -> TorchCVResult | None:
        """Return the last fold-local Torch cross-validation result.

        Stored on Session after :meth:`cross_validate_torch` for model selection review.

        Returns
        -------
        TorchCVResult or None
            ``None`` until :meth:`cross_validate_torch` has run."""
        return cast("TorchCVResult | None", self._dl_cv_result)

    @property
    def dl_search_result(self) -> Any | None:
        """Return the last inner-fold Torch hyperparameter search result.

        Stored on Session after :meth:`search_torch` for reviewing best params.

        Returns
        -------
        TorchSearchResult or None
            ``None`` until :meth:`search_torch` has run."""
        return cast("Any | None", self._dl_search_result)

    @property
    def dl_nested_cv_result(self) -> Any | None:
        """Return the last nested Torch CV result with inner search.

        Stored on Session after :meth:`nested_cv_torch` for unbiased performance review.

        Returns
        -------
        TorchNestedCVResult or None
            ``None`` until :meth:`nested_cv_torch` has run."""
        return cast("Any | None", self._dl_nested_cv_result)

    @property
    def dl_export_result(self) -> Any | None:
        """Return the last TorchScript or ONNX export result.

        Stored on Session after :meth:`export_torch` for serving and packaging flows.

        Returns
        -------
        TorchExportResult or None
            ``None`` until :meth:`export_torch` has run."""
        return cast("Any | None", self._dl_export_result)

    @property
    def dl_ddp_result(self) -> Any | None:
        """Return the last distributed data parallel training result.

        Stored on Session after :meth:`fit_torch_ddp` for multi-GPU run review.

        Returns
        -------
        DDPTrainResult or None
            ``None`` until :meth:`fit_torch_ddp` has run."""
        return cast("Any | None", self._dl_ddp_result)

    def torch_training_curve(self) -> TrainingCurveReport:
        """Return structured training-curve teaching data for the last Torch run.

        Session facade over :func:`buildml.session.dl_ops.torch_training_curve`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        TrainingCurve
            Epoch-wise loss/metric series for visualization or reporting.

        See Also
        --------
        :func:`buildml.session.dl_ops.torch_training_curve`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("TrainingCurveReport", dl_ops.torch_training_curve(self))

    def evaluate_torch(
        self,
        *,
        partition: Literal["train", "validation", "test"] = "test",
        device: str | None = None,
    ) -> DLEvaluateResult:
        """Evaluate the last Torch trainer on a named partition.

        Session facade over :func:`buildml.session.dl_ops.evaluate_torch`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        TorchEvalResult
            Partition metrics for the trained module.

        See Also
        --------
        :func:`buildml.session.dl_ops.evaluate_torch`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("DLEvaluateResult", dl_ops.evaluate_torch(self, partition=partition, device=device))

    def save_torch_bundle(self, path: str | Path) -> Path:
        """Persist the last Torch trainer as ``buildml.torch_bundle.v1``.

        Session facade over :func:`buildml.session.dl_ops.save_torch_bundle`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.

        See Also
        --------
        :func:`buildml.session.dl_ops.save_torch_bundle`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Path", dl_ops.save_torch_bundle(self, path=path))

    def load_torch_bundle(
        self,
        path: str | Path,
        module: Any,
        *,
        map_location: str | None = None,
        trusted: bool = False,
    ) -> Session:
        """Load a Torch trainer bundle into this Session.

        Session facade over :func:`buildml.session.dl_ops.load_torch_bundle`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        Session
            ``self`` with ``dl_train_result`` attached for chaining.

        See Also
        --------
        :func:`buildml.session.dl_ops.load_torch_bundle`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", dl_ops.load_torch_bundle(self, path=path, module=module, map_location=map_location, trusted=trusted))
