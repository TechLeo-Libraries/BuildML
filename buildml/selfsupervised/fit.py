"""Train-only self-supervised pretext fit (Torch default; sklearn legacy fallback)."""

from __future__ import annotations

import warnings
from typing import Any

from buildml.core.errors import MissingExtraError, ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition, frame_for_partition
from buildml.dl.extras import torch_available
from buildml.selfsupervised.encoder import MaskedTabularEncoder
from buildml.selfsupervised.features import (
    matrix_from_frame,
    representation_column_names,
    resolve_ssl_columns,
)
from buildml.selfsupervised.results import SelfSupervisedFitResult, SelfSupervisedPlan
from buildml.selfsupervised.torch.catalog import (
    LEGACY_SKLEARN_METHODS,
    TORCH_METHODS,
    method_modality,
    resolve_default_tabular_method,
)
from buildml.selfsupervised.types import SelfSupervisedConfig, SelfSupervisedMethod

_LEGACY_DEPRECATION = (
    "method='masked_tabular' uses the legacy sklearn MLP path and is deprecated. "
    "Install buildml[torch] and use simclr_tabular, byol_tabular, vicreg_tabular, "
    "mae_tabular, or vae_tabular instead. See guides/quickstart-selfsupervised.md."
)


def fit_ssl_pretext(
    dataset: Dataset,
    split_plan: SplitPlan | None,
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
    reduce_plan: Any | None = None,
    representation_prefix: str = "ssl_emb",
    backbone: str = "resnet18",
    weight_mode: str = "mock",
    hf_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cpu",
) -> tuple[SelfSupervisedPlan, SelfSupervisedFitResult]:
    """Fit a self-supervised pretext encoder on the train partition only.

    Industry default when ``buildml[torch]`` is installed: ``simclr_tabular``.
    Legacy ``masked_tabular`` remains as sklearn fallback with DeprecationWarning.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None

    resolved_method = _resolve_method(method)
    modality = method_modality(resolved_method)
    train = frame_for_partition(dataset, split_plan, "train")
    n_train = int(len(train))
    warnings_list: list[str] = []
    disclosures: list[str] = []
    explicit_legacy = method == "masked_tabular"

    if resolved_method in LEGACY_SKLEARN_METHODS:
        if explicit_legacy:
            warnings.warn(_LEGACY_DEPRECATION, DeprecationWarning, stacklevel=2)
            warnings_list.append(_LEGACY_DEPRECATION)
        return _fit_legacy_masked_tabular(
            dataset,
            train,
            split_plan,
            resolved_method,
            columns,
            random_state,
            latent_dim,
            hidden,
            mask_ratio,
            n_mask_views,
            max_iter,
            prefer_reduce_components,
            reduce_plan,
            representation_prefix,
            n_train,
            disclosures,
            warnings_list,
        )

    if resolved_method in TORCH_METHODS and not torch_available():
        raise MissingExtraError(
            "torch",
            f"SSL method {resolved_method!r} requires PyTorch (pip install 'buildml[torch]')",
        )

    if modality == "tabular":
        return _fit_torch_tabular(
            dataset,
            train,
            resolved_method,
            columns,
            random_state,
            latent_dim,
            hidden,
            mask_ratio,
            epochs,
            batch_size,
            learning_rate,
            temperature,
            projector_dim,
            projector_hidden,
            prefer_reduce_components,
            reduce_plan,
            representation_prefix,
            device,
            n_train,
            disclosures,
            warnings_list,
        )
    if modality == "text":
        return _fit_torch_text(
            dataset,
            train,
            resolved_method,
            text_column,
            random_state,
            latent_dim,
            epochs,
            batch_size,
            learning_rate,
            hf_model_name,
            weight_mode,
            device,
            representation_prefix,
            n_train,
            disclosures,
            warnings_list,
        )
    if modality == "vision":
        return _fit_torch_vision(
            dataset,
            train,
            resolved_method,
            image_column,
            random_state,
            latent_dim,
            projector_dim,
            epochs,
            batch_size,
            learning_rate,
            temperature,
            backbone,
            weight_mode,
            device,
            representation_prefix,
            n_train,
            disclosures,
            warnings_list,
        )
    raise ValidationError(f"Unsupported SSL method {resolved_method!r}")


def _resolve_method(method: SelfSupervisedMethod | None) -> str:
    if method is None:
        return resolve_default_tabular_method()
    return str(method)


def _fit_legacy_masked_tabular(
    dataset: Dataset,
    train: Any,
    split_plan: SplitPlan,
    method: str,
    columns: list[str] | None,
    random_state: int | None,
    latent_dim: int,
    hidden: tuple[int, ...] | list[int],
    mask_ratio: float,
    n_mask_views: int,
    max_iter: int,
    prefer_reduce_components: bool,
    reduce_plan: Any | None,
    representation_prefix: str,
    n_train: int,
    disclosures: list[str],
    warnings_list: list[str],
) -> tuple[SelfSupervisedPlan, SelfSupervisedFitResult]:
    cols, used_reduce, col_disclosures = resolve_ssl_columns(
        dataset,
        train,
        columns,
        reduce_plan=reduce_plan,
        prefer_reduce_components=prefer_reduce_components,
    )
    disclosures.extend(col_disclosures)
    x = matrix_from_frame(train, cols)
    encoder = MaskedTabularEncoder(
        latent_dim=latent_dim,
        hidden=tuple(hidden),
        mask_ratio=mask_ratio,
        n_mask_views=n_mask_views,
        max_iter=max_iter,
        random_state=random_state,
    )
    encoder.fit(x)
    rep_cols = representation_column_names(representation_prefix, latent_dim)
    disclosures.extend(_common_disclosures(encoder.reconstruction_mae_))
    config = SelfSupervisedConfig(
        method=method,  # type: ignore[arg-type]
        modality="tabular",
        columns=tuple(cols),
        random_state=random_state,
        latent_dim=int(latent_dim),
        hidden=tuple(int(h) for h in hidden),
        mask_ratio=float(mask_ratio),
        n_mask_views=int(n_mask_views),
        max_iter=int(max_iter),
        prefer_reduce_components=prefer_reduce_components,
        representation_prefix=representation_prefix,
    )
    plan = SelfSupervisedPlan(
        method=method,
        modality="tabular",
        columns=tuple(cols),
        n_train_rows=n_train,
        latent_dim=int(latent_dim),
        representation_prefix=representation_prefix,
        representation_columns=rep_cols,
        encoder_=encoder,
        reconstruction_mae_=float(encoder.reconstruction_mae_),
        pretext_loss_=None,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings_list),
        used_reduce_components=used_reduce,
        config=config.to_dict(),
        bundle_format="buildml.selfsupervised_bundle.v1",
    )
    result = SelfSupervisedFitResult(
        method=method,
        modality="tabular",
        n_train_rows=n_train,
        columns=tuple(cols),
        latent_dim=int(latent_dim),
        reconstruction_mae=float(encoder.reconstruction_mae_),
        pretext_loss=None,
        representation_columns=rep_cols,
        used_reduce_components=used_reduce,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings_list),
    )
    return plan, result


def _fit_torch_tabular(
    dataset: Dataset,
    train: Any,
    method: str,
    columns: list[str] | None,
    random_state: int | None,
    latent_dim: int,
    hidden: tuple[int, ...] | list[int],
    mask_ratio: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    temperature: float,
    projector_dim: int,
    projector_hidden: tuple[int, ...] | list[int],
    prefer_reduce_components: bool,
    reduce_plan: Any | None,
    representation_prefix: str,
    device: str,
    n_train: int,
    disclosures: list[str],
    warnings_list: list[str],
) -> tuple[SelfSupervisedPlan, SelfSupervisedFitResult]:
    from buildml.selfsupervised.torch.encoder import TorchTabularSSLEncoder

    cols, used_reduce, col_disclosures = resolve_ssl_columns(
        dataset,
        train,
        columns,
        reduce_plan=reduce_plan,
        prefer_reduce_components=prefer_reduce_components,
    )
    disclosures.extend(col_disclosures)
    x = matrix_from_frame(train, cols)
    encoder = TorchTabularSSLEncoder(
        method=method,
        latent_dim=latent_dim,
        hidden=tuple(int(h) for h in hidden),
        projector_hidden=tuple(int(h) for h in projector_hidden),
        projector_dim=projector_dim,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        temperature=temperature,
        mask_ratio=mask_ratio,
        random_state=random_state,
        device=device,
    )
    encoder.fit(x)
    rep_cols = representation_column_names(representation_prefix, latent_dim)
    disclosures.extend(
        _common_disclosures(encoder.reconstruction_mae_)
        + [
            f"Torch SSL method={method} (industry contrastive/generative default path).",
            f"Final pretext loss={encoder.pretext_loss_}.",
        ]
    )
    config = SelfSupervisedConfig(
        method=method,  # type: ignore[arg-type]
        modality="tabular",
        columns=tuple(cols),
        random_state=random_state,
        latent_dim=int(latent_dim),
        hidden=tuple(int(h) for h in hidden),
        mask_ratio=float(mask_ratio),
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        temperature=float(temperature),
        projector_dim=int(projector_dim),
        projector_hidden=tuple(int(h) for h in projector_hidden),
        prefer_reduce_components=prefer_reduce_components,
        representation_prefix=representation_prefix,
        device=device,
    )
    plan = SelfSupervisedPlan(
        method=method,
        modality="tabular",
        columns=tuple(cols),
        n_train_rows=n_train,
        latent_dim=int(latent_dim),
        representation_prefix=representation_prefix,
        representation_columns=rep_cols,
        encoder_=encoder,
        reconstruction_mae_=encoder.reconstruction_mae_,
        pretext_loss_=encoder.pretext_loss_,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings_list),
        used_reduce_components=used_reduce,
        config=config.to_dict(),
        bundle_format="buildml.ssl_bundle.v2",
    )
    result = SelfSupervisedFitResult(
        method=method,
        modality="tabular",
        n_train_rows=n_train,
        columns=tuple(cols),
        latent_dim=int(latent_dim),
        reconstruction_mae=encoder.reconstruction_mae_,
        pretext_loss=encoder.pretext_loss_,
        representation_columns=rep_cols,
        used_reduce_components=used_reduce,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings_list),
    )
    return plan, result


def _fit_torch_text(
    dataset: Dataset,
    train: Any,
    method: str,
    text_column: str | None,
    random_state: int | None,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hf_model_name: str,
    weight_mode: str,
    device: str,
    representation_prefix: str,
    n_train: int,
    disclosures: list[str],
    warnings_list: list[str],
) -> tuple[SelfSupervisedPlan, SelfSupervisedFitResult]:
    from buildml.selfsupervised.torch.text import HFTextSSLEncoder

    col = text_column or _first_text_column(dataset, train)
    texts = train[col].astype(str).tolist()
    encoder = HFTextSSLEncoder(
        model_name=hf_model_name,
        latent_dim=latent_dim,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        random_state=random_state,
        device=device,
        weight_mode=weight_mode,
    )
    encoder.fit(texts)
    out_dim = int(encoder.latent_dim or latent_dim)
    rep_cols = representation_column_names(representation_prefix, out_dim)
    disclosures.extend(
        _common_disclosures(None)
        + [
            f"Text SSL via sentence-transformers model={hf_model_name}.",
            f"text_column={col!r}.",
        ]
    )
    config = SelfSupervisedConfig(
        method=method,  # type: ignore[arg-type]
        modality="text",
        text_column=col,
        random_state=random_state,
        latent_dim=out_dim,
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        representation_prefix=representation_prefix,
        hf_model_name=hf_model_name,
        weight_mode=weight_mode,
        device=device,
    )
    plan = SelfSupervisedPlan(
        method=method,
        modality="text",
        columns=(col,),
        n_train_rows=n_train,
        latent_dim=out_dim,
        representation_prefix=representation_prefix,
        representation_columns=rep_cols,
        encoder_=encoder,
        reconstruction_mae_=None,
        pretext_loss_=encoder.pretext_loss_,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings_list),
        used_reduce_components=False,
        config=config.to_dict(),
        bundle_format="buildml.ssl_bundle.v2",
    )
    result = SelfSupervisedFitResult(
        method=method,
        modality="text",
        n_train_rows=n_train,
        columns=(col,),
        latent_dim=out_dim,
        reconstruction_mae=None,
        pretext_loss=encoder.pretext_loss_,
        representation_columns=rep_cols,
        used_reduce_components=False,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings_list),
    )
    return plan, result


def _fit_torch_vision(
    dataset: Dataset,
    train: Any,
    method: str,
    image_column: str | None,
    random_state: int | None,
    latent_dim: int,
    projector_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    temperature: float,
    backbone: str,
    weight_mode: str,
    device: str,
    representation_prefix: str,
    n_train: int,
    disclosures: list[str],
    warnings_list: list[str],
) -> tuple[SelfSupervisedPlan, SelfSupervisedFitResult]:
    from buildml.selfsupervised.torch.vision import VisionSSLEncoder

    col = image_column or _first_image_column(train)
    images = train[col].tolist()
    encoder = VisionSSLEncoder(
        architecture=backbone,
        weight_mode=weight_mode,
        latent_dim=latent_dim,
        projector_dim=projector_dim,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        temperature=temperature,
        random_state=random_state,
        device=device,
    )
    encoder.fit(images)
    out_dim = int(encoder.latent_dim)
    rep_cols = representation_column_names(representation_prefix, out_dim)
    disclosures.extend(
        _common_disclosures(None)
        + [
            f"Vision SSL backbone={backbone} weight_mode={weight_mode}.",
            f"image_column={col!r}.",
            f"Final pretext loss={encoder.pretext_loss_}.",
        ]
    )
    config = SelfSupervisedConfig(
        method=method,  # type: ignore[arg-type]
        modality="vision",
        image_column=col,
        random_state=random_state,
        latent_dim=out_dim,
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        temperature=float(temperature),
        projector_dim=int(projector_dim),
        representation_prefix=representation_prefix,
        backbone=backbone,
        weight_mode=weight_mode,
        device=device,
    )
    plan = SelfSupervisedPlan(
        method=method,
        modality="vision",
        columns=(col,),
        n_train_rows=n_train,
        latent_dim=out_dim,
        representation_prefix=representation_prefix,
        representation_columns=rep_cols,
        encoder_=encoder,
        reconstruction_mae_=None,
        pretext_loss_=encoder.pretext_loss_,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings_list),
        used_reduce_components=False,
        config=config.to_dict(),
        bundle_format="buildml.ssl_bundle.v2",
    )
    result = SelfSupervisedFitResult(
        method=method,
        modality="vision",
        n_train_rows=n_train,
        columns=(col,),
        latent_dim=out_dim,
        reconstruction_mae=None,
        pretext_loss=encoder.pretext_loss_,
        representation_columns=rep_cols,
        used_reduce_components=False,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings_list),
    )
    return plan, result


def _common_disclosures(reconstruction_mae: float | None) -> list[str]:
    rows = [
        "Self-supervised pretext fit uses the train partition only "
        "(labels ignored; all train rows contribute features).",
        "Validation/test never enter pretext fitting.",
        "Story: fit_ssl_pretext → transform_ssl / finetune_ssl_head "
        "(or semi-supervised fine-tune on exported embeddings).",
    ]
    if reconstruction_mae is not None:
        rows.append(f"Train reconstruction MAE (diagnostic)={reconstruction_mae:.6f}.")
    return rows


def _first_text_column(dataset: Dataset, train: Any) -> str:
    from buildml.core.types import ColumnRole

    for col in dataset.role_columns(ColumnRole.FEATURE):
        if col in train.columns and train[col].dtype == object:
            return str(col)
    for col in train.columns:
        if train[col].dtype == object:
            return str(col)
    raise ValidationError(
        "Text SSL requires a text column. Pass text_column= or ingest string features."
    )


def _first_image_column(train: Any) -> str:
    for col in train.columns:
        sample = train[col].iloc[0]
        if isinstance(sample, (str, bytes)) and str(sample):
            return str(col)
    raise ValidationError(
        "Vision SSL requires an image path/array column. Pass image_column= explicitly."
    )
