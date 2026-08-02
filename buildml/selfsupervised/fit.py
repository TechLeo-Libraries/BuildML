"""Train-only self-supervised pretext fit (masked tabular autoencoder lite)."""

from __future__ import annotations

from typing import Any

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition, frame_for_partition
from buildml.selfsupervised.encoder import MaskedTabularEncoder
from buildml.selfsupervised.features import (
    matrix_from_frame,
    representation_column_names,
    resolve_ssl_columns,
)
from buildml.selfsupervised.results import SelfSupervisedFitResult, SelfSupervisedPlan
from buildml.selfsupervised.types import SelfSupervisedConfig, SelfSupervisedMethod


def fit_ssl_pretext(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    method: SelfSupervisedMethod = "masked_tabular",
    columns: list[str] | None = None,
    random_state: int | None = 0,
    latent_dim: int = 16,
    hidden: tuple[int, ...] | list[int] = (64,),
    mask_ratio: float = 0.15,
    n_mask_views: int = 3,
    max_iter: int = 200,
    prefer_reduce_components: bool = True,
    reduce_plan: Any | None = None,
    representation_prefix: str = "ssl_emb",
) -> tuple[SelfSupervisedPlan, SelfSupervisedFitResult]:
    """Fit a self-supervised pretext encoder on the train partition only.

    Uses unlabeled *and* labeled train rows as features (labels ignored). Holdout
    partitions are never used to fit the pretext. Downstream supervised /
    semi-supervised fine-tune attaches a head on labeled train only.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None
    if method != "masked_tabular":
        raise ValidationError(
            f"Unsupported SSL method {method!r}. Shipped surface: 'masked_tabular' "
            "(compact tabular pretext). Contrastive zoos / BERT-from-scratch are out of scope. "
            "For vision/audio/speech transfer, use Session.load_pretrained_backbone + "
            "attach_backbone_head (buildml[torch] / speech extras)."
        )

    train = frame_for_partition(dataset, split_plan, "train")
    cols, used_reduce, disclosures = resolve_ssl_columns(
        dataset,
        train,
        columns,
        reduce_plan=reduce_plan,
        prefer_reduce_components=prefer_reduce_components,
    )
    x = matrix_from_frame(train, cols)
    n_train = int(x.shape[0])
    warnings: list[str] = []

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
    disclosures.extend(
        [
            "Self-supervised pretext fit uses the train partition only "
            "(labels ignored; all train rows contribute features).",
            "Validation/test never enter pretext fitting.",
            "Story: fit_ssl_pretext → transform_ssl / finetune_ssl_head "
            "(or semi-supervised fine-tune on exported embeddings).",
            "Not BERT-from-scratch and not a contrastive foundation-model zoo. "
            "Vision/audio/speech freeze/finetune continues via "
            "load_pretrained_backbone / attach_backbone_head.",
            f"Train reconstruction MAE (diagnostic)={encoder.reconstruction_mae_:.6f}.",
        ]
    )

    config = SelfSupervisedConfig(
        method=method,
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
        columns=tuple(cols),
        n_train_rows=n_train,
        latent_dim=int(latent_dim),
        representation_prefix=representation_prefix,
        representation_columns=rep_cols,
        encoder_=encoder,
        reconstruction_mae_=float(encoder.reconstruction_mae_),
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
        used_reduce_components=used_reduce,
        config=config.to_dict(),
    )
    result = SelfSupervisedFitResult(
        method=method,
        n_train_rows=n_train,
        columns=tuple(cols),
        latent_dim=int(latent_dim),
        reconstruction_mae=float(encoder.reconstruction_mae_),
        representation_columns=rep_cols,
        used_reduce_components=used_reduce,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
    return plan, result
