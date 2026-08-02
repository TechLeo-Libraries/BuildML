"""Self-supervised learning hooks (Torch industry defaults + legacy sklearn fallback).

Phase R1 (refinement): Torch tabular contrastive (SimCLR/BYOL/VICReg), generative
(MAE/VAE), HF text SSL, vision SSL with Session API + bundle v2.

Dependency policy: industry Torch/HF backends when ``buildml[torch]`` /
``buildml[ssl]`` installed; legacy ``masked_tabular`` sklearn path deprecated.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT",
    "BUNDLE_FORMAT_V1",
    "BUNDLE_FORMAT_V2",
    "CHECKPOINT_BOUNDARY",
    "MaskedTabularEncoder",
    "SSLHeadEstimator",
    "SSLHeadFitResult",
    "SSLHeadPlan",
    "SelfSupervisedConfig",
    "SelfSupervisedEvalResult",
    "SelfSupervisedFitResult",
    "SelfSupervisedMethod",
    "SelfSupervisedPlan",
    "SelfSupervisedTransformResult",
    "evaluate_ssl",
    "finetune_ssl_head",
    "fit_ssl_pretext",
    "list_ssl_methods",
    "load_ssl_bundle",
    "save_ssl_bundle",
    "selfsupervised_status",
    "selfsupervised_status_for_session",
    "transform_ssl",
]


def __getattr__(name: str) -> Any:
    if name in {"SelfSupervisedMethod", "SSLHeadEstimator", "SelfSupervisedConfig"}:
        from buildml.selfsupervised import types as types_mod

        return getattr(types_mod, name)
    if name in {
        "SelfSupervisedPlan",
        "SelfSupervisedFitResult",
        "SelfSupervisedTransformResult",
        "SSLHeadPlan",
        "SSLHeadFitResult",
        "SelfSupervisedEvalResult",
    }:
        from buildml.selfsupervised import results as results_mod

        return getattr(results_mod, name)
    if name == "MaskedTabularEncoder":
        from buildml.selfsupervised.encoder import MaskedTabularEncoder

        return MaskedTabularEncoder
    if name == "fit_ssl_pretext":
        from buildml.selfsupervised.fit import fit_ssl_pretext

        return fit_ssl_pretext
    if name == "transform_ssl":
        from buildml.selfsupervised.transform import transform_ssl

        return transform_ssl
    if name == "finetune_ssl_head":
        from buildml.selfsupervised.finetune import finetune_ssl_head

        return finetune_ssl_head
    if name == "evaluate_ssl":
        from buildml.selfsupervised.evaluate import evaluate_ssl

        return evaluate_ssl
    if name in {
        "BUNDLE_FORMAT",
        "BUNDLE_FORMAT_V1",
        "BUNDLE_FORMAT_V2",
        "CHECKPOINT_BOUNDARY",
        "save_ssl_bundle",
        "load_ssl_bundle",
    }:
        from buildml.selfsupervised import checkpoint as checkpoint_mod

        return getattr(checkpoint_mod, name)
    if name in {"selfsupervised_status", "selfsupervised_status_for_session"}:
        from buildml.selfsupervised import explain_hooks as hooks

        return getattr(hooks, name)
    if name == "list_ssl_methods":
        from buildml.selfsupervised.torch.catalog import list_ssl_methods

        return list_ssl_methods
    raise AttributeError(f"module 'buildml.selfsupervised' has no attribute {name!r}")
