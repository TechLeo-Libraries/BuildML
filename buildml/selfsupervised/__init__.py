"""Self-supervised learning hooks (tabular pretext → representation → head).

Phase coverage (internal tracker — depth-first; do not spray stubs)
------------------------------------------------------------------
Phase 1 (**complete**): unsupervised → ensembles → AutoML → forecasting → anomaly.

Phase 2:
  1. Semi-supervised learning — done (``buildml.semisupervised``).
  2. Self-supervised learning hooks — **this module (PASS)** (masked tabular AE
     lite + representation export + supervised head attach; zoo freeze/finetune
     remains ``load_pretrained_backbone`` / ``attach_backbone_head``).
  3. Active learning — done (``buildml.activelearning``).
  4. Online / continual (partial_fit) — done (``buildml.online``).
  5. Multi-task learning — done (``buildml.multitask``).
  6. Meta-learning — done (``buildml.metalearning``).
  7. Federated learning — done (``buildml.federated``).
  8. Bayesian / probabilistic — done (``buildml.probabilistic``); next = Causal.
  Later: graph, evolutionary,
  symbolic, CBR, IL+RL, TDA, recommenders / LTR / KG / optimisation / synthetic /
  NLP-CV deepenings. Speech: ASR keep/improve; TTS out.

Explicit non-goals (no product surfaces): neuromorphic/SNN, swarm zoo,
digital twins, AV stack, multi-agent world sims, TTS, robotics/control product,
full COCO detection/segmentation suite.

Honesty (this package):
  - Not BERT / SimCLR / MoCo product training from scratch.
  - Complete smaller surface: masked tabular reconstruction + embedding export +
    supervised head fine-tune, with leakage-safe train-only pretext.
  - Vision/audio/speech SSL-style transfer reuses existing Torch zoo hooks under
    ``buildml[torch]`` / ``buildml[speech]`` — not reimplemented here.

Dependency policy: core masked-tabular path uses numpy/pandas/sklearn (no
optional extra). Torch zoo transfer remains optional extras.

Lazy imports — core never grows heavy SSL stacks.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT",
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
        "CHECKPOINT_BOUNDARY",
        "save_ssl_bundle",
        "load_ssl_bundle",
    }:
        from buildml.selfsupervised import checkpoint as checkpoint_mod

        return getattr(checkpoint_mod, name)
    if name in {"selfsupervised_status", "selfsupervised_status_for_session"}:
        from buildml.selfsupervised import explain_hooks as hooks

        return getattr(hooks, name)
    raise AttributeError(f"module 'buildml.selfsupervised' has no attribute {name!r}")
