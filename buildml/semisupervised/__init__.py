"""Semi-supervised learning domain (scarce labels + unlabeled train rows).

Phase coverage (internal tracker — depth-first; do not spray stubs)
------------------------------------------------------------------
Phase 1 (**complete**): unsupervised → ensembles → AutoML → forecasting → anomaly.

Phase 2:
  1. Semi-supervised learning — **this module (PASS)**.
  2. Self-supervised learning hooks — done (``buildml.selfsupervised``).
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
  - Not active learning (querying an oracle) and not self-supervised pretext.
  - Anomaly novelty (normal-only fit) is a different Session path.
  - Validation/test never invent labels for model selection without disclosure.
  - Pseudo-labels are train-fit artifacts; eval scores labeled holdout only.

Dependency policy: core stays numpy/pandas/pyarrow/sklearn. Semi-supervised
methods use core sklearn — no optional extra required for ``import buildml``.

Lazy imports — core never grows heavy semi-supervised stacks.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT",
    "CHECKPOINT_BOUNDARY",
    "SemiSupervisedConfig",
    "SemiSupervisedEvalResult",
    "SemiSupervisedFitResult",
    "SemiSupervisedMethod",
    "SemiSupervisedPlan",
    "SemiSupervisedPredictResult",
    "evaluate_semisupervised",
    "fit_semisupervised",
    "load_semisupervised_bundle",
    "predict_semisupervised",
    "save_semisupervised_bundle",
    "semisupervised_status",
    "semisupervised_status_for_session",
]


def __getattr__(name: str) -> Any:
    if name in {"SemiSupervisedMethod", "SemiSupervisedConfig"}:
        from buildml.semisupervised import types as types_mod

        return getattr(types_mod, name)
    if name in {
        "SemiSupervisedPlan",
        "SemiSupervisedFitResult",
        "SemiSupervisedPredictResult",
        "SemiSupervisedEvalResult",
    }:
        from buildml.semisupervised import results as results_mod

        return getattr(results_mod, name)
    if name == "fit_semisupervised":
        from buildml.semisupervised.fit import fit_semisupervised

        return fit_semisupervised
    if name == "predict_semisupervised":
        from buildml.semisupervised.predict import predict_semisupervised

        return predict_semisupervised
    if name == "evaluate_semisupervised":
        from buildml.semisupervised.evaluate import evaluate_semisupervised

        return evaluate_semisupervised
    if name in {
        "BUNDLE_FORMAT",
        "CHECKPOINT_BOUNDARY",
        "save_semisupervised_bundle",
        "load_semisupervised_bundle",
    }:
        from buildml.semisupervised import checkpoint as checkpoint_mod

        return getattr(checkpoint_mod, name)
    if name in {"semisupervised_status", "semisupervised_status_for_session"}:
        from buildml.semisupervised import explain_hooks as hooks

        return getattr(hooks, name)
    raise AttributeError(f"module 'buildml.semisupervised' has no attribute {name!r}")
