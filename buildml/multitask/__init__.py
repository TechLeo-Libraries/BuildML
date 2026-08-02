"""Multi-task / multi-output learning domain (sklearn MultiOutput / Chain).

Phase coverage (internal tracker — depth-first; do not spray stubs)
------------------------------------------------------------------
Phase 1 (**complete**): unsupervised → ensembles → AutoML → forecasting → anomaly.

Phase 2:
  1. Semi-supervised learning — done (``buildml.semisupervised``).
  2. Self-supervised learning hooks — done (``buildml.selfsupervised``).
  3. Active learning — done (``buildml.activelearning``).
  4. Online / continual (partial_fit) — done (``buildml.online``).
  5. Multi-task learning — **this module** (done).
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
  - Shared-feature multi-target fitting via sklearn ``MultiOutputClassifier`` /
    ``MultiOutputRegressor`` / ``ClassifierChain`` / ``RegressorChain``.
  - Same-type tasks only (all classification or all regression). Mixed
    classification+regression targets are refused with a clear error.
  - Classical ``Session.fit`` remains single-target (``require_target()``);
    this path is distinct and requires ``>= 2`` target columns.
  - Train-only fit; validation/test are evaluation-only.
  - Not a deep multi-head MTL research platform, not multi-label binary
    relevance zoos, not causal multi-task.

Dependency policy: core stays numpy/pandas/pyarrow/sklearn. Multi-task uses
sklearn multi-output / chain façades — no optional extra required for
``import buildml``.

Lazy imports — core never grows heavy MTL stacks.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT",
    "CHECKPOINT_BOUNDARY",
    "MultiTaskBaseEstimator",
    "MultiTaskConfig",
    "MultiTaskEvalResult",
    "MultiTaskFitResult",
    "MultiTaskMethod",
    "MultiTaskPlan",
    "MultiTaskPredictResult",
    "MultiTaskTask",
    "evaluate_multitask",
    "fit_multitask",
    "load_multitask_bundle",
    "multitask_status",
    "multitask_status_for_session",
    "predict_multitask",
    "save_multitask_bundle",
]


def __getattr__(name: str) -> Any:
    if name in {
        "MultiTaskMethod",
        "MultiTaskTask",
        "MultiTaskBaseEstimator",
        "MultiTaskConfig",
    }:
        from buildml.multitask import types as types_mod

        return getattr(types_mod, name)
    if name in {
        "MultiTaskPlan",
        "MultiTaskFitResult",
        "MultiTaskEvalResult",
        "MultiTaskPredictResult",
    }:
        from buildml.multitask import results as results_mod

        return getattr(results_mod, name)
    if name == "fit_multitask":
        from buildml.multitask.fit import fit_multitask

        return fit_multitask
    if name == "predict_multitask":
        from buildml.multitask.predict import predict_multitask

        return predict_multitask
    if name == "evaluate_multitask":
        from buildml.multitask.evaluate import evaluate_multitask

        return evaluate_multitask
    if name in {
        "BUNDLE_FORMAT",
        "CHECKPOINT_BOUNDARY",
        "save_multitask_bundle",
        "load_multitask_bundle",
    }:
        from buildml.multitask import checkpoint as checkpoint_mod

        return getattr(checkpoint_mod, name)
    if name in {"multitask_status", "multitask_status_for_session"}:
        from buildml.multitask import explain_hooks as hooks

        return getattr(hooks, name)
    raise AttributeError(f"module 'buildml.multitask' has no attribute {name!r}")
