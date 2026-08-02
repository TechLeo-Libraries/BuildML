"""Meta-learning domain (tabular few-shot / episodic Session protocols).

Phase coverage (internal tracker — depth-first; do not spray stubs)
------------------------------------------------------------------
Phase 1 (**complete**): unsupervised → ensembles → AutoML → forecasting → anomaly.

Phase 2:
  1. Semi-supervised learning — done (``buildml.semisupervised``).
  2. Self-supervised learning hooks — done (``buildml.selfsupervised``).
  3. Active learning — done (``buildml.activelearning``).
  4. Online / continual (partial_fit) — done (``buildml.online``).
  5. Multi-task learning — done (``buildml.multitask``).
  6. Meta-learning — **this module**.
  7. Federated learning — done (``buildml.federated``).
  8. Bayesian / probabilistic — done (``buildml.probabilistic``); next = Causal.
  Later: graph, evolutionary,
  symbolic, CBR, IL+RL, TDA, recommenders / LTR / KG / optimisation / synthetic /
  NLP-CV deepenings. Speech: ASR keep/improve; TTS out.

Explicit non-goals (no product surfaces): neuromorphic/SNN, swarm zoo,
digital twins, AV stack, multi-agent world sims, TTS, robotics/control product,
full COCO detection/segmentation suite.

Honesty (this package):
  - Practical tabular few-shot / episodic protocols on Session data.
  - Task definition via a task/group column (role or ``task_column=``).
  - Algorithms shipped deeply: ``prototypical`` (nearest-centroid) and
    ``warm_start`` (pooled sklearn init + support adapt).
  - Train-only meta-train; validation/test are evaluation-only.
  - Not foundation-model meta-learning, not MAML-at-scale, not a paper zoo.

Dependency policy: core stays numpy/pandas/pyarrow/sklearn. Meta-learning
uses sklearn façades — no optional extra required for ``import buildml``.

Lazy imports — core never grows heavy meta-learning stacks.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT",
    "CHECKPOINT_BOUNDARY",
    "MetaAdaptResult",
    "MetaLearningBaseEstimator",
    "MetaLearningConfig",
    "MetaLearningEvalResult",
    "MetaLearningFitResult",
    "MetaLearningMethod",
    "MetaLearningPlan",
    "adapt_to_task",
    "evaluate_metalearning",
    "fit_metalearning",
    "load_metalearning_bundle",
    "metalearning_status",
    "metalearning_status_for_session",
    "save_metalearning_bundle",
]


def __getattr__(name: str) -> Any:
    if name in {
        "MetaLearningMethod",
        "MetaLearningBaseEstimator",
        "MetaLearningConfig",
    }:
        from buildml.metalearning import types as types_mod

        return getattr(types_mod, name)
    if name in {
        "MetaLearningPlan",
        "MetaLearningFitResult",
        "MetaLearningEvalResult",
        "MetaAdaptResult",
    }:
        from buildml.metalearning import results as results_mod

        return getattr(results_mod, name)
    if name == "fit_metalearning":
        from buildml.metalearning.fit import fit_metalearning

        return fit_metalearning
    if name == "adapt_to_task":
        from buildml.metalearning.adapt import adapt_to_task

        return adapt_to_task
    if name == "evaluate_metalearning":
        from buildml.metalearning.evaluate import evaluate_metalearning

        return evaluate_metalearning
    if name in {
        "BUNDLE_FORMAT",
        "CHECKPOINT_BOUNDARY",
        "save_metalearning_bundle",
        "load_metalearning_bundle",
    }:
        from buildml.metalearning import checkpoint as checkpoint_mod

        return getattr(checkpoint_mod, name)
    if name in {"metalearning_status", "metalearning_status_for_session"}:
        from buildml.metalearning import explain_hooks as hooks

        return getattr(hooks, name)
    raise AttributeError(f"module 'buildml.metalearning' has no attribute {name!r}")
