"""Active learning domain (human-in-the-loop train-pool querying).

Phase coverage (internal tracker — depth-first; do not spray stubs)
------------------------------------------------------------------
Phase 1 (**complete**): unsupervised → ensembles → AutoML → forecasting → anomaly.

Phase 2:
  1. Semi-supervised learning — done (``buildml.semisupervised``).
  2. Self-supervised learning hooks — done (``buildml.selfsupervised``).
  3. Active learning — **this module (PASS)**.
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
  - Labels come from the user. Core never invents a fake oracle (tests may simulate one).
  - Query pool is the train partition (NaN targets by default) — never validation/test.
  - Not semi-supervised propagation and not self-supervised pretext.
  - Budget caps are enforced; exhausted budgets return empty queries.

Dependency policy: core stays numpy/pandas/pyarrow/sklearn. Active learning
uses core sklearn classifiers + bagging committee — no optional extra required
for ``import buildml``.

Lazy imports — core never grows heavy active-learning stacks.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT",
    "CHECKPOINT_BOUNDARY",
    "ActiveLearningConfig",
    "ActiveLearningEstimator",
    "ActiveLearningEvalResult",
    "ActiveLearningFitResult",
    "ActiveLearningLabelResult",
    "ActiveLearningPlan",
    "ActiveLearningQueryResult",
    "ActiveLearningStrategy",
    "activelearning_status",
    "activelearning_status_for_session",
    "evaluate_active_learning",
    "fit_active_learner",
    "label_rows",
    "load_active_learning_bundle",
    "query_indices",
    "save_active_learning_bundle",
    "suggest_query",
]


def __getattr__(name: str) -> Any:
    if name in {
        "ActiveLearningStrategy",
        "ActiveLearningEstimator",
        "ActiveLearningConfig",
    }:
        from buildml.activelearning import types as types_mod

        return getattr(types_mod, name)
    if name in {
        "ActiveLearningPlan",
        "ActiveLearningFitResult",
        "ActiveLearningQueryResult",
        "ActiveLearningLabelResult",
        "ActiveLearningEvalResult",
    }:
        from buildml.activelearning import results as results_mod

        return getattr(results_mod, name)
    if name == "fit_active_learner":
        from buildml.activelearning.fit import fit_active_learner

        return fit_active_learner
    if name in {"suggest_query", "query_indices"}:
        from buildml.activelearning import query as query_mod

        return getattr(query_mod, name)
    if name == "label_rows":
        from buildml.activelearning.label import label_rows

        return label_rows
    if name == "evaluate_active_learning":
        from buildml.activelearning.evaluate import evaluate_active_learning

        return evaluate_active_learning
    if name in {
        "BUNDLE_FORMAT",
        "CHECKPOINT_BOUNDARY",
        "save_active_learning_bundle",
        "load_active_learning_bundle",
    }:
        from buildml.activelearning import checkpoint as checkpoint_mod

        return getattr(checkpoint_mod, name)
    if name in {"activelearning_status", "activelearning_status_for_session"}:
        from buildml.activelearning import explain_hooks as hooks

        return getattr(hooks, name)
    raise AttributeError(f"module 'buildml.activelearning' has no attribute {name!r}")
