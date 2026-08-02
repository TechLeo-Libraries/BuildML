"""Anomaly / fraud detection domain (train-fit / holdout-score Session path).

Phase coverage (internal tracker — depth-first; do not spray stubs)
------------------------------------------------------------------
Phase 1 (**complete** with this package):
  1. Unsupervised learning — done (see ``buildml.unsupervised``).
  2. Ensemble learning — done (see ``buildml.ensemble``).
  3. AutoML — done (see ``buildml.automl``).
  4. Time-series forecasting — done (see ``buildml.forecasting``).
  5. Anomaly / fraud detection — dedicated path beyond EDA IsolationForest.
     **This module.**

Phase 2 progress (depth-first; do not spray stubs):
  1. Semi-supervised — done (see ``buildml.semisupervised``).
  2. Self-supervised hooks — done (see ``buildml.selfsupervised``).
  3. Active learning — done (``buildml.activelearning``).
  4. Online / continual (partial_fit) — done (``buildml.online``); next = multi-task.
  Later: graph (causal done in ``buildml.causal``; probabilistic in ``buildml.probabilistic``)
  (separate assumption objects; EDA stays associational), graph ML,
  evolutionary search/NAS-lite, symbolic/neuro-symbolic, CBR, imitation + RL,
  allowlisted LLM tool agents, TDA, recommenders / LTR / KG / optimisation /
  synthetic-data / NLP-CV deepenings. Speech: ASR keep/improve; TTS out.

Explicit non-goals (no product surfaces): neuromorphic/SNN, swarm zoo,
digital twins, AV stack, multi-agent world sims, TTS, robotics/control product,
full COCO detection/segmentation suite.

Honesty (this package):
  - Not a full fraud platform (no graph fraud, no online streaming product).
  - No causal fraud claims; thresholds and alert rates are always disclosed.
  - EDA IsolationForest screens and ``Session.handle_outliers`` fences are not
    this API. Novelty mode is normal-only semi-supervised fit, not Phase 2
    semi-supervised representation learning.

Dependency policy: core stays numpy/pandas/pyarrow/sklearn. Anomaly detectors
use core sklearn — no optional extra required for ``import buildml``.

Lazy imports — core never grows heavy anomaly stacks.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT",
    "CHECKPOINT_BOUNDARY",
    "AnomalyConfig",
    "AnomalyEvalResult",
    "AnomalyFitResult",
    "AnomalyMethod",
    "AnomalyMode",
    "AnomalyPlan",
    "AnomalyScoreResult",
    "ThresholdPolicy",
    "anomaly_status",
    "anomaly_status_for_session",
    "evaluate_anomaly",
    "fit_detector",
    "load_anomaly_bundle",
    "save_anomaly_bundle",
    "score_anomalies",
]


def __getattr__(name: str) -> Any:
    if name in {"AnomalyMethod", "AnomalyMode", "ThresholdPolicy", "AnomalyConfig"}:
        from buildml.anomaly import types as types_mod

        return getattr(types_mod, name)
    if name in {
        "AnomalyPlan",
        "AnomalyFitResult",
        "AnomalyScoreResult",
        "AnomalyEvalResult",
    }:
        from buildml.anomaly import results as results_mod

        return getattr(results_mod, name)
    if name == "fit_detector":
        from buildml.anomaly.fit import fit_detector

        return fit_detector
    if name == "score_anomalies":
        from buildml.anomaly.score import score_anomalies

        return score_anomalies
    if name == "evaluate_anomaly":
        from buildml.anomaly.evaluate import evaluate_anomaly

        return evaluate_anomaly
    if name in {
        "BUNDLE_FORMAT",
        "CHECKPOINT_BOUNDARY",
        "save_anomaly_bundle",
        "load_anomaly_bundle",
    }:
        from buildml.anomaly import checkpoint as checkpoint_mod

        return getattr(checkpoint_mod, name)
    if name in {"anomaly_status", "anomaly_status_for_session"}:
        from buildml.anomaly import explain_hooks as hooks

        return getattr(hooks, name)
    raise AttributeError(f"module 'buildml.anomaly' has no attribute {name!r}")
