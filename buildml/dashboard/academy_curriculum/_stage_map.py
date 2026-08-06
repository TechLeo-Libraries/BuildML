"""Stage assignment for BuildML CONCEPT_NOTES inside the Academy hub.

Stages 00–05 follow the EDA → modeling readiness spine. Stage 06 holds
domain-depth concepts (NLP, RL, graphs, …) that are still first-class lessons,
not a thin reference dump.
"""

from __future__ import annotations

# Explicit overrides beat prefix heuristics (readiness-path vocabulary).
_STAGE_OVERRIDES: dict[str, int] = {
    # 00 Framing
    "diagnostic-uncertainty": 0,
    "feature-schema": 0,
    "column-roles": 0,
    "engine-choice": 0,
    "explain-learning-levels": 0,
    "dry-run-plans": 0,
    # 01 Data quality
    "missing-data": 1,
    "categorical-encoding": 1,
    "encoding-imputation-scaling": 1,
    "text-features": 1,
    "normality-screens": 1,
    "outlier-handling": 1,
    "custom-transforms": 1,
    "feature-binning": 1,
    "target-encoding": 1,
    "anomaly-eda-boundary": 1,
    # 02 Relationships
    "mutual-information": 2,
    "variance-inflation": 2,
    "feature-scaling": 2,
    "feature-selection": 2,
    "principal-components": 2,
    "pca-cluster-integration": 2,
    "cluster-kmeans-vs-density": 2,
    "cluster-validity-not-truth": 2,
    "causal-assumptions": 2,
    "causal-eda-boundary": 2,
    "feature-importance": 2,
    # 03 Validation
    "data-splitting": 3,
    "cross-validation": 3,
    "dataset-drift": 3,
    "leakage-boundary": 3,
    "batch-leakage": 3,
    "evaluation-partitions": 3,
    "early-stopping-partition": 3,
    "reproducibility": 3,
    "overfitting": 3,
    "checkpoint-integrity": 3,
    "operation-history": 3,
    "train-serve-parity": 3,
    # 04 Evaluation
    "class-imbalance": 4,
    "baselines": 4,
    "thresholds": 4,
    "probability-calibration": 4,
    "model-selection": 4,
    "training-curves": 4,
    "decision-cost-matrix": 4,
    "decision-operating-point": 4,
    "decision-allocation": 4,
    "anomaly-imbalance-metrics": 4,
    "anomaly-threshold-alert-rate": 4,
    # 05 Interpretation / handoff
    "causal-ate-backdoor": 5,
    "causal-ipw": 5,
    "causal-aipw": 5,
    "causal-t-learner": 5,
    "mi-vs-correlation": 5,
}

# Prefix → stage for domain families (honest peripheral staging).
_PREFIX_STAGE: tuple[tuple[str, int], ...] = (
    ("causal-", 5),
    ("anomaly-", 4),
    ("forecast-", 6),
    ("ts-", 6),
    ("nlp-", 6),
    ("rag-", 6),
    ("rl-", 6),
    ("imitation-", 6),
    ("graph-", 6),
    ("kg-", 6),
    ("tda-", 6),
    ("federated-", 6),
    ("metalearning-", 6),
    ("recommender-", 6),
    ("ltr-", 6),
    ("synthetic-", 6),
    ("ssl-", 6),
    ("semisupervised-", 6),
    ("activelearning-", 6),
    ("online-", 6),
    ("multitask-", 6),
    ("symbolic-", 6),
    ("neuro-symbolic", 6),
    ("cbr-", 6),
    ("automl-", 6),
    ("probabilistic-", 4),
    ("ensemble-", 4),
    ("unsupervised-", 2),
    ("decision-", 4),
    ("ai-", 6),
    ("feature-", 2),
)


def stage_for_concept(key: str) -> int:
    """Return Academy stage 0–6 for a CONCEPT_NOTES key."""
    if key in _STAGE_OVERRIDES:
        return _STAGE_OVERRIDES[key]
    for prefix, stage in _PREFIX_STAGE:
        if key.startswith(prefix):
            return stage
    # Classical leftovers default to validation/evaluation middle path.
    if any(token in key for token in ("leak", "split", "valid", "cv", "drift")):
        return 3
    if any(token in key for token in ("metric", "threshold", "calibr", "baseline", "imbalance")):
        return 4
    if any(token in key for token in ("missing", "encod", "impute", "dtype", "schema", "outlier")):
        return 1
    return 6


DOMAIN_STAGE = {
    "key": 6,
    "n": "06",
    "label": "Domain depth",
    "blurb": (
        "specialized BuildML domains (NLP, RL, graphs, forecasting, …) — "
        "full lessons, staged honestly outside the core EDA readiness spine"
    ),
}

__all__ = ["DOMAIN_STAGE", "stage_for_concept"]
