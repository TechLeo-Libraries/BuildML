"""Group disparity reporting for binary classifiers on Session holdouts.

Observational metrics only: not a legal audit, not causal fairness, and not a
certification product. Callers must declare sensitive column(s) and interpret
gaps in domain context. Optional mitigation helpers return weights/thresholds
explicitly — they never silently rewrite predictions.
"""

from buildml.fairness.catalog import fairness_capability_matrix
from buildml.fairness.classical_bridge import per_group_classical_metrics
from buildml.fairness.evaluate import evaluate_fairness, validate_positive_label
from buildml.fairness.groups import compose_group_keys, normalize_sensitive_columns
from buildml.fairness.mitigation import (
    GroupThresholdSuggestion,
    ReweighingSuggestion,
    apply_group_thresholds,
    suggest_group_thresholds,
    suggest_reweighing_weights,
)
from buildml.fairness.results import FairnessReport
from buildml.fairness.stability import FairnessStability, estimate_gap_stability

__all__ = [
    "FairnessReport",
    "FairnessStability",
    "GroupThresholdSuggestion",
    "ReweighingSuggestion",
    "apply_group_thresholds",
    "compose_group_keys",
    "estimate_gap_stability",
    "evaluate_fairness",
    "fairness_capability_matrix",
    "normalize_sensitive_columns",
    "per_group_classical_metrics",
    "suggest_group_thresholds",
    "suggest_reweighing_weights",
    "validate_positive_label",
]
