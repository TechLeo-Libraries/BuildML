"""Group disparity reporting for binary classifiers on Session holdouts.

Observational metrics only: not a legal audit, not causal fairness, and not a
certification product. Callers must declare the sensitive column and interpret
gaps in domain context.
"""

from buildml.fairness.catalog import fairness_capability_matrix
from buildml.fairness.evaluate import evaluate_fairness
from buildml.fairness.results import FairnessReport

__all__ = [
    "FairnessReport",
    "evaluate_fairness",
    "fairness_capability_matrix",
]
