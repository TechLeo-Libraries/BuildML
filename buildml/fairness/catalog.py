"""Honest capability matrix for fairness disparity reporting."""

from __future__ import annotations

from typing import Any


def fairness_capability_matrix() -> dict[str, Any]:
    """Report fairness metrics, requirements, and explicit non-goals.

    Returns
    -------
    dict[str, Any]
        Backend availability, metric list, install hints, and boundaries.
    """
    return {
        "backends": {
            "native": {
                "available": True,
                "extra": None,
                "metrics": [
                    "selection_rate_by_group",
                    "demographic_parity_difference",
                    "disparate_impact_ratio",
                    "equalized_odds_tpr_difference",
                    "equalized_odds_fpr_difference",
                ],
                "tasks": ["binary_classification"],
                "notes": (
                    "Holdout-only observational rates and gaps. Requires a fitted "
                    "classifier and a sensitive column on the evaluated partition. "
                    "positive_label must appear in y_true (hard-validated)."
                ),
            },
            "shap": {
                "available": False,
                "extra": "shap",
                "notes": (
                    "SHAP attribution is a separate Session path "
                    "(explain_shap), not a fairness metric."
                ),
            },
        },
        "default_backend": "native",
        "maturity": "observational_analysis",
        "requires_sensitive_column": True,
        "positive_label_validated": True,
        "partition_default": "test",
        "install_hints": {
            "shap": "pip install 'buildml[shap]'",
        },
        "non_goals": [
            "Legal disparate-impact certification or regulator filings",
            "Causal fair representation learning",
            "Multi-class / regression fairness suites",
            "Automatic bias mitigation / reweighing products",
            "Inferring protected class membership from features",
        ],
        "disclosures": [
            "Gaps are descriptive on one split: they do not prove discrimination "
            "or excuse a model.",
            "Sensitive attributes must be declared by the caller; BuildML never "
            "infers protected class membership.",
            "Misconfigured positive_label (e.g. default 1 with string labels) "
            "raises ValidationError instead of silent zero rates.",
        ],
    }
