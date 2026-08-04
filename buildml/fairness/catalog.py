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
                    "classical_metrics_by_group",
                    "stability_bands",
                ],
                "tasks": ["binary_classification"],
                "features": [
                    "intersectional_sensitive_columns",
                    "bootstrap_or_stratified_subsample_stability",
                    "per_group_classical_bridge",
                    "markdown_and_dict_reports",
                ],
                "notes": (
                    "Holdout observational rates, gaps, optional stability bands, "
                    "and per-group classical metrics. Requires a fitted classifier "
                    "and caller-declared sensitive column(s). positive_label must "
                    "appear in y_true (hard-validated). Intersectional audits join "
                    "multiple columns into composite group keys."
                ),
            },
            "mitigation_helpers": {
                "available": True,
                "extra": None,
                "opt_in": True,
                "tools": [
                    "suggest_group_thresholds",
                    "suggest_reweighing_weights",
                    "apply_group_thresholds",
                ],
                "notes": (
                    "Optional post-hoc helpers under buildml.fairness.mitigation "
                    "and session.fairness.suggest_*. They return thresholds or "
                    "sample weights only — never silent fairness washing, never "
                    "legal certification."
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
        "depth": "high",
        "requires_sensitive_column": True,
        "supports_intersectional": True,
        "positive_label_validated": True,
        "partition_default": "test",
        "stability_default": "off",
        "install_hints": {
            "shap": "pip install 'buildml[shap]'",
        },
        "session_paths": [
            "session.fairness.evaluate",
            "session.fairness.attach_to_last_eval",
            "session.fairness.suggest_thresholds",
            "session.fairness.suggest_reweighing",
            "session.fairness.capability_matrix",
            "session.fairness.last_report",
        ],
        "non_goals": [
            "Legal disparate-impact certification or regulator filings",
            "Causal fair representation learning",
            "Multi-class / regression fairness suites",
            "Automatic silent bias mitigation / fairness washing",
            "Inferring protected class membership from features",
        ],
        "disclosures": [
            "Gaps are descriptive on one split: they do not prove discrimination "
            "or excuse a model.",
            "Sensitive attributes must be declared by the caller; BuildML never "
            "infers protected class membership.",
            "Misconfigured positive_label (e.g. default 1 with string labels) "
            "raises ValidationError instead of silent zero rates.",
            "Stability bands disclose sampling variability of observational gaps; "
            "they are not causal uncertainty.",
            "Mitigation helpers are opt-in post-hoc tools that return weights or "
            "thresholds — applying them does not certify fairness.",
            "Intersectional group keys can be sparse; support and warnings are "
            "part of the report contract.",
        ],
    }
