"""Ensemble capability matrix: honest sklearn voting / stacking / blending."""

from __future__ import annotations

from typing import Any


def ensemble_capability_matrix() -> dict[str, Any]:
    """Report which ensemble strategies are available on this machine.

    Ensembles are core sklearn (plus an in-tree holdout blender). There is no
    industry extra and no platform marker skip: availability is always
    ``True`` when BuildML's core dependencies import. Read-only introspection.

    Returns
    -------
    dict[str, Any]
        Nested ``backends``, strategies, install hints (empty), and ``non_goals``.
    """
    return {
        "backends": {
            "sklearn": {
                "available": True,
                "extra": None,
                "strategies": ["voting", "stacking", "blending"],
                "notes": (
                    "VotingClassifier/Regressor and Stacking* from scikit-learn; "
                    "holdout blending is an in-tree estimator. Always available "
                    "with core BuildML dependencies."
                ),
            },
        },
        "strategies": [
            {
                "strategy": "voting",
                "task": ["classification", "regression"],
                "default": True,
                "requires_extra": None,
            },
            {
                "strategy": "stacking",
                "task": ["classification", "regression"],
                "default": False,
                "requires_extra": None,
            },
            {
                "strategy": "blending",
                "task": ["classification", "regression"],
                "default": False,
                "requires_extra": None,
            },
        ],
        "default_strategy": "voting",
        "default_backend_when_installed": "sklearn",
        "reporting": {
            "evaluate_enrichment": [
                "base_contributions",
                "diversity",
                "ensemble_report",
            ],
            "leakage_safe_defaults": (
                "evaluate scores train-fitted bases predict-only on the named "
                "partition; stacking OOF / blend holdout stay inside train at fit."
            ),
        },
        "install_hints": {},
        "platform_markers": [],
        "non_goals": [
            "AutoML recipe search (see buildml.automl)",
            "Industry GBDT zoo as base learners beyond what the caller supplies",
            "Distributed / multi-node ensemble product",
        ],
        "domain_floor": {
            "catalog": True,
            "checkpoint": True,
            "explain_hooks": True,
            "session_matrix": "ensemble_capability_matrix",
            "analysis_only": False,
        },
    }


def ensemble_status_payload() -> dict[str, Any]:
    """Build a compact install disclosure for walkthrough / teaching overlays.

    Summarises strategies and the fact that ensembles need no industry extra.
    Safe to call without a dataset.

    Returns
    -------
    dict[str, Any]
        Availability flag, default strategy, strategy list, and disclosures.
    """
    matrix = ensemble_capability_matrix()
    return {
        "available": True,
        "default_strategy": matrix["default_strategy"],
        "strategies": ["voting", "stacking", "blending"],
        "recommended_extra": None,
        "disclosures": [
            "Ensemble learning uses core scikit-learn voting/stacking plus "
            "in-tree holdout blending: no optional industry extra.",
            "Call Session.ensemble_capability_matrix() before choosing a strategy.",
        ],
    }
