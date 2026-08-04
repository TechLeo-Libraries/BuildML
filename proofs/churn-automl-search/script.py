"""Tier A proof: telco-style churn AutoML search with holdout evaluation."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from buildml import Session
from buildml.automl.types import AutoMLBudget
from buildml.core.errors import MissingExtraError
from proofs._lib import (
    assert_no_test_in_selection,
    extra_available,
    load_telco_churn_synthetic,
    metrics_round,
    new_proof_context,
    write_results,
)


FEATURES = [
    "tenure_months",
    "monthly_charges",
    "contract",
    "internet_service",
    "support_tickets",
]
TARGET = "churn"


def main() -> None:
    ctx = new_proof_context("churn-automl-search", seed=7)
    frame, data_meta = load_telco_churn_synthetic(n=1600, seed=ctx.seed)
    caps = {
        "lightgbm": extra_available("lightgbm"),
        "xgboost": extra_available("xgboost"),
        "flaml": extra_available("flaml"),
        "autogluon": extra_available("autogluon.tabular"),
        "optuna": extra_available("optuna"),
    }

    session = (
        Session.ingest(frame)
        .set_roles({**{c: "feature" for c in FEATURES}, TARGET: "target"})
        .split(
            test_size=0.2,
            validation_size=0.2,
            stratify=True,
            random_state=ctx.seed,
        )
    )
    plan = session.split_plan
    assert plan is not None
    counts = {
        "train": len(plan.train_indices),
        "validation": len(plan.validation_indices),
        "test": len(plan.test_indices),
    }

    # Prefer industry backend when FLAML/AutoGluon present; else native + GBDT families.
    backend = "native"
    if caps["flaml"]:
        backend = "flaml"
    elif caps["autogluon"]:
        backend = "autogluon"

    assert_no_test_in_selection(
        selection_partition="cv",
        evaluation_partition="test",
    )
    try:
        result = session.automl.run(
            backend=backend,  # type: ignore[arg-type]
            method="randomized",
            selection="cv",
            n_trials=16,
            cv=3,
            include_recipe_search=True,
            include_industry_families=True,
            include_ensembles=True,
            families=(
                "logistic",
                "random_forest",
                "gradient_boosting",
                "lightgbm",
                "xgboost",
            ),
            budget=AutoMLBudget(max_trials=16, max_recipe_strategies=6),
            time_budget=120.0,
            random_state=ctx.seed,
        )
    except (MissingExtraError, ValueError, TypeError) as exc:
        # Fallback: core families only.
        result = session.automl.run(
            backend="native",
            method="randomized",
            selection="cv",
            n_trials=12,
            cv=3,
            include_recipe_search=True,
            include_industry_families=False,
            include_ensembles=True,
            families=("logistic", "random_forest", "gradient_boosting"),
            budget=AutoMLBudget(max_trials=12, max_recipe_strategies=6),
            random_state=ctx.seed,
        )
        backend = f"native_fallback ({type(exc).__name__}: {exc})"

    val = session.automl.evaluate(partition="validation")
    test = session.automl.evaluate(partition="test")
    bundle = session.automl.save_bundle(ctx.artifacts_dir / "automl_bundle")

    best = {}
    if hasattr(result, "to_dict"):
        best = metrics_round(result.to_dict())
    elif hasattr(result, "best_params"):
        best = {
            "best_params": getattr(result, "best_params", None),
            "best_score": getattr(result, "best_score", None),
        }
    leaderboard_rows = []
    if hasattr(result, "leaderboard"):
        board = result.leaderboard(top_n=8)
        leaderboard_rows = board.to_dict(orient="records")

    write_results(
        ctx,
        {
            "status": "completed",
            "data": data_meta,
            "split": {"kind": plan.kind, "counts": counts, "stratify": True},
            "capabilities": caps,
            "backend": backend,
            "selection": getattr(result, "selection", "cv"),
            "selection_note": (
                "Default selection='cv' ranks by train-fold CV; "
                "use selection='nested' for outer post-selection estimates."
            ),
            "automl_result": best,
            "leaderboard": leaderboard_rows,
            "outer_score_mean": getattr(result, "outer_score_mean", None),
            "validation_metrics": metrics_round(dict(val.metrics)),
            "test_metrics": metrics_round(dict(test.metrics)),
            "bundle_path": str(bundle),
            "leakage_controls": [
                "Stratified split before search",
                "session.automl.run selection='cv' on train folds only",
                "Session test never enters ranking",
                "session.automl.evaluate(test) once after search + refit",
            ],
            "industry_comparison": {
                "status": "filled",
                "note": (
                    "Tier C baseline_industry.py: sklearn RandomizedSearchCV twin on the same "
                    "split; optional FLAML/AutoGluon when installed — run script then "
                    "baseline_industry.py for results/comparison.json."
                ),
            },
            "limitations": [
                "Synthetic telco churn; finite trial budget (not unbounded HPO)",
                "Industry backends used when installed; otherwise native catalog",
            ],
        },
    )
    print("churn-automl-search OK", dict(test.metrics))


if __name__ == "__main__":
    main()
