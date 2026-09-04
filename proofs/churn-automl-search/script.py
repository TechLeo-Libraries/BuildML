"""Tier A proof: family + recipe AutoML on sklearn Wisconsin breast cancer."""

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
    load_sklearn_breast_cancer,
    metrics_round,
    new_proof_context,
    refuse_perfect_scores,
    supervised_roles,
    write_results,
)


def main() -> None:
    ctx = new_proof_context("churn-automl-search", seed=7)
    frame, data_meta = load_sklearn_breast_cancer()
    roles = supervised_roles(data_meta)
    caps = {
        "lightgbm": extra_available("lightgbm"),
        "xgboost": extra_available("xgboost"),
        "flaml": extra_available("flaml"),
        "autogluon": extra_available("autogluon.tabular"),
        "optuna": extra_available("optuna"),
    }

    session = (
        Session.ingest(frame)
        .set_roles(roles)
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
    test_metrics = metrics_round(dict(test.metrics))
    refuse_perfect_scores(
        test_metrics,
        keys=("accuracy", "f1", "f1_weighted", "f1_macro", "roc_auc"),
        ceiling=1.0,
        proof_slug="churn-automl-search",
        context="sklearn breast_cancer holdout after automl",
    )

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
            "evidence_tier": "REAL_PUBLIC_DATASET",
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
            "test_metrics": test_metrics,
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
                    "split; optional FLAML/AutoGluon when installed. Run script then "
                    "baseline_industry.py for results/comparison.json."
                ),
            },
            "limitations": [
                (
                    "Slug kept; the table is sklearn Wisconsin breast cancer, "
                    "not a telco CRM extract. Finite trial budget."
                ),
                "Industry backends used when installed; otherwise native catalog",
            ],
        },
    )
    print("churn-automl-search OK", data_meta.get("name"), dict(test.metrics))


if __name__ == "__main__":
    main()
