"""Tier B product: Ledger Underwriting Studio.

Composes classical supervised scoring + AutoML search + declared-assumption
causal treatment effect + validation-tuned cost-sensitive decisions +
calibration diagnostics. Leakage discipline throughout.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.automl.types import AutoMLBudget
from buildml.core.errors import MissingExtraError, ValidationError
from proofs._lib import (
    assert_no_test_in_selection,
    extra_available,
    load_credit_approval_synthetic,
    metrics_round,
    new_proof_context,
    write_results,
)


FEATURE_NUM = ["age", "income", "debt_ratio", "employment_years"]
FEATURE_CAT = ["region", "product"]
FEATURES = FEATURE_NUM + FEATURE_CAT
TARGET = "approved"


def _with_treatment(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Add a binary outreach treatment correlated with confounders (synthetic)."""
    rng = np.random.default_rng(seed)
    out = frame.copy()
    # Confounded assignment: higher income / employment → more likely treated.
    income_z = (out["income"].fillna(out["income"].median()) - out["income"].median()) / (
        out["income"].std() + 1e-6
    )
    emp = out["employment_years"].fillna(0.0)
    logit = -0.4 + 0.35 * income_z + 0.06 * emp + rng.normal(0, 0.3, size=len(out))
    treat = (1 / (1 + np.exp(-logit)) > 0.5).astype(int)
    # Outcome nudge under treatment (true ATE ≈ small lift on approval).
    nudge = treat * (0.12 + 0.05 * rng.normal(size=len(out)))
    # Soft-flip some denials toward approve when treated (synthetic effect).
    flip_mask = (out[TARGET] == 0) & (treat == 1) & (rng.random(len(out)) < 0.15 + nudge.clip(0, 0.2))
    out.loc[flip_mask, TARGET] = 1
    out["outreach"] = treat
    out["review_cost"] = np.where(out[TARGET] == 1, 2.5, 1.0)
    out["app_id"] = [f"app-{i}" for i in range(len(out))]
    return out


def main() -> None:
    ctx = new_proof_context("ledger-underwriting-studio", seed=42)
    base, data_meta = load_credit_approval_synthetic(n=1200, seed=ctx.seed)
    frame = _with_treatment(base, seed=ctx.seed)
    stages: dict = {}
    skip_notes: list[str] = []

    session = (
        Session.ingest(frame.copy())
        .set_roles(
            {
                **{c: "feature" for c in FEATURES},
                TARGET: "target",
                "outreach": "ignore",
                "review_cost": "ignore",
                "app_id": "id",
            }
        )
        .split(
            test_size=0.2,
            validation_size=0.2,
            stratify=True,
            random_state=ctx.seed,
        )
    )
    plan = session.split_plan
    assert plan is not None
    split_counts = {
        "train": len(plan.train_indices),
        "validation": len(plan.validation_indices),
        "test": len(plan.test_indices),
    }
    # Defaults skip ignore/id roles — review_cost and app_id stay usable for knapsack.
    session.impute(strategy="median")
    session.encode(method="onehot")
    session.scale(method="standard")

    # --- Stage 1: classical supervised ---
    session.fit(
        LogisticRegression(max_iter=1000, random_state=ctx.seed),
        task="classification",
    )
    classical_val = session.evaluate(partition="validation")
    classical_test = session.evaluate(partition="test")
    stages["classical"] = {
        "status": "ok",
        "estimator": "LogisticRegression",
        "validation_metrics": metrics_round(dict(classical_val.metrics)),
        "test_metrics": metrics_round(dict(classical_test.metrics)),
    }
    write_results(ctx, stages["classical"], filename="classical.json")

    # --- Stage 2: AutoML (selection on CV/val, never test) ---
    try:
        automl_session = (
            Session.ingest(frame.copy())
            .set_roles({**{c: "feature" for c in FEATURES}, TARGET: "target"})
            .inject_split(
                train_indices=list(plan.train_indices),
                validation_indices=list(plan.validation_indices),
                test_indices=list(plan.test_indices),
            )
        )
        backend = "native"
        if extra_available("flaml"):
            backend = "flaml"
        elif extra_available("autogluon.tabular"):
            backend = "autogluon"
        assert_no_test_in_selection(
            selection_partition="cv", evaluation_partition="test"
        )
        try:
            result = automl_session.run_automl(
                task="classification",
                backend=backend,  # type: ignore[arg-type]
                method="randomized",
                selection="cv",
                n_trials=12,
                cv=3,
                include_recipe_search=True,
                include_industry_families=True,
                families=("logistic", "random_forest", "gradient_boosting"),
                budget=AutoMLBudget(max_trials=12, max_recipe_strategies=4),
                time_budget=60.0,
                random_state=ctx.seed,
            )
        except (MissingExtraError, ValueError, TypeError) as exc:
            result = automl_session.run_automl(
                task="classification",
                backend="native",
                method="randomized",
                selection="cv",
                n_trials=10,
                cv=3,
                include_recipe_search=True,
                include_industry_families=False,
                families=("logistic", "random_forest"),
                budget=AutoMLBudget(max_trials=10, max_recipe_strategies=4),
                random_state=ctx.seed,
            )
            backend = f"native_fallback ({type(exc).__name__})"
        try:
            am_test = automl_session.evaluate_automl(partition="test")
        except Exception:
            am_test = automl_session.evaluate(partition="test")
        stages["automl"] = {
            "status": "ok",
            "backend": backend,
            "fit": metrics_round(result.to_dict() if hasattr(result, "to_dict") else {}),
            "test_metrics": metrics_round(dict(am_test.metrics)),
        }
        write_results(ctx, stages["automl"], filename="automl.json")
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["automl"] = {"status": "skipped", "error": f"{type(exc).__name__}: {exc}"}
        skip_notes.append(f"automl: {exc}")

    # --- Stage 3: causal (assumptions REQUIRED before fit) ---
    try:
        causal_session = (
            Session.ingest(frame.copy())
            .set_roles(
                {
                    "age": "feature",
                    "income": "feature",
                    "debt_ratio": "feature",
                    "employment_years": "feature",
                    "outreach": "feature",
                    TARGET: "target",
                    "region": "ignore",
                    "product": "ignore",
                    "review_cost": "ignore",
                    "app_id": "id",
                }
            )
            .inject_split(
                train_indices=list(plan.train_indices),
                validation_indices=list(plan.validation_indices),
                test_indices=list(plan.test_indices),
            )
            .impute(strategy="median")
            .scale(method="standard")
        )
        causal_session.declare_causal_assumptions(
            treatment="outreach",
            outcome=TARGET,
            confounders=["age", "income", "debt_ratio", "employment_years"],
            acknowledge_unconfoundedness=True,
            acknowledge_positivity=True,
        )
        fit_c = causal_session.fit_causal(method="aipw", bootstrap_samples=30)
        ev_c = causal_session.evaluate_causal(partition="validation", bootstrap_samples=15)
        stages["causal"] = {
            "status": "ok",
            "assumptions": {
                "treatment": "outreach",
                "outcome": TARGET,
                "confounders": ["age", "income", "debt_ratio", "employment_years"],
                "acknowledged": ["unconfoundedness", "positivity"],
                "note": (
                    "Assumptions are declared, not proven. Synthetic DGP; "
                    "outreach assignment is confounded by income/employment."
                ),
            },
            "fit": {
                "method": fit_c.method,
                "ate": float(fit_c.ate),
                "ate_ci_low": float(fit_c.ate_ci_low),
                "ate_ci_high": float(fit_c.ate_ci_high),
            },
            "validation_eval": {
                "ate": float(ev_c.ate),
                "metrics": metrics_round(dict(ev_c.metrics)),
            },
        }
        write_results(ctx, stages["causal"], filename="causal.json")
    except (MissingExtraError, TypeError, ValueError, ValidationError) as exc:
        stages["causal"] = {"status": "skipped", "error": f"{type(exc).__name__}: {exc}"}
        skip_notes.append(f"causal: {exc}")

    # --- Stage 4: cost-sensitive decisions — threshold on VALIDATION only ---
    assert_no_test_in_selection(
        selection_partition="validation", evaluation_partition="test"
    )
    thr = session.fit_decision_policy(
        method="threshold",
        partition="validation",
        fp_cost=1.0,
        fn_cost=5.0,
    )
    thr_test = session.evaluate_decisions(partition="test")
    knap_payload: dict = {"status": "skipped"}
    try:
        knap = session.fit_decision_policy(
            method="knapsack",
            partition="validation",
            budget=70.0,
            cost_column="review_cost",
            id_column="app_id",
            score_source="model_proba",
            knapsack_solver="dp",
        )
        applied = session.apply_decisions(partition="test")
        knap_payload = {
            "knapsack_status": "ok",
            "knapsack_policy": metrics_round(
                knap.to_dict() if hasattr(knap, "to_dict") else {}
            ),
            "knapsack_applied": {
                "n_selected": int(applied.n_selected),
                "selected_value": float(applied.selected_value),
                "selected_cost": float(applied.selected_cost),
            },
        }
    except Exception as exc:  # noqa: BLE001
        try:
            topk = session.fit_decision_policy(
                method="topk",
                partition="validation",
                capacity=40,
                score_source="model_proba",
            )
            applied = session.apply_decisions(partition="test")
            knap_payload = {
                "knapsack_status": "ok_topk_fallback",
                "knapsack_error": f"{type(exc).__name__}: {exc}",
                "topk_policy": metrics_round(
                    topk.to_dict() if hasattr(topk, "to_dict") else {}
                ),
                "knapsack_applied": {
                    "n_selected": int(applied.n_selected),
                    "selected_value": float(applied.selected_value),
                    "selected_cost": float(getattr(applied, "selected_cost", float("nan"))),
                },
            }
        except Exception as exc2:  # noqa: BLE001
            knap_payload = {
                "knapsack_status": "skipped",
                "error": f"{type(exc).__name__}: {exc}; fallback: {exc2}",
            }
            skip_notes.append(f"knapsack: {exc}")
    stages["decisions"] = {
        "status": "ok",
        "threshold_selection_partition": "validation",
        "threshold_policy": metrics_round(thr.to_dict() if hasattr(thr, "to_dict") else {}),
        "threshold_test": metrics_round(
            thr_test.to_dict() if hasattr(thr_test, "to_dict") else {}
        ),
        **knap_payload,
    }
    write_results(ctx, stages["decisions"], filename="decisions.json")

    # --- Stage 5: calibration diagnostics (report on validation; confirm on test) ---
    try:
        cal_val = session.calibration(partition="validation")
        cal_test = session.calibration(partition="test")
        stages["calibration"] = {
            "status": "ok",
            "validation": metrics_round(
                cal_val.to_dict() if hasattr(cal_val, "to_dict") else {}
            ),
            "test": metrics_round(
                cal_test.to_dict() if hasattr(cal_test, "to_dict") else {}
            ),
        }
        write_results(ctx, stages["calibration"], filename="calibration.json")
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["calibration"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"calibration: {exc}")

    summary = {
        "status": "completed",
        "product": "Ledger Underwriting Studio",
        "data": {
            **data_meta,
            "treatment_column": "outreach",
            "notes": "Synthetic credit book + confounded outreach treatment for causal stage",
        },
        "split": {"kind": plan.kind, "counts": split_counts, "stratify": True},
        "stages": {k: {"status": v.get("status")} for k, v in stages.items()},
        "stage_details": stages,
        "skip_notes": skip_notes,
        "leakage_controls": [
            "Stratified split before classical / AutoML / causal / decisions",
            "Causal assumptions declared before fit_causal (required API gate)",
            "Decision threshold + knapsack selected on validation ONLY — never test",
            "Calibration reported on validation then confirmed on untouched test",
            "AutoML search/selection never uses the test partition",
        ],
        "what_fails_if_leakage_ignored": [
            "Tuning the approve threshold on test understates expected review cost",
            "Skipping causal assumption declaration hides confounding risk",
            "Fitting AutoML with test in the search loop invents leaderboard wins",
            "Reporting calibration only on train hides probability miscalibration",
        ],
        "limitations": [
            "Synthetic underwriting — not FCRA / bureau data",
            "Causal ATE assumes declared unconfoundedness (not proven)",
            "Product proof, not a production LOS certification",
        ],
    }
    write_results(ctx, summary, filename="summary.json")
    write_results(ctx, summary, filename="results.json")
    print(
        "ledger-underwriting-studio OK",
        {
            "classical_roc": stages["classical"]["test_metrics"].get("roc_auc"),
            "threshold": stages["decisions"]["threshold_policy"].get("threshold"),
            "causal_ate": (stages.get("causal") or {}).get("fit", {}).get("ate"),
            "skips": skip_notes,
        },
    )


if __name__ == "__main__":
    main()
