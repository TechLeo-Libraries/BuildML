"""Tier B product: Keystone Underwrite ML.

Composes stacking ensemble + AutoML search + declared-assumption causal
treatment effect for synthetic mortgage / credit underwriting.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.automl.types import AutoMLBudget
from buildml.core.errors import MissingExtraError, ValidationError
from proofs._lib import (
    assert_no_test_in_selection,
    extra_available,
    load_mortgage_default_synthetic,
    metrics_round,
    new_proof_context,
    write_results,
)

FEATURE_NUM = ["ltv", "dti", "credit_score", "note_rate", "term_years"]
FEATURE_CAT = ["property_type"]
FEATURES = FEATURE_NUM + FEATURE_CAT
TARGET = "defaulted"


def _with_outreach(frame, seed: int):
    rng = np.random.default_rng(seed)
    out = frame.copy()
    ltv_z = (out["ltv"] - out["ltv"].median()) / (out["ltv"].std() + 1e-6)
    logit = -0.3 + 0.5 * ltv_z + 0.4 * out["dti"] + rng.normal(0, 0.3, size=len(out))
    treat = (1 / (1 + np.exp(-logit)) > 0.5).astype(int)
    flip = (out[TARGET] == 1) & (treat == 1) & (rng.random(len(out)) < 0.16)
    out.loc[flip, TARGET] = 0
    out["outreach"] = treat
    out["app_id"] = [f"app-{i}" for i in range(len(out))]
    return out


def main() -> None:
    ctx = new_proof_context("keystone-underwrite-ml", seed=104)
    base, data_meta = load_mortgage_default_synthetic(n=1400, seed=ctx.seed)
    frame = _with_outreach(base, seed=ctx.seed)
    stages: dict = {}
    skip_notes: list[str] = []

    session = (
        Session.ingest(frame.copy())
        .set_roles(
            {
                **{c: "feature" for c in FEATURES},
                TARGET: "target",
                "outreach": "ignore",
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
    session.impute(strategy="median")
    session.encode(method="onehot")
    session.scale(method="standard")

    # --- Stage 1: stacking ensemble ---
    try:
        bases = {
            "lr": LogisticRegression(max_iter=1000, random_state=ctx.seed),
            "rf": RandomForestClassifier(
                n_estimators=60, max_depth=5, random_state=ctx.seed
            ),
        }
        fit = session.fit_stacking(
            bases,
            final_estimator=LogisticRegression(max_iter=1000, random_state=ctx.seed),
            cv=3,
            task="classification",
        )
        val = session.evaluate_ensemble(partition="validation")
        assert_no_test_in_selection(
            selection_partition="validation", evaluation_partition="test"
        )
        test = session.evaluate_ensemble(partition="test")
        stages["stacking"] = {
            "status": "ok",
            "cv": 3,
            "fit": metrics_round(fit.to_dict() if hasattr(fit, "to_dict") else {}),
            "validation_metrics": metrics_round(dict(val.metrics)),
            "test_metrics": metrics_round(dict(test.metrics)),
        }
        write_results(ctx, stages["stacking"], filename="stacking.json")
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["stacking"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"stacking: {exc}")

    # --- Stage 2: AutoML ---
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

    # --- Stage 3: causal ---
    try:
        causal_session = (
            Session.ingest(frame.copy())
            .set_roles(
                {
                    "ltv": "feature",
                    "dti": "feature",
                    "credit_score": "feature",
                    "note_rate": "feature",
                    "term_years": "feature",
                    "outreach": "feature",
                    TARGET: "target",
                    "property_type": "ignore",
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
            confounders=["ltv", "dti", "credit_score", "note_rate", "term_years"],
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
                "confounders": ["ltv", "dti", "credit_score", "note_rate", "term_years"],
                "acknowledged": ["unconfoundedness", "positivity"],
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

    ok_stages = sum(1 for v in stages.values() if v.get("status") == "ok")
    summary = {
        "status": "completed" if ok_stages >= 3 else "partial",
        "product": "Keystone Underwrite ML",
        "data": {
            **data_meta,
            "treatment_column": "outreach",
            "notes": "Synthetic mortgage book + confounded outreach treatment",
        },
        "split": {"kind": plan.kind, "counts": split_counts, "stratify": True},
        "stages": {k: {"status": v.get("status")} for k, v in stages.items()},
        "stage_details": stages,
        "skip_notes": skip_notes,
        "leakage_controls": [
            "Stratified split before stacking / AutoML / causal",
            "OOF meta features from train CV folds only (cv=3)",
            "AutoML search/selection never uses the test partition",
            "Causal assumptions declared before fit_causal",
        ],
        "what_fails_if_leakage_ignored": [
            "Stacking with test in OOF folds invents ensemble ROC",
            "Fitting AutoML with test in the search loop invents leaderboard wins",
            "Skipping causal assumption declaration hides confounding risk",
        ],
        "limitations": [
            "Synthetic mortgage — not FCRA / bureau data",
            "Causal ATE assumes declared unconfoundedness (not proven)",
        ],
    }
    write_results(ctx, summary, filename="summary.json")
    write_results(ctx, summary, filename="results.json")
    print(
        "keystone-underwrite-ml OK",
        {
            "stacking": (stages.get("stacking") or {}).get("status"),
            "automl": (stages.get("automl") or {}).get("status"),
            "causal_ate": (stages.get("causal") or {}).get("fit", {}).get("ate"),
            "skips": skip_notes,
        },
    )


if __name__ == "__main__":
    main()
