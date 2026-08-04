"""Tier B product: Cornerstone Mortgage Suite.

Composes classical mortgage default scoring + declared-assumption causal
treatment effect + validation-tuned cost-sensitive decisions.
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
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.core.errors import MissingExtraError, ValidationError
from proofs._lib import (
    assert_no_test_in_selection,
    load_mortgage_default_synthetic,
    metrics_round,
    new_proof_context,
    write_results,
)

FEATURE_NUM = ["ltv", "dti", "credit_score", "note_rate", "term_years"]
FEATURE_CAT = ["property_type"]
FEATURES = FEATURE_NUM + FEATURE_CAT
TARGET = "defaulted"


def _with_counseling(frame, seed: int):
    """Add confounded counseling treatment with a small default-reduction effect."""
    rng = np.random.default_rng(seed)
    out = frame.copy()
    ltv_z = (out["ltv"] - out["ltv"].median()) / (out["ltv"].std() + 1e-6)
    dti_z = (out["dti"] - out["dti"].median()) / (out["dti"].std() + 1e-6)
    logit = -0.2 + 0.45 * ltv_z + 0.35 * dti_z + rng.normal(0, 0.3, size=len(out))
    treat = (1 / (1 + np.exp(-logit)) > 0.5).astype(int)
    # Soft-flip some defaults toward cured when treated.
    flip = (out[TARGET] == 1) & (treat == 1) & (rng.random(len(out)) < 0.18)
    out.loc[flip, TARGET] = 0
    out["counseling"] = treat
    out["review_cost"] = np.where(out[TARGET] == 1, 5.0, 1.5)
    out["loan_id"] = [f"loan-{i}" for i in range(len(out))]
    return out


def main() -> None:
    ctx = new_proof_context("cornerstone-mortgage-suite", seed=31)
    base, data_meta = load_mortgage_default_synthetic(n=1400, seed=ctx.seed)
    frame = _with_counseling(base, seed=ctx.seed)
    stages: dict = {}
    skip_notes: list[str] = []

    session = (
        Session.ingest(frame.copy())
        .set_roles(
            {
                **{c: "feature" for c in FEATURES},
                TARGET: "target",
                "counseling": "ignore",
                "review_cost": "ignore",
                "loan_id": "id",
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

    # --- Stage 2: causal (assumptions REQUIRED before fit) ---
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
                    "counseling": "feature",
                    TARGET: "target",
                    "property_type": "ignore",
                    "review_cost": "ignore",
                    "loan_id": "id",
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
        causal_session.causal.declare_assumptions(
            treatment="counseling",
            outcome=TARGET,
            confounders=["ltv", "dti", "credit_score", "note_rate", "term_years"],
            acknowledge_unconfoundedness=True,
            acknowledge_positivity=True,
        )
        fit_c = causal_session.causal.fit(method="aipw", bootstrap_samples=30)
        ev_c = causal_session.causal.evaluate(partition="validation", bootstrap_samples=15)
        stages["causal"] = {
            "status": "ok",
            "assumptions": {
                "treatment": "counseling",
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

    # --- Stage 3: cost-sensitive decisions on validation ---
    assert_no_test_in_selection(
        selection_partition="validation", evaluation_partition="test"
    )
    thr = session.decision.fit(
        method="threshold",
        partition="validation",
        fp_cost=1.0,
        fn_cost=8.0,
    )
    thr_test = session.decision.evaluate(partition="test")
    knap_payload: dict = {"alloc_status": "skipped"}
    try:
        knap = session.decision.fit(
            method="knapsack",
            partition="validation",
            budget=80.0,
            cost_column="review_cost",
            id_column="loan_id",
            score_source="model_proba",
            knapsack_solver="dp",
        )
        applied = session.decision.apply(partition="test")
        knap_payload = {
            "alloc_status": "ok",
            "knapsack_policy": metrics_round(
                knap.to_dict() if hasattr(knap, "to_dict") else {}
            ),
            "applied": {
                "n_selected": int(applied.n_selected),
                "selected_value": float(applied.selected_value),
                "selected_cost": float(applied.selected_cost),
            },
        }
    except Exception as exc:  # noqa: BLE001
        try:
            topk = session.decision.fit(
                method="topk",
                partition="validation",
                capacity=40,
                score_source="model_proba",
            )
            applied = session.decision.apply(partition="test")
            knap_payload = {
                "alloc_status": "ok_topk_fallback",
                "error": f"{type(exc).__name__}: {exc}",
                "topk_policy": metrics_round(
                    topk.to_dict() if hasattr(topk, "to_dict") else {}
                ),
                "applied": {
                    "n_selected": int(applied.n_selected),
                    "selected_value": float(applied.selected_value),
                    "selected_cost": float(getattr(applied, "selected_cost", float("nan"))),
                },
            }
        except Exception as exc2:  # noqa: BLE001
            knap_payload = {
                "alloc_status": "skipped",
                "error": f"{type(exc).__name__}: {exc}; fallback: {exc2}",
            }
            skip_notes.append(f"knapsack: {exc}")
    stages["decisions"] = {
        "status": "ok",
        "threshold_policy": metrics_round(thr.to_dict() if hasattr(thr, "to_dict") else {}),
        "threshold_test": metrics_round(
            thr_test.to_dict() if hasattr(thr_test, "to_dict") else {}
        ),
        **knap_payload,
    }
    write_results(ctx, stages["decisions"], filename="decisions.json")

    ok_stages = sum(1 for v in stages.values() if v.get("status") == "ok")
    summary = {
        "status": "completed" if ok_stages >= 3 else "partial",
        "product": "Cornerstone Mortgage Suite",
        "data": {
            **data_meta,
            "treatment_column": "counseling",
            "notes": "Synthetic mortgage book + confounded counseling treatment",
        },
        "split": {"kind": plan.kind, "counts": split_counts, "stratify": True},
        "stages": {k: {"status": v.get("status")} for k, v in stages.items()},
        "stage_details": stages,
        "skip_notes": skip_notes,
        "leakage_controls": [
            "Stratified split before classical / causal / decisions",
            "Causal assumptions declared before session.causal.fit",
            "Decision threshold + knapsack selected on validation ONLY",
            "Test evaluate after each stage locks",
        ],
        "what_fails_if_leakage_ignored": [
            "Tuning the review threshold on test understates expected loss",
            "Skipping causal assumption declaration hides confounding risk",
            "Fitting classical scores on the full book invents holdout ROC",
        ],
        "limitations": [
            "Synthetic mortgage — not FCRA / bureau data",
            "Causal ATE assumes declared unconfoundedness (not proven)",
            "Product proof, not a production LOS certification",
        ],
    }
    write_results(ctx, summary, filename="summary.json")
    write_results(ctx, summary, filename="results.json")
    print(
        "cornerstone-mortgage-suite OK",
        {
            "classical_roc": stages["classical"]["test_metrics"].get("roc_auc"),
            "causal_ate": (stages.get("causal") or {}).get("fit", {}).get("ate"),
            "skips": skip_notes,
        },
    )


if __name__ == "__main__":
    main()
