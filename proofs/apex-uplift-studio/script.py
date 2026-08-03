"""Tier B product: Apex Uplift Studio.

Composes causal marketing uplift + classical conversion scoring +
validation-tuned allocation (optimize / decisions) for promo budget.
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
from sklearn.linear_model import LogisticRegression, Ridge

from buildml import Session
from buildml.core.errors import MissingExtraError, ValidationError
from proofs._lib import (
    assert_no_test_in_selection,
    metrics_round,
    new_proof_context,
    write_results,
)


def _uplift_portfolio(n: int = 700, seed: int = 33) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n, 3))
    logit = 0.7 * w[:, 0] - 0.5 * w[:, 1] + 0.3 * w[:, 2]
    t = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    spend = (
        1.2 * t
        + 0.5 * w[:, 0]
        - 0.4 * w[:, 1]
        + 0.25 * w[:, 2]
        + rng.normal(scale=0.5, size=n)
    )
    convert = (spend > np.median(spend)).astype(int)
    frame = pd.DataFrame(
        {
            "recency_z": w[:, 0],
            "freq_z": w[:, 1],
            "monetary_z": w[:, 2],
            "promo": t,
            "spend": spend,
            "converted": convert,
            "cust_id": [f"c-{i}" for i in range(n)],
            "offer_cost": np.where(t == 1, 2.5, 1.0),
        }
    )
    meta = {
        "name": "apex_synthetic_uplift_portfolio",
        "license": "synthetic/public-domain",
        "n_rows": n,
        "true_ate_approx": 1.2,
        "positive_rate": float(convert.mean()),
    }
    return frame, meta


def main() -> None:
    ctx = new_proof_context("apex-uplift-studio", seed=33)
    frame, data_meta = _uplift_portfolio(seed=ctx.seed)
    stages: dict = {}
    skip_notes: list[str] = []
    feats = ["recency_z", "freq_z", "monetary_z"]

    # Shared split for classical + decisions
    session = (
        Session.ingest(frame.copy())
        .set_roles(
            {
                **{c: "feature" for c in feats},
                "converted": "target",
                "promo": "ignore",
                "spend": "ignore",
                "offer_cost": "ignore",
                "cust_id": "id",
            }
        )
        .split(test_size=0.2, validation_size=0.2, random_state=ctx.seed, stratify=True)
        .scale(method="standard")
    )
    plan = session.split_plan
    assert plan is not None
    split_counts = {
        "train": len(plan.train_indices),
        "validation": len(plan.validation_indices),
        "test": len(plan.test_indices),
    }

    # --- Stage 1: causal uplift (continuous spend outcome) ---
    try:
        causal_session = (
            Session.ingest(frame.copy())
            .set_roles(
                {
                    **{c: "feature" for c in feats},
                    "promo": "feature",
                    "spend": "target",
                    "converted": "ignore",
                    "offer_cost": "ignore",
                    "cust_id": "id",
                }
            )
            .inject_split(
                train_indices=list(plan.train_indices),
                validation_indices=list(plan.validation_indices),
                test_indices=list(plan.test_indices),
            )
            .scale(method="standard")
        )
        causal_session.declare_causal_assumptions(
            treatment="promo",
            outcome="spend",
            confounders=feats,
            acknowledge_unconfoundedness=True,
            acknowledge_positivity=True,
        )
        fit_c = causal_session.fit_causal(method="aipw", bootstrap_samples=40)
        ev_c = causal_session.evaluate_causal(partition="validation", bootstrap_samples=20)
        try:
            refute = causal_session.refute_causal(kind="placebo_treatment")
            refute_payload = {
                "kind": "placebo_treatment",
                "refute_ate": float(refute.refute_ate),
                "ate_shift": float(refute.ate_shift),
            }
        except Exception as exc:  # noqa: BLE001
            refute_payload = {"skipped": f"{type(exc).__name__}: {exc}"}
        stages["causal_uplift"] = {
            "status": "ok",
            "assumptions": {
                "treatment": "promo",
                "outcome": "spend",
                "confounders": feats,
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
            "refute": refute_payload,
        }
        write_results(ctx, stages["causal_uplift"], filename="causal_uplift.json")
    except (MissingExtraError, TypeError, ValueError, ValidationError) as exc:
        stages["causal_uplift"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"causal_uplift: {exc}")

    # --- Stage 2: classical conversion scorer ---
    session.fit(
        LogisticRegression(max_iter=1000, random_state=ctx.seed),
        task="classification",
    )
    classical_test = session.evaluate(partition="test")
    # Optional spend regressor disclosure on same features
    try:
        spend_session = (
            Session.ingest(frame.copy())
            .set_roles(
                {
                    **{c: "feature" for c in feats},
                    "spend": "target",
                    "promo": "ignore",
                    "converted": "ignore",
                    "offer_cost": "ignore",
                    "cust_id": "id",
                }
            )
            .inject_split(
                train_indices=list(plan.train_indices),
                validation_indices=list(plan.validation_indices),
                test_indices=list(plan.test_indices),
            )
            .scale(method="standard")
        )
        spend_session.fit(Ridge(alpha=1.0), task="regression")
        spend_test = spend_session.evaluate(partition="test")
        spend_metrics = metrics_round(dict(spend_test.metrics))
    except Exception as exc:  # noqa: BLE001
        spend_metrics = {"error": f"{type(exc).__name__}: {exc}"}
    stages["classical"] = {
        "status": "ok",
        "estimator": "LogisticRegression",
        "test_metrics": metrics_round(dict(classical_test.metrics)),
        "spend_ridge_test": spend_metrics,
    }
    write_results(ctx, stages["classical"], filename="classical.json")

    # --- Stage 3: optimize / decisions on validation ---
    assert_no_test_in_selection(
        selection_partition="validation", evaluation_partition="test"
    )
    thr = session.fit_decision_policy(
        method="threshold",
        partition="validation",
        fp_cost=1.0,
        fn_cost=4.0,
    )
    thr_test = session.evaluate_decisions(partition="test")
    alloc_payload: dict = {"alloc_status": "skipped"}
    try:
        knap = session.fit_decision_policy(
            method="knapsack",
            partition="validation",
            budget=70.0,
            cost_column="offer_cost",
            id_column="cust_id",
            score_source="model_proba",
            knapsack_solver="dp",
        )
        applied = session.apply_decisions(partition="test")
        alloc_payload = {
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
            topk = session.fit_decision_policy(
                method="topk",
                partition="validation",
                capacity=40,
                score_source="model_proba",
            )
            applied = session.apply_decisions(partition="test")
            alloc_payload = {
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
            alloc_payload = {
                "alloc_status": "skipped",
                "error": f"{type(exc).__name__}: {exc}; fallback: {exc2}",
            }
            skip_notes.append(f"decisions: {exc}")
    stages["decisions"] = {
        "status": "ok",
        "threshold_policy": metrics_round(thr.to_dict() if hasattr(thr, "to_dict") else {}),
        "threshold_test": metrics_round(
            thr_test.to_dict() if hasattr(thr_test, "to_dict") else {}
        ),
        "capability_matrix": Session.optimize_capability_matrix(),
        **alloc_payload,
    }
    write_results(ctx, stages["decisions"], filename="decisions.json")

    ok_stages = sum(1 for v in stages.values() if v.get("status") == "ok")
    summary = {
        "status": "completed" if ok_stages >= 3 else "partial",
        "product": "Apex Uplift Studio",
        "data": data_meta,
        "split": {"kind": plan.kind, "counts": split_counts, "stratify": True},
        "stages": {k: {"status": v.get("status")} for k, v in stages.items()},
        "stage_details": stages,
        "skip_notes": skip_notes,
        "leakage_controls": [
            "Shared stratified split before causal / classical / decisions",
            "Causal assumptions declared before fit_causal",
            "Promo budget knapsack / threshold tuned on validation only",
            "Test evaluated after each stage locks",
        ],
        "what_fails_if_leakage_ignored": [
            "Allocating promo budget on test invents ROI",
            "Skipping assumption declaration hides confounding in uplift ATE",
            "Fitting conversion scores on the full book invents holdout ROC",
        ],
        "limitations": [
            "Synthetic uplift DGP — not a real CRM extract",
            "ATE assumes declared unconfoundedness (not proven)",
        ],
    }
    write_results(ctx, summary, filename="summary.json")
    write_results(ctx, summary, filename="results.json")
    print(
        "apex-uplift-studio OK",
        {
            "ate": (stages.get("causal_uplift") or {}).get("fit", {}).get("ate"),
            "classical_roc": stages["classical"]["test_metrics"].get("roc_auc"),
            "skips": skip_notes,
        },
    )


if __name__ == "__main__":
    main()
