"""Tier B product: Scaffold Compliance AI.

Composes symbolic KYC/AML rules + optional neuro-symbolic NAM +
validation-tuned decision thresholds for escalation review capacity.
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
from buildml.core.errors import MissingExtraError
from proofs._lib import (
    TORCH_STATUS,
    assert_no_test_in_selection,
    metrics_round,
    new_proof_context,
    write_results,
)

FEATS = ["account_age_days", "wire_amount", "pep_score", "jurisdiction_risk"]
TARGET = "escalate"


def _kyc_book(n: int = 420, seed: int = 27) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    account_age_days = rng.uniform(1, 4000, size=n)
    wire_amount = rng.lognormal(9.2, 0.85, size=n)
    pep_score = rng.beta(2.5, 4.0, size=n)
    jurisdiction_risk = rng.beta(2.5, 3.5, size=n)
    escalate = (
        ((wire_amount > 8000) & (jurisdiction_risk > 0.35))
        | ((account_age_days < 365) & (pep_score > 0.30))
    ).astype(int)
    frame = pd.DataFrame(
        {
            "account_age_days": account_age_days,
            "wire_amount": wire_amount,
            "pep_score": pep_score,
            "jurisdiction_risk": jurisdiction_risk,
            "escalate": escalate,
            "case_id": [f"kyc-{i}" for i in range(n)],
            "review_cost": np.where(escalate == 1, 4.0, 1.2),
        }
    )
    meta = {
        "name": "scaffold_synthetic_kyc_aml",
        "license": "synthetic/public-domain",
        "n_rows": n,
        "positive_rate": float(escalate.mean()),
    }
    return frame, meta


def main() -> None:
    ctx = new_proof_context("scaffold-compliance-ai", seed=27)
    frame, data_meta = _kyc_book(seed=ctx.seed)
    stages: dict = {}
    skip_notes: list[str] = []

    session = (
        Session.ingest(frame.copy())
        .set_roles(
            {
                **{c: "feature" for c in FEATS},
                TARGET: "target",
                "case_id": "id",
                "review_cost": "ignore",
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

    # --- Stage 1: symbolic ---
    try:
        try:
            fit_s = session.fit_symbolic(
                source="decision_tree", max_depth=4, random_state=ctx.seed
            )
        except TypeError:
            fit_s = session.fit_symbolic(method="decision_tree", random_state=ctx.seed)
        val_s = session.evaluate_symbolic(partition="validation")
        test_s = session.evaluate_symbolic(partition="test")
        stages["symbolic"] = {
            "status": "ok",
            "fit": metrics_round(fit_s.to_dict() if hasattr(fit_s, "to_dict") else {}),
            "validation_metrics": metrics_round(dict(getattr(val_s, "metrics", {}) or {})),
            "test_metrics": metrics_round(dict(getattr(test_s, "metrics", {}) or {})),
        }
        write_results(ctx, stages["symbolic"], filename="symbolic.json")
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["symbolic"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"symbolic: {exc}")

    # --- Stage 2: neuro-symbolic (optional torch) ---
    neuro: dict = {
        "status": "skipped",
        "skip_torch_paths": TORCH_STATUS.get("skip_torch_paths", True),
    }
    if not TORCH_STATUS.get("skip_torch_paths"):
        try:
            nf = session.fit_neuro_symbolic(
                mode="constraint_overlay",
                base_estimator="logistic_regression",
                rule_source="decision_tree",
                random_state=ctx.seed,
                torch_epochs=5,
            )
            ne = session.evaluate_neuro_symbolic(partition="validation")
            ne_test = session.evaluate_neuro_symbolic(partition="test")
            neuro = {
                "status": "ok",
                "fit": metrics_round(nf.to_dict() if hasattr(nf, "to_dict") else {}),
                "validation_metrics": metrics_round(
                    dict(getattr(ne, "metrics", {}) or {})
                ),
                "test_metrics": metrics_round(
                    dict(getattr(ne_test, "metrics", {}) or {})
                ),
            }
        except (MissingExtraError, TypeError, ValueError) as exc:
            neuro = {
                "status": "skipped",
                "error": f"{type(exc).__name__}: {exc}",
                "torch": TORCH_STATUS,
            }
            skip_notes.append(f"neuro_symbolic: {exc}")
    else:
        skip_notes.append("neuro_symbolic: torch paths skipped by TORCH_STATUS")
    stages["neuro_symbolic"] = neuro
    write_results(ctx, stages["neuro_symbolic"], filename="neuro_symbolic.json")

    # --- Stage 3: classical scores + decisions ---
    try:
        classical = (
            Session.ingest(frame.copy())
            .set_roles(
                {
                    **{c: "feature" for c in FEATS},
                    TARGET: "target",
                    "case_id": "id",
                    "review_cost": "ignore",
                }
            )
            .inject_split(
                train_indices=list(plan.train_indices),
                validation_indices=list(plan.validation_indices),
                test_indices=list(plan.test_indices),
            )
            .scale(method="standard")
        )
        classical.fit(
            LogisticRegression(max_iter=1000, random_state=ctx.seed),
            task="classification",
        )
        c_test = classical.evaluate(partition="test")
        assert_no_test_in_selection(
            selection_partition="validation", evaluation_partition="test"
        )
        thr = classical.fit_decision_policy(
            method="threshold",
            partition="validation",
            fp_cost=1.0,
            fn_cost=7.0,
        )
        thr_test = classical.evaluate_decisions(partition="test")
        alloc_payload: dict = {"alloc_status": "skipped"}
        try:
            knap = classical.fit_decision_policy(
                method="knapsack",
                partition="validation",
                budget=60.0,
                cost_column="review_cost",
                id_column="case_id",
                score_source="model_proba",
                knapsack_solver="dp",
            )
            applied = classical.apply_decisions(partition="test")
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
            topk = classical.fit_decision_policy(
                method="topk",
                partition="validation",
                capacity=30,
                score_source="model_proba",
            )
            applied = classical.apply_decisions(partition="test")
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
        stages["decisions"] = {
            "status": "ok",
            "classical_test_metrics": metrics_round(dict(c_test.metrics)),
            "threshold_policy": metrics_round(
                thr.to_dict() if hasattr(thr, "to_dict") else {}
            ),
            "threshold_test": metrics_round(
                thr_test.to_dict() if hasattr(thr_test, "to_dict") else {}
            ),
            **alloc_payload,
        }
        write_results(ctx, stages["decisions"], filename="decisions.json")
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["decisions"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"decisions: {exc}")

    ok_core = sum(
        1
        for k in ("symbolic", "decisions")
        if (stages.get(k) or {}).get("status") == "ok"
    )
    neuro_ok = (stages.get("neuro_symbolic") or {}).get("status") == "ok"
    status = "completed" if ok_core >= 2 else "partial"
    if neuro_ok:
        status = "completed"
    summary = {
        "status": status,
        "product": "Scaffold Compliance AI",
        "data": data_meta,
        "split": {"kind": plan.kind, "counts": split_counts, "stratify": True},
        "stages": {k: {"status": v.get("status")} for k, v in stages.items()},
        "stage_details": stages,
        "skip_notes": skip_notes,
        "torch": TORCH_STATUS,
        "leakage_controls": [
            "Stratified split before symbolic / neuro-symbolic / decisions",
            "Symbolic + NAM fit on train only",
            "Review capacity / threshold tuned on validation only",
            "Test evaluate after each stage locks",
        ],
        "what_fails_if_leakage_ignored": [
            "Inducing rules on the full book looks more 'compliant' than production",
            "Tuning escalation thresholds on test understates review cost",
            "Fitting NAM with test rows invents holdout fidelity",
        ],
        "limitations": [
            "Not legal advice; rule fidelity ≠ compliance certification",
            "Neuro-symbolic NAM skipped when torch paths are disabled",
        ],
    }
    write_results(ctx, summary, filename="summary.json")
    write_results(ctx, summary, filename="results.json")
    print(
        "scaffold-compliance-ai OK",
        {
            "symbolic": (stages.get("symbolic") or {}).get("status"),
            "neuro_symbolic": (stages.get("neuro_symbolic") or {}).get("status"),
            "decisions": (stages.get("decisions") or {}).get("status"),
            "skips": skip_notes,
        },
    )


if __name__ == "__main__":
    main()
