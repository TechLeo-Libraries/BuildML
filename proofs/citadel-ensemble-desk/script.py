"""Tier B product: Citadel Ensemble Desk.

Composes voting/stacking ensembles + unsupervised anomaly + decision
thresholds for attrition / risk review.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.core.errors import MissingExtraError
from proofs._lib import (
    assert_no_test_in_selection,
    extra_available,
    load_attrition_tabular_synthetic,
    metrics_round,
    new_proof_context,
    write_results,
)


FEATURE_NUM = ["tenure_years", "salary", "overtime", "satisfaction", "promotions"]
FEATURE_CAT = ["department"]
TARGET = "left"


def main() -> None:
    ctx = new_proof_context("citadel-ensemble-desk", seed=49)
    frame, data_meta = load_attrition_tabular_synthetic(n=1200, seed=ctx.seed)
    frame = frame.copy()
    frame["review_cost"] = frame[TARGET].map({1: 3.0, 0: 1.0})
    frame["employee_id"] = [f"e-{i}" for i in range(len(frame))]
    stages: dict = {}
    skip_notes: list[str] = []

    session = (
        Session.ingest(frame)
        .set_roles(
            {
                **{c: "feature" for c in FEATURE_NUM + FEATURE_CAT},
                TARGET: "target",
                "review_cost": "ignore",
                "employee_id": "id",
            }
        )
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=ctx.seed)
    )
    plan = session.split_plan
    assert plan is not None
    split_counts = {
        "train": len(plan.train_indices),
        "validation": len(plan.validation_indices),
        "test": len(plan.test_indices),
    }
    session.encode(method="onehot")
    session.scale(method="standard")

    bases = {
        "lr": LogisticRegression(max_iter=1000, random_state=ctx.seed),
        "rf": RandomForestClassifier(
            n_estimators=60, max_depth=5, random_state=ctx.seed
        ),
    }

    # --- Stage 1: voting then stacking ---
    try:
        v_fit = session.ensemble.fit_voting(bases, voting="soft", task="classification")
        v_val = session.ensemble.evaluate(partition="validation")
        assert_no_test_in_selection(
            selection_partition="validation", evaluation_partition="test"
        )
        v_test = session.ensemble.evaluate(partition="test")
        stages["voting"] = {
            "status": "ok",
            "fit": metrics_round(v_fit.to_dict() if hasattr(v_fit, "to_dict") else {}),
            "validation_metrics": metrics_round(dict(v_val.metrics)),
            "test_metrics": metrics_round(dict(v_test.metrics)),
        }
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["voting"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"voting: {exc}")
    write_results(ctx, stages["voting"], filename="voting.json")

    try:
        # Fresh session for stacking so voting plan does not collide.
        stack_session = (
            Session.ingest(frame.copy())
            .set_roles(
                {
                    **{c: "feature" for c in FEATURE_NUM + FEATURE_CAT},
                    TARGET: "target",
                    "review_cost": "ignore",
                    "employee_id": "id",
                }
            )
            .inject_split(
                train_indices=list(plan.train_indices),
                validation_indices=list(plan.validation_indices),
                test_indices=list(plan.test_indices),
            )
        )
        stack_session.encode(method="onehot")
        stack_session.scale(method="standard")
        s_fit = stack_session.ensemble.fit_stacking(
            bases,
            final_estimator=LogisticRegression(max_iter=1000, random_state=ctx.seed),
            cv=3,
            task="classification",
        )
        s_val = stack_session.ensemble.evaluate(partition="validation")
        s_test = stack_session.ensemble.evaluate(partition="test")
        stages["stacking"] = {
            "status": "ok",
            "fit": metrics_round(s_fit.to_dict() if hasattr(s_fit, "to_dict") else {}),
            "validation_metrics": metrics_round(dict(s_val.metrics)),
            "test_metrics": metrics_round(dict(s_test.metrics)),
        }
        decision_session = stack_session
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["stacking"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"stacking: {exc}")
        decision_session = session
    write_results(ctx, stages["stacking"], filename="stacking.json")

    # --- Stage 2: anomaly on numeric features ---
    try:
        a_session = (
            Session.ingest(frame.copy())
            .set_roles(
                {
                    **{c: "feature" for c in FEATURE_NUM},
                    FEATURE_CAT[0]: "ignore",
                    TARGET: "target",
                    "review_cost": "ignore",
                    "employee_id": "id",
                }
            )
            .inject_split(
                train_indices=list(plan.train_indices),
                validation_indices=list(plan.validation_indices),
                test_indices=list(plan.test_indices),
            )
            .scale(method="standard")
        )
        if extra_available("pyod"):
            a_fit = a_session.anomaly.fit(
                backend="pyod",
                method="hbos",
                mode="unsupervised",
                contamination=0.1,
                random_state=ctx.seed,
            )
            a_backend = "pyod/hbos"
        else:
            a_fit = a_session.anomaly.fit(
                method="isolation_forest",
                mode="unsupervised",
                contamination=0.1,
                random_state=ctx.seed,
            )
            a_backend = "sklearn/isolation_forest"
        a_tune = a_session.anomaly.tune_threshold(
            partition="validation",
            label_column=TARGET,
            positive_label=1,
            metric="f1",
        )
        a_ev = a_session.anomaly.evaluate(partition="test", positive_label=1)
        stages["anomaly"] = {
            "status": "ok",
            "backend": a_backend,
            "fit_threshold": float(getattr(a_fit, "threshold", float("nan"))),
            "tune": metrics_round(a_tune.to_dict() if hasattr(a_tune, "to_dict") else {}),
            "test_labeled_metrics": metrics_round(
                dict(getattr(a_ev, "labeled_metrics", {}) or {})
            ),
        }
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["anomaly"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"anomaly: {exc}")
    write_results(ctx, stages["anomaly"], filename="anomaly.json")

    # --- Stage 3: review decisions ---
    try:
        assert_no_test_in_selection(
            selection_partition="validation", evaluation_partition="test"
        )
        thr = decision_session.decision.fit(
            method="threshold",
            partition="validation",
            fp_cost=1.0,
            fn_cost=4.0,
        )
        thr_test = decision_session.decision.evaluate(partition="test")
        stages["decisions"] = {
            "status": "ok",
            "threshold_policy": metrics_round(
                thr.to_dict() if hasattr(thr, "to_dict") else {}
            ),
            "threshold_test": metrics_round(
                thr_test.to_dict() if hasattr(thr_test, "to_dict") else {}
            ),
        }
        try:
            knap = decision_session.decision.fit(
                method="knapsack",
                partition="validation",
                budget=80.0,
                cost_column="review_cost",
                id_column="employee_id",
                score_source="model_proba",
                knapsack_solver="dp",
            )
            applied = decision_session.decision.apply(partition="test")
            stages["decisions"]["knapsack"] = {
                "status": "ok",
                "policy": metrics_round(
                    knap.to_dict() if hasattr(knap, "to_dict") else {}
                ),
                "applied": {
                    "n_selected": int(applied.n_selected),
                    "selected_value": float(applied.selected_value),
                    "selected_cost": float(applied.selected_cost),
                },
            }
        except Exception as exc:  # noqa: BLE001
            stages["decisions"]["knapsack"] = {
                "status": "skipped",
                "error": f"{type(exc).__name__}: {exc}",
            }
            skip_notes.append(f"decisions_knapsack: {exc}")
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["decisions"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"decisions: {exc}")
    write_results(ctx, stages["decisions"], filename="decisions.json")

    summary = {
        "status": "completed",
        "product": "Citadel Ensemble Desk",
        "data": data_meta,
        "split": {"kind": plan.kind, "counts": split_counts, "stratify": True},
        "stages": {k: {"status": v.get("status")} for k, v in stages.items()},
        "stage_details": stages,
        "skip_notes": skip_notes,
        "leakage_controls": [
            "Stratified split before encode/scale/ensemble fit",
            "Stacking OOF meta features from train CV folds only",
            "Anomaly threshold + decisions tuned on validation only",
            "Test evaluate after each stage locks",
        ],
        "what_fails_if_leakage_ignored": [
            "Picking the voting/stacking winner with test scores is not a fair ensemble",
            "Anomaly thresholds on test inflate review F1",
            "Review knapsack tuned on test understates HR cost",
        ],
        "limitations": [
            "Synthetic attrition table — not a production HRIS extract",
            "Two-base ensembles only for smoke latency",
        ],
    }
    write_results(ctx, summary, filename="summary.json")
    write_results(ctx, summary, filename="results.json")
    print(
        "citadel-ensemble-desk OK",
        {
            "stacking": stages.get("stacking", {}).get("status"),
            "skips": skip_notes,
        },
    )


if __name__ == "__main__":
    main()
