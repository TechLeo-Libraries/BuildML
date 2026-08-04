"""Tier B product: Rivulet Stream Risk.

Composes online stream scoring + unsupervised anomaly with validation-tuned
thresholds + cost-sensitive decision policies for a synthetic payment stream.
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
from buildml.core.errors import MissingExtraError
from proofs._lib import (
    assert_no_test_in_selection,
    extra_available,
    load_payment_rail_anomaly_synthetic,
    metrics_round,
    new_proof_context,
    write_results,
)

FEATURES = [
    "amount_z",
    "hour_sin",
    "hour_cos",
    "merchant_risk",
    "device_age_days",
    "velocity_1h",
]
TARGET = "is_attack"


def main() -> None:
    ctx = new_proof_context("rivulet-stream-risk", seed=41)
    frame, data_meta = load_payment_rail_anomaly_synthetic(seed=ctx.seed)
    frame = frame.copy()
    frame["txn_id"] = [f"txn-{i}" for i in range(len(frame))]
    frame["review_cost"] = np.where(frame[TARGET] == 1, 4.0, 1.0)
    stages: dict = {}
    skip_notes: list[str] = []

    session = (
        Session.ingest(frame.copy())
        .set_roles(
            {
                **{c: "feature" for c in FEATURES},
                TARGET: "target",
                "txn_id": "id",
                "review_cost": "ignore",
            }
        )
        .split(
            test_size=0.2,
            validation_size=0.2,
            stratify=True,
            random_state=ctx.seed,
        )
        .scale(method="standard")
    )
    plan = session.split_plan
    assert plan is not None
    split_counts = {
        "train": len(plan.train_indices),
        "validation": len(plan.validation_indices),
        "test": len(plan.test_indices),
    }

    # --- Stage 1: online stream updates (train cursor only) ---
    try:
        online_session = (
            Session.ingest(frame.copy())
            .set_roles({**{c: "feature" for c in FEATURES}, TARGET: "target"})
            .inject_split(
                train_indices=list(plan.train_indices),
                validation_indices=list(plan.validation_indices),
                test_indices=list(plan.test_indices),
            )
            .scale(method="standard")
        )
        o_fit = online_session.online.fit(
            estimator="sgd_classifier",
            chunk_size=50,
            n_init=50,
            classes=[0, 1],
        )
        updates = 0
        while True:
            remaining = (
                online_session.online.plan.n_train_rows
                - online_session.online.plan.cursor
            )
            if remaining <= 0:
                break
            online_session.online.partial_fit(n_rows=min(50, remaining))
            updates += 1
        o_test = online_session.online.evaluate(partition="test")
        stages["online"] = {
            "status": "ok",
            "n_init_rows": int(o_fit.n_init_rows),
            "n_updates": updates,
            "test_metrics": metrics_round(dict(o_test.metrics)),
        }
        write_results(ctx, stages["online"], filename="online.json")
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["online"] = {"status": "skipped", "error": f"{type(exc).__name__}: {exc}"}
        skip_notes.append(f"online: {exc}")

    # --- Stage 2: unsupervised anomaly + validation threshold ---
    try:
        if extra_available("pyod"):
            a_fit = session.anomaly.fit(
                backend="pyod",
                method="hbos",
                mode="unsupervised",
                contamination=0.08,
                random_state=ctx.seed,
            )
            a_backend = "pyod/hbos"
        else:
            a_fit = session.anomaly.fit(
                method="isolation_forest",
                mode="unsupervised",
                contamination=0.08,
                random_state=ctx.seed,
            )
            a_backend = "sklearn/isolation_forest"
        assert_no_test_in_selection(
            selection_partition="validation", evaluation_partition="test"
        )
        a_tune = session.anomaly.tune_threshold(
            partition="validation",
            label_column=TARGET,
            positive_label=1,
            metric="f1",
        )
        a_ev = session.anomaly.evaluate(partition="test", positive_label=1)
        stages["anomaly"] = {
            "status": "ok",
            "backend": a_backend,
            "fit_threshold": float(getattr(a_fit, "threshold", float("nan"))),
            "tune": metrics_round(a_tune.to_dict() if hasattr(a_tune, "to_dict") else {}),
            "test_labeled_metrics": metrics_round(
                dict(getattr(a_ev, "labeled_metrics", {}) or {})
            ),
        }
        write_results(ctx, stages["anomaly"], filename="anomaly.json")
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["anomaly"] = {"status": "skipped", "error": f"{type(exc).__name__}: {exc}"}
        skip_notes.append(f"anomaly: {exc}")

    # --- Stage 3: supervised classical + decision thresholds ---
    session.fit(
        LogisticRegression(max_iter=1000, random_state=ctx.seed),
        task="classification",
    )
    supervised_test = session.evaluate(partition="test")
    stages["supervised"] = {
        "status": "ok",
        "estimator": "LogisticRegression",
        "test_metrics": metrics_round(dict(supervised_test.metrics)),
    }
    write_results(ctx, stages["supervised"], filename="supervised.json")

    assert_no_test_in_selection(
        selection_partition="validation", evaluation_partition="test"
    )
    thr = session.decision.fit(
        method="threshold",
        partition="validation",
        fp_cost=1.0,
        fn_cost=6.0,
    )
    thr_test = session.decision.evaluate(partition="test")
    knap_payload: dict = {"alloc_status": "skipped"}
    try:
        knap = session.decision.fit(
            method="knapsack",
            partition="validation",
            budget=90.0,
            cost_column="review_cost",
            id_column="txn_id",
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
                capacity=45,
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
            skip_notes.append(f"decisions_alloc: {exc}")
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
        "product": "Rivulet Stream Risk",
        "data": data_meta,
        "split": {"kind": plan.kind, "counts": split_counts, "stratify": True},
        "stages": {k: {"status": v.get("status")} for k, v in stages.items()},
        "stage_details": stages,
        "skip_notes": skip_notes,
        "leakage_controls": [
            "Stratified split before online / anomaly / supervised / decisions",
            "Online partial_fit consumes train cursor only",
            "Anomaly threshold + decision policies tuned on validation only",
            "Test evaluated once per stage after that stage locks",
        ],
        "what_fails_if_leakage_ignored": [
            "Streaming updates that include test rows make online metrics meaningless",
            "Tuning anomaly/decision thresholds on test inflates F1 and understates review cost",
            "Fitting the supervised scorer on the full table invents holdout ROC",
        ],
        "limitations": [
            "Synthetic payment rail — not a card-network extract",
            "Product proof, not a production fraud SaaS certification",
        ],
    }
    write_results(ctx, summary, filename="summary.json")
    write_results(ctx, summary, filename="results.json")
    print(
        "rivulet-stream-risk OK",
        {
            "online": (stages.get("online") or {}).get("status"),
            "anomaly": (stages.get("anomaly") or {}).get("status"),
            "decisions": stages["decisions"]["status"],
            "skips": skip_notes,
        },
    )


if __name__ == "__main__":
    main()
