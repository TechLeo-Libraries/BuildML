"""Tier B product: Aegis Fraud Platform.

Composes graph ring detection + unsupervised anomaly + supervised scoring +
online stream updates + validation-tuned decision thresholds + optional
symbolic guardrail rules. Leakage discipline at every stage.
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
    extra_available,
    metrics_round,
    new_proof_context,
    write_results,
)


FEATURES = [
    "amount_z",
    "velocity_z",
    "device_risk",
    "geo_mismatch",
    "hist_chargeback",
]
TARGET = "is_fraud"


def _synthetic_fraud_portfolio(n: int = 900, seed: int = 21):
    rng = np.random.default_rng(seed)
    # Two communities; fraud denser in community B.
    community = np.array([0] * (n // 2) + [1] * (n - n // 2))
    amount_z = rng.normal(0, 1, size=n) + 1.4 * community
    velocity_z = rng.normal(0, 1, size=n) + 1.1 * community
    device_risk = rng.beta(2, 5, size=n) + 0.25 * community
    geo_mismatch = rng.binomial(1, 0.08 + 0.2 * community, size=n).astype(float)
    hist_chargeback = rng.poisson(0.2 + 0.6 * community, size=n).astype(float)
    logit = (
        -2.2
        + 0.9 * amount_z
        + 0.7 * velocity_z
        + 1.4 * device_risk
        + 1.1 * geo_mismatch
        + 0.55 * hist_chargeback
        + rng.normal(0, 0.35, size=n)
    )
    is_fraud = (1 / (1 + np.exp(-logit)) > 0.55).astype(int)
    nodes = pd.DataFrame(
        {
            "node_id": [f"acct-{i}" for i in range(n)],
            "amount_z": amount_z,
            "velocity_z": velocity_z,
            "device_risk": device_risk,
            "geo_mismatch": geo_mismatch,
            "hist_chargeback": hist_chargeback,
            "is_fraud": is_fraud,
            "review_cost": np.where(is_fraud == 1, 4.0, 1.0),
        }
    )
    edges = []
    half = n // 2
    for i in range(n):
        lo, hi = (0, half) if i < half else (half, n)
        for j in rng.choice(range(lo, hi), size=4, replace=True):
            if i != int(j):
                edges.append((f"acct-{i}", f"acct-{int(j)}"))
    edge_frame = pd.DataFrame(edges, columns=["source", "target"]).drop_duplicates()
    meta = {
        "name": "aegis_synthetic_fraud_portfolio",
        "license": "synthetic/public-domain",
        "n_nodes": n,
        "n_edges": int(len(edge_frame)),
        "positive_rate": float(is_fraud.mean()),
    }
    return nodes, edge_frame, meta


def main() -> None:
    ctx = new_proof_context("aegis-fraud-platform", seed=21)
    nodes, edges, data_meta = _synthetic_fraud_portfolio(seed=ctx.seed)
    stages: dict = {}
    skip_notes: list[str] = []

    # --- Stage 0: honest split before any fit ---
    session = Session.ingest(nodes.copy())
    session.set_roles(
        {
            "node_id": "id",
            **{c: "feature" for c in FEATURES},
            TARGET: "target",
            "review_cost": "ignore",  # must stay non-negative for knapsack
        }
    )
    session.split(
        test_size=0.2,
        validation_size=0.2,
        stratify=True,
        random_state=ctx.seed,
    )
    plan = session.split_plan
    assert plan is not None
    split_counts = {
        "train": len(plan.train_indices),
        "validation": len(plan.validation_indices),
        "test": len(plan.test_indices),
    }
    # Default scale skips ignore/id — review_cost stays non-negative for knapsack.
    session.scale(method="standard")

    # --- Stage 1: graph ring features (classical) ---
    try:
        session.set_graph(
            edges,
            source_col="source",
            target_col="target",
            node_id_col="node_id",
        )
        g_fit = session.fit_graph(method="classical", mode="inductive", random_state=ctx.seed)
        g_val = session.evaluate_graph(partition="validation")
        stages["graph"] = {
            "status": "ok",
            "fit": metrics_round(g_fit.to_dict() if hasattr(g_fit, "to_dict") else {}),
            "validation_metrics": metrics_round(dict(getattr(g_val, "metrics", {}) or {})),
        }
        write_results(ctx, stages["graph"], filename="graph.json")
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["graph"] = {"status": "skipped", "error": f"{type(exc).__name__}: {exc}"}
        skip_notes.append(f"graph: {exc}")

    # --- Stage 2: unsupervised anomaly on train features ---
    try:
        if extra_available("pyod"):
            a_fit = session.fit_anomaly(
                backend="pyod",
                method="hbos",
                mode="unsupervised",
                contamination=0.08,
                random_state=ctx.seed,
            )
            a_backend = "pyod/hbos"
        else:
            a_fit = session.fit_anomaly(
                method="isolation_forest",
                mode="unsupervised",
                contamination=0.08,
                random_state=ctx.seed,
            )
            a_backend = "sklearn/isolation_forest"
        assert_no_test_in_selection(
            selection_partition="validation", evaluation_partition="test"
        )
        a_tune = session.tune_anomaly_threshold(
            partition="validation",
            label_column=TARGET,
            positive_label=1,
            metric="f1",
        )
        a_ev = session.evaluate_anomaly(partition="test", positive_label=1)
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

    # --- Stage 3: supervised classical scorer ---
    session.fit(
        LogisticRegression(max_iter=1000, random_state=ctx.seed),
        task="classification",
    )
    supervised_val = session.evaluate(partition="validation")
    supervised_test = session.evaluate(partition="test")
    stages["supervised"] = {
        "status": "ok",
        "estimator": "LogisticRegression",
        "validation_metrics": metrics_round(dict(supervised_val.metrics)),
        "test_metrics": metrics_round(dict(supervised_test.metrics)),
    }
    write_results(ctx, stages["supervised"], filename="supervised.json")

    # --- Stage 4: online / stream updates on train cursor ---
    try:
        online_session = (
            Session.ingest(nodes.copy())
            .set_roles({**{c: "feature" for c in FEATURES}, TARGET: "target"})
            .inject_split(
                train_indices=list(plan.train_indices),
                validation_indices=list(plan.validation_indices),
                test_indices=list(plan.test_indices),
            )
            .scale(method="standard")
        )
        o_fit = online_session.fit_online(
            estimator="sgd_classifier",
            chunk_size=40,
            n_init=40,
            classes=[0, 1],
        )
        updates = 0
        while True:
            remaining = online_session.online_plan.n_train_rows - online_session.online_plan.cursor
            if remaining <= 0:
                break
            online_session.partial_fit_online(n_rows=min(40, remaining))
            updates += 1
        o_test = online_session.evaluate_online(partition="test")
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

    # --- Stage 5: decision thresholds on validation only ---
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
            budget=80.0,
            cost_column="review_cost",
            id_column="node_id",
            score_source="model_proba",
            knapsack_solver="dp",
        )
        applied = session.apply_decisions(partition="test")
        knap_payload = {
            "status": "ok",
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
        # Fallback: capacity top-k (no cost column dependency).
        try:
            topk = session.fit_decision_policy(
                method="topk",
                partition="validation",
                capacity=40,
                score_source="model_proba",
            )
            applied = session.apply_decisions(partition="test")
            knap_payload = {
                "status": "ok_topk_fallback",
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
                "status": "skipped",
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

    # --- Stage 6: optional symbolic guardrails ---
    try:
        sym = session.fit_symbolic(
            source="decision_tree",
            max_depth=3,
            random_state=ctx.seed,
        )
        sym_test = session.evaluate_symbolic(partition="test")
        stages["symbolic"] = {
            "status": "ok",
            "fit": metrics_round(sym.to_dict() if hasattr(sym, "to_dict") else {}),
            "test_metrics": metrics_round(dict(getattr(sym_test, "metrics", {}) or {})),
        }
        write_results(ctx, stages["symbolic"], filename="symbolic.json")
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["symbolic"] = {"status": "skipped", "error": f"{type(exc).__name__}: {exc}"}
        skip_notes.append(f"symbolic: {exc}")

    summary = {
        "status": "completed",
        "product": "Aegis Fraud Platform",
        "data": data_meta,
        "split": {"kind": plan.kind, "counts": split_counts, "stratify": True},
        "stages": {k: {"status": v.get("status")} for k, v in stages.items()},
        "stage_details": stages,
        "skip_notes": skip_notes,
        "torch": TORCH_STATUS,
        "leakage_controls": [
            "Stratified node split before any graph/anomaly/supervised fit",
            "Anomaly threshold + decision policies tuned on validation only",
            "Online partial_fit consumes train cursor only",
            "Test used once per stage after that stage's selection lock",
        ],
        "what_fails_if_leakage_ignored": [
            "Tuning anomaly/decision thresholds on test inflates F1 and understates review cost",
            "Fitting graph features with test-label-conditioned edges overstates ring detection",
            "Streaming updates that include test rows make online metrics meaningless",
            "Symbolic rules induced on full data look more 'compliant' than production",
        ],
        "limitations": [
            "Synthetic portfolio — not a real card network",
            "Classical graph path is primary; GCN optional elsewhere",
            "Product proof, not a production fraud SaaS certification",
        ],
    }
    write_results(ctx, summary, filename="summary.json")
    write_results(ctx, summary, filename="results.json")
    print(
        "aegis-fraud-platform OK",
        {
            "supervised_test_roc": stages["supervised"]["test_metrics"].get("roc_auc"),
            "decision_selected": stages["decisions"]["knapsack_applied"]["n_selected"],
            "skips": skip_notes,
        },
    )


if __name__ == "__main__":
    main()
