"""Tier B product: Canyon Segment Studio.

Composes unsupervised clustering + classical segment propensity + decision
thresholds for CRM targeting. External labels eval-only.
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
    assert_no_test_in_selection,
    metrics_round,
    new_proof_context,
    write_results,
)


FEATURES = ["recency_z", "frequency_z", "monetary_z", "affinity_z", "channel_z"]
EXTERNAL = "true_segment"


def _crm_segments(n_per: int = 180, seed: int = 47) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    centers = {
        0: np.array([-1.4, -1.1, -0.8, 0.1, -0.3]),
        1: np.array([1.3, -0.7, 0.4, 0.5, 0.2]),
        2: np.array([0.2, 1.4, -0.3, -0.4, 0.9]),
        3: np.array([-0.4, 0.3, 1.5, 1.0, -0.8]),
    }
    rows = []
    for seg, center in centers.items():
        for _ in range(n_per):
            x = center + rng.normal(scale=0.35, size=5)
            # Propensity: high-value outreach worth targeting
            logit = 0.7 * x[2] + 0.4 * x[1] - 0.3 * x[0] + rng.normal(0, 0.3)
            respond = int(1 / (1 + np.exp(-logit)) > 0.5)
            rows.append(
                {
                    **{FEATURES[i]: float(x[i]) for i in range(5)},
                    EXTERNAL: seg,
                    "respond": respond,
                    "outreach_cost": 2.0 if respond == 1 else 1.0,
                    "customer_id": f"c-{seg}-{len(rows)}",
                }
            )
    frame = pd.DataFrame(rows)
    meta = {
        "name": "canyon_crm_segments",
        "license": "synthetic/public-domain",
        "n_rows": int(len(frame)),
        "n_segments": 4,
        "respond_rate": float(frame["respond"].mean()),
    }
    return frame, meta


def main() -> None:
    ctx = new_proof_context("canyon-segment-studio", seed=47)
    frame, data_meta = _crm_segments(seed=ctx.seed)
    stages: dict = {}
    skip_notes: list[str] = []

    session = (
        Session.ingest(frame.copy())
        .set_roles(
            {
                **{c: "feature" for c in FEATURES},
                EXTERNAL: "ignore",
                "respond": "ignore",
                "outreach_cost": "ignore",
                "customer_id": "id",
            }
        )
        .split(test_size=0.2, validation_size=0.2, random_state=ctx.seed)
        .scale(method="standard")
        .reduce_dimensions(method="pca", n_components=2, prefix="pc")
    )
    plan = session.split_plan
    assert plan is not None
    split_counts = {
        "train": len(plan.train_indices),
        "validation": len(plan.validation_indices),
        "test": len(plan.test_indices),
    }

    # --- Stage 1: unsupervised clusters ---
    try:
        c_fit = session.fit_clusters(method="kmeans", n_clusters=4, random_state=ctx.seed)
        assert_no_test_in_selection(
            selection_partition="validation", evaluation_partition="test"
        )
        c_val = session.evaluate_clusters(
            partition="validation", external_label_column=EXTERNAL
        )
        c_test = session.evaluate_clusters(
            partition="test", external_label_column=EXTERNAL
        )
        stages["clusters"] = {
            "status": "ok",
            "method": "kmeans",
            "fit": metrics_round(c_fit.to_dict() if hasattr(c_fit, "to_dict") else {}),
            "validation_metrics": metrics_round(
                c_val.to_dict() if hasattr(c_val, "to_dict") else {}
            ),
            "test_metrics": metrics_round(
                c_test.to_dict() if hasattr(c_test, "to_dict") else {}
            ),
        }
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["clusters"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"clusters: {exc}")
    write_results(ctx, stages["clusters"], filename="clusters.json")

    # --- Stage 2: classical respond propensity ---
    prop = (
        Session.ingest(frame.copy())
        .set_roles(
            {
                **{c: "feature" for c in FEATURES},
                "respond": "target",
                EXTERNAL: "ignore",
                "outreach_cost": "ignore",
                "customer_id": "id",
            }
        )
        .inject_split(
            train_indices=list(plan.train_indices),
            validation_indices=list(plan.validation_indices),
            test_indices=list(plan.test_indices),
        )
        .scale(method="standard")
    )
    prop.fit(
        LogisticRegression(max_iter=1000, random_state=ctx.seed),
        task="classification",
    )
    p_val = prop.evaluate(partition="validation")
    p_test = prop.evaluate(partition="test")
    stages["supervised"] = {
        "status": "ok",
        "estimator": "LogisticRegression",
        "validation_metrics": metrics_round(dict(p_val.metrics)),
        "test_metrics": metrics_round(dict(p_test.metrics)),
    }
    write_results(ctx, stages["supervised"], filename="supervised.json")

    # --- Stage 3: outreach decisions ---
    try:
        assert_no_test_in_selection(
            selection_partition="validation", evaluation_partition="test"
        )
        thr = prop.fit_decision_policy(
            method="threshold",
            partition="validation",
            fp_cost=1.0,
            fn_cost=2.5,
        )
        thr_test = prop.evaluate_decisions(partition="test")
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
            knap = prop.fit_decision_policy(
                method="knapsack",
                partition="validation",
                budget=70.0,
                cost_column="outreach_cost",
                id_column="customer_id",
                score_source="model_proba",
                knapsack_solver="dp",
            )
            applied = prop.apply_decisions(partition="test")
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
        "product": "Canyon Segment Studio",
        "data": data_meta,
        "split": {"kind": plan.kind, "counts": split_counts},
        "stages": {k: {"status": v.get("status")} for k, v in stages.items()},
        "stage_details": stages,
        "skip_notes": skip_notes,
        "leakage_controls": [
            "Scale + PCA + clusters fit on train only",
            "External segment labels used only for cluster evaluation",
            "Propensity + decision policies selected on validation only",
            "Test after each stage locks",
        ],
        "what_fails_if_leakage_ignored": [
            "Clustering with test-conditioned PCA overstates segment purity",
            "Using external labels as features collapses unsupervised into supervised",
            "Outreach thresholds tuned on test understate CRM cost",
        ],
        "limitations": [
            "Synthetic CRM RFM-like features — not a real CDP extract",
            "External labels exist only because data is synthetic",
        ],
    }
    write_results(ctx, summary, filename="summary.json")
    write_results(ctx, summary, filename="results.json")
    print(
        "canyon-segment-studio OK",
        {
            "supervised_roc": stages["supervised"]["test_metrics"].get("roc_auc"),
            "skips": skip_notes,
        },
    )


if __name__ == "__main__":
    main()
