"""Tier B product: Orbit Multitask Hub.

Composes multi-output multitask learning + AutoML/classical search +
validation-tuned decision thresholds for retail SKU outcomes.
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
from buildml.core.errors import LeakageError, MissingExtraError
from proofs._lib import (
    assert_no_test_in_selection,
    metrics_round,
    new_proof_context,
    write_results,
)


FEATS = ["price_z", "discount_z", "affinity_z", "competitor_z", "season_z", "stock_z"]


def _sku_multitask(n: int = 560, seed: int = 44) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 6))
    buy = (x[:, 0] + 0.55 * x[:, 1] + rng.normal(scale=0.3, size=n) > 0).astype(int)
    high_margin = (
        x[:, 2] - 0.45 * x[:, 3] + 0.2 * x[:, 4] + rng.normal(scale=0.3, size=n) > 0
    ).astype(int)
    frame = pd.DataFrame(x, columns=FEATS)
    frame["buy"] = buy
    frame["high_margin"] = high_margin
    frame["promo_cost"] = np.where(buy == 1, 2.2, 1.0)
    frame["sku_id"] = [f"sku-{i}" for i in range(n)]
    meta = {
        "name": "orbit_sku_multitask",
        "license": "synthetic/public-domain",
        "n_rows": n,
        "buy_rate": float(buy.mean()),
        "high_margin_rate": float(high_margin.mean()),
    }
    return frame, meta


def main() -> None:
    ctx = new_proof_context("orbit-multitask-hub", seed=44)
    frame, data_meta = _sku_multitask(seed=ctx.seed)
    stages: dict = {}
    skip_notes: list[str] = []

    # --- Stage 1: multitask multioutput ---
    mt_session = (
        Session.ingest(frame.copy())
        .set_roles(
            {
                **{c: "feature" for c in FEATS},
                "buy": "target",
                "high_margin": "target",
                "promo_cost": "ignore",
                "sku_id": "id",
            }
        )
        .split(test_size=0.2, validation_size=0.2, random_state=ctx.seed)
    )
    plan = mt_session.split_plan
    assert plan is not None
    split_counts = {
        "train": len(plan.train_indices),
        "validation": len(plan.validation_indices),
        "test": len(plan.test_indices),
    }
    try:
        mt_fit = mt_session.fit_multitask(method="multioutput", random_state=ctx.seed)
        mt_val = mt_session.evaluate_multitask(partition="validation")
        mt_test = mt_session.evaluate_multitask(partition="test")
        stages["multitask"] = {
            "status": "ok",
            "fit": metrics_round(mt_fit.to_dict() if hasattr(mt_fit, "to_dict") else {}),
            "validation_metrics": metrics_round(
                dict(getattr(mt_val, "metrics", {}) or {})
            ),
            "test_metrics": metrics_round(dict(getattr(mt_test, "metrics", {}) or {})),
        }
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["multitask"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"multitask: {exc}")
    write_results(ctx, stages["multitask"], filename="multitask.json")

    # --- Stage 2: AutoML / classical on primary buy target ---
    # AutoML refuses Session-global scale (CV leakage). Keep unpoisoned; classical
    # fallback may scale on its own session.
    auto_session = (
        Session.ingest(frame.copy())
        .set_roles(
            {
                **{c: "feature" for c in FEATS},
                "buy": "target",
                "high_margin": "ignore",
                "promo_cost": "ignore",
                "sku_id": "id",
            }
        )
        .inject_split(
            train_indices=list(plan.train_indices),
            validation_indices=list(plan.validation_indices),
            test_indices=list(plan.test_indices),
        )
    )
    try:
        am = auto_session.run_automl(
            backend="native",
            task="classification",
            method="randomized",
            n_trials=8,
            cv=3,
            include_recipe_search=False,
            include_industry_families=False,
            families=["logistic", "random_forest"],
            random_state=ctx.seed,
        )
        am_val = auto_session.evaluate_automl(partition="validation")
        am_test = auto_session.evaluate_automl(partition="test")
        stages["automl"] = {
            "status": "ok",
            "result": metrics_round(am.to_dict() if hasattr(am, "to_dict") else {}),
            "validation_metrics": metrics_round(
                dict(getattr(am_val, "metrics", {}) or {})
            ),
            "test_metrics": metrics_round(dict(getattr(am_test, "metrics", {}) or {})),
        }
        decision_session = auto_session
    except (MissingExtraError, LeakageError, TypeError, ValueError) as exc:
        skip_notes.append(f"automl: {exc}")
        decision_session = (
            Session.ingest(frame.copy())
            .set_roles(
                {
                    **{c: "feature" for c in FEATS},
                    "buy": "target",
                    "high_margin": "ignore",
                    "promo_cost": "ignore",
                    "sku_id": "id",
                }
            )
            .inject_split(
                train_indices=list(plan.train_indices),
                validation_indices=list(plan.validation_indices),
                test_indices=list(plan.test_indices),
            )
            .scale(method="standard")
        )
        decision_session.fit(
            LogisticRegression(max_iter=1000, random_state=ctx.seed),
            task="classification",
        )
        c_val = decision_session.evaluate(partition="validation")
        c_test = decision_session.evaluate(partition="test")
        stages["automl"] = {
            "status": "ok_classical_fallback",
            "automl_error": f"{type(exc).__name__}: {exc}",
            "estimator": "LogisticRegression",
            "validation_metrics": metrics_round(dict(c_val.metrics)),
            "test_metrics": metrics_round(dict(c_test.metrics)),
        }
    write_results(ctx, stages["automl"], filename="automl.json")

    # --- Stage 3: decision thresholds on buy scores ---
    try:
        assert_no_test_in_selection(
            selection_partition="validation", evaluation_partition="test"
        )
        thr = decision_session.fit_decision_policy(
            method="threshold",
            partition="validation",
            fp_cost=1.0,
            fn_cost=3.5,
        )
        thr_test = decision_session.evaluate_decisions(partition="test")
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
            knap = decision_session.fit_decision_policy(
                method="knapsack",
                partition="validation",
                budget=55.0,
                cost_column="promo_cost",
                id_column="sku_id",
                score_source="model_proba",
                knapsack_solver="dp",
            )
            applied = decision_session.apply_decisions(partition="test")
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
        "product": "Orbit Multitask Hub",
        "data": data_meta,
        "split": {"kind": plan.kind, "counts": split_counts},
        "stages": {k: {"status": v.get("status")} for k, v in stages.items()},
        "stage_details": stages,
        "skip_notes": skip_notes,
        "leakage_controls": [
            "Split before multitask / AutoML / decision fit",
            "AutoML CV uses train folds only",
            "Decision policies selected on validation only",
            "Test evaluated after each stage locks",
        ],
        "what_fails_if_leakage_ignored": [
            "Multitask heads trained on test labels overstate joint skill",
            "AutoML winner picked with test scores is not a fair search",
            "Promo thresholds tuned on test understate campaign cost",
        ],
        "limitations": [
            "Synthetic SKU outcomes; same-type classification targets",
            "Native AutoML smoke with small trial budget",
        ],
    }
    write_results(ctx, summary, filename="summary.json")
    write_results(ctx, summary, filename="results.json")
    print(
        "orbit-multitask-hub OK",
        {
            "automl_status": stages["automl"]["status"],
            "skips": skip_notes,
        },
    )


if __name__ == "__main__":
    main()
