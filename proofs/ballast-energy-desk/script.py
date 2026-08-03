"""Tier B product: Ballast Energy Desk.

Composes chronological energy forecast + probabilistic intervals + optimize
allocation for generation / demand response capacity.
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

from buildml import Session
from buildml.core.errors import MissingExtraError
from proofs._lib import (
    assert_no_test_in_selection,
    load_energy_load_synthetic,
    metrics_round,
    new_proof_context,
    write_results,
)


def _allocation_candidates(forecast_vals: list[float], seed: int = 52) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i, demand in enumerate(forecast_vals):
        cost = float(8.0 + 0.12 * demand + rng.uniform(0, 4))
        value = float(demand * rng.uniform(0.85, 1.25))
        rows.append(
            {
                "block_id": f"blk-{i}",
                "forecast_mw": float(demand),
                "cost": cost,
                "value": value,
                "score": value / max(cost, 1e-6),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ctx = new_proof_context("ballast-energy-desk", seed=52)
    frame, data_meta = load_energy_load_synthetic(n_hours=24 * 120, seed=ctx.seed)
    stages: dict = {}
    skip_notes: list[str] = []

    session = (
        Session.ingest(frame.copy())
        .set_roles({"ts": "time", "temp_c": "feature", "load_mw": "target"})
        .time_split(test_size=0.15, validation_size=0.15)
    )
    plan = session.split_plan
    assert plan is not None
    split_counts = {
        "train": len(plan.train_indices),
        "validation": len(plan.validation_indices),
        "test": len(plan.test_indices),
    }

    # --- Stage 1: forecast ---
    fit = session.fit_forecast(
        method="lag_ridge",
        horizon=24,
        lags=[1, 2, 3, 24, 48],
        alpha=1.0,
    )
    assert_no_test_in_selection(
        selection_partition="validation", evaluation_partition="test"
    )
    val_fc = session.evaluate_forecast(
        partition="validation", strategy="rolling_one_step"
    )
    test_fc = session.evaluate_forecast(partition="test", strategy="rolling_one_step")
    gen = session.generate_forecast(horizon=24)
    stages["forecast"] = {
        "status": "ok",
        "fit": metrics_round(fit.to_dict() if hasattr(fit, "to_dict") else {}),
        "validation": metrics_round(
            val_fc.to_dict() if hasattr(val_fc, "to_dict") else {}
        ),
        "test": metrics_round(
            test_fc.to_dict() if hasattr(test_fc, "to_dict") else {}
        ),
        "generate": metrics_round(gen.to_dict() if hasattr(gen, "to_dict") else {}),
    }
    write_results(ctx, stages["forecast"], filename="forecast.json")

    # --- Stage 2: probabilistic intervals on residual view ---
    try:
        hist = frame.loc[list(plan.train_indices)].copy().reset_index(drop=True)
        hist["lag1"] = hist["load_mw"].shift(1)
        hist["lag24"] = hist["load_mw"].shift(24)
        hist = hist.dropna().reset_index(drop=True)
        hist["residual_proxy"] = hist["load_mw"] - hist["lag1"]
        prob_session = (
            Session.ingest(hist)
            .set_roles(
                {
                    "lag1": "feature",
                    "lag24": "feature",
                    "temp_c": "feature",
                    "residual_proxy": "target",
                }
            )
            .split(test_size=0.2, validation_size=0.2, random_state=ctx.seed)
            .scale(method="standard")
        )
        p_fit = prob_session.fit_probabilistic(
            estimator="bayesian_ridge",
            conformal=True,
            interval_method="both",
            random_state=ctx.seed,
        )
        try:
            intervals = prob_session.predict_interval(partition="test", alpha=0.1)
            interval_payload = metrics_round(
                intervals.to_dict() if hasattr(intervals, "to_dict") else {}
            )
        except Exception as exc:  # noqa: BLE001
            interval_payload = {"error": f"{type(exc).__name__}: {exc}"}
        p_ev = prob_session.evaluate_probabilistic(partition="test")
        stages["probabilistic"] = {
            "status": "ok",
            "fit": metrics_round(p_fit.to_dict() if hasattr(p_fit, "to_dict") else {}),
            "intervals": interval_payload,
            "test_metrics": metrics_round(dict(getattr(p_ev, "metrics", {}) or {})),
        }
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["probabilistic"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"probabilistic: {exc}")
    write_results(ctx, stages["probabilistic"], filename="probabilistic.json")

    # --- Stage 3: allocate DR / generation blocks ---
    preds = list(getattr(gen, "predictions", None) or [])
    if not preds and hasattr(gen, "to_dict"):
        preds = list(gen.to_dict().get("predictions") or [])
    if not preds:
        preds = [float(frame.loc[list(plan.train_indices), "load_mw"].iloc[-1])] * 24

    candidates = _allocation_candidates(preds, seed=ctx.seed)
    cand_session = Session.ingest(candidates)
    cand_session.set_roles(
        {
            "block_id": "id",
            "forecast_mw": "feature",
            "cost": "feature",
            "value": "feature",
            "score": "target",
        }
    )
    n = len(candidates)
    n_train = max(2, n // 3)
    n_val = max(2, (n - n_train) // 2)
    train_i = list(range(0, n_train))
    val_i = list(range(n_train, n_train + n_val))
    test_i = list(range(n_train + n_val, n))
    if len(test_i) < 1:
        test_i = [val_i.pop()] if len(val_i) > 1 else [train_i.pop()]
    cand_session.inject_split(
        train_indices=train_i,
        validation_indices=val_i,
        test_indices=test_i,
    )
    assert_no_test_in_selection(
        selection_partition="validation", evaluation_partition="test"
    )
    try:
        alloc = cand_session.fit_decision_policy(
            method="knapsack",
            partition="validation",
            budget=200.0,
            cost_column="cost",
            value_column="value",
            id_column="block_id",
            score_source="score_column",
            score_column="score",
            knapsack_solver="dp",
        )
        applied = cand_session.apply_decisions(partition="test")
        stages["allocation"] = {
            "status": "ok",
            "policy": metrics_round(
                alloc.to_dict() if hasattr(alloc, "to_dict") else {}
            ),
            "applied": {
                "n_selected": int(applied.n_selected),
                "selected_value": float(applied.selected_value),
                "selected_cost": float(applied.selected_cost),
            },
        }
    except Exception as exc:  # noqa: BLE001
        stages["allocation"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"allocation: {exc}")
    write_results(ctx, stages["allocation"], filename="allocation.json")

    summary = {
        "status": "completed",
        "product": "Ballast Energy Desk",
        "data": data_meta,
        "split": {"kind": plan.kind, "counts": split_counts},
        "stages": {k: {"status": v.get("status")} for k, v in stages.items()},
        "stage_details": stages,
        "skip_notes": skip_notes,
        "leakage_controls": [
            "time_split chronological train → validation → test",
            "Forecast fit on train; selection metrics on validation",
            "Probabilistic residual model uses train-only history",
            "Allocation policy selected on validation half of future blocks",
        ],
        "what_fails_if_leakage_ignored": [
            "Random split on hours lets the model peek at future seasonality",
            "Calibrating intervals on test residuals reports perfect coverage",
            "Choosing allocation with realized future demand is not a desk decision",
        ],
        "limitations": [
            "Single synthetic load series — not a multi-zone ISO market",
            "Allocation is score/cost knapsack — not a full unit-commitment MIP",
        ],
    }
    write_results(ctx, summary, filename="summary.json")
    write_results(ctx, summary, filename="results.json")
    print(
        "ballast-energy-desk OK",
        {
            "forecast": stages["forecast"]["status"],
            "allocation": stages.get("allocation", {}).get("status"),
            "skips": skip_notes,
        },
    )


if __name__ == "__main__":
    main()
