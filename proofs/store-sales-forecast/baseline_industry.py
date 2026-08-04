"""Tier C: seasonal naive + optional SARIMAX twin for store-sales-forecast."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from buildml import Session
from proofs._lib import (
    extra_available,
    extract_buildml_test_metrics,
    load_buildml_results,
    load_store_sales_synthetic,
    metrics_round,
    new_proof_context,
    write_comparison,
)


def _seasonal_naive(y_train: np.ndarray, y_test: np.ndarray, period: int = 7) -> np.ndarray:
    """One-step seasonal naive over the test window using expanding history."""
    hist = list(y_train.astype(float))
    preds = []
    for actual in y_test:
        if len(hist) >= period:
            preds.append(hist[-period])
        else:
            preds.append(hist[-1] if hist else 0.0)
        hist.append(float(actual))
    return np.asarray(preds, dtype=float)


def main() -> None:
    ctx = new_proof_context("store-sales-forecast", seed=3)
    frame, _ = load_store_sales_synthetic(n_days=730, seed=ctx.seed)
    session = (
        Session.ingest(frame.copy())
        .set_roles({"date": "time", "promo": "feature", "sales": "target"})
        .time_split(test_size=0.15, validation_size=0.15)
    )
    plan = session.split_plan
    assert plan is not None
    train_idx = list(plan.train_indices)
    val_idx = list(plan.validation_indices)
    test_idx = list(plan.test_indices)

    y_train = frame.loc[train_idx, "sales"].to_numpy(dtype=float)
    y_test = frame.loc[test_idx, "sales"].to_numpy(dtype=float)
    # Selection uses validation only for method choice disclosure.
    y_val = frame.loc[val_idx, "sales"].to_numpy(dtype=float)
    sn_val = _seasonal_naive(y_train, y_val, period=7)
    sn_val_mae = float(mean_absolute_error(y_val, sn_val))

    preds = _seasonal_naive(np.concatenate([y_train, y_val]), y_test, period=7)
    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mape = float(np.mean(np.abs((y_test - preds) / np.clip(np.abs(y_test), 1e-6, None))) * 100)

    industry_backend = "seasonal_naive(period=7)"
    industry_metrics = {"mae": mae, "rmse": rmse, "mape": mape}
    sm_ok = extra_available("statsmodels")
    sarimax_note = None
    if sm_ok:
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX

            # Fit on train only; evaluate one-step rolling on test (honest).
            endog_train = np.concatenate([y_train, y_val])
            model = SARIMAX(
                endog_train,
                order=(1, 0, 0),
                seasonal_order=(1, 0, 0, 7),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False)
            # Append test recursively with true past (rolling one-step).
            hist = list(endog_train.astype(float))
            s_preds = []
            for actual in y_test:
                mod = SARIMAX(
                    np.asarray(hist),
                    order=(1, 0, 0),
                    seasonal_order=(1, 0, 0, 7),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                # Use previously fitted params for speed/stability.
                res = mod.filter(fitted.params)
                s_preds.append(float(res.forecast(1).iloc[0]))
                hist.append(float(actual))
            s_preds_a = np.asarray(s_preds)
            industry_backend = "statsmodels.SARIMAX(1,0,0)x(1,0,0,7) rolling"
            industry_metrics = {
                "mae": float(mean_absolute_error(y_test, s_preds_a)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, s_preds_a))),
                "mape": float(
                    np.mean(np.abs((y_test - s_preds_a) / np.clip(np.abs(y_test), 1e-6, None)))
                    * 100
                ),
            }
            sarimax_note = f"validation_seasonal_naive_mae={sn_val_mae:.4f}"
        except Exception as exc:  # noqa: BLE001
            sarimax_note = f"SARIMAX failed ({type(exc).__name__}: {exc}); seasonal_naive used"

    industry_metrics = metrics_round(industry_metrics)
    bml_raw = load_buildml_results(ctx.project_dir)
    bml_metrics = extract_buildml_test_metrics(bml_raw, prefer=("test_metrics",))
    # Normalize BuildML forecast metric keys if nested.
    if "mae" not in bml_metrics and "metrics" in bml_raw.get("test_metrics", {}):
        nested = bml_raw["test_metrics"].get("metrics", {})
        if isinstance(nested, dict):
            bml_metrics = metrics_round(
                {k: nested[k] for k in ("mae", "rmse", "mape") if k in nested}
            )
    # Also try common BuildML forecast result shapes.
    for key in ("mae", "rmse", "mape", "smape"):
        tm = bml_raw.get("test_metrics", {})
        if key not in bml_metrics and isinstance(tm, dict) and key in tm:
            bml_metrics[key] = tm[key]
    bml_metrics = metrics_round(bml_metrics)

    write_comparison(
        ctx,
        buildml={
            "backend": "buildml.session.forecast.fit",
            "method": bml_raw.get("fit", {}).get("method", "lag_ridge"),
            "test_metrics": bml_metrics,
        },
        industry={
            "backend": industry_backend,
            "test_metrics": industry_metrics,
            "selection_note": sarimax_note,
            "leakage_controls": [
                "time_split chronological train→validation→test",
                "Industry model fit uses only pre-test history",
                "Rolling one-step evaluation mirrors BuildML strategy",
            ],
        },
        split_counts={
            "train": len(train_idx),
            "validation": len(val_idx),
            "test": len(test_idx),
        },
        delta_keys=("mae", "rmse", "mape"),
    )
    print("store-sales-forecast Tier C OK", industry_metrics)


if __name__ == "__main__":
    main()
