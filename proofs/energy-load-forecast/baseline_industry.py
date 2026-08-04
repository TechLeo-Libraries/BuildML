"""Tier C: seasonal naive + Ridge lag twin for energy-load-forecast."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from buildml import Session
from proofs._lib import (
    extract_buildml_test_metrics,
    load_buildml_results,
    load_energy_load_synthetic,
    metrics_round,
    new_proof_context,
    write_comparison,
)


def _seasonal_naive(y_train: np.ndarray, y_test: np.ndarray, period: int = 24) -> np.ndarray:
    hist = list(y_train.astype(float))
    preds = []
    for actual in y_test:
        if len(hist) >= period:
            preds.append(hist[-period])
        else:
            preds.append(hist[-1] if hist else 0.0)
        hist.append(float(actual))
    return np.asarray(preds, dtype=float)


def _ridge_lag_preds(
    y_hist: np.ndarray,
    y_test: np.ndarray,
    lags: list[int],
    alpha: float = 1.0,
) -> np.ndarray:
    """Fit Ridge on lag features from history; rolling one-step on test."""
    max_lag = max(lags)
    rows, targets = [], []
    for t in range(max_lag, len(y_hist)):
        rows.append([y_hist[t - lag] for lag in lags])
        targets.append(y_hist[t])
    if len(rows) < 10:
        return _seasonal_naive(y_hist, y_test, period=24)
    model = Ridge(alpha=alpha)
    model.fit(np.asarray(rows), np.asarray(targets))
    hist = list(y_hist.astype(float))
    preds = []
    for actual in y_test:
        x = np.asarray([[hist[-lag] for lag in lags]], dtype=float)
        preds.append(float(model.predict(x)[0]))
        hist.append(float(actual))
    return np.asarray(preds, dtype=float)


def main() -> None:
    ctx = new_proof_context("energy-load-forecast", seed=110)
    frame, _ = load_energy_load_synthetic(n_hours=24 * 120, seed=ctx.seed)
    session = (
        Session.ingest(frame.copy())
        .set_roles({"ts": "time", "temp_c": "feature", "load_mw": "target"})
        .time_split(test_size=0.15, validation_size=0.15)
    )
    plan = session.split_plan
    assert plan is not None
    train_idx = list(plan.train_indices)
    val_idx = list(plan.validation_indices)
    test_idx = list(plan.test_indices)

    y_train = frame.loc[train_idx, "load_mw"].to_numpy(dtype=float)
    y_val = frame.loc[val_idx, "load_mw"].to_numpy(dtype=float)
    y_test = frame.loc[test_idx, "load_mw"].to_numpy(dtype=float)

    # Select between seasonal naive and Ridge lag using validation MAE.
    sn_val = _seasonal_naive(y_train, y_val, period=24)
    ridge_val = _ridge_lag_preds(y_train, y_val, lags=[1, 2, 3, 24, 48])
    sn_mae = float(mean_absolute_error(y_val, sn_val))
    ridge_mae = float(mean_absolute_error(y_val, ridge_val))
    use_ridge = ridge_mae <= sn_mae
    hist = np.concatenate([y_train, y_val])
    if use_ridge:
        preds = _ridge_lag_preds(hist, y_test, lags=[1, 2, 3, 24, 48])
        industry_backend = "sklearn.Ridge lag features (selected on val)"
    else:
        preds = _seasonal_naive(hist, y_test, period=24)
        industry_backend = "seasonal_naive(period=24)"

    industry_metrics = metrics_round(
        {
            "mae": float(mean_absolute_error(y_test, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
            "mape": float(
                np.mean(np.abs((y_test - preds) / np.clip(np.abs(y_test), 1e-6, None))) * 100
            ),
        }
    )

    bml_raw = load_buildml_results(ctx.project_dir)
    bml_metrics = extract_buildml_test_metrics(bml_raw, prefer=("test_metrics",))
    if "mae" not in bml_metrics and "metrics" in bml_raw.get("test_metrics", {}):
        nested = bml_raw["test_metrics"].get("metrics", {})
        if isinstance(nested, dict):
            bml_metrics = metrics_round(
                {k: nested[k] for k in ("mae", "rmse", "mape") if k in nested}
            )
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
            "validation_mae": {"seasonal_naive": sn_mae, "ridge_lag": ridge_mae},
            "test_metrics": industry_metrics,
            "leakage_controls": [
                "time_split chronological train→validation→test",
                "Method selected on validation MAE only",
                "Rolling one-step evaluation on test",
            ],
        },
        split_counts={
            "train": len(train_idx),
            "validation": len(val_idx),
            "test": len(test_idx),
        },
        delta_keys=("mae", "rmse", "mape"),
    )
    print("energy-load-forecast Tier C OK", industry_metrics)


if __name__ == "__main__":
    main()
