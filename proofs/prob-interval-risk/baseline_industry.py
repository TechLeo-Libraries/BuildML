"""Tier C: BayesianRidge + residual quantile twin for prob-interval-risk."""

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
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from buildml import Session
from proofs._lib import (
    extract_buildml_test_metrics,
    load_buildml_results,
    metrics_round,
    new_proof_context,
    write_comparison,
)


def main() -> None:
    ctx = new_proof_context("prob-interval-risk", seed=0)
    rng = np.random.default_rng(ctx.seed)
    n = 400
    x = rng.normal(size=(n, 4))
    y = 2.0 * x[:, 0] - x[:, 1] + rng.normal(scale=0.5, size=n)
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(4)])
    frame["loss"] = y

    session = (
        Session.ingest(frame.copy())
        .set_roles({**{f"f{i}": "feature" for i in range(4)}, "loss": "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=ctx.seed)
    )
    plan = session.split_plan
    assert plan is not None
    train_idx = list(plan.train_indices)
    val_idx = list(plan.validation_indices)
    test_idx = list(plan.test_indices)
    feats = [f"f{i}" for i in range(4)]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(frame.loc[train_idx, feats])
    y_train = frame.loc[train_idx, "loss"].to_numpy()
    x_val = scaler.transform(frame.loc[val_idx, feats])
    y_val = frame.loc[val_idx, "loss"].to_numpy()
    x_test = scaler.transform(frame.loc[test_idx, feats])
    y_test = frame.loc[test_idx, "loss"].to_numpy()

    model = BayesianRidge()
    model.fit(x_train, y_train)
    # Conformal-style residual calibration on validation only.
    val_pred = model.predict(x_val)
    resid = np.abs(y_val - val_pred)
    q = float(np.quantile(resid, 0.9))
    test_pred, test_std = model.predict(x_test, return_std=True)
    lo, hi = test_pred - q, test_pred + q
    cover = float(np.mean((y_test >= lo) & (y_test <= hi)))

    industry_metrics = metrics_round(
        {
            "mae": float(mean_absolute_error(y_test, test_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, test_pred))),
            "r2": float(r2_score(y_test, test_pred)),
            "interval_coverage": cover,
            "mean_interval_width": float(np.mean(hi - lo)),
            "mean_predictive_std": float(np.mean(test_std)),
        }
    )

    bml_raw = load_buildml_results(ctx.project_dir)
    bml_metrics = extract_buildml_test_metrics(bml_raw, prefer=("test_metrics",))
    # Pull coverage from intervals block if present.
    intervals = bml_raw.get("intervals", {})
    if isinstance(intervals, dict):
        for src, dst in (
            ("coverage", "interval_coverage"),
            ("empirical_coverage", "interval_coverage"),
            ("mean_width", "mean_interval_width"),
        ):
            if src in intervals and dst not in bml_metrics:
                bml_metrics[dst] = intervals[src]
    bml_metrics = metrics_round(bml_metrics)

    write_comparison(
        ctx,
        buildml={
            "backend": "buildml.Session.fit_probabilistic",
            "test_metrics": bml_metrics,
        },
        industry={
            "backend": "sklearn.BayesianRidge + val residual quantile",
            "test_metrics": industry_metrics,
            "leakage_controls": [
                "Model fit on train only",
                "Interval half-width calibrated on validation residuals only",
                "Test evaluated once after calibration lock",
                "Same SplitPlan as BuildML Session",
            ],
        },
        split_counts={
            "train": len(train_idx),
            "validation": len(val_idx),
            "test": len(test_idx),
        },
        delta_keys=("mae", "rmse", "r2", "interval_coverage", "mean_interval_width"),
    )
    print("prob-interval-risk Tier C OK", industry_metrics)


if __name__ == "__main__":
    main()
