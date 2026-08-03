"""Tier C: sklearn Ridge twin for claim-severity-regression."""

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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from buildml import Session
from proofs._lib import (
    extract_buildml_test_metrics,
    load_buildml_results,
    load_claim_severity_synthetic,
    metrics_round,
    new_proof_context,
    write_comparison,
)

FEATURES = ["vehicle_age", "driver_age", "prior_claims", "urban", "deductible"]
TARGET = "severity"


def main() -> None:
    ctx = new_proof_context("claim-severity-regression", seed=102)
    frame, _ = load_claim_severity_synthetic(n=1100, seed=ctx.seed)
    session = (
        Session.ingest(frame.copy())
        .set_roles({**{c: "feature" for c in FEATURES}, TARGET: "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=ctx.seed)
    )
    plan = session.split_plan
    assert plan is not None
    train_idx = list(plan.train_indices)
    val_idx = list(plan.validation_indices)
    test_idx = list(plan.test_indices)

    x_train = frame.loc[train_idx, FEATURES]
    y_train = frame.loc[train_idx, TARGET]
    x_test = frame.loc[test_idx, FEATURES]
    y_test = frame.loc[test_idx, TARGET]

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0, random_state=ctx.seed)),
        ]
    )
    pipe.fit(x_train, y_train)
    pred = pipe.predict(x_test)
    industry_metrics = metrics_round(
        {
            "r2": float(r2_score(y_test, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "mae": float(mean_absolute_error(y_test, pred)),
        }
    )

    bml_raw = load_buildml_results(ctx.project_dir)
    bml_metrics = extract_buildml_test_metrics(
        bml_raw,
        prefer=("test_metrics",),
        keys=("r2", "rmse", "mae", "mse"),
    )
    if "rmse" not in bml_metrics and "mse" in bml_metrics:
        bml_metrics["rmse"] = float(np.sqrt(float(bml_metrics["mse"])))

    write_comparison(
        ctx,
        buildml={
            "backend": "buildml.Session",
            "estimator": bml_raw.get("estimator", "HistGradientBoostingRegressor"),
            "test_metrics": bml_metrics,
        },
        industry={
            "backend": "sklearn.Pipeline",
            "estimator": "Ridge",
            "test_metrics": industry_metrics,
            "leakage_controls": [
                "Scaler + Ridge fit on train indices only",
                "Test evaluated once after lock",
                "Same SplitPlan as BuildML Session",
            ],
        },
        split_counts={
            "train": len(train_idx),
            "validation": len(val_idx),
            "test": len(test_idx),
        },
        delta_keys=("r2", "rmse", "mae"),
    )
    print("claim-severity-regression Tier C OK", industry_metrics)


if __name__ == "__main__":
    main()
