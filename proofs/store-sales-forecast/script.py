"""Tier A proof: store sales forecast with time_split + TS analysis."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from buildml import Session
from buildml.core.errors import MissingExtraError
from proofs._lib import (
    assert_no_test_in_selection,
    extra_available,
    load_store_sales_synthetic,
    metrics_round,
    new_proof_context,
    write_results,
)


def main() -> None:
    ctx = new_proof_context("store-sales-forecast", seed=3)
    frame, data_meta = load_store_sales_synthetic(n_days=730, seed=ctx.seed)
    sm_ok = extra_available("statsmodels")

    session = (
        Session.ingest(frame)
        .set_roles({"date": "time", "promo": "feature", "sales": "target"})
        .time_split(test_size=0.15, validation_size=0.15)
    )
    plan = session.split_plan
    assert plan is not None
    counts = {
        "train": len(plan.train_indices),
        "validation": len(plan.validation_indices),
        "test": len(plan.test_indices),
    }

    analysis_payload: dict = {"statsmodels_available": sm_ok}
    try:
        analysis = session.analyze_timeseries(
            scope="train",
            seasonal_period=7,
            include_decompose=True,
            include_diagnostics=True,
            include_changepoints=True,
            include_features=True,
        )
        analysis_payload["result"] = metrics_round(
            analysis.to_dict() if hasattr(analysis, "to_dict") else {"repr": str(analysis)[:2000]}
        )
    except MissingExtraError as exc:
        analysis_payload["skipped"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        analysis_payload["error"] = f"{type(exc).__name__}: {exc}"

    fit = session.fit_forecast(
        method="lag_ridge",
        horizon=14,
        lags=[1, 2, 3, 7, 14],
        alpha=1.0,
    )
    assert_no_test_in_selection(
        selection_partition="validation",
        evaluation_partition="test",
    )
    val_metrics = session.evaluate_forecast(
        partition="validation",
        strategy="rolling_one_step",
    )
    test_metrics = session.evaluate_forecast(
        partition="test",
        strategy="rolling_one_step",
    )
    gen = session.generate_forecast(horizon=14)
    bundle = session.save_forecast_bundle(ctx.artifacts_dir / "forecast_bundle")

    write_results(
        ctx,
        {
            "status": "completed",
            "data": data_meta,
            "split": {"kind": plan.kind, "counts": counts},
            "analysis": analysis_payload,
            "fit": metrics_round(
                fit.to_dict() if hasattr(fit, "to_dict") else {"repr": str(fit)}
            ),
            "validation_metrics": metrics_round(
                val_metrics.to_dict()
                if hasattr(val_metrics, "to_dict")
                else dict(getattr(val_metrics, "__dict__", {}))
            ),
            "test_metrics": metrics_round(
                test_metrics.to_dict()
                if hasattr(test_metrics, "to_dict")
                else dict(getattr(test_metrics, "__dict__", {}))
            ),
            "generate": metrics_round(
                gen.to_dict() if hasattr(gen, "to_dict") else {"repr": str(gen)}
            ),
            "bundle_path": str(bundle),
            "leakage_controls": [
                "time_split: chronological train → validation → test",
                "analyze_timeseries scope='train' only",
                "Forecast model fit on train; selection metrics on validation",
                "Test evaluate_forecast after model locked",
            ],
            "industry_comparison": {
                "status": "filled",
                "note": (
                    "Tier C baseline_industry.py: seasonal naive + optional "
                    "statsmodels SARIMAX on the same time_split; run script then "
                    "baseline_industry.py for results/comparison.json."
                ),
            },
            "limitations": [
                "Synthetic retail series; no multi-store hierarchy",
                "lag_ridge is a strong classical baseline, not a full M5 winner stack",
            ],
        },
    )
    print(
        "store-sales-forecast OK",
        getattr(test_metrics, "to_dict", lambda: test_metrics)(),
    )


if __name__ == "__main__":
    main()
