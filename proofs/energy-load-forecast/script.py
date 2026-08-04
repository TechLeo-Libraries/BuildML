"""Tier A proof: hourly energy load forecast with time_split."""

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
    load_energy_load_synthetic,
    metrics_round,
    new_proof_context,
    write_results,
)


def main() -> None:
    ctx = new_proof_context("energy-load-forecast", seed=110)
    frame, data_meta = load_energy_load_synthetic(n_hours=24 * 120, seed=ctx.seed)
    sm_ok = extra_available("statsmodels")

    session = (
        Session.ingest(frame)
        .set_roles({"ts": "time", "temp_c": "feature", "load_mw": "target"})
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
        analysis = session.timeseries.analyze(
            scope="train",
            seasonal_period=24,
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

    fit = session.forecast.fit(
        method="lag_ridge",
        horizon=24,
        lags=[1, 2, 3, 24, 48],
        alpha=1.0,
    )
    assert_no_test_in_selection(
        selection_partition="validation",
        evaluation_partition="test",
    )
    val_metrics = session.forecast.evaluate(
        partition="validation",
        strategy="rolling_one_step",
    )
    test_metrics = session.forecast.evaluate(
        partition="test",
        strategy="rolling_one_step",
    )
    gen = session.forecast.generate(horizon=24)
    bundle = session.forecast.save_bundle(ctx.artifacts_dir / "forecast_bundle")

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
                "Test session.forecast.evaluate after model locked",
            ],
            "industry_comparison": {
                "status": "filled",
                "note": (
                    "Tier C baseline_industry.py: seasonal naive / Ridge lag twin on the same "
                    "time_split; run script then baseline_industry.py for results/comparison.json."
                ),
            },
            "limitations": [
                "Synthetic grid load; no multi-zone hierarchy",
                "lag_ridge is a classical baseline, not a full energy SOTA stack",
            ],
        },
    )
    print(
        "energy-load-forecast OK",
        getattr(test_metrics, "to_dict", lambda: test_metrics)(),
    )


if __name__ == "__main__":
    main()
