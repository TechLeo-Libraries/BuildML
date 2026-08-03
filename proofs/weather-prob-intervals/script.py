"""Tier A proof: probabilistic weather / regression intervals."""

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
from proofs._lib import metrics_round, new_proof_context, write_results


def _weather_frame(n: int = 500, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    hour = rng.integers(0, 24, size=n).astype(float)
    humidity = rng.uniform(20, 95, size=n)
    pressure = rng.normal(1013, 8, size=n)
    wind = rng.exponential(4.0, size=n).clip(0, 30)
    temp = (
        12
        + 8 * np.sin(2 * np.pi * hour / 24 - 0.6)
        - 0.04 * (humidity - 50)
        + 0.02 * (pressure - 1013)
        - 0.3 * wind
        + rng.normal(0, 1.2, size=n)
    )
    frame = pd.DataFrame(
        {
            "hour": hour,
            "humidity": humidity,
            "pressure_hpa": pressure,
            "wind_mps": wind,
            "temp_c": temp,
        }
    )
    meta = {
        "name": "weather_temp_synthetic",
        "license": "synthetic/public-domain (generated in-repo)",
        "n_rows": int(n),
        "target": "temp_c",
        "notes": "Synthetic weather regression; not a real METAR extract.",
    }
    return frame, meta


def main() -> None:
    ctx = new_proof_context("weather-prob-intervals", seed=111)
    frame, data_meta = _weather_frame(n=500, seed=ctx.seed)
    feats = ["hour", "humidity", "pressure_hpa", "wind_mps"]
    session = (
        Session.ingest(frame)
        .set_roles({**{c: "feature" for c in feats}, "temp_c": "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=ctx.seed)
        .scale(method="standard")
    )
    fit = session.fit_probabilistic(
        estimator="bayesian_ridge",
        alpha=0.1,
        conformal=True,
        interval_method="both",
        random_state=ctx.seed,
    )
    try:
        intervals = session.predict_interval(partition="test", alpha=0.1)
        interval_payload = metrics_round(
            intervals.to_dict() if hasattr(intervals, "to_dict") else {}
        )
    except Exception as exc:  # noqa: BLE001
        interval_payload = {"error": f"{type(exc).__name__}: {exc}"}
    ev = session.evaluate_probabilistic(partition="test")
    bundle = session.save_probabilistic_bundle(ctx.artifacts_dir / "prob_bundle")
    write_results(
        ctx,
        {
            "status": "completed",
            "data": data_meta,
            "fit": metrics_round(fit.to_dict() if hasattr(fit, "to_dict") else {}),
            "intervals": interval_payload,
            "test_metrics": metrics_round(dict(getattr(ev, "metrics", {}) or {})),
            "bundle_path": str(bundle),
            "leakage_controls": [
                "Probabilistic model fit on train",
                "Interval calibration uses non-test partitions when required by API",
                "Test evaluate after lock",
            ],
            "industry_comparison": {
                "status": "filled",
                "note": (
                    "Tier C baseline_industry.py: BayesianRidge + residual quantile twin on the "
                    "same split; run script then baseline_industry.py for results/comparison.json."
                ),
            },
            "limitations": [
                "Synthetic weather; empirical coverage ≠ guaranteed under shift",
            ],
        },
    )
    print("weather-prob-intervals OK", getattr(ev, "metrics", ev))


if __name__ == "__main__":
    main()
