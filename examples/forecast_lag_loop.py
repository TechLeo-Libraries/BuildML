"""Runnable classical forecasting loop (core BuildML — no extra)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session


def main() -> None:
    rng = np.random.default_rng(0)
    n = 120
    t = pd.date_range("2024-01-01", periods=n, freq="D")
    y = 10 + 0.05 * np.arange(n) + np.sin(np.arange(n) / 7) + rng.normal(0, 0.3, n)
    frame = pd.DataFrame({"ts": t, "y": y})

    session = (
        Session.ingest(frame)
        .set_roles({"ts": "time", "y": "target"})
        .time_split(test_size=0.2, validation_size=0.2)
    )
    fit = session.fit_forecast(
        method="lag_ridge",
        horizon=7,
        lags=[1, 2, 3, 7],
        alpha=1.0,
    )
    metrics = session.evaluate_forecast(
        partition="validation",
        strategy="rolling_one_step",
    )
    gen = session.generate_forecast(horizon=7)
    print("fit:", fit.to_dict())
    print("eval:", metrics.to_dict())
    print("generate:", gen.to_dict())

    out = Path(".buildml-artifacts") / "forecast_bundle"
    path = session.save_forecast_bundle(out)
    print("bundle:", path)


if __name__ == "__main__":
    main()
