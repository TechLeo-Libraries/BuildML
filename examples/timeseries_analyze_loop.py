"""Mirror of guides/quickstart-timeseries-analysis.md — analyze, not forecast."""

from __future__ import annotations

import numpy as np
import pandas as pd

from buildml import Session


def main() -> None:
    rng = np.random.default_rng(0)
    n = 150
    t = pd.date_range("2024-01-01", periods=n, freq="D")
    y = 12 + 0.03 * np.arange(n) + 2 * np.sin(2 * np.pi * np.arange(n) / 7)
    y += rng.normal(0, 0.25, n)
    frame = pd.DataFrame({"ts": t, "y": y})

    session = (
        Session.ingest(frame)
        .set_roles({"ts": "time", "y": "target"})
        .time_split(test_size=0.2, validation_size=0.2)
    )

    report = session.timeseries.analyze(scope="train", seasonal_period=7)
    report.show()
    session.timeseries.decompose(decompose_method="stl", seasonal_period=7)
    session.timeseries.diagnostics(acf_lags=30)
    print("analysis only; forecast is session.forecast")


if __name__ == "__main__":
    main()
