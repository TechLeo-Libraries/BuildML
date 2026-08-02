#!/usr/bin/env python3
"""Smoke benchmark for time-series analysis APIs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from buildml import Session
from buildml.timeseries.extras import statsmodels_available


def main() -> int:
    rng = np.random.default_rng(0)
    n = 180
    t = pd.date_range("2022-01-01", periods=n, freq="D")
    y = 5 + 0.02 * np.arange(n) + np.sin(np.arange(n) / 7) + rng.normal(0, 0.1, n)
    frame = pd.DataFrame({"date": t, "value": y})

    session = (
        Session.ingest(frame)
        .set_roles({"date": "time", "value": "target"})
        .time_split(test_size=0.2, validation_size=0.1)
    )
    result = session.analyze_timeseries(scope="train", seasonal_period=7)
    payload = {
        "n_points": result.n_points,
        "decompose_method": None if result.decompose is None else result.decompose.method,
        "adf_pvalue": None
        if result.diagnostics is None
        else result.diagnostics.adf_pvalue,
        "n_changepoints": 0
        if result.changepoints is None
        else len(result.changepoints.changepoint_indices),
        "statsmodels": statsmodels_available(),
    }
    out = Path(__file__).resolve().parent / "results" / "analysis_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
