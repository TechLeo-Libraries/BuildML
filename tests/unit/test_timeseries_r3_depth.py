"""Phase R3 time-series analysis depth: Session, walkthrough, AI allowlist."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.tools import build_default_registry
from buildml.core.errors import LeakageError
from buildml.timeseries.analyze import analyze_timeseries, ts_decompose
from buildml.timeseries.extras import statsmodels_available


def _frame(n: int = 120, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range("2024-01-01", periods=n, freq="D")
    y = 10 + 0.04 * np.arange(n) + 2 * np.sin(2 * np.pi * np.arange(n) / 7)
    y += rng.normal(0, 0.2, n)
    return pd.DataFrame({"ts": t, "y": y})


def test_low_level_analyze_train_scope() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"ts": "time", "y": "target"})
        .time_split(test_size=0.2, validation_size=0.2)
    )
    result = analyze_timeseries(session.dataset, session.split_plan, scope="train")
    assert result.n_points > 0
    assert result.decompose is not None
    assert result.diagnostics is not None
    assert len(result.diagnostics.acf_values) > 1
    if statsmodels_available():
        assert result.decompose.method in {"stl", "classical", "moving_average"}
        assert result.diagnostics.adf_pvalue is not None
    else:
        assert result.decompose.method == "moving_average"


def test_session_analyze_and_walkthrough() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"ts": "time", "y": "target"})
        .time_split(test_size=0.25)
    )
    out = session.analyze_timeseries(scope="train")
    assert out.decompose is not None
    decomp = session.ts_decompose(seasonal_period=7)
    assert decomp.decompose is not None
    diag = session.ts_diagnostics(acf_lags=20, pacf_lags=10)
    assert diag.diagnostics is not None

    report = session.walkthrough()
    ts_status = report.to_dict()["timeseries_status"]
    assert ts_status["has_analysis_result"] is True

    registry = build_default_registry()
    for name in ("analyze_timeseries", "ts_decompose", "ts_diagnostics"):
        assert registry.get(name) is not None


def test_refuses_random_split() -> None:
    frame = _frame(n=80)
    session = Session.ingest(frame).set_roles({"ts": "time", "y": "target"})
    session.split(test_size=0.2, random_state=0)
    with pytest.raises(LeakageError):
        session.analyze_timeseries()


def test_ts_decompose_only_low_level() -> None:
    session = (
        Session.ingest(_frame(n=90))
        .set_roles({"ts": "time", "y": "target"})
        .time_split(test_size=0.2)
    )
    result = ts_decompose(session.dataset, session.split_plan, seasonal_period=7)
    assert result.decompose is not None
    assert len(result.decompose.trend) == result.n_points
