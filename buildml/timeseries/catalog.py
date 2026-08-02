"""Time-series analysis method catalog and install hints."""

from __future__ import annotations

from typing import Any

from buildml.timeseries.extras import (
    ruptures_available,
    statsmodels_available,
)

DECOMPOSE_METHODS: frozenset[str] = frozenset({"stl", "classical", "moving_average"})
DEFAULT_DECOMPOSE = "stl" if statsmodels_available() else "moving_average"

CHANGEPOINT_METHODS_CORE: frozenset[str] = frozenset({"cusum"})
CHANGEPOINT_METHODS_EXTRA: frozenset[str] = frozenset({"pelt", "binseg"})
ALL_CHANGEPOINT_METHODS = CHANGEPOINT_METHODS_CORE | CHANGEPOINT_METHODS_EXTRA
DEFAULT_CHANGEPOINT = "pelt" if ruptures_available() else "cusum"


def list_decompose_methods() -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for name in sorted(DECOMPOSE_METHODS):
        extra = None
        if name in {"stl", "classical"}:
            extra = None if statsmodels_available() else "timeseries"
        rows.append(
            {
                "method": name,
                "backend": "statsmodels" if name in {"stl", "classical"} else "core",
                "default": name == DEFAULT_DECOMPOSE,
                "requires_extra": extra,
            }
        )
    return tuple(rows)


def list_changepoint_methods() -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for name in sorted(ALL_CHANGEPOINT_METHODS):
        extra = "timeseries" if name in CHANGEPOINT_METHODS_EXTRA else None
        if name in CHANGEPOINT_METHODS_EXTRA and not ruptures_available():
            extra = "timeseries"
        rows.append(
            {
                "method": name,
                "backend": "ruptures" if name in CHANGEPOINT_METHODS_EXTRA else "core",
                "default": name == DEFAULT_CHANGEPOINT,
                "requires_extra": extra,
            }
        )
    return tuple(rows)


def timeseries_status_payload() -> dict[str, Any]:
    """Install / backend disclosure for walkthrough and guides."""
    return {
        "statsmodels_available": statsmodels_available(),
        "ruptures_available": ruptures_available(),
        "default_decompose": DEFAULT_DECOMPOSE,
        "default_changepoint": DEFAULT_CHANGEPOINT,
        "recommended_extra": "timeseries",
        "disclosures": [
            "Time-series analysis defaults to statsmodels STL/ACF/ADF/KPSS when "
            "buildml[timeseries] is installed.",
            "Core fallback: moving-average decomposition, numpy ACF, lightweight CUSUM "
            "changepoints — stationarity tests require statsmodels.",
            "Analysis APIs refuse shuffled splits; prefer time_split with scope='train'.",
        ],
    }


def timeseries_capability_matrix() -> dict[str, Any]:
    """Honest capability matrix for time-series analysis backends."""
    return {
        "backends": {
            "core": {
                "available": True,
                "extra": None,
                "decompose": ["moving_average"],
                "changepoint": sorted(CHANGEPOINT_METHODS_CORE),
                "notes": "Numpy moving-average + CUSUM — always available.",
            },
            "statsmodels": {
                "available": statsmodels_available(),
                "extra": "timeseries",
                "decompose": ["stl", "classical"],
                "notes": "STL/classical decompose + ACF/ADF/KPSS (buildml[timeseries]).",
            },
            "ruptures": {
                "available": ruptures_available(),
                "extra": "timeseries",
                "changepoint": sorted(CHANGEPOINT_METHODS_EXTRA),
                "notes": "PELT / BinSeg changepoints when ruptures is installed.",
            },
        },
        "default_decompose": DEFAULT_DECOMPOSE,
        "default_changepoint": DEFAULT_CHANGEPOINT,
        "decompose_methods": list(list_decompose_methods()),
        "changepoint_methods": list(list_changepoint_methods()),
        "install_hints": {
            "timeseries": "pip install 'buildml[timeseries]'  # statsmodels + ruptures",
        },
        "non_goals": [
            "Full forecasting product (see buildml.forecasting)",
            "Streaming anomaly product",
        ],
    }
