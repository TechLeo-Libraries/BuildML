"""History / catalog / walkthrough helpers for time-series analysis."""

from __future__ import annotations

from typing import Any


def analysis_result_summary(result: Any) -> dict[str, Any]:
    """Compact result_summary for analyze_timeseries history."""
    if result is None:
        return {}
    payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    out = {
        "target_column": payload.get("target_column"),
        "time_column": payload.get("time_column"),
        "scope": payload.get("scope"),
        "n_points": payload.get("n_points"),
    }
    if payload.get("has_decompose"):
        out["decompose"] = True
    if payload.get("has_diagnostics"):
        out["diagnostics"] = True
    return out


def timeseries_status(
    analysis_result: Any = None,
    *,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Factual walkthrough disclosure for time-series analysis."""
    from buildml.timeseries.catalog import timeseries_status_payload

    records = list(history or [])
    saw = any(
        str(r.get("operation_id") or r.get("action"))
        in {
            "analyze_timeseries",
            "ts_decompose",
            "ts_diagnostics",
        }
        for r in records
    )
    enabled = analysis_result is not None
    base = timeseries_status_payload()
    disclosures = list(base.get("disclosures", []))
    if enabled:
        payload = (
            analysis_result.to_dict()
            if hasattr(analysis_result, "to_dict")
            else dict(analysis_result)
        )
        disclosures.append(
            f"Last analysis: scope={payload.get('scope')}, "
            f"n={payload.get('n_points')}, target={payload.get('target_column')}."
        )
    elif saw:
        disclosures.append(
            "Time-series analysis appears in history but no live result is attached."
        )

    return {
        "enabled": enabled,
        "present": enabled or saw,
        "has_analysis_result": enabled,
        "backends": {
            "statsmodels": base.get("statsmodels_available"),
            "ruptures": base.get("ruptures_available"),
        },
        "defaults": {
            "decompose": base.get("default_decompose"),
            "changepoint": base.get("default_changepoint"),
        },
        "disclosures": disclosures,
        "boundary": (
            "Time-series analysis is descriptive EDA on temporal data — "
            "distinct from fit_forecast and from supervised shuffled-row fit."
        ),
    }


def timeseries_status_for_session(session: Any) -> dict[str, Any]:
    """Session-facing status helper."""
    return timeseries_status(
        getattr(session, "_ts_analysis_result", None),
        history=list(getattr(session, "_history", ()) or ()),
    )
