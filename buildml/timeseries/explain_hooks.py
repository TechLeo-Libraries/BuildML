"""History / catalog / walkthrough helpers for time-series analysis."""

from __future__ import annotations

from typing import Any


def analysis_result_summary(result: Any) -> dict[str, Any]:
    """Build a compact history summary from a :class:`TSAnalysisResult`.

    Strips heavy component vectors so Session audit logs stay small while still
    recording scope, column names, and which analysis blocks ran.

    Parameters
    ----------
    result:
        :class:`~buildml.timeseries.results.TSAnalysisResult` or ``None``.

    Returns
    -------
    dict[str, Any]
        Keys ``target_column``, ``time_column``, ``scope``, ``n_points``, and
        boolean flags for decomposition/diagnostics when present. Empty dict
        when ``result`` is ``None``.
    """
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
    """Build factual walkthrough disclosure for time-series analysis state.

    Combines install/backend facts from :func:`timeseries_status_payload` with
    whether a live analysis result exists or analysis appears in Session history.
    Used by teaching overlays and dashboard walkthrough panels.

    Parameters
    ----------
    analysis_result:
        Last :class:`~buildml.timeseries.results.TSAnalysisResult`, if any.
    history:
        Session operation history records to detect past analyze calls without a
        live result attached.

    Returns
    -------
    dict[str, Any]
        Keys ``enabled``, ``present``, ``has_analysis_result``, ``backends``,
        ``defaults``, ``disclosures``, and ``boundary`` separating analysis from
        forecasting.
    """
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

    from buildml.explain.capability_status import attach_capability_matrix

    return attach_capability_matrix(
        {
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
                "Time-series analysis is descriptive EDA on temporal data: "
                "distinct from fit_forecast and from supervised shuffled-row fit."
            ),
        },
        "timeseries_capability_matrix",
    )


def timeseries_status_for_session(session: Any) -> dict[str, Any]:
    """Report time-series analysis status for a Session walkthrough panel.

    Reads ``_ts_analysis_result`` and ``_history`` from the Session without
    mutating state. Convenience wrapper around :func:`timeseries_status`.

    Parameters
    ----------
    session:
        :class:`~buildml.session.session.Session` instance.

    Returns
    -------
    dict[str, Any]
        Same payload as :func:`timeseries_status` for the Session's last result
        and history.
    """
    return timeseries_status(
        getattr(session, "_ts_analysis_result", None),
        history=list(getattr(session, "_history", ()) or ()),
    )
