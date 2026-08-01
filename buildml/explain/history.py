"""Versioned, JSON-safe operation history records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

HISTORY_SCHEMA_VERSION = 2

_EMPTY_STATE: dict[str, Any] = {
    "has_dataset": False,
    "columns": [],
    "roles": {},
    "has_split": False,
    "split_kind": None,
    "has_fit": False,
    "has_dl_train": False,
    "has_impute_plan": False,
    "has_encode_plan": False,
    "has_scale_plan": False,
    "has_outlier_plan": False,
    "has_binning_plan": False,
    "has_feature_select_plan": False,
    "has_text_plan": False,
    "has_reduce_plan": False,
    "has_custom_plan": False,
    "has_date_plan": False,
    "has_resample_plan": False,
}


def json_safe(value: Any) -> Any:
    """Return a compact JSON-compatible representation of runtime values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return json_safe(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [json_safe(item) for item in value]
    if hasattr(value, "to_dict"):
        return json_safe(value.to_dict())
    return repr(value)


def session_state(session: Any) -> dict[str, Any]:
    """Capture the workflow state relevant to explanations."""
    dataset = getattr(session, "_dataset", None)
    split = getattr(session, "_split_plan", None)
    return {
        "has_dataset": dataset is not None,
        "columns": [] if dataset is None else list(dataset.columns),
        "roles": {}
        if dataset is None
        else {name: role.value for name, role in dataset.roles.items()},
        "has_split": split is not None,
        "split_kind": None if split is None else split.kind,
        "has_fit": getattr(session, "_fit_result", None) is not None,
        "has_dl_train": getattr(session, "_dl_train_result", None) is not None,
        "has_rag_corpus": getattr(session, "_rag_corpus", None) is not None,
        "has_rag_index": getattr(session, "_rag_index_result", None) is not None,
        "has_impute_plan": getattr(session, "_impute_plan", None) is not None,
        "has_encode_plan": getattr(session, "_encode_plan", None) is not None,
        "has_scale_plan": getattr(session, "_scale_plan", None) is not None,
        "has_outlier_plan": getattr(session, "_outlier_plan", None) is not None,
        "has_binning_plan": getattr(session, "_binning_plan", None) is not None,
        "has_feature_select_plan": getattr(session, "_feature_select_plan", None) is not None,
        "has_text_plan": getattr(session, "_text_plan", None) is not None,
        "has_reduce_plan": getattr(session, "_reduce_plan", None) is not None,
        "has_custom_plan": getattr(session, "_custom_plan", None) is not None,
        "has_date_plan": getattr(session, "_date_plan", None) is not None,
        "has_resample_plan": getattr(session, "_resample_plan", None) is not None,
    }


def state_changes(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    """Describe changed state keys precisely and without domain speculation."""
    changes: list[str] = []
    for key in sorted(set(before) | set(after)):
        old, new = before.get(key), after.get(key)
        if old != new:
            changes.append(f"{key}: {old!r} -> {new!r}")
    return changes


def make_operation_record(
    *,
    sequence: int,
    operation_id: str,
    parameters: Mapping[str, Any] | None,
    decision_origin: str,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    warnings: Sequence[str] = (),
    result_summary: Mapping[str, Any] | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build one backward-compatible v2 history record."""
    safe_parameters = json_safe(dict(parameters or {}))
    safe_before = json_safe(dict(before))
    safe_after = json_safe(dict(after))
    return {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "sequence": sequence,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "operation_id": operation_id,
        # Retain v1 keys for code that still reads action/details.
        "action": operation_id,
        "parameters": safe_parameters,
        "details": safe_parameters,
        "decision_origin": decision_origin,
        "state_transition": {
            "before": safe_before,
            "after": safe_after,
            "changes": state_changes(safe_before, safe_after),
        },
        "warnings": [str(item) for item in warnings],
        "result_summary": json_safe(dict(result_summary or parameters or {})),
    }


def normalize_history(history: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    """Normalize v1/v2 checkpoint history into ordered v2 records."""
    normalized: list[dict[str, Any]] = []
    previous = dict(_EMPTY_STATE)
    for index, raw in enumerate(history or (), start=1):
        operation_id = str(raw.get("operation_id") or raw.get("action") or "unknown")
        transition = raw.get("state_transition")
        if isinstance(transition, Mapping):
            before = transition.get("before", previous)
            after = transition.get("after", before)
        else:
            before = previous
            after = previous
        details = raw.get("parameters", raw.get("details", {}))
        raw_warnings = raw.get("warnings", ())
        warnings = (
            raw_warnings
            if isinstance(raw_warnings, Sequence)
            and not isinstance(raw_warnings, (str, bytes, bytearray))
            else (str(raw_warnings),)
            if raw_warnings
            else ()
        )
        record = make_operation_record(
            sequence=index,
            operation_id=operation_id,
            parameters=details if isinstance(details, Mapping) else {"value": details},
            decision_origin=str(raw.get("decision_origin", "explicit")),
            before=before if isinstance(before, Mapping) else previous,
            after=after if isinstance(after, Mapping) else previous,
            warnings=warnings,
            result_summary=raw.get("result_summary", {})
            if isinstance(raw.get("result_summary", {}), Mapping)
            else {"value": raw.get("result_summary")},
            timestamp=str(raw.get("timestamp") or datetime.now(timezone.utc).isoformat()),
        )
        normalized.append(record)
        previous = dict(record["state_transition"]["after"])
    return normalized


def prior_state(history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return the state after the latest record, or the empty workflow state."""
    if history:
        transition = history[-1].get("state_transition", {})
        after = transition.get("after", {}) if isinstance(transition, Mapping) else {}
        if isinstance(after, Mapping):
            return dict(after)
    return dict(_EMPTY_STATE)
