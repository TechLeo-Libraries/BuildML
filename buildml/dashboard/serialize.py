"""Helpers for normalizing analyzer payloads into JSON-safe dashboard shapes."""

from __future__ import annotations

from typing import Any


def flagged_column_names(flagged: Any) -> list[str]:
    """Return column names from string lists or analyzer flag row dicts."""
    names: list[str] = []
    if not isinstance(flagged, list):
        return names
    for item in flagged:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict) and item.get("column") is not None:
            names.append(str(item["column"]))
    return names


def json_safe(value: Any) -> Any:
    """Best-effort conversion of numpy/pandas scalars for API responses."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return json_safe(item())
        except Exception:
            return str(value)
    return str(value)
