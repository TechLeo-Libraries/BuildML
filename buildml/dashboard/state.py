"""In-memory store for a single local EDA app process."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from buildml.eda.report import EDAReport


@dataclass(slots=True)
class DashboardState:
    """Process-local payload for one EDA app instance."""

    report: EDAReport
    report_dict: dict[str, Any]
    title: str = "BuildML EDA Studio"
    session_meta: dict[str, Any] = field(default_factory=dict)


_LOCK = RLock()
_STATE: DashboardState | None = None


def set_state(state: DashboardState) -> None:
    global _STATE
    with _LOCK:
        _STATE = state


def get_state() -> DashboardState:
    with _LOCK:
        if _STATE is None:
            raise RuntimeError("EDA dashboard state is not initialized.")
        return _STATE


def clear_state() -> None:
    global _STATE
    with _LOCK:
        _STATE = None
