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
    """Install the report the dashboard serves, replacing any previous one.

    Called once before the server starts. The lock guards against a request
    arriving mid-assignment, which is possible because the web server runs on
    its own threads.

    Parameters
    ----------
    state:
        The report and metadata to serve.

    Returns
    -------
    None

    Notes
    -----
    **There is one slot.** A second dashboard in the same process replaces the
    first, and the first one's routes then serve the second one's data. One
    dashboard per process is the supported arrangement.

    See Also
    --------
    get_state : Reading it back.
    clear_state : Releasing it.
    """
    global _STATE
    with _LOCK:
        _STATE = state


def get_state() -> DashboardState:
    """Return the installed report, or refuse if there is none.

    Every request handler starts here. Raising rather than returning ``None``
    means a route cannot accidentally serve an empty page when the state was
    never installed: the failure is loud and points at the setup, not at the
    route.

    Returns
    -------
    DashboardState
        The report and metadata currently being served.

    Raises
    ------
    RuntimeError
        If no state has been installed, or it was cleared. Usually means the
        app was created without going through
        :func:`~buildml.dashboard.launch.launch_eda_app`.

    See Also
    --------
    set_state : Installing it.
    """
    with _LOCK:
        if _STATE is None:
            raise RuntimeError("EDA dashboard state is not initialized.")
        return _STATE


def clear_state() -> None:
    """Drop the installed report, releasing its memory.

    Called when a dashboard shuts down. An EDA report holds analyzer output and
    possibly rendered figures, so leaving it installed after the server stops
    keeps all of that alive for the life of the process: which matters in a
    notebook, where the process lives a long time.

    Returns
    -------
    None

    Notes
    -----
    **Requests after this raise.** Any handler still running will get a
    ``RuntimeError`` from :func:`get_state`, which is the intended behaviour
    during shutdown.

    See Also
    --------
    set_state : Installing state.
    """
    global _STATE
    with _LOCK:
        _STATE = None
