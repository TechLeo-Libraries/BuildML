"""Launch helpers for the local EDA Teaching Studio."""

from __future__ import annotations

import socket
import threading
import time
import webbrowser
from dataclasses import dataclass
from typing import Any

from buildml.core.errors import BuildMLError, MissingExtraError, ValidationError
from buildml.dashboard.state import DashboardState, clear_state, set_state
from buildml.eda.report import EDAReport


class DashboardLaunchError(BuildMLError):
    """Raised when the local EDA Teaching Studio cannot bind or start."""


@dataclass(slots=True)
class EDAAppHandle:
    """Handle for a running local EDA app process thread."""

    host: str
    port: int
    url: str
    title: str
    _server: Any
    _thread: threading.Thread

    def stop(self) -> None:
        """Stop the background uvicorn server and clear app state."""
        server = self._server
        if server is not None:
            server.should_exit = True
        if self._thread.is_alive():
            self._thread.join(timeout=5)
        clear_state()

    @property
    def is_running(self) -> bool:
        """Whether the server thread is still alive.

        Reads the thread rather than asking the server, because a server that
        crashed on startup leaves a dead thread and no other outward sign. This
        is the reliable check.

        Returns
        -------
        bool
            ``True`` while the thread is running. ``False`` after
            :meth:`stop`, or if the server died on its own.

        Notes
        -----
        **Alive is not the same as serving.** There is a brief window during
        startup where the thread exists and the port is not yet accepting; a
        handle returned from a non-blocking launch is past that window.
        """
        return self._thread.is_alive()


def launch_eda_app(
    report: EDAReport,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    title: str = "BuildML EDA Studio",
    session_meta: dict[str, Any] | None = None,
    blocking: bool = False,
) -> EDAAppHandle:
    """Start a local web server showing the studio for a report.

    Turns a report you already have into an interactive dashboard on
    ``localhost``. Non-blocking by default, so a notebook cell returns
    immediately and the server keeps running in the background: call
    :meth:`EDAAppHandle.stop` when done.

    The port is checked before the server starts. That check exists because
    uvicorn's own bind failure surfaces from a background thread as an obscure
    traceback with no obvious remedy; failing early gives a message that names
    the port and says what to do.

    Parameters
    ----------
    report:
        The EDA result, from ``session.eda(...)`` or
        :func:`~buildml.eda.profile.explore_dataset`.
    host:
        Bind address. **Keep this at ``127.0.0.1``.** Binding to ``0.0.0.0``
        exposes the dashboard: and your data: to everything that can reach the
        machine, with no authentication of any kind.
    port:
        Bind port.
    open_browser:
        Open the default browser once the server is accepting connections.
    title:
        Header and window title.
    session_meta:
        Extra Session facts for the cockpit board.
    blocking:
        Run on the calling thread and do not return until the server stops.
        Right for a script whose only job is to serve; wrong for a notebook,
        where it would hang the kernel.

    Returns
    -------
    EDAAppHandle
        The URL and the controls to stop it. Under ``blocking=True`` this is
        returned only after the server has already stopped.

    Raises
    ------
    MissingExtraError
        If the dashboard dependencies are not installed. Install with
        ``pip install 'buildml[dashboard]'``.
    DashboardLaunchError
        If the port is occupied or the server fails to start within 15 seconds.
    ValidationError
        If the host is empty or the port is outside 1–65535.

    Notes
    -----
    **There is no authentication.** Anyone who can reach the address can read
    the report, which includes column names, distributions, and example values.
    Local-only binding is the entire protection.

    **One dashboard per process.** State is process-global, so a second launch
    replaces the first one's data.

    **State is cleared on stop**, releasing the report and its figures.

    Examples
    --------
    ::

        report = session.eda()
        handle = launch_eda_app(report)
        print(handle.url)
        ...
        handle.stop()

    See Also
    --------
    buildml.dashboard.offline.export_studio_html : A file instead of a server.
    """
    try:
        import uvicorn
    except ImportError as exc:
        raise MissingExtraError("dashboard", "EDA Teaching Studio app") from exc

    _validate_bind_target(host, port)
    _ensure_port_available(host, port)

    from buildml.dashboard.app import create_app

    set_state(
        DashboardState(
            report=report,
            report_dict=report.to_dict(),
            title=title,
            session_meta=session_meta or {},
        )
    )
    try:
        app = create_app()
    except MissingExtraError:
        clear_state()
        raise
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    url = f"http://{host}:{port}/"

    if blocking:
        if open_browser:
            threading.Thread(
                target=_open_when_ready,
                args=(server, url),
                daemon=True,
            ).start()
        try:
            server.run()
        finally:
            clear_state()
        return EDAAppHandle(
            host=host,
            port=port,
            url=url,
            title=title,
            _server=server,
            _thread=threading.current_thread(),
        )

    error_box: list[BaseException] = []

    def _run() -> None:
        try:
            server.run()
        except BaseException as exc:  # noqa: BLE001 - surface bind failures to waiter
            error_box.append(exc)

    thread = threading.Thread(target=_run, name="buildml-eda-app", daemon=True)
    thread.start()
    try:
        _wait_until_started(server, thread, error_box, host=host, port=port, timeout=15.0)
    except Exception:
        server.should_exit = True
        if thread.is_alive():
            thread.join(timeout=2)
        clear_state()
        raise
    if open_browser:
        webbrowser.open(url)
    return EDAAppHandle(
        host=host,
        port=port,
        url=url,
        title=title,
        _server=server,
        _thread=thread,
    )


def open_eda_dashboard(
    report: EDAReport,
    **kwargs: Any,
) -> EDAAppHandle:
    """Launch the studio, under the name people reach for first.

    An alias for :func:`launch_eda_app`, kept because "open the dashboard" is
    how the action is usually described. Identical behaviour; every argument is
    forwarded.

    Parameters
    ----------
    report:
        The EDA result to serve.
    **kwargs:
        Passed through: ``host``, ``port``, ``open_browser``, ``title``,
        ``session_meta``, ``blocking``.

    Returns
    -------
    EDAAppHandle
        The URL and controls for the running server.

    Raises
    ------
    MissingExtraError
        If the dashboard dependencies are missing.
    DashboardLaunchError
        If the port is occupied or startup fails.
    ValidationError
        If the host or port is invalid.

    See Also
    --------
    launch_eda_app : The same function, with the full documentation.
    """
    return launch_eda_app(report, **kwargs)


def _validate_bind_target(host: str, port: int) -> None:
    if not isinstance(host, str) or not host.strip():
        raise ValidationError("EDA app host must be a non-empty string such as '127.0.0.1'.")
    if not isinstance(port, int) or isinstance(port, bool) or not (1 <= port <= 65535):
        raise ValidationError("EDA app port must be an integer between 1 and 65535.")


def _ensure_port_available(host: str, port: int) -> None:
    """Fail fast with a clear next action when the bind port is occupied."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # Do not set SO_REUSEADDR here: on Windows it can mask an occupied port.
            sock.bind((host, port))
    except OSError as exc:
        alternate = port + 1 if port < 65535 else 8766
        raise DashboardLaunchError(
            f"Cannot start EDA Teaching Studio: port {port} on {host} is already in use "
            f"({exc}). Next actions: stop the other process, or launch on another port, "
            f"e.g. session.eda_app(port={alternate}). "
            "If dependencies are missing, install with: pip install 'buildml[dashboard]'."
        ) from exc


def _wait_until_started(
    server: Any,
    thread: threading.Thread,
    error_box: list[BaseException],
    *,
    host: str,
    port: int,
    timeout: float,
) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if error_box:
            raise _launch_failure(host, port, error_box[0])
        if getattr(server, "started", False):
            return
        if not thread.is_alive():
            detail = error_box[0] if error_box else None
            raise _launch_failure(host, port, detail)
        time.sleep(0.05)
    if error_box:
        raise _launch_failure(host, port, error_box[0])
    if getattr(server, "started", False):
        return
    alternate = port + 1 if port < 65535 else 8766
    raise DashboardLaunchError(
        f"EDA Teaching Studio did not become ready on http://{host}:{port}/ "
        f"within {timeout:.0f}s. Confirm nothing else is bound to that port, try "
        f"session.eda_app(port={alternate}), and ensure dependencies are installed with "
        "pip install 'buildml[dashboard]'."
    )


def _launch_failure(host: str, port: int, exc: BaseException | None) -> DashboardLaunchError:
    alternate = port + 1 if port < 65535 else 8766
    detail = f" Underlying error: {exc}." if exc is not None else ""
    message = str(exc).lower() if exc is not None else ""
    if "address already in use" in message or "10048" in message or "eaddrinuse" in message:
        return DashboardLaunchError(
            f"Cannot start EDA Teaching Studio: port {port} on {host} is already in use."
            f"{detail} Next actions: stop the other process, or launch on another port, "
            f"e.g. session.eda_app(port={alternate}). "
            "If dependencies are missing, install with: pip install 'buildml[dashboard]'."
        )
    return DashboardLaunchError(
        f"EDA Teaching Studio failed to start on http://{host}:{port}/.{detail} "
        f"Try session.eda_app(port={alternate}) or reinstall with "
        "pip install 'buildml[dashboard]'."
    )


def _open_when_ready(server: Any, url: str) -> None:
    deadline = time.time() + 20.0
    while time.time() < deadline:
        if getattr(server, "started", False):
            webbrowser.open(url)
            return
        time.sleep(0.05)
