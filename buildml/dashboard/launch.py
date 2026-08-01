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
    """Serve the EDA Teaching Studio for an existing :class:`EDAReport`.

    Parameters
    ----------
    report:
        Structured EDA result from ``session.eda(...)``.
    host, port:
        Local bind address.
    open_browser:
        Open the default browser when the server accepts connections.
    title:
        Window/header title.
    session_meta:
        Optional lightweight Session facts for the cockpit.
    blocking:
        If True, run the server on the current thread (useful for scripts).

    Raises
    ------
    MissingExtraError
        When ``buildml[dashboard]`` dependencies are not installed.
    DashboardLaunchError
        When the requested port is already in use or the server fails to start.
    ValidationError
        When host/port values are invalid.
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
    """Alias for :func:`launch_eda_app`."""
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
