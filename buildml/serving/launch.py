"""Launch helpers for BuildML managed model serving."""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from buildml.core.errors import BuildMLError, MissingExtraError, ValidationError
from buildml.serving.app import BundleKind, clear_serving_state, create_serving_app


class ServingLaunchError(BuildMLError):
    """Raised when the managed serving process cannot bind or start."""


@dataclass(slots=True)
class ServeHandle:
    """Handle for a running local model server thread."""

    host: str
    port: int
    url: str
    kind: str
    path: str
    _server: Any
    _thread: threading.Thread

    def stop(self) -> None:
        """Stop the background uvicorn server and clear serving state."""
        server = self._server
        if server is not None:
            server.should_exit = True
        if self._thread.is_alive():
            self._thread.join(timeout=5)
        clear_serving_state()

    @property
    def is_running(self) -> bool:
        return self._thread.is_alive()


def _validate_bind_target(host: str, port: int) -> None:
    if not host or not str(host).strip():
        raise ValidationError("host must be non-empty")
    if not isinstance(port, int) or port < 1 or port > 65535:
        raise ValidationError("port must be an integer in 1..65535")


def _ensure_port_available(host: str, port: int) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
    except OSError as exc:
        raise ServingLaunchError(
            f"Cannot bind managed serving to {host}:{port}: {exc}"
        ) from exc
    finally:
        sock.close()


def serve_bundle(
    path: str | Path,
    *,
    kind: BundleKind | Literal["pipeline", "torchscript"] = "pipeline",
    host: str = "127.0.0.1",
    port: int = 8080,
    title: str = "BuildML Serve",
    blocking: bool = False,
    map_location: str = "cpu",
    api_keys: str | list[str] | tuple[str, ...] | None = None,
) -> ServeHandle:
    """Serve a classical pipeline or TorchScript bundle over HTTP.

    Parameters
    ----------
    path:
        Pipeline bundle directory or TorchScript file.
    kind:
        ``pipeline`` (classical ``buildml.pipeline_bundle``) or ``torchscript``.
    host, port:
        Bind address. **Defaults to localhost**.
    api_keys:
        Optional API key(s) enabling Bearer / ``X-API-Key`` middleware.
        Still not a managed IAM / cloud auth product.
    blocking:
        If True, run uvicorn on the current thread.

    Notes
    -----
    Prefer TLS + auth at a reverse proxy for any non-local exposure.
    This is a library-owned local server, not a Kubernetes multi-cluster product.
    """
    try:
        import uvicorn
    except ImportError as exc:
        raise MissingExtraError("serve", "Managed model serving") from exc

    _validate_bind_target(host, port)
    if host not in {"127.0.0.1", "localhost", "::1"} and host != "0.0.0.0":
        # Allow other binds; optional api_keys middleware is still not IAM-as-a-service.
        pass
    _ensure_port_available(host, port)

    app = create_serving_app(
        path,
        kind=kind,  # type: ignore[arg-type]
        title=title,
        map_location=map_location,
        api_keys=api_keys,
    )
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="buildml-serve", daemon=True)
    handle = ServeHandle(
        host=host,
        port=port,
        url=f"http://{host}:{port}",
        kind=str(kind),
        path=str(path),
        _server=server,
        _thread=thread,
    )
    if blocking:
        try:
            server.run()
        finally:
            clear_serving_state()
        return handle
    thread.start()
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if getattr(server, "started", False):
            break
        if not thread.is_alive():
            clear_serving_state()
            raise ServingLaunchError("Managed serving thread exited during startup")
        time.sleep(0.05)
    return handle
