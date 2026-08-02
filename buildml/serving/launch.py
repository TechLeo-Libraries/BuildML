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
    tls: bool = False

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


def _is_loopback_host(host: str) -> bool:
    normalized = str(host).strip().lower()
    return normalized in {"127.0.0.1", "localhost", "::1"}


def _has_api_keys(api_keys: str | list[str] | tuple[str, ...] | None) -> bool:
    if api_keys is None:
        return False
    if isinstance(api_keys, str):
        return bool(api_keys.strip())
    return any(str(key).strip() for key in api_keys)


def _ensure_bind_security(
    host: str,
    *,
    api_keys: str | list[str] | tuple[str, ...] | None,
    allow_insecure_public_bind: bool,
) -> None:
    """Refuse non-loopback binds without API keys unless explicitly overridden."""
    if _is_loopback_host(host):
        return
    if _has_api_keys(api_keys):
        return
    if allow_insecure_public_bind:
        return
    raise ValidationError(
        f"Refusing to bind managed serving to non-loopback host {host!r} without "
        "api_keys. Pass api_keys= (or CLI --api-key) for public/non-localhost binds, "
        "or set allow_insecure_public_bind=True / --allow-insecure-public-bind to "
        "override deliberately. Localhost defaults remain open-with-honesty."
    )


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
    allow_insecure_public_bind: bool = False,
    ssl_certfile: str | Path | None = None,
    ssl_keyfile: str | Path | None = None,
) -> ServeHandle:
    """Serve a classical pipeline or TorchScript bundle over HTTP(S).

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
        Still not a managed IAM / cloud auth product. **Required** for
        non-loopback binds unless ``allow_insecure_public_bind=True``.
    allow_insecure_public_bind:
        Loud opt-in to bind ``0.0.0.0`` / other non-loopback hosts without
        ``api_keys``. Prefer API keys + reverse-proxy TLS instead.
    ssl_certfile, ssl_keyfile:
        Optional local PEM paths for uvicorn HTTPS. Both required together.
        Library-owned local TLS only — not managed certificate infrastructure.
    blocking:
        If True, run uvicorn on the current thread.

    Notes
    -----
    Prefer TLS + auth at a reverse proxy for any non-local exposure.
    This is a library-owned local server, not a Kubernetes multi-cluster product.
    """
    _validate_bind_target(host, port)
    _ensure_bind_security(
        host,
        api_keys=api_keys,
        allow_insecure_public_bind=allow_insecure_public_bind,
    )
    cert = None if ssl_certfile is None else Path(ssl_certfile)
    key = None if ssl_keyfile is None else Path(ssl_keyfile)
    if (cert is None) ^ (key is None):
        raise ValidationError(
            "ssl_certfile and ssl_keyfile must be provided together for local HTTPS."
        )
    if cert is not None and not cert.is_file():
        raise ValidationError(f"ssl_certfile not found: {cert}")
    if key is not None and not key.is_file():
        raise ValidationError(f"ssl_keyfile not found: {key}")

    try:
        import uvicorn
    except ImportError as exc:
        raise MissingExtraError("serve", "Managed model serving") from exc

    _ensure_port_available(host, port)

    app = create_serving_app(
        path,
        kind=kind,  # type: ignore[arg-type]
        title=title,
        map_location=map_location,
        api_keys=api_keys,
    )
    tls = cert is not None and key is not None
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        ssl_certfile=str(cert) if cert is not None else None,
        ssl_keyfile=str(key) if key is not None else None,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="buildml-serve", daemon=True)
    scheme = "https" if tls else "http"
    handle = ServeHandle(
        host=host,
        port=port,
        url=f"{scheme}://{host}:{port}",
        kind=str(kind),
        path=str(path),
        _server=server,
        _thread=thread,
        tls=tls,
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
