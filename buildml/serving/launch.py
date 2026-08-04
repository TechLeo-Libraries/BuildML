"""Start a server for a bundle, with the unsafe defaults refused up front.

The app in :mod:`buildml.serving.app` is a FastAPI object. Getting it running is
this module's job, and the interesting part is what it refuses to do.

Binding to ``0.0.0.0`` without authentication exposes an unauthenticated
prediction endpoint to every host that can route to the machine, and it is the
kind of thing that happens by copying a command from a tutorial. That
combination raises rather than starting. Localhost stays open, because that is a
development loop and adding friction there teaches nothing. An explicit override
exists for the case where a reverse proxy really is handling auth, and it is
named to be conspicuous in a code review.

A server also starts by default in a background thread, so a notebook stays
usable and the handle can be stopped again. Blocking mode is there for
containers, where the process should not exit.

See Also
--------
buildml.serving.app.create_serving_app : The app being served.
buildml.serving.cli : The command-line entry point onto this.
"""

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
    """Raised when the server cannot take the port or dies during startup.

    Separate from :class:`~buildml.core.errors.ValidationError` because the
    request was valid and the environment refused it: the port is taken, the
    address is not assignable, or the thread exited before the server reported
    itself started. Nothing in the calling code is wrong; something outside it
    has to change.
    """


@dataclass(slots=True)
class ServeHandle:
    """A running background server, and the means to stop it.

    Returned by :func:`serve_bundle` in non-blocking mode. Holding it matters:
    the thread is a daemon, so losing the reference leaves a server bound to a
    port with no way to shut it down short of ending the process.

    Attributes
    ----------
    host, port:
        Where the server bound.
    url:
        The base URL, with the scheme reflecting whether TLS was configured, so
        it can be handed straight to a client.
    kind:
        Which artifact kind is being served.
    path:
        The artifact location, for tracing a running server back to a file.
    tls:
        Whether local HTTPS is in effect.

    Notes
    -----
    **Always stop the handle when finished.** In a notebook this is what frees
    the port for the next run; :meth:`stop` also clears the loaded bundle, which
    releases the model.

    Examples
    --------
    Serve, use, and stop::

        handle = serve_bundle("artifacts/churn-pipeline", port=8123, trusted=True)
        try:
            requests.get(handle.url + "/health").json()
        finally:
            handle.stop()

    See Also
    --------
    serve_bundle : Producing this.
    """

    host: str
    port: int
    url: str
    kind: str
    path: str
    _server: Any
    _thread: threading.Thread
    tls: bool = False

    def stop(self) -> None:
        """Ask the server to exit, wait briefly, and release the loaded bundle.

        Signals uvicorn to stop accepting connections and finish what it is
        handling, then joins the thread with a five-second timeout and clears
        the process-global serving state so the model can be collected.

        Notes
        -----
        **The join times out rather than hanging.** A request still in flight
        after five seconds leaves the thread running; since it is a daemon, it
        will not keep the interpreter alive. The bundle is cleared either way,
        so a straggling request may fail: which is the right trade for a stop
        that always returns.

        **Calling this twice is harmless.**
        """
        server = self._server
        if server is not None:
            server.should_exit = True
        if self._thread.is_alive():
            self._thread.join(timeout=5)
        clear_serving_state()

    @property
    def is_running(self) -> bool:
        """Report whether the server thread is still alive.

        Returns
        -------
        bool
            ``True`` while the thread runs.

        Notes
        -----
        **A live thread is not the same as a ready server.** The thread starts
        before uvicorn finishes binding, and :func:`serve_bundle` already waits
        for readiness before returning: so use this to detect a server that has
        *stopped*, and ``/health`` to confirm one that is *working*.
        """
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


def _has_basic_auth(
    basic_auth: str
    | tuple[str, str]
    | list[tuple[str, str]]
    | dict[str, str]
    | None,
) -> bool:
    if basic_auth is None:
        return False
    if isinstance(basic_auth, str):
        return bool(basic_auth.strip())
    if isinstance(basic_auth, dict):
        user = basic_auth.get("username", basic_auth.get("user"))
        password = basic_auth.get("password", basic_auth.get("pass"))
        return bool(user and str(user).strip()) and password is not None
    if isinstance(basic_auth, tuple) and len(basic_auth) == 2:
        return bool(str(basic_auth[0]).strip())
    if isinstance(basic_auth, list):
        return len(basic_auth) > 0
    return False


def _has_auth(
    api_keys: str | list[str] | tuple[str, ...] | None,
    basic_auth: str
    | tuple[str, str]
    | list[tuple[str, str]]
    | dict[str, str]
    | None,
) -> bool:
    return _has_api_keys(api_keys) or _has_basic_auth(basic_auth)


def _ensure_bind_security(
    host: str,
    *,
    api_keys: str | list[str] | tuple[str, ...] | None,
    basic_auth: str
    | tuple[str, str]
    | list[tuple[str, str]]
    | dict[str, str]
    | None = None,
    allow_insecure_public_bind: bool,
) -> None:
    """Refuse non-loopback binds without auth unless explicitly overridden."""
    if _is_loopback_host(host):
        return
    if _has_auth(api_keys, basic_auth):
        return
    if allow_insecure_public_bind:
        return
    raise ValidationError(
        f"Refusing to bind managed serving to non-loopback host {host!r} without "
        "api_keys or basic_auth. Pass api_keys= / basic_auth= (or CLI --api-key / "
        "--basic-auth) for public/non-localhost binds, or set "
        "allow_insecure_public_bind=True / --allow-insecure-public-bind to "
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
    path: str | Path | None = None,
    *,
    kind: BundleKind | Literal["pipeline", "torchscript"] = "pipeline",
    host: str = "127.0.0.1",
    port: int = 8080,
    title: str = "BuildML Serve",
    blocking: bool = False,
    map_location: str = "cpu",
    api_keys: str | list[str] | tuple[str, ...] | None = None,
    basic_auth: str
    | tuple[str, str]
    | list[tuple[str, str]]
    | dict[str, str]
    | None = None,
    docs_enabled: bool | None = None,
    allow_insecure_public_bind: bool = False,
    ssl_certfile: str | Path | None = None,
    ssl_keyfile: str | Path | None = None,
    trusted: bool = False,
    config: Any | None = None,
) -> ServeHandle:
    """Load a bundle and start serving it, refusing the unsafe configurations.

    Validates the bind target, refuses a public bind without auth, confirms the
    port is free, loads the artifact, and starts uvicorn. In the default
    non-blocking mode it waits until the server reports itself started before
    returning, so a successful return means the endpoint is live rather than
    merely scheduled.

    The checks run in order of how cheap they are to fail. A bad port number is
    rejected before the artifact is read; a taken port is found before the model
    is loaded into memory.

    Parameters
    ----------
    path:
        The pipeline bundle directory or TorchScript file to serve. Optional
        when ``config.bundle`` is set.
    kind:
        ``'pipeline'`` or ``'torchscript'``.
    host:
        The bind address, defaulting to loopback. Anything else triggers the
        security check below.
    port:
        The TCP port, 1 to 65535. Tested for availability before loading, so a
        clash fails in a second rather than after the model is read.
    title:
        The name shown in the docs page and health response.
    blocking:
        Run uvicorn on the calling thread and never return until it stops. Use
        this in a container, where the process must stay alive. The default runs
        in a background thread, which keeps a notebook usable.
    map_location:
        Device placement for TorchScript.
    api_keys:
        One key or several, enabling the Bearer and ``X-API-Key`` middleware.
        Either API keys or Basic auth satisfy non-loopback bind requirements.
    basic_auth:
        HTTP Basic credentials (``'user:pass'``, pair, or mapping). Either
        Basic or API-key may authorize a request when both are configured.
    docs_enabled:
        OpenAPI/docs toggle. ``None`` auto-closes docs when auth is enabled.
    allow_insecure_public_bind:
        Bind a public address with no auth at all. Named at length because it
        should be conspicuous in review. Only defensible when something in front
        (a proxy, a service mesh) is doing the authentication.
    ssl_certfile:
        A PEM certificate for local HTTPS. Must be paired with ``ssl_keyfile``.
    ssl_keyfile:
        The matching PEM private key. Must be paired with ``ssl_certfile``.
    trusted:
        Must be ``True`` to deserialize the artifact (pickle/joblib/TorchScript).
    config:
        Optional :class:`~buildml.serving.config.ServeConfig` base. Explicit
        kwargs overlay it.

    Returns
    -------
    ServeHandle
        The running server. Keep it: it is the only way to stop the thread.

    Raises
    ------
    ValidationError
        If the host is empty or the port out of range; if a non-loopback host is
        requested without auth and without the override; if only one of the two
        TLS files is given, or either does not exist; or if the artifact cannot
        be loaded.
    MissingExtraError
        If uvicorn or FastAPI is missing. Install with ``pip install
        'buildml[serve]'``.
    ServingLaunchError
        If the port cannot be bound, or the server thread exits during the
        ten-second startup window: usually a failure inside uvicorn's own
        startup, such as an unreadable certificate.

    Notes
    -----
    **A non-loopback bind without auth is refused, not warned about.** An
    unauthenticated prediction endpoint on ``0.0.0.0`` is reachable from
    anywhere that can route to the host, and a warning in a log is not a
    control. Supply ``api_keys`` / ``basic_auth``, or state the override
    explicitly.

    **The TLS support here is genuinely local.** Uvicorn will terminate HTTPS
    with the PEM files you give it, and that is all: no certificate issuance,
    renewal, or rotation. For anything that outlives an afternoon, terminate TLS
    at a proxy.

    **Local HTTPS with a self-signed certificate needs client cooperation.**
    Most clients reject it by default; ``verify=False`` is the usual local
    workaround and should not follow the code anywhere else.

    Examples
    --------
    A development server on localhost::

        handle = serve_bundle("artifacts/churn-pipeline", port=8123, trusted=True)
        try:
            print(handle.url)
        finally:
            handle.stop()

    Reachable from other hosts, and therefore authenticated::

        handle = serve_bundle(
            "artifacts/churn-pipeline",
            host="0.0.0.0",
            port=8080,
            api_keys=["rotate-me-2026-q1"],
            trusted=True,
        )

    In a container, holding the process open::

        serve_bundle("/models/churn", host="0.0.0.0",
                     api_keys=os.environ["SERVE_KEY"], blocking=True,
                     trusted=True)

    See Also
    --------
    buildml.serving.app.create_serving_app : Building the app without serving.
    buildml.serving.config.ServeConfig : Declarative YAML/env/CLI config.
    """
    if config is not None:
        from buildml.serving.config import ServeConfig

        if not isinstance(config, ServeConfig):
            raise ValidationError("config must be a ServeConfig instance")
        overlay: dict[str, Any] = {}
        if path is not None:
            overlay["bundle"] = str(path)
        if kind != "pipeline" or config.kind == "pipeline":
            overlay["kind"] = kind
        if host != "127.0.0.1" or config.host == "127.0.0.1":
            overlay["host"] = host
        if port != 8080 or config.port == 8080:
            overlay["port"] = port
        if title != "BuildML Serve" or config.title == "BuildML Serve":
            overlay["title"] = title
        if map_location != "cpu" or config.map_location == "cpu":
            overlay["map_location"] = map_location
        if api_keys is not None:
            overlay["api_keys"] = api_keys
        if basic_auth is not None:
            overlay["basic_auth"] = basic_auth
        if docs_enabled is not None:
            overlay["docs_enabled"] = docs_enabled
        if allow_insecure_public_bind:
            overlay["allow_insecure_public_bind"] = True
        if ssl_certfile is not None:
            overlay["ssl_certfile"] = str(ssl_certfile)
        if ssl_keyfile is not None:
            overlay["ssl_keyfile"] = str(ssl_keyfile)
        if trusted:
            overlay["trusted"] = True
        elif config.trusted:
            overlay["trusted"] = True
        # Prefer config host/port/title/kind when kwargs left at defaults.
        cfg = config.merge(**overlay)
        if host == "127.0.0.1" and config.host != "127.0.0.1":
            cfg = cfg.merge(host=config.host)
        if port == 8080 and config.port != 8080:
            cfg = cfg.merge(port=config.port)
        if title == "BuildML Serve" and config.title != "BuildML Serve":
            cfg = cfg.merge(title=config.title)
        if kind == "pipeline" and config.kind != "pipeline":
            cfg = cfg.merge(kind=config.kind)
        if map_location == "cpu" and config.map_location != "cpu":
            cfg = cfg.merge(map_location=config.map_location)
        if not allow_insecure_public_bind and config.allow_insecure_public_bind:
            cfg = cfg.merge(allow_insecure_public_bind=True)
        if ssl_certfile is None and config.ssl_certfile:
            cfg = cfg.merge(ssl_certfile=config.ssl_certfile)
        if ssl_keyfile is None and config.ssl_keyfile:
            cfg = cfg.merge(ssl_keyfile=config.ssl_keyfile)
        if docs_enabled is None and config.docs_enabled is not None:
            cfg = cfg.merge(docs_enabled=config.docs_enabled)
        if api_keys is None and config.api_keys:
            cfg = cfg.merge(api_keys=list(config.api_keys))
        if basic_auth is None and config.basic_auth is not None:
            cfg = cfg.merge(basic_auth=config.basic_auth)
        path = cfg.require_bundle()
        kind = cfg.kind  # type: ignore[assignment]
        host = cfg.host
        port = cfg.port
        title = cfg.title
        map_location = cfg.map_location
        api_keys = list(cfg.api_keys) if cfg.api_keys else None
        basic_auth = cfg.basic_auth
        docs_enabled = cfg.docs_enabled
        allow_insecure_public_bind = cfg.allow_insecure_public_bind
        ssl_certfile = cfg.ssl_certfile
        ssl_keyfile = cfg.ssl_keyfile
        trusted = cfg.trusted
    elif path is None:
        raise ValidationError(
            "serve_bundle requires path= or a ServeConfig with bundle set"
        )

    _validate_bind_target(host, port)
    _ensure_bind_security(
        host,
        api_keys=api_keys,
        basic_auth=basic_auth,
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
        basic_auth=basic_auth,
        docs_enabled=docs_enabled,
        trusted=trusted,
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
