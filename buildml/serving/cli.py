"""Serve a bundle from a shell, for containers and one-off deployments.

The command-line face of :func:`~buildml.serving.launch.serve_bundle`. Every
flag maps to a keyword argument there, including the security ones, so the same
refusal to bind a public address without keys applies from the shell.

Always blocking, because that is what a container entrypoint or a systemd unit
needs: the process must stay in the foreground so the supervisor can see it
running and restart it when it stops::

    buildml-serve --bundle artifacts/churn --port 8080
    python -m buildml.serving --bundle artifacts/churn --api-key "$SERVE_KEY"

See Also
--------
buildml.serving.launch.serve_bundle : The function underneath, for use in code.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser, exposed separately so it can be tested.

    Kept apart from :func:`main` so tests can parse arguments and assert on the
    result without starting a server, and so ``--help`` can be rendered by
    documentation tooling.

    Returns
    -------
    argparse.ArgumentParser
        The parser for ``buildml-serve``, with every flag documented in its own
        help text.

    Notes
    -----
    **``--api-key`` is repeatable**, collecting into a list, which is how key
    rotation is done: add the new key, deploy, remove the old one.

    See Also
    --------
    main : Where the parsed arguments are used.
    """
    parser = argparse.ArgumentParser(
        prog="buildml-serve",
        description=(
            "BuildML managed model server (alpha). Loads a classical pipeline "
            "bundle or TorchScript file and exposes /health + /predict. "
            "Binds to 127.0.0.1 by default; no authentication. Put a reverse "
            "proxy in front for non-local exposure."
        ),
    )
    parser.add_argument(
        "--bundle",
        required=True,
        help="Path to a classical pipeline bundle directory or TorchScript file.",
    )
    parser.add_argument(
        "--kind",
        choices=("pipeline", "torchscript"),
        default="pipeline",
        help="Bundle kind (default: pipeline).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Bind port (default: 8080).",
    )
    parser.add_argument(
        "--title",
        default="BuildML Serve",
        help="Service title shown in /health.",
    )
    parser.add_argument(
        "--api-key",
        action="append",
        default=None,
        dest="api_keys",
        help=(
            "Optional API key (repeatable). Enables Bearer / X-API-Key auth on "
            "/predict. Required for non-loopback binds unless "
            "--allow-insecure-public-bind is set. Still not a managed IAM product; "
            "prefer TLS at a reverse proxy."
        ),
    )
    parser.add_argument(
        "--allow-insecure-public-bind",
        action="store_true",
        default=False,
        help=(
            "Dangerous override: allow binding 0.0.0.0 / other non-loopback hosts "
            "without --api-key. Prefer --api-key + reverse-proxy TLS instead."
        ),
    )
    parser.add_argument(
        "--ssl-certfile",
        default=None,
        help="Optional PEM certificate for local HTTPS (requires --ssl-keyfile).",
    )
    parser.add_argument(
        "--ssl-keyfile",
        default=None,
        help="Optional PEM private key for local HTTPS (pairs with --ssl-certfile).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, announce the configuration, and serve until stopped.

    Prints a one-line summary to stderr before starting — the scheme, address,
    bundle, and whether authentication is on. That last part is the reason the
    line exists: a server started without keys should say so where an operator
    will see it, not only in a docstring.

    Parameters
    ----------
    argv:
        Arguments to parse. ``None`` reads ``sys.argv``, which is the normal
        path; pass a list in tests.

    Returns
    -------
    int
        ``0`` after the server stops cleanly. In practice this only returns when
        the process is being shut down, since serving is blocking.

    Raises
    ------
    SystemExit
        From argparse, on ``--help`` or a malformed argument list.
    ValidationError
        If the configuration is refused — most often a public bind without
        ``--api-key``.
    MissingExtraError
        If the serving extra is not installed.
    ServingLaunchError
        If the port is unavailable.

    Notes
    -----
    **The summary goes to stderr, not stdout**, so it does not contaminate a
    pipeline that captures stdout, and so it appears in container logs
    interleaved with uvicorn's own output.

    See Also
    --------
    build_parser : The flags this accepts.
    """
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    from buildml.serving.launch import serve_bundle

    auth_note = "api-key auth on" if args.api_keys else "auth off (localhost-oriented)"
    tls = bool(args.ssl_certfile and args.ssl_keyfile)
    scheme = "https" if tls else "http"
    print(
        f"Starting BuildML serve kind={args.kind} bundle={args.bundle} "
        f"at {scheme}://{args.host}:{args.port} ({auth_note})",
        file=sys.stderr,
    )
    serve_bundle(
        args.bundle,
        kind=args.kind,
        host=args.host,
        port=args.port,
        title=args.title,
        blocking=True,
        api_keys=args.api_keys,
        allow_insecure_public_bind=args.allow_insecure_public_bind,
        ssl_certfile=args.ssl_certfile,
        ssl_keyfile=args.ssl_keyfile,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
