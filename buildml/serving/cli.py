"""CLI entry point for ``buildml-serve`` / ``python -m buildml.serving``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
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
