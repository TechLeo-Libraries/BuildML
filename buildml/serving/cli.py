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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    from buildml.serving.launch import serve_bundle

    print(
        f"Starting BuildML serve kind={args.kind} bundle={args.bundle} "
        f"at http://{args.host}:{args.port} (no auth; localhost-oriented)",
        file=sys.stderr,
    )
    serve_bundle(
        args.bundle,
        kind=args.kind,
        host=args.host,
        port=args.port,
        title=args.title,
        blocking=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
