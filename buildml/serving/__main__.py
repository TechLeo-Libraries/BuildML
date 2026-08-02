"""``python -m buildml.serving`` entry point."""

from __future__ import annotations

from buildml.serving.cli import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
