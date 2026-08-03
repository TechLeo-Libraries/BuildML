#!/usr/bin/env python3
"""Remove mistakenly injected trusted params from non-pickle load_* overlays."""

from __future__ import annotations

import re
from pathlib import Path

OVERLAYS = Path(__file__).resolve().parents[1] / "buildml" / "explain" / "overlays"

# Nested broken form: required=True,\n            _p("trusted"...),
NESTED = re.compile(
    r'(required=True)\s*,\s*_p\(\s*"trusted",\s*"bool",\s*"[^"]*",\s*required=False\s*\)\s*,?',
    re.DOTALL,
)

# Standalone trusted tuple entries that shouldn't be on non-bundle ops — handled per-file after parse fails.


def main() -> int:
    for path in sorted(OVERLAYS.glob("*.py")):
        src = path.read_text(encoding="utf-8")
        new = NESTED.sub(r"\1", src)
        # Remove empty double commas / leftover ),), from nesting
        new = re.sub(r"required=True\),", "required=True),", new)
        try:
            compile(new, str(path), "exec")
        except SyntaxError as exc:
            print("still broken", path.name, exc)
            continue
        if new != src:
            path.write_text(new, encoding="utf-8", newline="\n")
            print("scrubbed", path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
