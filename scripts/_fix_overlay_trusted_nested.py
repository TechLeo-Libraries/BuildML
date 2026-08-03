#!/usr/bin/env python3
"""Fix overlays where trusted _p was nested as the path _p default argument."""

from __future__ import annotations

import re
from pathlib import Path

OVERLAYS = Path(__file__).resolve().parents[1] / "buildml" / "explain" / "overlays"

PATTERN = re.compile(
    r'parameters=\(\s*_p\(\s*"path",\s*"str \| Path",\s*"(?P<desc>[^"]*)",\s*'
    r'_p\(\s*"trusted",\s*"bool",\s*"[^"]*",\s*required=False\s*\)\s*,?\s*\)\s*,?\s*\)\s*,?',
    re.DOTALL,
)

REPL = (
    "parameters=(\n"
    '            _p("path", "str | Path", "{desc}", required=True),\n'
    "            _p(\n"
    '                "trusted",\n'
    '                "bool",\n'
    '                "Must be True to deserialize pickle/joblib/torch payloads '
    '(default False).",\n'
    "                False,\n"
    "            ),\n"
    "        ),"
)


def main() -> int:
    for path in sorted(OVERLAYS.glob("*.py")):
        src = path.read_text(encoding="utf-8")
        new, n = PATTERN.subn(lambda m: REPL.format(desc=m.group("desc")), src)
        if n:
            compile(new, str(path), "exec")
            path.write_text(new, encoding="utf-8", newline="\n")
            print(f"fixed {path.name} ({n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
