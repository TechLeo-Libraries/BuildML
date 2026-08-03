#!/usr/bin/env python3
"""Repair broken trusted-parameter injections in explain overlays."""

from __future__ import annotations

import re
from pathlib import Path

OVERLAYS = Path(__file__).resolve().parents[1] / "buildml" / "explain" / "overlays"

TRUSTED_LINE = (
    '_p(\n'
    '                "trusted",\n'
    '                "bool",\n'
    '                "Must be True to deserialize pickle/joblib/torch payloads '
    '(default False).",\n'
    '                False,\n'
    '            )'
)


def repair(src: str) -> str:
    # Collapse the broken nested _p(..., _p("trusted"...)) pattern.
    # Match from parameters=(_p("path"... through the broken trusted injection.
    pattern = re.compile(
        r'parameters=\(\s*_p\(\s*"path",\s*"str \| Path",\s*"(?P<desc>[^"]*)",\s*'
        r'required=True\s*,\s*'
        r'_p\(\s*"trusted",\s*"bool",\s*"[^"]*",\s*required=False\s*\)\s*,?\s*\)\s*,?\s*\)\s*,?',
        re.DOTALL,
    )

    def repl(m: re.Match[str]) -> str:
        desc = m.group("desc")
        return (
            "parameters=(\n"
            f'            _p("path", "str | Path", "{desc}", required=True),\n'
            f"            {TRUSTED_LINE},\n"
            "        ),"
        )

    return pattern.sub(repl, src)


def main() -> int:
    for path in sorted(OVERLAYS.glob("*.py")):
        src = path.read_text(encoding="utf-8")
        if "trusted" not in src or "required=False" not in src:
            # still try if broken
            if '_p("trusted"' not in src and "\"trusted\"" not in src:
                continue
        new = repair(src)
        try:
            compile(new, str(path), "exec")
        except SyntaxError as exc:
            print("STILL BROKEN", path.name, exc)
            continue
        if new != src:
            path.write_text(new, encoding="utf-8", newline="\n")
            print("fixed", path.name)
        else:
            # try compile original
            try:
                compile(src, str(path), "exec")
            except SyntaxError as exc:
                print("UNFIXED", path.name, exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
