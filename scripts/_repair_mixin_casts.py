#!/usr/bin/env python3
"""Repair broken cast() wraps on multi-line mixin returns."""

from __future__ import annotations

import re
from pathlib import Path

MIXINS = Path(__file__).resolve().parents[1] / "buildml" / "session" / "mixins"

# Pattern: return cast(Type, call(\n  args...\n)
BROKEN = re.compile(
    r"return cast\((?P<ann>[A-Za-z0-9_\[\]\|\s\.,\'\" ]+), (?P<call>[A-Za-z0-9_\.]+)\(\)\n"
    r"(?P<body>(?:^[ \t]+.+\n)+)"
    r"(?P<close>^[ \t]+\))\n",
    re.MULTILINE,
)


def repair(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        ann = m.group("ann").strip()
        call = m.group("call")
        body = m.group("body")
        close = m.group("close")
        return f"return cast({ann}, {call}(\n{body}{close})\n"

    prev = None
    while prev != text:
        prev = text
        text = BROKEN.sub(repl, text)
    return text


def main() -> int:
    for path in sorted(MIXINS.glob("*.py")):
        if path.name.startswith("_"):
            continue
        src = path.read_text(encoding="utf-8")
        new = repair(src)
        if new != src:
            path.write_text(new, encoding="utf-8", newline="\n")
            print("repaired", path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
