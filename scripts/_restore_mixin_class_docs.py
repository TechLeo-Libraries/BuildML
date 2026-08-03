#!/usr/bin/env python3
"""Restore class docstrings displaced below TYPE_CHECKING attr blocks."""

from __future__ import annotations

import re
from pathlib import Path

MIXINS = Path(__file__).resolve().parents[1] / "buildml" / "session" / "mixins"

PATTERN = re.compile(
    r"(class (?P<name>\w+SessionMixin):\n)"
    r"(?P<attrs>    # mypy: session private attrs \(owned by Session\.__init__\)\n"
    r"    if TYPE_CHECKING:\n"
    r"(?:        _\w+: Any\n)+)"
    r'    """(?P<doc>[^"]*)"""\n',
    re.MULTILINE,
)


def main() -> int:
    for path in sorted(MIXINS.glob("*.py")):
        if path.name.startswith("_"):
            continue
        src = path.read_text(encoding="utf-8")
        new, n = PATTERN.subn(
            lambda m: (
                f"{m.group(1)}"
                f'    """{m.group("doc")}"""\n'
                f"{m.group('attrs')}"
            ),
            src,
            count=1,
        )
        if n:
            path.write_text(new, encoding="utf-8", newline="\n")
            print("restored", path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
