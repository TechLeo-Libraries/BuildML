#!/usr/bin/env python3
"""Quote cast() type arguments so TYPE_CHECKING-only names work at runtime."""

from __future__ import annotations

import ast
import re
from pathlib import Path

MIXINS = Path(__file__).resolve().parents[1] / "buildml" / "session" / "mixins"


def repair_broken_quotes(src: str) -> str:
    """Fix earlier bad quoting that split on commas inside generics."""
    # cast("dict[str", Any], -> cast("dict[str, Any]",
    src = re.sub(
        r'cast\("dict\[str",\s*Any\]\s*,',
        'cast("dict[str, Any]",',
        src,
    )
    # cast("AdvisorResult | PlanResult | ExecutorResult | PlanExecutionResult | None",  already ok
    # cast("list[str", Path] or similar — general: cast("X", Y], -> cast("X, Y]",
    src = re.sub(
        r'cast\("([^"]+)",\s*([A-Za-z_][\w\.]*)\]\s*,',
        r'cast("\1, \2]",',
        src,
    )
    return src


def quote_casts(src: str) -> str:
    out: list[str] = []
    i = 0
    n = len(src)
    while i < n:
        j = src.find("cast(", i)
        if j == -1:
            out.append(src[i:])
            break
        out.append(src[i:j])
        k = j + 5  # after cast(
        while k < n and src[k].isspace():
            k += 1
        if k < n and src[k] in "'\"":
            # already quoted — copy until we finish this cast( call's type arg comma
            quote = src[k]
            k += 1
            while k < n:
                if src[k] == "\\" and k + 1 < n:
                    k += 2
                    continue
                if src[k] == quote:
                    k += 1
                    break
                k += 1
            while k < n and src[k].isspace():
                k += 1
            if k < n and src[k] == ",":
                k += 1
            out.append(src[j:k])
            i = k
            continue
        start = k
        depth = 0
        while k < n:
            ch = src[k]
            if ch in "[(":
                depth += 1
            elif ch in "])":
                depth = max(0, depth - 1)
            elif ch == "," and depth == 0:
                break
            k += 1
        typ = src[start:k].strip()
        typ_q = typ.replace("\\", "\\\\").replace('"', '\\"')
        out.append(f'cast("{typ_q}",')
        if k < n and src[k] == ",":
            k += 1
        i = k
    return "".join(out)


def main() -> int:
    for path in sorted(MIXINS.glob("*.py")):
        if path.name.startswith("_"):
            continue
        src = path.read_text(encoding="utf-8")
        src = repair_broken_quotes(src)
        new = quote_casts(src)
        try:
            ast.parse(new)
        except SyntaxError as exc:
            print("FAIL", path.name, exc)
            continue
        path.write_text(new, encoding="utf-8", newline="\n")
        print("ok", path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
