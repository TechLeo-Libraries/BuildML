#!/usr/bin/env python3
"""Wrap mixin property returns of self._attr in cast(ReturnType, ...)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

MIXINS = Path(__file__).resolve().parents[1] / "buildml" / "session" / "mixins"


def _is_property(node: ast.FunctionDef) -> bool:
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        name = getattr(target, "attr", None) or getattr(target, "id", "")
        if name in {"property", "cached_property"}:
            return True
    return False


def process(path: Path) -> None:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    edits: list[tuple[int, str]] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for child in node.body:
            if not isinstance(child, ast.FunctionDef) or not _is_property(child):
                continue
            if child.returns is None:
                continue
            ann = ast.unparse(child.returns)
            for stmt in child.body:
                if not isinstance(stmt, ast.Return) or stmt.value is None:
                    continue
                line = lines[stmt.lineno - 1]
                if "cast(" in line:
                    continue
                indent = re.match(r"^(\s*)", line).group(1)  # type: ignore[union-attr]
                expr = line.strip()[len("return ") :]
                edits.append((stmt.lineno - 1, f"{indent}return cast({ann}, {expr})\n"))
    for idx, new in sorted(edits, key=lambda x: x[0], reverse=True):
        lines[idx] = new
    if "cast" not in src and edits:
        src2 = "".join(lines)
        if "from typing import" in src2:
            src2 = src2.replace("from typing import", "from typing import cast,", 1)
        else:
            src2 = src2.replace(
                "from __future__ import annotations\n",
                "from __future__ import annotations\n\nfrom typing import cast\n",
                1,
            )
        path.write_text(src2, encoding="utf-8", newline="\n")
    elif edits:
        path.write_text("".join(lines), encoding="utf-8", newline="\n")
    if edits:
        print(path.name, len(edits))


def main() -> int:
    for path in sorted(MIXINS.glob("*.py")):
        if path.name.startswith("_"):
            continue
        process(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
