#!/usr/bin/env python3
"""Shorten Returns bodies on already-shrunk Session mixin facades."""

from __future__ import annotations

import ast
import re
from pathlib import Path

MIXINS = Path(__file__).resolve().parents[1] / "buildml" / "session" / "mixins"


def shorten_returns(doc: str) -> str:
    if "Session facade over" not in doc and "Canonical Parameters" not in doc:
        return doc
    lines = doc.splitlines()
    out: list[str] = []
    i = 0
    section_names = {
        "Parameters",
        "Raises",
        "Notes",
        "Examples",
        "See Also",
        "Warns",
        "Yields",
    }
    while i < len(lines):
        if (
            lines[i].strip() == "Returns"
            and i + 1 < len(lines)
            and set(lines[i + 1].strip()) == {"-"}
        ):
            out.append(lines[i])
            out.append(lines[i + 1])
            i += 2
            type_line = None
            desc_line = None
            while i < len(lines):
                s = lines[i].strip()
                if (
                    s in section_names
                    and i + 1 < len(lines)
                    and set(lines[i + 1].strip()) <= {"-"}
                ):
                    break
                if not s:
                    i += 1
                    continue
                if type_line is None:
                    type_line = lines[i]
                elif desc_line is None and (
                    lines[i].startswith(" ") or lines[i].startswith("\t")
                ):
                    desc_line = lines[i]
                i += 1
            if type_line:
                out.append(type_line)
            if desc_line:
                text = desc_line.strip()
                if len(text) > 90:
                    text = text[:87] + "..."
                indent = re.match(r"^(\s*)", desc_line).group(1)  # type: ignore[union-attr]
                out.append(indent + text)
            out.append("")
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out).rstrip() + "\n"


def main() -> int:
    changed = 0
    for path in sorted(MIXINS.glob("*.py")):
        if path.name.startswith("_"):
            continue
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
        reps: list[tuple[ast.FunctionDef, str]] = []
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            for child in node.body:
                if not isinstance(child, ast.FunctionDef):
                    continue
                doc = ast.get_docstring(child)
                if not doc or "Session facade over" not in doc:
                    continue
                new = shorten_returns(doc)
                if new.strip() != doc.strip():
                    reps.append((child, new))
        if not reps:
            continue
        lines = src.splitlines(keepends=True)
        for child, new in sorted(reps, key=lambda x: x[0].lineno, reverse=True):
            dn = child.body[0]
            indent = re.match(r"^(\s*)", lines[dn.lineno - 1]).group(1)  # type: ignore[union-attr]
            doc_lines = new.strip("\n").splitlines() or [""]
            if len(doc_lines) == 1:
                block = f'{indent}"""{doc_lines[0]}"""\n'
            else:
                block = f'{indent}"""{doc_lines[0]}\n'
                for dl in doc_lines[1:]:
                    block += f"{indent}{dl}\n" if dl else "\n"
                block += f'{indent}"""\n'
            lines = lines[: dn.lineno - 1] + [block] + lines[dn.end_lineno :]
        path.write_text("".join(lines), encoding="utf-8", newline="\n")
        changed += len(reps)
    print(f"shortened returns on {changed} facades")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
