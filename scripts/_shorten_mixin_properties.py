#!/usr/bin/env python3
"""Shorten long property docstrings on Session mixins (keep summary + None note)."""

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


def _short_property_doc(node: ast.FunctionDef, doc: str) -> str:
    if len(doc) < 280:
        return doc
    summary = next((ln.strip() for ln in doc.splitlines() if ln.strip()), "Session property.")
    if len(summary.split()) < 4:
        summary = f"Return the cached ``{node.name}`` value for this Session."
    ann = ast.unparse(node.returns) if node.returns else ""
    optional = ann.endswith("| None") or ann.startswith("Optional[")
    lines = [summary, "", f"Session-held result for ``{node.name}``."]
    if optional or "None" in doc:
        lines.append(
            "``None`` until the matching Session fit/score/load call populates it."
        )
    return "\n".join(lines) + "\n"


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
                if not isinstance(child, ast.FunctionDef) or not _is_property(child):
                    continue
                doc = ast.get_docstring(child)
                if not doc:
                    continue
                new = _short_property_doc(child, doc)
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
        print(f"{path.name}: shortened {len(reps)} properties")
    print(f"TOTAL properties shortened={changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
