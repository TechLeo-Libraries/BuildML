#!/usr/bin/env python3
"""Insert trusted parameter docs into load_* functions that gained the kwarg."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "buildml"

TRUSTED_DOC = (
    "trusted:\n"
    "    Must be ``True`` to deserialize pickle/joblib/torch payloads. Pass\n"
    "    only for artifacts you created or fully trust. Defaults to ``False``.\n"
)


def needs_trusted_doc(doc: str, args: list[str]) -> bool:
    if "trusted" not in args:
        return False
    return not re.search(r"(?m)^trusted\s*:", doc or "")


def inject(doc: str) -> str:
    if re.search(r"(?m)^trusted\s*:", doc):
        return doc
    if "Parameters" not in doc:
        # Insert before Returns
        insert = "\nParameters\n----------\n" + TRUSTED_DOC
        for marker in ("\nReturns\n", "\nRaises\n", "\nNotes\n"):
            if marker in doc:
                return doc.replace(marker, insert + marker, 1)
        return doc.rstrip() + insert
    lines = doc.splitlines()
    out: list[str] = []
    i = 0
    injected = False
    while i < len(lines):
        out.append(lines[i])
        if (
            not injected
            and lines[i].strip() == "Parameters"
            and i + 1 < len(lines)
            and set(lines[i + 1].strip()) == {"-"}
        ):
            out.append(lines[i + 1])
            # append trusted after the underline; keep existing params
            i += 2
            # find end of parameters to append at end of section? Prefer after path:
            # Insert immediately after underline, then continue — actually after path is nicer.
            # Collect remaining param section and inject after first param block.
            rest: list[str] = []
            while i < len(lines):
                s = lines[i].strip()
                if s in {
                    "Returns",
                    "Raises",
                    "Notes",
                    "Examples",
                    "See Also",
                    "Warns",
                    "Yields",
                } and i + 1 < len(lines) and set(lines[i + 1].strip()) <= {"-"}:
                    break
                rest.append(lines[i])
                i += 1
            # Insert trusted before section ends (after existing params)
            # Ensure blank line separation
            while rest and not rest[-1].strip():
                rest.pop()
            out.extend(rest)
            out.append("trusted:")
            out.append(
                "    Must be ``True`` to deserialize pickle/joblib/torch payloads. Pass"
            )
            out.append(
                "    only for artifacts you created or fully trust. Defaults to ``False``."
            )
            out.append("")
            injected = True
            continue
        i += 1
    text = "\n".join(out)
    return text + ("\n" if doc.endswith("\n") else "")


def patch_file(path: Path) -> bool:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    reps: list[tuple[ast.FunctionDef, str]] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if not node.name.startswith("load_"):
            continue
        args = [a.arg for a in (*node.args.args, *node.args.kwonlyargs)]
        doc = ast.get_docstring(node)
        if not needs_trusted_doc(doc or "", args):
            continue
        new_doc = inject(doc or "")
        reps.append((node, new_doc))
    if not reps:
        return False
    lines = src.splitlines(keepends=True)
    for node, new_doc in sorted(reps, key=lambda x: x[0].lineno, reverse=True):
        if not (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        ):
            continue
        dn = node.body[0]
        indent_m = re.match(r"^(\s*)", lines[dn.lineno - 1])
        indent = indent_m.group(1) if indent_m else ""
        doc_lines = new_doc.strip("\n").splitlines() or [""]
        if len(doc_lines) == 1:
            block = f'{indent}"""{doc_lines[0]}"""\n'
        else:
            block = f'{indent}"""{doc_lines[0]}\n'
            for dl in doc_lines[1:]:
                block += f"{indent}{dl}\n" if dl else "\n"
            block += f'{indent}"""\n'
        lines = lines[: dn.lineno - 1] + [block] + lines[dn.end_lineno :]
    path.write_text("".join(lines), encoding="utf-8", newline="\n")
    return True


def main() -> int:
    n = 0
    for path in sorted(ROOT.rglob("*.py")):
        if path.name.startswith("_"):
            continue
        if patch_file(path):
            print(path.relative_to(ROOT.parent))
            n += 1
    print(f"patched {n} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
