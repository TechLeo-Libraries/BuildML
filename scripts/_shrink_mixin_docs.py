#!/usr/bin/env python3
"""One-shot: move long mixin docstrings to ops; leave short facade docs on mixins.

Not a CI tool — run manually during the navigation/docstring migration.

Usage
-----
::

    python scripts/_shrink_mixin_docs.py          # dry-run counts
    python scripts/_shrink_mixin_docs.py --write  # apply
"""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIXINS = ROOT / "buildml" / "session" / "mixins"
SESSION = ROOT / "buildml" / "session"

OPS_CALL_RE = re.compile(r"(?P<mod>\w+_ops)\.(?P<fn>\w+)\s*\(")


@dataclass
class MixinRewrite:
    path: Path
    lineno: int
    end_lineno: int
    has_docstring: bool
    insert_lineno: int
    indent: str
    new_doc: str


@dataclass
class OpsEnrich:
    path: Path
    lineno: int
    end_lineno: int
    has_docstring: bool
    insert_lineno: int
    indent: str
    new_doc: str


def _is_property(node: ast.FunctionDef) -> bool:
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        name = getattr(target, "attr", None) or getattr(target, "id", "")
        if name in {"property", "cached_property", "setter", "deleter"}:
            return True
    return False


def _summary_line(doc: str) -> str:
    for line in doc.strip().splitlines():
        s = line.strip()
        if s:
            return s
    return "Session facade method."


def _return_annotation(node: ast.FunctionDef) -> str | None:
    if node.returns is None:
        return None
    return ast.unparse(node.returns)


def _extract_section(doc: str, name: str) -> str:
    lines = doc.splitlines()
    section_names = {
        "Parameters",
        "Returns",
        "Raises",
        "Notes",
        "Examples",
        "See Also",
        "Warns",
        "Yields",
        "References",
        "Attributes",
        "Other Parameters",
    }
    for i, line in enumerate(lines):
        if line.strip() != name:
            continue
        if i + 1 >= len(lines) or set(lines[i + 1].strip()) != {"-"}:
            continue
        body: list[str] = []
        j = i + 2
        while j < len(lines):
            nxt = lines[j].strip()
            if (
                nxt in section_names
                and j + 1 < len(lines)
                and set(lines[j + 1].strip()) <= {"-"}
                and len(lines[j + 1].strip()) >= len(nxt)
            ):
                break
            body.append(lines[j])
            j += 1
        text = "\n".join(body).rstrip()
        if text.strip():
            return f"{name}\n{'-' * len(name)}\n{text}\n"
    return ""


def _returns_section(node: ast.FunctionDef, mixin_doc: str) -> str:
    existing = _extract_section(mixin_doc, "Returns")
    if existing:
        return existing
    ann = _return_annotation(node)
    if ann is None or ann.strip() in {"None", "'None'", '"None"'}:
        return ""
    if "Session" in ann:
        return (
            "Returns\n-------\nSession\n"
            "    ``self`` for fluent chaining.\n"
        )
    if "Path" in ann:
        return (
            "Returns\n-------\npathlib.Path\n"
            "    Resolved path written or loaded.\n"
        )
    return f"Returns\n-------\n{ann}\n    Result of the underlying ops call.\n"


def _facade_doc(node: ast.FunctionDef, ops_mod: str, ops_fn: str, mixin_doc: str) -> str:
    summary = _summary_line(mixin_doc)
    if len(summary.split()) < 4:
        summary = f"Session facade for ``{ops_mod}.{ops_fn}``."
    returns = _returns_section(node, mixin_doc)
    body = (
        f"{summary}\n\n"
        f"Session facade over :func:`buildml.session.{ops_mod}.{ops_fn}`. "
        f"Canonical Parameters, Raises, Notes, and Examples live on that ops "
        f"function — keep this method as a thin delegate.\n\n"
    )
    if returns:
        body += returns + "\n"
    body += (
        "See Also\n"
        "--------\n"
        f":func:`buildml.session.{ops_mod}.{ops_fn}`\n"
        "    Canonical documentation for parameters, raises, and examples.\n"
    )
    return body


def _ensure_session_param(doc: str) -> str:
    if re.search(r"(?m)^session\s*:", doc):
        return doc if doc.endswith("\n") else doc + "\n"
    insert = (
        "\nParameters\n----------\n"
        "session:\n"
        "    Active Session instance this operation mutates or reads.\n"
    )
    if "Parameters" not in doc:
        for marker in (
            "\nReturns\n",
            "\nRaises\n",
            "\nNotes\n",
            "\nExamples\n",
            "\nSee Also\n",
        ):
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
            out.append("session:")
            out.append("    Active Session instance this operation mutates or reads.")
            injected = True
            i += 2
            continue
        i += 1
    text = "\n".join(out)
    return text + "\n"


def _doc_block(indent: str, new_doc: str) -> str:
    doc_lines = new_doc.strip("\n").splitlines() or [""]
    if len(doc_lines) == 1:
        return f'{indent}"""{doc_lines[0]}"""\n'
    block = f'{indent}"""{doc_lines[0]}\n'
    for dl in doc_lines[1:]:
        block += f"{indent}{dl}\n" if dl else "\n"
    block += f'{indent}"""\n'
    return block


def _apply_rewrites(source: str, rewrites: list[MixinRewrite] | list[OpsEnrich]) -> str:
    lines = source.splitlines(keepends=True)
    for rw in sorted(rewrites, key=lambda r: r.lineno, reverse=True):
        block = _doc_block(rw.indent, rw.new_doc)
        if rw.has_docstring:
            lines = lines[: rw.lineno - 1] + [block] + lines[rw.end_lineno :]
        else:
            lines = lines[: rw.insert_lineno - 1] + [block] + lines[rw.insert_lineno - 1 :]
    return "".join(lines)


def _doc_span(node: ast.FunctionDef, source_lines: list[str]) -> tuple[bool, int, int, int, str]:
    """Return has_doc, start_lineno, end_lineno, insert_lineno, indent."""
    if (
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ):
        doc_node = node.body[0]
        indent_m = re.match(r"^(\s*)", source_lines[doc_node.lineno - 1])
        indent = indent_m.group(1) if indent_m else "        "
        return True, doc_node.lineno, doc_node.end_lineno or doc_node.lineno, 0, indent
    # Insert before first body stmt.
    insert_at = node.body[0].lineno if node.body else node.lineno + 1
    raw = source_lines[insert_at - 1]
    indent_m = re.match(r"^(\s*)", raw)
    indent = indent_m.group(1) if indent_m else "        "
    return False, 0, 0, insert_at, indent


def _find_ops_target(node: ast.FunctionDef) -> tuple[str, str] | None:
    try:
        body_src = ast.unparse(node)
    except Exception:
        return None
    matches = list(OPS_CALL_RE.finditer(body_src))
    if not matches:
        return None
    m = matches[-1]
    return m.group("mod"), m.group("fn")


def collect_from_mixin(mixin_path: Path) -> tuple[list[MixinRewrite], list[OpsEnrich]]:
    source = mixin_path.read_text(encoding="utf-8")
    source_lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    mixin_rewrites: list[MixinRewrite] = []
    ops_enrich: list[OpsEnrich] = []

    # Load ops AST cache lazily.
    ops_asts: dict[str, tuple[Path, str, list[str], dict[str, ast.FunctionDef]]] = {}

    methods: list[ast.FunctionDef] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if (
                    isinstance(child, ast.FunctionDef)
                    and not child.name.startswith("_")
                    and not _is_property(child)
                ):
                    methods.append(child)

    for method in methods:
        target = _find_ops_target(method)
        if target is None:
            continue
        ops_mod, ops_fn = target
        mixin_doc = ast.get_docstring(method) or ""
        if "Canonical Parameters, Raises, Notes" in mixin_doc:
            continue
        facade = _facade_doc(method, ops_mod, ops_fn, mixin_doc)
        has_doc, start, end, insert_at, indent = _doc_span(method, source_lines)
        mixin_rewrites.append(
            MixinRewrite(
                path=mixin_path,
                lineno=start or method.lineno,
                end_lineno=end,
                has_docstring=has_doc,
                insert_lineno=insert_at,
                indent=indent,
                new_doc=facade,
            )
        )

        ops_path = SESSION / f"{ops_mod}.py"
        if not ops_path.exists():
            continue
        if ops_mod not in ops_asts:
            ops_src = ops_path.read_text(encoding="utf-8")
            ops_lines = ops_src.splitlines(keepends=True)
            ops_tree = ast.parse(ops_src)
            fns = {
                n.name: n
                for n in ops_tree.body
                if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")
            }
            ops_asts[ops_mod] = (ops_path, ops_src, ops_lines, fns)
        ops_path_c, _ops_src, ops_lines, fns = ops_asts[ops_mod]
        ops_node = fns.get(ops_fn)
        if ops_node is None:
            continue
        ops_doc = ast.get_docstring(ops_node) or ""
        if len(mixin_doc) <= len(ops_doc) + 80:
            continue
        moved = _ensure_session_param(mixin_doc.strip() + "\n")
        o_has, o_start, o_end, o_insert, o_indent = _doc_span(ops_node, ops_lines)
        ops_enrich.append(
            OpsEnrich(
                path=ops_path_c,
                lineno=o_start or ops_node.lineno,
                end_lineno=o_end,
                has_docstring=o_has,
                insert_lineno=o_insert,
                indent=o_indent,
                new_doc=moved,
            )
        )
    return mixin_rewrites, ops_enrich


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    write = "--write" in argv

    all_mixin: list[MixinRewrite] = []
    all_ops: list[OpsEnrich] = []
    for path in sorted(MIXINS.glob("*.py")):
        if path.name.startswith("_"):
            continue
        mr, oe = collect_from_mixin(path)
        print(f"{path.name}: facades={len(mr)} ops_enrich={len(oe)}")
        all_mixin.extend(mr)
        all_ops.extend(oe)

    print(f"TOTAL facades={len(all_mixin)} ops_enrich={len(all_ops)} write={write}")
    if not write:
        return 0

    # Apply mixin rewrites per file.
    by_mixin: dict[Path, list[MixinRewrite]] = {}
    for rw in all_mixin:
        by_mixin.setdefault(rw.path, []).append(rw)
    for path, rewrites in by_mixin.items():
        src = path.read_text(encoding="utf-8")
        path.write_text(_apply_rewrites(src, rewrites), encoding="utf-8", newline="\n")

    # Deduplicate ops enrichments: keep longest new_doc per (path, lineno).
    best: dict[tuple[Path, int], OpsEnrich] = {}
    for oe in all_ops:
        key = (oe.path, oe.lineno)
        prev = best.get(key)
        if prev is None or len(oe.new_doc) > len(prev.new_doc):
            best[key] = oe
    by_ops: dict[Path, list[OpsEnrich]] = {}
    for oe in best.values():
        by_ops.setdefault(oe.path, []).append(oe)
    for path, rewrites in by_ops.items():
        src = path.read_text(encoding="utf-8")
        path.write_text(_apply_rewrites(src, rewrites), encoding="utf-8", newline="\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
