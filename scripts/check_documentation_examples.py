"""Inventory documentation code blocks and check each Python block in isolation.

No imports, data, or earlier blocks are injected into an example. Execution is
opt-in because examples may download models, start servers, or need credentials.
An inventory records non-Python blocks as well, rather than silently discarding
them. A static pass checks syntax and unresolved global names; it is not proof
of successful execution.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import dataclasses
import doctest
import json
import os
import re
import subprocess
import symtable
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = {"python", "py", "python3", "pycon", "python-console"}


@dataclasses.dataclass
class Example:
    path: str
    line: int
    language: str
    source: str
    kind: str


def blocks(text: str, path: str, offset: int = 0) -> list[Example]:
    """Extract fences, code directives, and indented RST literal blocks.

    Unclassified literal blocks remain in the inventory for manual review.
    Python files use Python as their default literal-block language.
    """
    lines = text.splitlines()
    found = []
    i = 0
    while i < len(lines):
        fence = re.match(r"^\s*(`{3,}|~{3,})\s*([^\s]*)", lines[i])
        directive = re.match(r"^(\s*)\.\.\s+(?:code-block|code|sourcecode)::\s*(\S*)", lines[i])
        if fence:
            marker, language = fence.groups()
            language = language.strip("{}").lower()
            start = i + 1
            i += 1
            while i < len(lines) and not re.match(r"^\s*" + re.escape(marker) + r"\s*$", lines[i]):
                i += 1
            found.append(Example(path, start + 1 + offset, language, textwrap.dedent("\n".join(lines[start:i])), "fence"))
        elif directive:
            indent, language = directive.groups()
            i += 1
            while i < len(lines) and (not lines[i].strip() or lines[i].lstrip().startswith(":")):
                i += 1
            start = i
            while i < len(lines) and (not lines[i].strip() or len(lines[i]) - len(lines[i].lstrip()) > len(indent)):
                i += 1
            found.append(Example(path, start + 1 + offset, language.lower(), textwrap.dedent("\n".join(lines[start:i])).rstrip(), "directive"))
            continue
        elif lines[i].rstrip().endswith("::") and not lines[i].lstrip().startswith(".. "):
            parent_indent = len(lines[i]) - len(lines[i].lstrip())
            start = i + 1
            while start < len(lines) and not lines[start].strip():
                start += 1
            end = start
            while end < len(lines) and (not lines[end].strip() or len(lines[end]) - len(lines[end].lstrip()) > parent_indent):
                end += 1
            source = textwrap.dedent("\n".join(lines[start:end])).rstrip()
            if source:
                language = "pycon" if source.lstrip().startswith(">>>") else ""
                if not language and path.endswith(".py"):
                    language = "python"
                elif not language:
                    try:
                        parsed = ast.parse(source)
                        if any(isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.Call)) for node in ast.walk(parsed)):
                            language = "python"
                    except SyntaxError:
                        pass
                found.append(Example(path, start + 1 + offset, language, source, "literal"))
                i = end
                continue
        i += 1
    return found


def inventory(paths: list[Path]) -> list[Example]:
    found = []
    for path in paths:
        text = path.read_text(encoding="utf-8-sig")
        relative = path.resolve().relative_to(ROOT).as_posix()
        if path.suffix != ".py":
            found.extend(blocks(text, relative))
            continue
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "example":
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    continue
                if isinstance(value, str):
                    source = value
                elif isinstance(value, (list, tuple)) and all(isinstance(line, str) for line in value):
                    source = "\n".join(value)
                else:
                    continue
                found.append(Example(relative, node.lineno, "python", source, "teaching"))
            if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            doc = ast.get_docstring(node, clean=False)
            if not doc:
                continue
            base = node.body[0].lineno - 1
            found.extend(blocks(textwrap.dedent(doc), relative, base))
            try:
                examples = doctest.DocTestParser().get_examples(doc)
            except ValueError as exc:
                found.append(Example(relative, base + 1, "pycon", str(exc), "invalid-doctest"))
                continue
            if examples:
                found.append(Example(relative, base + examples[0].lineno + 1, "python", "".join(e.source for e in examples), "doctest"))
    return found


def python_source(example: Example) -> str:
    if example.kind == "invalid-doctest":
        raise ValueError(example.source)
    if example.language in {"pycon", "python-console"}:
        examples = doctest.DocTestParser().get_examples(example.source)
        if not examples:
            raise ValueError("Python console block contains no prompts")
        return "".join(e.source for e in examples)
    return example.source


def static_check(source: str) -> dict:
    """Report unresolved names, including globals referenced by nested functions."""
    try:
        table = symtable.symtable(source, "<example>", "exec")
        ast.parse(source)
    except SyntaxError as exc:
        return {"status": "syntax-error", "detail": str(exc)}
    defined = {s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
    available = defined | set(dir(builtins)) | {"__name__", "__file__"}
    missing = set()

    def visit(scope):
        for symbol in scope.get_symbols():
            if symbol.is_referenced() and symbol.is_global() and symbol.get_name() not in available:
                missing.add(symbol.get_name())
        for child in scope.get_children():
            visit(child)

    visit(table)
    return {"status": "undefined-names" if missing else "static-pass", "undefined_names": sorted(missing)}


def execute(example: Example, timeout: int) -> dict:
    """Run exactly the authored source in its own process and working directory."""
    with tempfile.TemporaryDirectory(prefix="buildml-doc-example-") as directory:
        script = Path(directory) / "example.py"
        script.write_text(python_source(example), encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(ROOT), "MPLBACKEND": "Agg", "LOKY_MAX_CPU_COUNT": "2", "PYTHONIOENCODING": "utf-8"}
        try:
            result = subprocess.run([sys.executable, str(script)], cwd=directory, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "seconds": timeout}
        return {"status": "passed" if result.returncode == 0 else "failed", "exit_code": result.returncode, "stdout": result.stdout[-8000:], "stderr": result.stderr[-8000:]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = args.paths or subprocess.check_output(["git", "ls-files", "*.md", "*.rst", "buildml/*.py"], cwd=ROOT, text=True).splitlines()
    paths = [ROOT / name for name in names if (ROOT / name).is_file()]
    rows = []
    for example in inventory(paths):
        row = dataclasses.asdict(example)
        if example.language in PYTHON:
            try:
                row["static"] = static_check(python_source(example))
                row["execution"] = execute(example, args.timeout) if args.execute else {"status": "not-run"}
            except (ValueError, SyntaxError) as exc:
                row["static"] = {"status": "parse-error", "detail": str(exc)}
        else:
            row["static"] = {"status": "non-python"}
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"files": names, "examples": rows}, indent=2), encoding="utf-8")
    failures = [row for row in rows if row["static"]["status"] not in {"static-pass", "non-python"} or row.get("execution", {}).get("status") in {"failed", "timeout"}]
    print(f"Inventoried {len(rows)} blocks across {len(paths)} files; {len(failures)} blocks need attention")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
