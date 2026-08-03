#!/usr/bin/env python3
"""Add Session import + TYPE_CHECKING attrs + cast() on mixin returns for mypy."""

from __future__ import annotations

import ast
import re
from pathlib import Path

MIXINS = Path(__file__).resolve().parents[1] / "buildml" / "session" / "mixins"
SHARED = MIXINS / "_shared.py"


def ensure_session_in_shared() -> None:
    text = SHARED.read_text(encoding="utf-8")
    if "from buildml.session.session import Session" in text:
        return
    # Insert inside TYPE_CHECKING block if present, else create one.
    if "if TYPE_CHECKING:" in text:
        text = text.replace(
            "if TYPE_CHECKING:",
            "if TYPE_CHECKING:\n    from buildml.session.session import Session  # noqa: F401",
            1,
        )
    else:
        text += (
            "\nfrom typing import TYPE_CHECKING\n\n"
            "if TYPE_CHECKING:\n"
            "    from buildml.session.session import Session  # noqa: F401\n"
        )
    # Also provide a runtime alias so star-imports always expose the name to mypy/runtime.
    if "Session = Any" not in text:
        text += "\n# Runtime placeholder so ``from ._shared import *`` always binds Session.\n"
        text += "Session = Any  # type: ignore[misc,assignment]\n"
    SHARED.write_text(text, encoding="utf-8", newline="\n")
    print("updated _shared.py with Session")


def collect_private_attrs(tree: ast.Module) -> dict[str, set[str]]:
    """Map class name -> private attrs accessed on self."""
    out: dict[str, set[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        attrs: set[str] = set()
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Attribute)
                and isinstance(child.value, ast.Name)
                and child.value.id == "self"
                and child.attr.startswith("_")
            ):
                attrs.add(child.attr)
        out[node.name] = attrs
    return out


def inject_attrs(source: str, class_name: str, attrs: set[str]) -> str:
    if not attrs:
        return source
    # Find class body start
    pattern = re.compile(rf"class {class_name}\([^)]*\):\s*\n")
    m = pattern.search(source)
    if not m:
        pattern = re.compile(rf"class {class_name}:\s*\n")
        m = pattern.search(source)
    if not m:
        return source
    insert_at = m.end()
    # Skip if already injected
    if "# mypy: session private attrs" in source[insert_at : insert_at + 400]:
        return source
    block_lines = [
        "    # mypy: session private attrs (owned by Session.__init__)\n",
        "    if TYPE_CHECKING:\n",
    ]
    for attr in sorted(attrs):
        block_lines.append(f"        {attr}: Any\n")
    block = "".join(block_lines)
    return source[:insert_at] + block + source[insert_at:]


def ensure_imports(source: str) -> str:
    if "from typing import" in source and "cast" in source and "TYPE_CHECKING" in source:
        # may still need cast
        pass
    if "TYPE_CHECKING" not in source.split("from buildml.session.mixins._shared")[0]:
        # _shared star-import may already bring TYPE_CHECKING if exported — add local
        if "TYPE_CHECKING" not in source:
            source = source.replace(
                "from __future__ import annotations\n",
                "from __future__ import annotations\n\nfrom typing import TYPE_CHECKING, Any, cast\n",
                1,
            )
        elif "cast" not in source:
            source = source.replace(
                "from typing import",
                "from typing import cast, ",
                1,
            )
    elif "cast" not in source:
        if "from typing import" in source:
            source = source.replace("from typing import", "from typing import cast,", 1)
        else:
            source = source.replace(
                "from __future__ import annotations\n",
                "from __future__ import annotations\n\nfrom typing import Any, cast, TYPE_CHECKING\n",
                1,
            )
    return source


def wrap_returns_with_cast(source: str, tree: ast.Module) -> str:
    """Wrap `return ops_call(...)` in cast(ReturnType, ...) when annotated."""
    lines = source.splitlines(keepends=True)
    # Work bottom-up on return statements inside methods with return annotations
    returns: list[tuple[int, str, str]] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for child in node.body:
            if not isinstance(child, ast.FunctionDef) or child.returns is None:
                continue
            if _is_property(child):
                # property bodies usually return self._attr — attrs handle that
                continue
            ret_ann = ast.unparse(child.returns)
            if ret_ann in {"None", "Any"}:
                continue
            for stmt in child.body:
                if isinstance(stmt, ast.Return) and stmt.value is not None:
                    # only wrap calls
                    if isinstance(stmt.value, ast.Call):
                        returns.append((stmt.lineno, ret_ann, ast.unparse(stmt.value)))

    for lineno, ret_ann, call_src in sorted(returns, key=lambda x: x[0], reverse=True):
        line = lines[lineno - 1]
        if "cast(" in line:
            continue
        indent = re.match(r"^(\s*)", line).group(1)  # type: ignore[union-attr]
        # Replace return <call> with return cast(Ann, <call>)
        stripped = line.strip()
        if not stripped.startswith("return "):
            continue
        expr = stripped[len("return ") :]
        if expr.startswith("cast("):
            continue
        # Keep original call expression from source line (preserve formatting roughly)
        new_line = f"{indent}return cast({ret_ann}, {expr})\n"
        lines[lineno - 1] = new_line
    return "".join(lines)


def _is_property(node: ast.FunctionDef) -> bool:
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        name = getattr(target, "attr", None) or getattr(target, "id", "")
        if name in {"property", "cached_property"}:
            return True
    return False


def process_mixin(path: Path) -> None:
    if path.name.startswith("_"):
        return
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    attrs_map = collect_private_attrs(tree)
    source = ensure_imports(source)
    for cls, attrs in attrs_map.items():
        source = inject_attrs(source, cls, attrs)
    # Re-parse after attr injection for casts
    tree = ast.parse(source)
    source = wrap_returns_with_cast(source, tree)
    path.write_text(source, encoding="utf-8", newline="\n")
    print(f"typed {path.name}")


def main() -> int:
    ensure_session_in_shared()
    for path in sorted(MIXINS.glob("*.py")):
        process_mixin(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
