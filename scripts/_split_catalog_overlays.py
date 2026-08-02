"""One-shot helper: split catalog.py into domain overlay modules.

Run from repo root:
  python scripts/_split_catalog_overlays.py
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "buildml" / "explain" / "catalog.py"
OVERLAYS = ROOT / "buildml" / "explain" / "overlays"

DOMAIN_OF: dict[str, str] = {}


def domain_for(name: str) -> str:
    if name.startswith("rag_") or name in {"save_rag_bundle", "load_rag_bundle"}:
        return "rag"
    if (
        "torch" in name
        or name.startswith("dl_")
        or name in {"save_torch_bundle", "load_torch_bundle"}
    ):
        return "dl"
    if name.startswith("ai_") or name in {"save_ai_transcript", "load_ai_transcript"}:
        return "ai"
    if name in {
        "explain",
        "workflow",
        "walkthrough",
        "dry_run",
        "eda",
        "eda_app",
        "metadata",
        "head",
        "summarize_history",
        "list_transforms",
        "register_transform",
    }:
        return "workflow"
    return "classical"


def main() -> None:
    text = CATALOG.read_text(encoding="utf-8")
    # Locate _OPERATIONS = ( ... )
    match = re.search(r"^_OPERATIONS(?::[^=]+)? = \(", text, re.M)
    if not match:
        raise SystemExit("Could not find _OPERATIONS assignment")
    header = text[: match.start()]
    rest = text[match.end() :]
    # Find matching close of the big tuple before OPERATION_CATALOG
    catalog_assign = rest.find("\nOPERATION_CATALOG:")
    if catalog_assign < 0:
        raise SystemExit("Could not find OPERATION_CATALOG")
    body = rest[:catalog_assign].rstrip()
    if body.endswith(")"):
        body = body[:-1].rstrip()
    if body.endswith(","):
        body = body[:-1].rstrip()

    # Split on top-level _operation( occurrences
    parts = re.split(r"(?=\n    _operation\()", "\n" + body)
    ops: list[tuple[str, str]] = []
    for part in parts:
        chunk = part.strip("\n")
        if not chunk.strip():
            continue
        name_match = re.match(r'\s*_operation\(\s*\n\s*"([^"]+)"', chunk)
        if not name_match:
            raise SystemExit(f"Could not parse operation name from chunk:\n{chunk[:120]}")
        name = name_match.group(1)
        # Ensure trailing comma for tuple membership
        chunk = chunk.rstrip()
        if not chunk.endswith(","):
            chunk = chunk + ","
        ops.append((name, chunk))

    by_domain: dict[str, list[str]] = {
        "classical": [],
        "dl": [],
        "rag": [],
        "ai": [],
        "workflow": [],
    }
    for name, chunk in ops:
        by_domain[domain_for(name)].append(chunk)

    OVERLAYS.mkdir(parents=True, exist_ok=True)

    # Extract helper preamble from header (everything after module docstring/imports
    # through _operation function inclusive).
    common_end = header.rfind("def _operation(")
    if common_end < 0:
        raise SystemExit("Could not find _operation helper")
    # include full _operation function
    after = header[common_end:]
    func_end = after.find("\n\n\n")
    if func_end < 0:
        # fall back: end of header
        common_helpers = header
    else:
        # Keep imports + prerequisites + helpers
        common_helpers = header[: common_end + func_end + 1]

    # Rewrite common helpers module
    common_text = common_helpers
    # Drop catalog module docstring; overlays/_common owns helpers.
    common_text = re.sub(
        r'^"""[\s\S]*?"""\n',
        '"""Shared catalog helpers and prerequisites for teaching overlays."""\n',
        common_text,
        count=1,
    )
    # Fix relative imports - keep buildml.explain.*
    (OVERLAYS / "_common.py").write_text(common_text.rstrip() + "\n", encoding="utf-8")

    domain_header = '''# ruff: noqa: E501
"""{title} operation overlays (human teaching prose)."""

from __future__ import annotations

from buildml.explain.overlays._common import (
    AI,
    AI_PROVIDER,
    DASHBOARD,
    DATASET,
    FIT,
    FIT_TORCH,
    RAG,
    RAG_CORPUS,
    RAG_INDEX,
    ROLES,
    SPLIT,
    TORCH,
    VIZ,
    OperationKind,
    _operation,
    _p,
)
from buildml.explain.schemas import OperationSpec

_OPERATIONS: tuple[OperationSpec, ...] = (
{body}
)
'''

    titles = {
        "classical": "Classical Session",
        "dl": "Torch / deep-learning Session",
        "rag": "RAG Session",
        "ai": "AI operator Session",
        "workflow": "Workflow / teaching Session",
    }
    for domain, chunks in by_domain.items():
        body = "\n".join(chunks)
        # Indent chunks already have 4-space indent from original
        (OVERLAYS / f"{domain}.py").write_text(
            domain_header.format(title=titles[domain], body=body),
            encoding="utf-8",
        )
        print(f"{domain}: {len(chunks)} ops")

    init_text = '''"""Human teaching overlays for Session operations, split by domain."""

from __future__ import annotations

from buildml.explain.overlays.ai import _OPERATIONS as _AI
from buildml.explain.overlays.classical import _OPERATIONS as _CLASSICAL
from buildml.explain.overlays.dl import _OPERATIONS as _DL
from buildml.explain.overlays.rag import _OPERATIONS as _RAG
from buildml.explain.overlays.workflow import _OPERATIONS as _WORKFLOW
from buildml.explain.schemas import OperationSpec

_OPERATIONS: tuple[OperationSpec, ...] = (
    *_CLASSICAL,
    *_DL,
    *_RAG,
    *_AI,
    *_WORKFLOW,
)

__all__ = ["_OPERATIONS"]
'''
    (OVERLAYS / "__init__.py").write_text(init_text, encoding="utf-8")
    print(f"Wrote overlays for {len(ops)} operations")


if __name__ == "__main__":
    main()
