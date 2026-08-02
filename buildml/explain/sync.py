"""Teaching-surface sync: Session API index, catalog, and AI tool drift checks.

Session public callables are the source of truth for *which* operations exist.
Human overlays under ``buildml.explain.overlays`` supply teaching prose.
``operation_index.json`` is generated from Session signatures and checked in CI.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

INDEX_SCHEMA_VERSION = 1
GENERATED_DIR = Path(__file__).resolve().parent / "generated"
OPERATION_INDEX_PATH = GENERATED_DIR / "operation_index.json"

# AI tools are an intentional allowlist, not a full Session mirror.
# Every tool must still resolve to a real Session method / catalog op.
# Teaching-critical Phase C surfaces must appear in the default registry.
REQUIRED_AI_TOOL_SESSION_METHODS: frozenset[str] = frozenset(
    {
        "rag_retrieve",
        "rag_generate",
        "rag_ingest_corpus",
        "rag_embed_and_index",
        "make_torch_loaders",
        "make_text_torch_loaders",
        "fit_torch",
        "evaluate_torch",
        "cross_validate_torch",
        "fit",
        "evaluate",
        "split",
        "set_roles",
        "explain",
        "workflow",
        "eda",
        "walkthrough",
    }
)

# Tools that intentionally have no Session method (builtins / status helpers).
AI_TOOL_BUILTINS: frozenset[str] = frozenset({"describe_dataset", "ai_status"})


@dataclass(frozen=True, slots=True)
class ParameterIndex:
    name: str
    kind: str
    annotation: str
    default: str | None
    required: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "annotation": self.annotation,
            "default": self.default,
            "required": self.required,
        }


@dataclass(frozen=True, slots=True)
class OperationIndexEntry:
    name: str
    qualname: str
    doc_summary: str
    parameters: tuple[ParameterIndex, ...]
    is_classmethod: bool
    is_staticmethod: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "qualname": self.qualname,
            "doc_summary": self.doc_summary,
            "parameters": [item.to_dict() for item in self.parameters],
            "is_classmethod": self.is_classmethod,
            "is_staticmethod": self.is_staticmethod,
        }


@dataclass(slots=True)
class DriftReport:
    """Structured teaching-surface drift findings."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> None:
        if self.errors:
            joined = "\n".join(f"- {item}" for item in self.errors)
            raise AssertionError(f"Teaching-surface drift detected:\n{joined}")


def public_session_operations(session_cls: type | None = None) -> dict[str, Callable[..., Any]]:
    """Return public Session callables keyed by name (catalog surface)."""
    if session_cls is None:
        from buildml.session import Session

        session_cls = Session
    return {
        name: member
        for name, member in inspect.getmembers(session_cls, predicate=callable)
        if not name.startswith("_")
    }


def _annotation_str(annotation: Any) -> str:
    if annotation is inspect.Parameter.empty:
        return ""
    if isinstance(annotation, str):
        return annotation
    return inspect.formatannotation(annotation)


def _default_str(default: Any) -> str | None:
    if default is inspect.Parameter.empty:
        return None
    try:
        return repr(default)
    except Exception:
        return type(default).__name__


def _doc_summary(func: Callable[..., Any]) -> str:
    doc = inspect.getdoc(func) or ""
    for line in doc.splitlines():
        text = line.strip()
        if text:
            return text
    return ""


def build_operation_index(session_cls: type | None = None) -> dict[str, Any]:
    """Build the machine-readable Session operation index."""
    if session_cls is None:
        from buildml.session import Session

        session_cls = Session
    operations = public_session_operations(session_cls)
    entries: dict[str, Any] = {}
    for name in sorted(operations):
        member = operations[name]
        raw = inspect.getattr_static(session_cls, name)
        is_classmethod = isinstance(raw, classmethod)
        is_staticmethod = isinstance(raw, staticmethod)
        try:
            signature = inspect.signature(member)
        except (TypeError, ValueError):
            signature = inspect.Signature()
        parameters: list[ParameterIndex] = []
        for param_name, param in signature.parameters.items():
            if param_name in {"self", "cls"}:
                continue
            parameters.append(
                ParameterIndex(
                    name=param_name,
                    kind=param.kind.name,
                    annotation=_annotation_str(param.annotation),
                    default=_default_str(param.default),
                    required=param.default is inspect.Parameter.empty
                    and param.kind
                    not in (
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
                    ),
                )
            )
        entry = OperationIndexEntry(
            name=name,
            qualname=getattr(member, "__qualname__", name),
            doc_summary=_doc_summary(member),
            parameters=tuple(parameters),
            is_classmethod=is_classmethod,
            is_staticmethod=is_staticmethod,
        )
        entries[name] = entry.to_dict()
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "source": "buildml.session.Session",
        "n_operations": len(entries),
        "operations": entries,
    }


def write_operation_index(
    path: Path | None = None,
    *,
    session_cls: type | None = None,
) -> Path:
    """Regenerate ``operation_index.json`` from the live Session surface."""
    destination = path or OPERATION_INDEX_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = build_operation_index(session_cls)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def load_operation_index(path: Path | None = None) -> dict[str, Any]:
    """Load the checked-in generated operation index."""
    from typing import cast

    source = path or OPERATION_INDEX_PATH
    return cast(dict[str, Any], json.loads(source.read_text(encoding="utf-8")))


def _catalog_names(catalog: Mapping[str, Any] | None = None) -> set[str]:
    if catalog is None:
        from buildml.explain.catalog import OPERATION_CATALOG

        catalog = OPERATION_CATALOG
    return set(catalog)


def check_session_catalog_parity(
    *,
    session_cls: type | None = None,
    catalog: Mapping[str, Any] | None = None,
) -> DriftReport:
    report = DriftReport()
    session_names = set(public_session_operations(session_cls))
    catalog_names = _catalog_names(catalog)
    missing = sorted(session_names - catalog_names)
    extra = sorted(catalog_names - session_names)
    if missing:
        report.errors.append(f"Catalog missing Session operations: {missing}")
    if extra:
        report.errors.append(f"Catalog has unknown operations (not on Session): {extra}")
    return report


def check_catalog_parameters_vs_signatures(
    *,
    session_cls: type | None = None,
    catalog: Mapping[str, Any] | None = None,
) -> DriftReport:
    """Require catalog parameters ⊆ signature and signature ⊆ catalog (excl. self/cls)."""
    report = DriftReport()
    if session_cls is None:
        from buildml.session import Session

        session_cls = Session
    if catalog is None:
        from buildml.explain.catalog import OPERATION_CATALOG

        catalog = OPERATION_CATALOG
    for name, spec in catalog.items():
        member = getattr(session_cls, name, None)
        if member is None or not callable(member):
            report.errors.append(f"{name}: not a public Session callable")
            continue
        try:
            signature = inspect.signature(member)
        except (TypeError, ValueError):
            report.warnings.append(f"{name}: signature unavailable for parameter check")
            continue
        available = {
            parameter.name
            for parameter in signature.parameters.values()
            if parameter.name not in {"self", "cls"}
            and parameter.kind
            not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        }
        documented = {parameter.name for parameter in getattr(spec, "parameters", ())}
        unknown = sorted(documented - available)
        missing = sorted(available - documented)
        if unknown:
            report.errors.append(f"{name}: catalog parameters not in signature: {unknown}")
        if missing:
            report.errors.append(
                f"{name}: signature parameters missing from catalog: {missing}"
            )
    return report


def check_operation_index_fresh(
    *,
    path: Path | None = None,
    session_cls: type | None = None,
) -> DriftReport:
    """Fail when the checked-in index does not match live Session signatures."""
    report = DriftReport()
    source = path or OPERATION_INDEX_PATH
    if not source.is_file():
        report.errors.append(f"Missing generated operation index: {source}")
        return report
    on_disk = load_operation_index(source)
    expected = build_operation_index(session_cls)
    if on_disk.get("schema_version") != INDEX_SCHEMA_VERSION:
        report.errors.append(
            f"operation_index schema_version {on_disk.get('schema_version')!r} "
            f"!= {INDEX_SCHEMA_VERSION}"
        )
    disk_ops = on_disk.get("operations") or {}
    expected_ops = expected["operations"]
    if set(disk_ops) != set(expected_ops):
        missing = sorted(set(expected_ops) - set(disk_ops))
        extra = sorted(set(disk_ops) - set(expected_ops))
        if missing:
            report.errors.append(f"operation_index missing Session ops: {missing}")
        if extra:
            report.errors.append(f"operation_index has stale ops: {extra}")
    # Compare parameter names / required flags (stable teaching contract).
    for name in sorted(set(disk_ops) & set(expected_ops)):
        disk_params = {
            item["name"]: (item.get("required"), item.get("kind"))
            for item in disk_ops[name].get("parameters") or []
        }
        live_params = {
            item["name"]: (item.get("required"), item.get("kind"))
            for item in expected_ops[name].get("parameters") or []
        }
        if disk_params != live_params:
            report.errors.append(
                f"operation_index[{name}] parameters drifted; "
                "run: python scripts/sync_teaching_surface.py --write"
            )
    return report


def check_ai_tools_vs_catalog(
    *,
    session_cls: type | None = None,
    catalog: Mapping[str, Any] | None = None,
) -> DriftReport:
    report = DriftReport()
    if session_cls is None:
        from buildml.session import Session

        session_cls = Session
    catalog_names = _catalog_names(catalog)
    session_names = set(public_session_operations(session_cls))
    from buildml.ai.tools import build_default_registry

    registry = build_default_registry()
    tool_session_methods: set[str] = set()
    for tool in registry.tools:
        if tool.name in AI_TOOL_BUILTINS:
            if tool.session_method is not None:
                report.warnings.append(
                    f"AI builtin tool {tool.name!r} unexpectedly sets session_method"
                )
            if tool.catalog_operation and tool.catalog_operation not in catalog_names:
                report.errors.append(
                    f"AI tool {tool.name!r} catalog_operation "
                    f"{tool.catalog_operation!r} not in catalog"
                )
            continue
        method = tool.session_method or tool.name
        tool_session_methods.add(method)
        if method not in session_names:
            report.errors.append(f"AI tool {tool.name!r} maps to missing Session.{method}")
        catalog_op = tool.catalog_operation or method
        if catalog_op not in catalog_names:
            report.errors.append(
                f"AI tool {tool.name!r} catalog_operation {catalog_op!r} not in catalog"
            )
    missing_required = sorted(REQUIRED_AI_TOOL_SESSION_METHODS - tool_session_methods)
    if missing_required:
        report.errors.append(
            "Default AI registry missing teaching-critical Session methods: "
            f"{missing_required}"
        )
    return report


def check_dashboard_teaching_concepts() -> DriftReport:
    """Fail when Teaching Studio concept chips reference unknown CONCEPT_NOTES keys."""
    report = DriftReport()
    from buildml.explain.concepts import CONCEPT_NOTES

    try:
        from buildml.dashboard.teaching import build_teaching_studios
    except Exception as exc:  # pragma: no cover - optional dashboard extra
        report.warnings.append(f"dashboard teaching import skipped: {exc}")
        return report

    studios = build_teaching_studios({})
    unknown: set[str] = set()
    for studio in studios.values():
        for key in studio.get("concepts") or ():
            if key not in CONCEPT_NOTES:
                unknown.add(str(key))
    if unknown:
        report.errors.append(
            "dashboard teaching.py references unknown concept keys: "
            f"{sorted(unknown)}"
        )
    return report


def check_teaching_surface(
    *,
    session_cls: type | None = None,
    catalog: Mapping[str, Any] | None = None,
    index_path: Path | None = None,
) -> DriftReport:
    """Run the full Session ↔ index ↔ catalog ↔ AI tool ↔ dashboard sync suite."""
    report = DriftReport()
    for partial in (
        check_session_catalog_parity(session_cls=session_cls, catalog=catalog),
        check_catalog_parameters_vs_signatures(session_cls=session_cls, catalog=catalog),
        check_operation_index_fresh(path=index_path, session_cls=session_cls),
        check_ai_tools_vs_catalog(session_cls=session_cls, catalog=catalog),
        check_dashboard_teaching_concepts(),
    ):
        report.errors.extend(partial.errors)
        report.warnings.extend(partial.warnings)
    return report
