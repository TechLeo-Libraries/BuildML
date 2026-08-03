"""Public Session operation catalog (overlay merge + generated index sync).

Teaching prose lives in :mod:`buildml.explain.overlays` (human-authored, split
by domain). The machine-readable Session signature index lives in
``buildml/explain/generated/operation_index.json`` and is refreshed by
``python scripts/sync_teaching_surface.py --write``. CI fails when Session,
catalog overlays, the index, or AI tool bindings diverge.

Missing overlay parameter rows are auto-filled from the generated Session
signature index so hand lists cannot silently omit public knobs (richer
descriptions remain editable in overlays).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from buildml.explain.overlays import _OPERATIONS
from buildml.explain.schemas import OperationSpec, ParameterSpec
from buildml.explain.sync import load_operation_index


def _parameter_from_index(item: dict[str, Any]) -> ParameterSpec:
    return ParameterSpec(
        name=str(item["name"]),
        type_name=str(item.get("annotation") or "Any"),
        description=(
            f"Session.{item['name']} argument "
            "(auto-filled from operation_index; overlay may supply richer prose)."
        ),
        required=bool(item.get("required")),
        default=item.get("default"),
    )


def _merge_index_parameters(spec: OperationSpec) -> OperationSpec:
    """Ensure every indexed Session parameter appears in the teaching catalog."""
    index = load_operation_index()
    entry = (index.get("operations") or {}).get(spec.name)
    if not entry:
        return spec
    documented = {item.name: item for item in spec.parameters}
    merged: list[ParameterSpec] = []
    for item in entry.get("parameters") or []:
        name = str(item["name"])
        if name in {"self", "cls"}:
            continue
        existing = documented.get(name)
        merged.append(existing if existing is not None else _parameter_from_index(item))
    for name, item in documented.items():
        if name not in {entry.name for entry in merged}:
            merged.append(item)
    if tuple(item.name for item in merged) == tuple(item.name for item in spec.parameters) and all(
        documented.get(item.name) is item for item in merged if item.name in documented
    ):
        return spec
    return replace(spec, parameters=tuple(merged))


_RAW_OPERATIONS: tuple[OperationSpec, ...] = _OPERATIONS
OPERATION_CATALOG: dict[str, OperationSpec] = {
    operation.name: _merge_index_parameters(operation) for operation in _RAW_OPERATIONS
}

if len(OPERATION_CATALOG) != len(_RAW_OPERATIONS):
    raise RuntimeError("Duplicate operation name in BuildML operation catalog")


def get_operation(name: str) -> OperationSpec:
    """Fetch the editorial contract for one public session operation.

    The specification is what every explanation surface reads from: the
    resolver, the beginner primer, the AI tool registry, and the docs build: so
    an unknown name is raised rather than defaulted. A silent miss would produce
    an explanation that describes nothing.

    Parameters
    ----------
    name:
        A session operation name, such as ``'split'`` or ``'fit_forecast'``.

    Returns
    -------
    ~buildml.explain.schemas.OperationSpec
        The catalog entry, with index-derived parameters already merged in.

    Raises
    ------
    KeyError
        No catalog operation has that name.

    See Also
    --------
    list_operations : Every entry, in a stable order.
    """
    try:
        return OPERATION_CATALOG[name]
    except KeyError as exc:
        raise KeyError(f"Unknown Session operation: {name}") from exc


def list_operations() -> tuple[OperationSpec, ...]:
    """List every catalog entry in a stable order.

    Ordering is alphabetical rather than insertion-based so that generated
    artifacts: the operation index, the docs tables, the drift reports: do not
    churn when an overlay module is edited.

    Returns
    -------
    tuple of ~buildml.explain.schemas.OperationSpec
        All operations, sorted by name.
    """
    return tuple(OPERATION_CATALOG[name] for name in sorted(OPERATION_CATALOG))
