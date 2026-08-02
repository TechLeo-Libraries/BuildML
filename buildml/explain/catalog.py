"""Public Session operation catalog (overlay merge + generated index sync).

Teaching prose lives in :mod:`buildml.explain.overlays` (human-authored, split
by domain). The machine-readable Session signature index lives in
``buildml/explain/generated/operation_index.json`` and is refreshed by
``python scripts/sync_teaching_surface.py --write``. CI fails when Session,
catalog overlays, the index, or AI tool bindings diverge.
"""

from __future__ import annotations

from buildml.explain.overlays import _OPERATIONS
from buildml.explain.schemas import OperationSpec

OPERATION_CATALOG: dict[str, OperationSpec] = {
    operation.name: operation for operation in _OPERATIONS
}

if len(OPERATION_CATALOG) != len(_OPERATIONS):
    raise RuntimeError("Duplicate operation name in BuildML operation catalog")


def get_operation(name: str) -> OperationSpec:
    """Return one public operation specification."""
    try:
        return OPERATION_CATALOG[name]
    except KeyError as exc:
        raise KeyError(f"Unknown Session operation: {name}") from exc


def list_operations() -> tuple[OperationSpec, ...]:
    """Return operation specifications in stable alphabetical order."""
    return tuple(OPERATION_CATALOG[name] for name in sorted(OPERATION_CATALOG))
