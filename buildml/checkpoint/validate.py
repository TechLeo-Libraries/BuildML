"""Reattach validation for checkpoint bundles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from buildml.core.types import TableSchema
from buildml.data.splits import SplitPlan

ReattachStatus = Literal[
    "resume",
    "fresh_ingest",
    "resume_roles_needed",
    "blocked",
    "splits_invalidated",
]


@dataclass(slots=True)
class ReattachResult:
    """Outcome of validating a reattached dataset against checkpoint metadata."""

    status: ReattachStatus
    messages: list[str] = field(default_factory=list)
    split_plan: SplitPlan | None = None
    details: dict[str, Any] = field(default_factory=dict)


def validate_reattach(
    *,
    current_schema: TableSchema,
    current_columns: list[str],
    current_n_rows: int,
    meta: dict[str, Any] | None,
    splits_payload: dict[str, Any] | None,
) -> ReattachResult:
    """Validate reattached data against optional checkpoint metadata.

    Parameters
    ----------
    current_schema / current_columns / current_n_rows:
        Freshly loaded data characteristics.
    meta:
        Parsed ``meta.json`` or ``None`` if missing (data-only import).
    splits_payload:
        Parsed ``splits.json`` or ``None``.

    Returns
    -------
    ReattachResult
        Status and guidance for the session.
    """
    if meta is None:
        return ReattachResult(
            status="fresh_ingest",
            messages=[
                "No checkpoint metadata found. Treating as a fresh ingest; "
                "re-assign roles and recreate splits before modeling."
            ],
        )

    saved_schema = TableSchema.from_dict(meta.get("schema", {}))
    saved_columns = list(meta.get("columns", saved_schema.columns))
    saved_n_rows = int(meta.get("n_rows", -1))
    messages: list[str] = []

    removed = [c for c in saved_columns if c not in current_columns]
    added = [c for c in current_columns if c not in saved_columns]

    if removed:
        return ReattachResult(
            status="blocked",
            messages=[
                f"Cannot safely resume: required column(s) removed since checkpoint: {removed}"
            ],
            details={"removed": removed, "added": added},
        )

    split_plan: SplitPlan | None = None
    if splits_payload:
        split_plan = SplitPlan.from_dict(splits_payload)
        max_idx = -1
        for part in (
            split_plan.train_indices,
            split_plan.validation_indices,
            split_plan.test_indices,
        ):
            if part:
                max_idx = max(max_idx, max(part))
        if current_n_rows != saved_n_rows or (max_idx >= 0 and max_idx >= current_n_rows):
            messages.append(
                "Row count/order no longer matches checkpoint split membership. "
                "Splits invalidated; create a new split or inject partitions."
            )
            return ReattachResult(
                status="splits_invalidated",
                messages=messages,
                split_plan=None,
                details={"saved_n_rows": saved_n_rows, "current_n_rows": current_n_rows},
            )

    if added:
        messages.append(
            f"New column(s) detected: {added}. Resume allowed; assign roles before modeling."
        )
        return ReattachResult(
            status="resume_roles_needed",
            messages=messages,
            split_plan=split_plan,
            details={"added": added},
        )

    # dtype drift warnings (non-blocking for Phase 1)
    saved_dtypes = {f.name: f.dtype for f in saved_schema.fields}
    for schema_field in current_schema.fields:
        previous = saved_dtypes.get(schema_field.name)
        if previous is not None and previous != schema_field.dtype:
            messages.append(
                f"Dtype changed for column '{schema_field.name}': "
                f"{previous} -> {schema_field.dtype}"
            )

    messages.append("Checkpoint metadata matches; session can resume.")
    return ReattachResult(status="resume", messages=messages, split_plan=split_plan)
