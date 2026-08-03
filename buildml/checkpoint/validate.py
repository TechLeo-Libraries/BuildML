"""Decide whether saved roles and splits still apply to the data on disk.

A checkpoint records what the data looked like when it was written. When it is
loaded, the data may have changed: a column dropped, rows appended, a dtype
altered upstream. Reusing the old split membership across such a change is the
quiet failure this module exists to prevent: partition indices are positions,
so appending rows or reordering them assigns rows to partitions they were never
in, and the holdout stops being a holdout.

The verdict is graded rather than binary, because most changes do not invalidate
everything. A new column means roles need attention but the split is fine. A
changed row count means the split is gone but the data is still usable. Only a
*missing* column blocks the resume outright, since nothing downstream can
proceed without it.

The statuses, in decreasing order of trust: ``resume`` (everything holds),
``resume_roles_needed`` (new columns need roles), ``splits_invalidated`` (the
partitions no longer apply), ``fresh_ingest`` (no metadata at all), and
``blocked`` (a required column is gone).

See Also
--------
buildml.checkpoint.bundle.load_checkpoint : Where this verdict is consumed.
"""

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
    """How much of a checkpoint survived contact with the current data.

    Attributes
    ----------
    status:
        The verdict, from ``'resume'`` down to ``'blocked'``.
    messages:
        Plain-language explanations naming the specific columns or counts that
        caused the verdict. These are what a user needs in order to fix the
        problem or accept it.
    split_plan:
        The restored partition membership, or ``None`` when it could not be
        trusted. Deliberately cleared rather than returned with a warning.
    details:
        The structured facts behind the verdict: added and removed columns,
        saved and current row counts: for programmatic handling.

    Notes
    -----
    **A ``None`` split plan is a real result, not an error case.** It means the
    saved partitions no longer describe the current rows, and continuing
    requires either a new split or explicitly injected partitions.

    **Dtype drift is reported but never blocks.** A column that changed from
    integer to float is usually harmless and occasionally not, so it appears in
    ``messages`` for a human to weigh.

    See Also
    --------
    validate_reattach : Producing this.
    """

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
    """Compare the data as it is now against the data as it was when saved.

    Works down from the most damaging change to the least. A column that is gone
    blocks the resume, because nothing downstream can be recomputed without it.
    A row count that no longer matches, or a split index pointing past the end of
    the frame, invalidates the partitions: the split is dropped rather than
    applied to the wrong rows. A new column allows the resume but flags that it
    has no role yet. Anything else passes, with dtype changes noted for the
    record.

    Parameters
    ----------
    current_schema:
        The schema inferred from the data just loaded, used for dtype
        comparison.
    current_columns:
        The column names present now.
    current_n_rows:
        The row count now, checked against the saved count because split
        membership is positional.
    meta:
        The parsed ``meta.json``, or ``None`` for a data-only import: in which
        case there is nothing to validate against and the result is a fresh
        ingest.
    splits_payload:
        The parsed ``splits.json``, or ``None`` when the checkpoint saved no
        split.

    Returns
    -------
    ReattachResult
        The verdict, the messages explaining it, the split plan if it survived,
        and the structured details.

    Notes
    -----
    **Row count equality is a proxy for row identity, and an imperfect one.**
    Replacing the data with a different frame of the same length passes this
    check and produces partitions that are silently wrong. The check catches
    appends, filters, and truncations, which are the common cases; it cannot
    catch a substitution.

    **The order of checks is the order of severity**, so a checkpoint with both
    a removed column and a changed row count reports the removal, which is the
    one that has to be fixed first.

    See Also
    --------
    ReattachResult : What the fields of the verdict mean.
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
