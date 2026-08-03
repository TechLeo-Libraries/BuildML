"""Record what each operation did, as data that outlives the objects involved.

A Session's history is its audit trail: what was called, with what arguments,
and what changed as a result. Reconstructing that from a fitted model afterwards
is impossible, so it is recorded as it happens.

Two constraints shape the format. Records must be JSON-safe, because history
travels in checkpoints and reports where a live object cannot go: so
:func:`json_safe` converts everything and falls back to ``repr`` rather than
failing. And records must stay readable across versions, because a checkpoint
written months ago should still load: v1 keys are kept alongside their v2
replacements, and :func:`normalize_history` upgrades old records on read.

The distinctive part is the state transition. Each record holds the workflow
state before and after, plus the differences between them, which is what turns
"impute was called" into "impute was called and this is what it changed".

See Also
--------
buildml.explain.sync : Keeping operation metadata aligned with the real API.
buildml.checkpoint.bundle : Where history is persisted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

HISTORY_SCHEMA_VERSION = 2

_EMPTY_STATE: dict[str, Any] = {
    "has_dataset": False,
    "columns": [],
    "roles": {},
    "has_split": False,
    "split_kind": None,
    "has_fit": False,
    "has_dl_train": False,
    "has_rag_corpus": False,
    "has_rag_index": False,
    "has_cluster_plan": False,
    "has_ensemble_plan": False,
    "has_automl_plan": False,
    "has_forecast_plan": False,
    "has_anomaly_plan": False,
    "has_impute_plan": False,
    "has_encode_plan": False,
    "has_scale_plan": False,
    "has_outlier_plan": False,
    "has_binning_plan": False,
    "has_feature_select_plan": False,
    "has_text_plan": False,
    "has_reduce_plan": False,
    "has_custom_plan": False,
    "has_date_plan": False,
    "has_resample_plan": False,
}


def json_safe(value: Any) -> Any:
    """Convert a value into something JSON can hold, whatever it started as.

    Operation parameters are whatever the caller passed: enums, ``Path``
    objects, fitted estimators, nested config dicts. All of that has to survive
    into a checkpoint as JSON.

    Handled in a deliberate order. Primitives pass through. Enums become their
    values, so ``ColumnRole.TARGET`` reads as ``'target'`` rather than as a
    repr. Paths become strings. Mappings and sequences recurse. Anything with
    ``to_dict`` uses it, which covers BuildML's own result objects. Everything
    else becomes its ``repr``.

    Parameters
    ----------
    value:
        Anything.

    Returns
    -------
    Any
        A structure of dicts, lists, strings, numbers, booleans, and ``None``.

    Notes
    -----
    **The ``repr`` fallback never fails, which is the point.** History recording
    must not break the operation it is recording. The cost is that an
    unrecognised object lands as text: informative for a human, useless for
    reloading.

    **Strings and bytes are not treated as sequences**, which would otherwise
    explode them into lists of characters.

    Examples
    --------
    >>> from pathlib import Path
    >>> json_safe({"path": Path("x.csv"), "n": 5})
    {'path': 'x.csv', 'n': 5}
    >>> from buildml.core.types import ColumnRole
    >>> json_safe(ColumnRole.TARGET)
    'target'
    >>> json_safe(["a", 1, None])
    ['a', 1, None]
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return json_safe(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [json_safe(item) for item in value]
    if hasattr(value, "to_dict"):
        return json_safe(value.to_dict())
    return repr(value)


def session_state(session: Any) -> dict[str, Any]:
    """Take a flat snapshot of what a Session currently has.

    Not the Session's data: a set of booleans and small values saying what
    exists: is there a dataset, a split, a fit, an imputation plan. Comparing two
    of these before and after an operation is what produces the change list in a
    history record.

    Everything is read with ``getattr`` and a default, so this works against a
    Session in any state, including one restored from an older checkpoint that
    predates half these attributes.

    Parameters
    ----------
    session:
        A :class:`~buildml.session.Session`, or anything shaped like one.

    Returns
    -------
    dict
        ``has_dataset``, ``columns``, ``roles``, ``has_split``, ``split_kind``,
        ``has_fit``, and a ``has_*`` flag for each domain result and
        preprocessing plan.

    Notes
    -----
    **Presence, not content.** ``has_impute_plan`` says a plan exists, not what
    it does. That keeps the snapshot small enough to store twice in every
    history record.

    **Columns and roles are the exceptions**, carried in full because a change
    to either is exactly what a reader wants to see in a transition.

    See Also
    --------
    state_changes : Diffing two of these.
    """
    dataset = getattr(session, "_dataset", None)
    split = getattr(session, "_split_plan", None)
    return {
        "has_dataset": dataset is not None,
        "columns": [] if dataset is None else list(dataset.columns),
        "roles": {}
        if dataset is None
        else {name: role.value for name, role in dataset.roles.items()},
        "has_split": split is not None,
        "split_kind": None if split is None else split.kind,
        "has_fit": getattr(session, "_fit_result", None) is not None,
        "has_dl_train": getattr(session, "_dl_train_result", None) is not None,
        "has_rag_corpus": getattr(session, "_rag_corpus", None) is not None,
        "has_rag_index": getattr(session, "_rag_index_result", None) is not None,
        "has_cluster_plan": getattr(session, "_cluster_plan", None) is not None,
        "has_ensemble_plan": getattr(session, "_ensemble_plan", None) is not None,
        "has_automl_plan": getattr(session, "_automl_plan", None) is not None,
        "has_forecast_plan": getattr(session, "_forecast_plan", None) is not None,
        "has_anomaly_plan": getattr(session, "_anomaly_plan", None) is not None,
        "has_impute_plan": getattr(session, "_impute_plan", None) is not None,
        "has_encode_plan": getattr(session, "_encode_plan", None) is not None,
        "has_scale_plan": getattr(session, "_scale_plan", None) is not None,
        "has_outlier_plan": getattr(session, "_outlier_plan", None) is not None,
        "has_binning_plan": getattr(session, "_binning_plan", None) is not None,
        "has_feature_select_plan": getattr(session, "_feature_select_plan", None) is not None,
        "has_text_plan": getattr(session, "_text_plan", None) is not None,
        "has_reduce_plan": getattr(session, "_reduce_plan", None) is not None,
        "has_custom_plan": getattr(session, "_custom_plan", None) is not None,
        "has_date_plan": getattr(session, "_date_plan", None) is not None,
        "has_resample_plan": getattr(session, "_resample_plan", None) is not None,
    }


def state_changes(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    """List what differs between two snapshots, as literal before-and-after.

    Produces lines like ``has_split: False -> True``. Deliberately mechanical:
    it reports the keys that changed and their values, and infers nothing about
    what the change means. Interpretation belongs in the walkthrough, which can
    be revised; a history record should be able to state what happened without
    committing to a reading of it.

    Keys are sorted, so the same change always produces the same line in the
    same position: which is what makes two runs comparable.

    Parameters
    ----------
    before:
        The snapshot before the operation.
    after:
        The snapshot after.

    Returns
    -------
    list of str
        One ``key: old -> new`` line per difference, sorted by key. Empty when
        nothing changed, which is itself informative: an operation that altered
        no state either did nothing or did something this snapshot does not
        capture.

    Notes
    -----
    **Keys missing from one side compare against ``None``**, so a snapshot from
    an older version produces sensible lines rather than an error.

    Examples
    --------
    >>> state_changes({"has_split": False}, {"has_split": True})
    ['has_split: False -> True']
    >>> state_changes({"has_fit": True}, {"has_fit": True})
    []

    See Also
    --------
    session_state : Producing the snapshots.
    """
    changes: list[str] = []
    for key in sorted(set(before) | set(after)):
        old, new = before.get(key), after.get(key)
        if old != new:
            changes.append(f"{key}: {old!r} -> {new!r}")
    return changes


def make_operation_record(
    *,
    sequence: int,
    operation_id: str,
    parameters: Mapping[str, Any] | None,
    decision_origin: str,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    warnings: Sequence[str] = (),
    result_summary: Mapping[str, Any] | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Assemble one history record, readable by both old and new consumers.

    The single constructor for history entries, so every record has the same
    shape. Parameters and both state snapshots are passed through
    :func:`json_safe`, making the result immediately serialisable.

    Some fields are duplicated on purpose. ``action`` repeats ``operation_id``
    and ``details`` repeats ``parameters``, because v1 consumers read the old
    names. The cost is a slightly larger record; the benefit is that a report or
    script written against the old format keeps working.

    Parameters
    ----------
    sequence:
        Position in the history, from 1. Ordering does not depend on timestamps,
        which can collide at this resolution.
    operation_id:
        What was called: ``'impute'``, ``'fit'``, ``'split'``.
    parameters:
        The arguments, converted for storage.
    decision_origin:
        Who decided. ``'explicit'`` for a user's choice, ``'auto'`` for one
        BuildML made. **This is the field that keeps automation honest**: it is
        how a reader tells their own decisions from the library's.
    before:
        State snapshot before.
    after:
        State snapshot after.
    warnings:
        Anything the operation surfaced. Stringified.
    result_summary:
        A compact description of what came out. Falls back to the parameters
        when omitted.
    timestamp:
        ISO 8601. Defaults to now, in UTC.

    Returns
    -------
    dict
        A v2 record with the schema version, identity, parameters, the state
        transition and its change list, warnings, and the result summary.

    Notes
    -----
    **Timestamps are UTC.** Comparing across machines in different timezones is
    otherwise a source of confusion in a shared history.

    **The record is a snapshot, not a reference.** Nothing here points at live
    objects, so it stays valid after the Session is gone.

    See Also
    --------
    normalize_history : Upgrading older records to this shape.
    """
    safe_parameters = json_safe(dict(parameters or {}))
    safe_before = json_safe(dict(before))
    safe_after = json_safe(dict(after))
    return {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "sequence": sequence,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "operation_id": operation_id,
        # Retain v1 keys for code that still reads action/details.
        "action": operation_id,
        "parameters": safe_parameters,
        "details": safe_parameters,
        "decision_origin": decision_origin,
        "state_transition": {
            "before": safe_before,
            "after": safe_after,
            "changes": state_changes(safe_before, safe_after),
        },
        "warnings": [str(item) for item in warnings],
        "result_summary": json_safe(dict(result_summary or parameters or {})),
    }


def normalize_history(history: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    """Upgrade a history of any vintage into uniform v2 records.

    A checkpoint written by an older BuildML holds records in the old shape:
    ``action`` instead of ``operation_id``, ``details`` instead of
    ``parameters``, and no state transitions at all. Rather than making every
    consumer handle both, they are converted once on read.

    The interesting case is a v1 record with no transition. Since v1 did not
    record state, the previous record's "after" is carried forward as this
    record's "before" and "after": which produces an empty change list. That is
    the honest answer: the change was not recorded, and inventing one would be
    worse than reporting none.

    Parameters
    ----------
    history:
        Records in v1 shape, v2 shape, or a mixture. ``None`` yields an empty
        list.

    Returns
    -------
    list of dict
        v2 records, resequenced from 1 in the order given.

    Notes
    -----
    **Sequence numbers are reassigned**, so gaps or duplicates in a stored
    history come out contiguous.

    **Malformed records are coerced rather than rejected.** Non-mapping
    parameters get wrapped as ``{'value': ...}``, a non-sequence warnings field
    becomes a one-element tuple. A corrupt entry should not make a checkpoint
    unloadable.

    **Missing timestamps are filled with the current time**, which will look
    misleading in an old history. The sequence number is the reliable ordering.

    See Also
    --------
    make_operation_record : The target shape.
    """
    normalized: list[dict[str, Any]] = []
    previous = dict(_EMPTY_STATE)
    for index, raw in enumerate(history or (), start=1):
        operation_id = str(raw.get("operation_id") or raw.get("action") or "unknown")
        transition = raw.get("state_transition")
        if isinstance(transition, Mapping):
            before = transition.get("before", previous)
            after = transition.get("after", before)
        else:
            before = previous
            after = previous
        details = raw.get("parameters", raw.get("details", {}))
        raw_warnings = raw.get("warnings", ())
        warnings = (
            raw_warnings
            if isinstance(raw_warnings, Sequence)
            and not isinstance(raw_warnings, (str, bytes, bytearray))
            else (str(raw_warnings),)
            if raw_warnings
            else ()
        )
        record = make_operation_record(
            sequence=index,
            operation_id=operation_id,
            parameters=details if isinstance(details, Mapping) else {"value": details},
            decision_origin=str(raw.get("decision_origin", "explicit")),
            before=before if isinstance(before, Mapping) else previous,
            after=after if isinstance(after, Mapping) else previous,
            warnings=warnings,
            result_summary=raw.get("result_summary", {})
            if isinstance(raw.get("result_summary", {}), Mapping)
            else {"value": raw.get("result_summary")},
            timestamp=str(raw.get("timestamp") or datetime.now(timezone.utc).isoformat()),
        )
        normalized.append(record)
        previous = dict(record["state_transition"]["after"])
    return normalized


def prior_state(history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recover the current workflow state from the end of the history.

    The state after the last recorded operation is the state now, so the next
    record's "before" can be read off the history rather than recomputed from
    the Session. That matters after a checkpoint restore, when the history has
    been reloaded and the Session may not yet hold everything it describes.

    Parameters
    ----------
    history:
        The records, in order.

    Returns
    -------
    dict
        The last record's "after" snapshot. The empty workflow state: every
        flag ``False``: when the history is empty or the last record has no
        usable transition.

    Notes
    -----
    **A copy is returned**, so a caller cannot mutate the record through it.

    **The fallback is a full empty state, not an empty dict**, so every key a
    consumer expects is present.

    See Also
    --------
    session_state : Reading the state off a live Session instead.
    """
    if history:
        transition = history[-1].get("state_transition", {})
        after = transition.get("after", {}) if isinstance(transition, Mapping) else {}
        if isinstance(after, Mapping):
            return dict(after)
    return dict(_EMPTY_STATE)
