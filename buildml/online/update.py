"""Incremental partial_fit updates for online / continual learning."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition
from buildml.online.features import (
    align_external_frame,
    carve_train_chunk,
    chunk_drift_notes,
    encode_classification_targets,
    matrix_from_frame,
    regression_targets,
)
from buildml.online.fit import _maybe_refit_fallback
from buildml.online.results import OnlinePlan, OnlineUpdateResult


def partial_fit_online(
    dataset: Dataset,
    plan: OnlinePlan,
    split_plan: SplitPlan | None,
    *,
    n_rows: int | None = None,
    indices: Sequence[Any] | None = None,
    frame: pd.DataFrame | None = None,
) -> tuple[OnlinePlan, OnlineUpdateResult]:
    """Apply one incremental update via ``partial_fit`` (train-only).

    Chunk sources (exactly one style):
    - Default: next ``n_rows`` (or plan.chunk_size) unused train rows after cursor
    - ``indices=``: explicit train-partition dataset indices
    - ``frame=``: user-provided incremental frame with role-aligned columns

    Validation/test are never used for updates. Silent full refits are refused
    unless the plan was created with ``allow_refit_fallback=True`` (disclosed).

    Parameters
    ----------
    dataset:
        BuildML dataset backing the split (not used for external ``frame=``).
    plan:
        Fitted :class:`~buildml.online.results.OnlinePlan` from :func:`fit_online`.
    split_plan:
        Train/validation/test split; required unless ``frame=`` is provided.
    n_rows:
        Rows to take from unused train indices; defaults to ``plan.chunk_size``.
    indices:
        Optional explicit train-partition dataset indices.
    frame:
        Optional user-provided incremental frame with role-aligned columns.

    Returns
    -------
    tuple[OnlinePlan, OnlineUpdateResult]
        Updated plan and a serializable update summary.

    Raises
    ------
    ValidationError
        When plan, chunk source, or column preconditions are invalid.
    """
    if plan is None:
        raise ValidationError("No OnlinePlan. Call fit_online first.")
    if split_plan is None and frame is None:
        raise ValidationError(
            "partial_fit_online requires a SplitPlan (or an external frame=)."
        )

    sources = sum(x is not None for x in (indices, frame))
    if sources > 1:
        raise ValidationError(
            "Provide at most one of indices= or frame= for partial_fit_online."
        )

    allow_refit = bool((plan.config or {}).get("allow_refit_fallback", False))
    drift_on = bool((plan.config or {}).get("drift_disclose", True))
    warnings: list[str] = []
    disclosures: list[str] = [
        "partial_fit_online updates the estimator on a train chunk only.",
        "Holdout partitions are never used for incremental updates.",
    ]

    external = False
    if frame is not None:
        external = True
        chunk = align_external_frame(
            frame,
            columns=plan.columns,
            target_column=plan.target_column,
        )
        chunk_indices: list[Any] = list(chunk.index)
        new_cursor = plan.cursor
        disclosures.append(
            "Update used a user-provided incremental frame (role-aligned "
            "columns); Session train cursor was not advanced."
        )
    else:
        assert_fit_partition(split_plan, "train")
        assert split_plan is not None
        take = int(n_rows) if n_rows is not None else int(plan.chunk_size)
        chunk, chunk_indices, new_cursor = carve_train_chunk(
            dataset,
            split_plan,
            cursor=plan.cursor,
            n_rows=take,
            indices=indices,
        )

    x = matrix_from_frame(chunk, list(plan.columns))
    if plan.estimator_name == "multinomial_nb" and (x < 0).any():
        raise ValidationError(
            "multinomial_nb requires non-negative features on every chunk."
        )

    if plan.task == "classification":
        y, encoder, _classes = encode_classification_targets(
            chunk[plan.target_column],
            label_encoder=plan.label_encoder_,
            classes=plan.classes_,
        )
        label_encoder = encoder
    else:
        y = regression_targets(chunk[plan.target_column])
        label_encoder = plan.label_encoder_

    estimator_obj = plan.estimator_
    used_refit = False
    update_mode = "partial_fit"
    drift_events_before = (
        len(getattr(estimator_obj, "drift_events_", []) or [])
        if hasattr(estimator_obj, "drift_events_")
        else 0
    )
    try:
        if hasattr(estimator_obj, "partial_fit"):
            if plan.task == "classification":
                class_codes = np.arange(len(plan.classes_ or ()))
                estimator_obj.partial_fit(x, y, classes=class_codes)
            else:
                estimator_obj.partial_fit(x, y)
        else:
            # Cumulative refit over all seen rows + this chunk (disclosed).
            used_refit, update_mode, estimator_obj = _refit_cumulative(
                dataset,
                plan,
                split_plan,
                chunk=chunk,
                chunk_indices=chunk_indices,
                external=external,
                allow_refit_fallback=allow_refit,
                warnings=warnings,
                disclosures=disclosures,
            )
    except ValidationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(
            f"partial_fit_online failed for estimator={plan.estimator_name!r}: {exc}"
        ) from exc

    drift_note_list = list(
        chunk_drift_notes(
            x,
            plan.init_feature_means_,
            columns=plan.columns,
            enabled=drift_on
            and str((plan.config or {}).get("drift_detector", "mean_shift")) == "mean_shift",
        )
    )
    if drift_on and hasattr(estimator_obj, "drift_events_"):
        after_events = getattr(estimator_obj, "drift_events_", []) or []
        if len(after_events) > drift_events_before:
            for event in after_events[drift_events_before:]:
                drift_note_list.append(
                    f"River {event.get('detector')} drift on update "
                    f"(n_seen={event.get('n_seen')})."
                )
    drift_notes = tuple(drift_note_list)
    disclosures.extend(drift_notes)

    seen = list(plan.seen_train_indices)
    if not external:
        for idx in chunk_indices:
            if idx not in seen:
                seen.append(idx)
    n_seen = len(seen) if not external else plan.n_seen_rows + len(chunk_indices)
    n_updates = int(plan.n_updates) + 1
    history = list(plan.update_history)
    history.append(
        {
            "round": n_updates,
            "kind": "external_frame" if external else "train_chunk",
            "n_rows": len(chunk_indices),
            "indices": list(chunk_indices) if not external else [],
            "update_mode": update_mode,
            "used_refit_fallback": used_refit,
            "drift_notes": list(drift_notes),
            "backend": plan.backend,
        }
    )
    remaining = 0
    if split_plan is not None and not external:
        remaining = max(0, len(split_plan.train_indices) - new_cursor)
    elif split_plan is not None:
        remaining = max(0, len(split_plan.train_indices) - plan.cursor)

    disclosures.append(
        f"Update #{n_updates}: n_chunk_rows={len(chunk_indices)}, "
        f"n_seen_rows={n_seen}, mode={update_mode}."
    )

    new_plan = OnlinePlan(
        estimator_name=plan.estimator_name,
        task=plan.task,
        columns=plan.columns,
        target_column=plan.target_column,
        n_train_rows=plan.n_train_rows,
        n_seen_rows=n_seen,
        n_updates=n_updates,
        cursor=new_cursor if not external else plan.cursor,
        chunk_size=plan.chunk_size,
        classes_=plan.classes_,
        seen_train_indices=tuple(seen),
        update_history=tuple(history),
        backend=plan.backend,
        estimator_=estimator_obj,
        label_encoder_=label_encoder,
        init_feature_means_=plan.init_feature_means_,
        used_refit_fallback=bool(plan.used_refit_fallback or used_refit),
        disclosures=tuple(dict.fromkeys([*plan.disclosures, *disclosures])),
        warnings=tuple(dict.fromkeys([*plan.warnings, *warnings])),
        used_reduce_components=plan.used_reduce_components,
        config=dict(plan.config),
    )
    result = OnlineUpdateResult(
        estimator_name=plan.estimator_name,
        task=plan.task,
        n_chunk_rows=len(chunk_indices),
        n_seen_rows=n_seen,
        n_updates=n_updates,
        n_remaining_train=remaining,
        update_mode=update_mode,
        drift_notes=drift_notes,
        used_refit_fallback=used_refit,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
    return new_plan, result


# Package-level alias matching sklearn naming.
partial_fit = partial_fit_online


def _refit_cumulative(
    dataset: Dataset,
    plan: OnlinePlan,
    split_plan: SplitPlan | None,
    *,
    chunk: pd.DataFrame,
    chunk_indices: Sequence[Any],
    external: bool,
    allow_refit_fallback: bool,
    warnings: list[str],
    disclosures: list[str],
) -> tuple[bool, str, Any]:
    """Explicit disclosed full refit on cumulative seen + current chunk."""
    if not allow_refit_fallback:
        # Reuse shared error messaging.
        _maybe_refit_fallback(
            plan.estimator_,
            np.empty((0, len(plan.columns))),
            np.empty((0,)),
            allow_refit_fallback=False,
            estimator_name=plan.estimator_name,
            warnings=warnings,
            disclosures=disclosures,
        )
    frames: list[pd.DataFrame] = []
    if split_plan is not None and plan.seen_train_indices:
        full = dataset._ensure_pandas()
        frames.append(full.loc[list(plan.seen_train_indices)])
    frames.append(chunk)
    combined = pd.concat(frames, axis=0)
    # Drop duplicate indices when current chunk overlaps seen.
    combined = combined[~combined.index.duplicated(keep="last")]
    x = matrix_from_frame(combined, list(plan.columns))
    if plan.task == "classification":
        y, _, _ = encode_classification_targets(
            combined[plan.target_column],
            label_encoder=plan.label_encoder_,
            classes=plan.classes_,
        )
    else:
        y = regression_targets(combined[plan.target_column])
    plan.estimator_.fit(x, y)
    msg = (
        f"REFIT FALLBACK (disclosed): estimator={plan.estimator_name!r} lacks "
        f"partial_fit; full .fit on n={len(combined)} cumulative rows "
        f"(external={external}). This is not incremental online learning."
    )
    warnings.append(msg)
    disclosures.append(msg)
    return True, "refit_fallback", plan.estimator_
