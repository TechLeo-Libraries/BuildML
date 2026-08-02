"""Train subsampling helpers for TDA max_points_guard."""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.data.splits import SplitPlan

SubsampleStrategy = Literal["error", "random", "stratified"]


def apply_train_subsample(
    split_plan: SplitPlan,
    frame: pd.DataFrame,
    *,
    max_points: int,
    strategy: SubsampleStrategy,
    target_column: str | None,
    random_state: int | None,
) -> tuple[SplitPlan, pd.DataFrame, list[str], list[str]]:
    """Subsample train rows when above ``max_points`` per ``strategy``.

    Returns a shallow copy of ``split_plan`` with reduced train indices,
    the subsampled train frame, disclosures, and warnings.
    """
    disclosures: list[str] = []
    warnings: list[str] = []
    train_idx = list(split_plan.train_indices)
    n_train = len(train_idx)
    cap = int(max_points)
    if n_train <= cap:
        return split_plan, frame, disclosures, warnings

    if strategy == "error":
        raise ValidationError(
            f"Train size {n_train} exceeds max_points_guard={cap}. "
            "Subsample before fit_tda, raise max_points_guard, or set "
            "subsample_strategy='random'|'stratified'."
        )

    rng = np.random.default_rng(random_state)
    if strategy == "stratified":
        if target_column is None or target_column not in frame.columns:
            warnings.append(
                "subsample_strategy='stratified' requires a target column; "
                "falling back to random subsample."
            )
            chosen = _random_subsample(train_idx, cap, rng)
        else:
            chosen = _stratified_subsample(
                train_idx, frame[target_column], cap, rng
            )
        disclosures.append(
            f"Stratified train subsample: {n_train} → {len(chosen)} rows "
            f"(max_points_guard={cap})."
        )
    elif strategy == "random":
        chosen = _random_subsample(train_idx, cap, rng)
        disclosures.append(
            f"Random train subsample: {n_train} → {len(chosen)} rows "
            f"(max_points_guard={cap})."
        )
    else:
        raise ValidationError(
            f"Unknown subsample_strategy {strategy!r}; expected error, random, stratified."
        )

    new_plan = SplitPlan(
        kind=split_plan.kind,
        test_size=split_plan.test_size,
        validation_size=split_plan.validation_size,
        random_state=split_plan.random_state,
        stratify_column=split_plan.stratify_column,
        train_indices=tuple(chosen),
        validation_indices=split_plan.validation_indices,
        test_indices=split_plan.test_indices,
    )
    sub_frame = frame.loc[chosen].reset_index(drop=True)
    return new_plan, sub_frame, disclosures, warnings


def _random_subsample(
    train_idx: Sequence[int], cap: int, rng: np.random.Generator
) -> list[int]:
    pool = list(train_idx)
    if len(pool) <= cap:
        return pool
    pick = rng.choice(len(pool), size=cap, replace=False)
    return [pool[int(i)] for i in sorted(pick)]


def _stratified_subsample(
    train_idx: Sequence[int],
    y: pd.Series,
    cap: int,
    rng: np.random.Generator,
) -> list[int]:
    idx = list(train_idx)
    if len(idx) <= cap:
        return idx
    labels = y.iloc[idx]
    groups: dict[str, list[int]] = {}
    for i, lab in zip(idx, labels, strict=True):
        key = str(lab)
        groups.setdefault(key, []).append(i)
    per_class = max(1, cap // max(len(groups), 1))
    chosen: list[int] = []
    for class_idx in groups.values():
        if len(chosen) >= cap:
            break
        take = min(per_class, len(class_idx), cap - len(chosen))
        if take >= len(class_idx):
            chosen.extend(class_idx)
        else:
            pick = rng.choice(len(class_idx), size=take, replace=False)
            chosen.extend(class_idx[int(j)] for j in pick)
    if len(chosen) < cap:
        remaining = [i for i in idx if i not in set(chosen)]
        need = cap - len(chosen)
        if remaining and need > 0:
            pick = rng.choice(len(remaining), size=min(need, len(remaining)), replace=False)
            chosen.extend(remaining[int(j)] for j in pick)
    return chosen[:cap]
