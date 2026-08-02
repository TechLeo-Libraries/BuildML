"""Session / Dataset partitions → Torch DataLoaders."""

from __future__ import annotations

from typing import Any

import pandas as pd

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition
from buildml.dl.dataset import arrays_to_tensor_dataset, build_feature_contract
from buildml.dl.extras import require_torch
from buildml.dl.results import LoaderReport, TorchLoaderBundle
from buildml.dl.types import LoaderConfig, TaskSpec


def _group_column(dataset: Dataset, split_plan: SplitPlan) -> str | None:
    if split_plan.kind != "group":
        return None
    if split_plan.stratify_column:
        return split_plan.stratify_column
    cols = dataset.role_columns(ColumnRole.GROUP)
    return cols[0] if cols else None


def _time_column(dataset: Dataset, split_plan: SplitPlan) -> str | None:
    if split_plan.kind != "time":
        return None
    if split_plan.stratify_column:
        return split_plan.stratify_column
    cols = dataset.role_columns(ColumnRole.TIME)
    return cols[0] if cols else None


def _verify_group_disjoint(
    dataset: Dataset,
    split_plan: SplitPlan,
    group_column: str,
) -> tuple[bool, list[str]]:
    """Confirm group membership does not cross partition loaders."""
    warnings: list[str] = []
    frame = dataset._ensure_pandas()
    groups = frame[group_column]
    membership: dict[str, set[Any]] = {}
    for name in ("train", "validation", "test"):
        idx = list(split_plan.indices_for(name))  # type: ignore[arg-type]
        if not idx:
            membership[name] = set()
            continue
        membership[name] = set(groups.iloc[idx].tolist())
    overlaps: list[str] = []
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    for left, right in pairs:
        shared = membership[left] & membership[right]
        if shared:
            overlaps.append(f"{left}/{right}: {len(shared)} shared group(s)")
    if overlaps:
        raise ValidationError(
            "Group split leakage: the same group appears in multiple partitions — "
            + "; ".join(overlaps)
        )
    return True, warnings


def _verify_time_order(
    dataset: Dataset,
    split_plan: SplitPlan,
    time_column: str,
) -> tuple[bool, list[str]]:
    """Confirm chronological partition boundaries (max train < min val < min test)."""
    warnings: list[str] = []
    stamps = pd.to_datetime(dataset._ensure_pandas()[time_column], errors="coerce")
    bounds: dict[str, tuple[Any, Any] | None] = {}
    for name in ("train", "validation", "test"):
        idx = list(split_plan.indices_for(name))  # type: ignore[arg-type]
        if not idx:
            bounds[name] = None
            continue
        part_stamps = stamps.iloc[idx]
        if part_stamps.isna().any():
            raise ValidationError(
                f"Time split partition '{name}' has unparseable timestamps in '{time_column}'."
            )
        bounds[name] = (part_stamps.min(), part_stamps.max())

    def _after(earlier: str, later: str) -> None:
        left = bounds.get(earlier)
        right = bounds.get(later)
        if left is None or right is None:
            return
        if left[1] > right[0]:
            raise ValidationError(
                f"Time split leakage: max({earlier})={left[1]} is after min({later})={right[0]}."
            )

    _after("train", "validation")
    _after("train", "test")
    _after("validation", "test")
    return True, warnings


def make_loaders(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    config: LoaderConfig | None = None,
    task: TaskSpec = "auto",
    classical_plans: dict[str, Any] | None = None,
) -> TorchLoaderBundle:
    """Build train / validation / test DataLoaders from a split Dataset.

    Shuffle is applied to the **train** loader only. Normalize statistics, when
    enabled, are fit on train and frozen for validation/test.

    Group and time :class:`~buildml.data.splits.SplitPlan` kinds are honored via
    partition index membership. Group splits are checked for cross-partition
    group leakage; time splits are checked for chronological boundary order.

    When ``classical_plans`` is provided (plan name → object/summary), the
    report discloses that loaders read the **current** frame. Session impute /
    encode / scale already mutate that frame with train-fitted transforms;
    this bridge records the relationship rather than silently refitting.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None
    require_torch(feature="Torch DataLoaders")
    cfg = config or LoaderConfig()
    if cfg.batch_size < 1:
        raise ValidationError("batch_size must be >= 1")

    contract, arrays = build_feature_contract(
        dataset,
        split_plan,
        task=task,
        normalize=cfg.normalize,
    )

    warnings: list[str] = []
    if classical_plans:
        names = sorted(str(k) for k, v in classical_plans.items() if v is not None)
        if names:
            warnings.append(
                "Classical preprocess plans attached ("
                + ", ".join(names)
                + "). Loaders materialize tensors from the current Session frame "
                "(train-fitted Session transforms already applied when you called "
                "impute/encode/scale). Torch normalize, when enabled, is an additional "
                "train-fit mean/std on top of that frame."
            )
    group_column = _group_column(dataset, split_plan)
    time_column = _time_column(dataset, split_plan)
    groups_disjoint: bool | None = None
    time_order_ok: bool | None = None

    if split_plan.kind == "group":
        if group_column is None:
            raise ValidationError(
                "Group SplitPlan requires a group column (role or stratify_column)."
            )
        groups_disjoint, extra = _verify_group_disjoint(dataset, split_plan, group_column)
        warnings.extend(extra)
    elif split_plan.kind == "time":
        if time_column is None:
            raise ValidationError(
                "Time SplitPlan requires a time column (role or stratify_column)."
            )
        time_order_ok, extra = _verify_time_order(dataset, split_plan, time_column)
        warnings.extend(extra)
        if cfg.shuffle_train:
            warnings.append(
                "Time split: train loader shuffle is within the past window only; "
                "it does not pull future rows into train. Sequence models that need "
                "ordered batches should set shuffle_train=False."
            )

    torch = require_torch(feature="Torch DataLoaders")
    generator = torch.Generator()
    generator.manual_seed(int(cfg.seed))

    loaders: dict[str, Any] = {}
    for name in ("train", "validation", "test"):
        x, y = arrays[name]
        if len(x) == 0:
            if name == "train":
                raise ValidationError("Train partition is empty")
            warnings.append(f"Partition '{name}' is empty; no DataLoader created.")
            continue
        # Row-count gate: tensors must match SplitPlan membership size.
        expected = len(split_plan.indices_for(name))  # type: ignore[arg-type]
        if len(x) != expected:
            raise ValidationError(
                f"Loader partition '{name}' has {len(x)} rows but SplitPlan lists "
                f"{expected} indices — refusing to proceed (possible leakage)."
            )
        dataset_t = arrays_to_tensor_dataset(x, y, task=contract.task)
        shuffle = bool(cfg.shuffle_train and name == "train")
        loaders[name] = torch.utils.data.DataLoader(
            dataset_t,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory and name == "train",
            drop_last=cfg.drop_last and name == "train",
            generator=generator if shuffle else None,
        )

    report = LoaderReport(
        batch_size=cfg.batch_size,
        shuffle_train=cfg.shuffle_train,
        normalize=cfg.normalize,
        feature_columns=contract.feature_columns,
        target_column=contract.target_column,
        task=contract.task,
        n_train=int(len(arrays["train"][0])),
        n_validation=int(len(arrays["validation"][0])),
        n_test=int(len(arrays["test"][0])),
        class_labels=contract.class_labels,
        warnings=warnings,
        split_kind=split_plan.kind,
        group_column=group_column,
        time_column=time_column,
        groups_disjoint=groups_disjoint,
        time_order_ok=time_order_ok,
    )
    return TorchLoaderBundle(loaders=loaders, contract=contract, report=report)
