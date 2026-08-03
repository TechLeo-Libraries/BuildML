"""Turn a split Dataset into DataLoaders, checking the split as it goes.

A DataLoader is the thing a Torch training loop iterates: it batches rows,
optionally shuffles them, and hands tensors to the model. Building one from a
BuildML Dataset is mostly mechanical, and this module does that work — but it
also verifies the split before creating anything, which is the part worth
knowing about.

Three checks run here. Group splits are verified to have no group appearing in
more than one partition, because a customer whose rows straddle train and test
lets the model memorise that customer and score well on them. Time splits are
verified to be chronologically ordered, because training on rows dated after the
test rows is predicting the past from the future. And every partition's tensor
count is compared against the split plan's index count, since a mismatch means
rows moved somewhere between the plan and the tensors.

Each of these raises rather than warns. They are all forms of leakage, and
leakage produces a good score and a bad model — the failure mode most worth
stopping early.

See Also
--------
buildml.dl.dataset : Building the arrays these loaders wrap.
buildml.data.splits : Where the partitions come from.
buildml.dl.train : What consumes the result.
"""

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
    """Build the DataLoaders a training loop needs, verifying the split first.

    Materialises each partition as tensors, wraps them as DataLoaders, and
    returns them alongside the feature contract and a report of what was built.
    Before any of that, the split is checked for the leakage patterns its kind
    is prone to.

    Parameters
    ----------
    dataset:
        The data, with roles and a target assigned.
    split_plan:
        Which rows belong to which partition. Must include a train partition.
    config:
        Batching and normalisation settings. Defaults are reasonable for
        tabular work.
    task:
        ``'auto'`` to infer classification versus regression from the training
        targets, or an explicit choice.
    classical_plans:
        Preprocessing plans already applied to the Session frame. Passing them
        does not change what is built; it adds a disclosure to the report
        recording that Torch normalisation sits on top of transforms that
        already ran.

    Returns
    -------
    TorchLoaderBundle
        The loaders keyed by partition, the feature contract, and the report.
        Partitions with no rows are absent from the loaders rather than present
        and empty.

    Raises
    ------
    MissingExtraError
        If PyTorch is not installed. Install with ``pip install buildml[dl]``.
    ValidationError
        If there is no train partition or it is empty, if ``batch_size`` is
        below 1, if a group or time split lacks the column it needs, if group
        membership crosses partitions, if time boundaries are out of order, or
        if a partition's row count disagrees with the split plan.

    Notes
    -----
    **Only the train loader shuffles.** Shuffling changes the order gradients
    arrive in, which matters for learning and not at all for scoring — so
    validation and test iterate in a fixed order, keeping their metrics
    reproducible.

    **Shuffling a time split is safe but may not be what you want.** It reorders
    rows within the training window and never pulls future rows in, so there is
    no leakage. But a sequence model that expects consecutive batches to be
    consecutive in time will be fed nonsense; set ``shuffle_train=False`` for
    those.

    **Read ``bundle.report.warnings``.** Empty partitions and the classical-plan
    disclosure appear there and nowhere else.

    Examples
    --------
    Build loaders and inspect what was created::

        bundle = make_loaders(dataset, split_plan)
        bundle.report.n_train
        bundle.report.groups_disjoint  # None unless this is a group split

    See Also
    --------
    buildml.dl.train.train_supervised_module : Consumes the bundle.
    buildml.dl.types.LoaderConfig : The settings.
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
