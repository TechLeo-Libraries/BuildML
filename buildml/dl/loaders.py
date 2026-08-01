"""Session / Dataset partitions → Torch DataLoaders."""

from __future__ import annotations

from typing import Any

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition
from buildml.dl.dataset import arrays_to_tensor_dataset, build_feature_contract
from buildml.dl.extras import require_torch
from buildml.dl.results import LoaderReport, TorchLoaderBundle
from buildml.dl.types import LoaderConfig, TaskSpec


def make_loaders(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    config: LoaderConfig | None = None,
    task: TaskSpec = "auto",
) -> TorchLoaderBundle:
    """Build train / validation / test DataLoaders from a split Dataset.

    Shuffle is applied to the **train** loader only. Normalize statistics, when
    enabled, are fit on train and frozen for validation/test.
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

    torch = require_torch(feature="Torch DataLoaders")
    generator = torch.Generator()
    generator.manual_seed(int(cfg.seed))

    loaders: dict[str, Any] = {}
    warnings: list[str] = []
    for name in ("train", "validation", "test"):
        x, y = arrays[name]
        if len(x) == 0:
            if name == "train":
                raise ValidationError("Train partition is empty")
            warnings.append(f"Partition '{name}' is empty; no DataLoader created.")
            continue
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
    )
    return TorchLoaderBundle(loaders=loaders, contract=contract, report=report)
