"""Train-only imbalance resampling with a strategy registry.

Leakage contract
----------------
Only the **train** partition is resampled. Validation and test row sets are
concatenated unchanged (order preserved within each holdout partition). A new
``SplitPlan`` is rebuilt so membership stays disjoint.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from buildml.core.errors import MissingExtraError, ValidationError
from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition, frame_for_partition
from buildml.ingest.detect import schema_from_dataframe

SamplerName = Literal[
    "smote",
    "random_oversample",
    "random_undersample",
    "adasyn",
    "borderline_smote",
]


@dataclass(frozen=True, slots=True)
class SamplerStrategy:
    """Registered resampling strategy metadata."""

    name: SamplerName
    family: Literal["over", "under", "synthetic"]
    requires_numeric_features: bool
    description: str
    when_to_use: str
    factory: Callable[[int], Any] = field(repr=False, compare=False)


def _load_imblearn() -> Any:
    try:
        import imblearn
    except ImportError as exc:
        raise MissingExtraError("imbalanced", "Train-only resampling") from exc
    return imblearn


def _strategy_registry() -> dict[str, SamplerStrategy]:
    """Build strategy registry (lazy imblearn imports inside factories)."""

    def smote(rs: int) -> Any:
        from imblearn.over_sampling import SMOTE

        return SMOTE(random_state=rs)

    def random_over(rs: int) -> Any:
        from imblearn.over_sampling import RandomOverSampler

        return RandomOverSampler(random_state=rs)

    def random_under(rs: int) -> Any:
        from imblearn.under_sampling import RandomUnderSampler

        return RandomUnderSampler(random_state=rs)

    def adasyn(rs: int) -> Any:
        from imblearn.over_sampling import ADASYN

        return ADASYN(random_state=rs)

    def borderline(rs: int) -> Any:
        from imblearn.over_sampling import BorderlineSMOTE

        return BorderlineSMOTE(random_state=rs)

    strategies = [
        SamplerStrategy(
            name="smote",
            family="synthetic",
            requires_numeric_features=True,
            description="Synthetic Minority Over-sampling Technique (k-NN interpolation).",
            when_to_use=(
                "Numeric feature spaces with moderate imbalance; avoid with "
                "pure categoricals or tiny minority counts (<k_neighbors)."
            ),
            factory=smote,
        ),
        SamplerStrategy(
            name="random_oversample",
            family="over",
            requires_numeric_features=False,
            description="Random oversampling with replacement of minority classes.",
            when_to_use=(
                "Quick baseline; works with mixed types after encoding. "
                "Can overfit duplicated minority rows."
            ),
            factory=random_over,
        ),
        SamplerStrategy(
            name="random_undersample",
            family="under",
            requires_numeric_features=False,
            description="Random undersampling of majority classes.",
            when_to_use=(
                "Large majority classes where dropping rows is acceptable; "
                "risks discarding useful majority diversity."
            ),
            factory=random_under,
        ),
        SamplerStrategy(
            name="adasyn",
            family="synthetic",
            requires_numeric_features=True,
            description="Adaptive synthetic sampling focusing on hard minority regions.",
            when_to_use=(
                "Numeric features where minority examples near the decision "
                "boundary need denser coverage."
            ),
            factory=adasyn,
        ),
        SamplerStrategy(
            name="borderline_smote",
            family="synthetic",
            requires_numeric_features=True,
            description="Borderline-SMOTE synthesizing near class boundaries.",
            when_to_use=(
                "Numeric features with overlap; concentrates synthesis on "
                "borderline minority samples."
            ),
            factory=borderline,
        ),
    ]
    return {s.name: s for s in strategies}


def list_resample_strategies() -> list[dict[str, Any]]:
    """Return public metadata for available resampling strategies."""
    return [
        {
            "name": s.name,
            "family": s.family,
            "requires_numeric_features": s.requires_numeric_features,
            "description": s.description,
            "when_to_use": s.when_to_use,
            "extra": "imbalanced",
        }
        for s in _strategy_registry().values()
    ]


@dataclass(slots=True)
class ResamplePlan:
    """Structured outcome of a train-only resample operation."""

    sampler: SamplerName
    n_train_before: int
    n_train_after: int
    class_counts_before: dict[str, int]
    class_counts_after: dict[str, int]
    n_validation_unchanged: int = 0
    n_test_unchanged: int = 0
    feature_columns: tuple[str, ...] = ()
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sampler": self.sampler,
            "n_train_before": self.n_train_before,
            "n_train_after": self.n_train_after,
            "class_counts_before": dict(self.class_counts_before),
            "class_counts_after": dict(self.class_counts_after),
            "n_validation_unchanged": self.n_validation_unchanged,
            "n_test_unchanged": self.n_test_unchanged,
            "feature_columns": list(self.feature_columns),
            "notes": list(self.notes),
            "delta_train_rows": int(self.n_train_after - self.n_train_before),
        }


def resample_train(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    sampler: SamplerName | str = "smote",
    random_state: int = 42,
    sampling_strategy: str | float | dict[str, float] = "auto",
) -> tuple[Dataset, SplitPlan, ResamplePlan]:
    """Resample **train rows only**, then rebuild dataset/split membership.

    Parameters
    ----------
    dataset:
        Dataset with a target role assigned.
    split_plan:
        Required split; fit-scope must be train.
    sampler:
        Strategy name from :func:`list_resample_strategies`.
    random_state:
        Sampler RNG seed.
    sampling_strategy:
        Forwarded to imbalanced-learn (``"auto"`` balances classes).

    Returns
    -------
    tuple[Dataset, SplitPlan, ResamplePlan]
        New dataset, rebuilt split plan, and before/after class-count report.

    Notes
    -----
    **Leakage:** Validation/test rows are never resampled or used to fit the
    sampler. Call only after :meth:`Session.split`.

    **Extra:** Requires ``pip install 'buildml[imbalanced]'``.

    Raises
    ------
    LeakageError
        If no split exists or fit partition is not train.
    MissingExtraError
        If imbalanced-learn is not installed.
    ValidationError
        For unknown strategies, non-numeric features with synthetic samplers,
        or insufficient minority support.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None
    _load_imblearn()  # clear MissingExtraError early

    registry = _strategy_registry()
    if sampler not in registry:
        known = ", ".join(sorted(registry))
        raise ValidationError(
            f"Unknown sampler '{sampler}'. Known strategies: {known}. "
            "Call buildml.preprocess.imbalance.list_resample_strategies()."
        )
    strategy = registry[str(sampler)]

    target = dataset.require_target()
    feature_cols = dataset.role_columns(ColumnRole.FEATURE)
    if not feature_cols:
        feature_cols = [c for c in dataset.columns if c != target]
    if not feature_cols:
        raise ValidationError("No feature columns available for resampling")

    train = frame_for_partition(dataset, split_plan, "train")
    valid = (
        frame_for_partition(dataset, split_plan, "validation")
        if split_plan.validation_indices
        else None
    )
    test = frame_for_partition(dataset, split_plan, "test")

    # Holdout fingerprints — must remain byte-identical after concat rebuild.
    valid_fingerprint = None if valid is None else valid.reset_index(drop=True).copy()
    test_fingerprint = test.reset_index(drop=True).copy()

    x_train = train[feature_cols]
    y_train = train[target]
    before_counts = y_train.astype(str).value_counts().to_dict()
    notes: list[str] = []

    if len(pd.unique(y_train)) < 2:
        raise ValidationError("Resampling requires at least two classes in the train partition")

    minority = int(y_train.astype(str).value_counts().min())
    if minority < 2 and strategy.family == "synthetic":
        raise ValidationError(
            f"Sampler '{strategy.name}' needs ≥2 minority samples in train; "
            f"found minority_count={minority}. Use random_oversample or gather more data."
        )

    if strategy.requires_numeric_features:
        non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(x_train[c])]
        if non_numeric:
            raise ValidationError(
                f"Sampler '{strategy.name}' requires numeric features; "
                f"non-numeric columns present: {non_numeric[:12]}. "
                "Encode categoricals first, or use random_oversample / "
                "random_undersample."
            )
        if x_train.isna().any().any():
            raise ValidationError(
                f"Sampler '{strategy.name}' cannot run with NaNs in train features. "
                "Call session.impute(...) before resample."
            )

    sampler_obj = strategy.factory(random_state)
    if hasattr(sampler_obj, "sampling_strategy") and sampling_strategy != "auto":
        sampler_obj.set_params(sampling_strategy=sampling_strategy)

    try:
        x_res, y_res = sampler_obj.fit_resample(x_train, y_train)
    except ValueError as exc:
        raise ValidationError(
            f"Resampling with '{strategy.name}' failed: {exc}. "
            "Check class counts, k_neighbors, and that features are finite."
        ) from exc

    train_res = pd.DataFrame(x_res, columns=feature_cols)
    train_res[target] = y_res
    for col in train.columns:
        if col not in train_res.columns:
            # Preserve non-feature/target columns with a train-mode fill.
            if col not in feature_cols + [target]:
                mode = train[col].mode(dropna=True)
                train_res[col] = mode.iloc[0] if len(mode) else pd.NA
            else:
                train_res[col] = pd.NA

    # Column order aligned to original dataset.
    ordered_cols = list(dataset.columns)
    for col in ordered_cols:
        if col not in train_res.columns:
            train_res[col] = pd.NA
    train_res = train_res[ordered_cols]

    parts = [train_res.reset_index(drop=True)]
    if valid_fingerprint is not None and len(valid_fingerprint):
        parts.append(valid_fingerprint)
    parts.append(test_fingerprint)
    combined = pd.concat(parts, ignore_index=True)

    n_train = len(train_res)
    n_valid = len(valid_fingerprint) if valid_fingerprint is not None else 0
    train_idx = tuple(range(0, n_train))
    valid_idx = tuple(range(n_train, n_train + n_valid)) if n_valid else ()
    test_idx = tuple(range(n_train + n_valid, len(combined)))

    new_plan = SplitPlan(
        kind=f"resampled_{split_plan.kind}",
        test_size=split_plan.test_size,
        validation_size=split_plan.validation_size,
        random_state=split_plan.random_state,
        stratify_column=split_plan.stratify_column,
        train_indices=train_idx,
        validation_indices=valid_idx,
        test_indices=test_idx,
    )
    new_plan.assert_disjoint()

    # Leakage guard: holdout frames unchanged.
    if valid_fingerprint is not None and n_valid:
        rebuilt_valid = combined.iloc[list(valid_idx)].reset_index(drop=True)
        if not rebuilt_valid.equals(valid_fingerprint[ordered_cols]):
            raise ValidationError(
                "Internal leakage guard failed: validation partition changed during resample"
            )
    rebuilt_test = combined.iloc[list(test_idx)].reset_index(drop=True)
    if not rebuilt_test.equals(test_fingerprint[ordered_cols]):
        raise ValidationError(
            "Internal leakage guard failed: test partition changed during resample"
        )

    roles = dict(dataset.roles)
    new_dataset = Dataset.from_transformed(
        dataset,
        combined,
        schema=schema_from_dataframe(combined),
        roles=roles,
    )
    after_counts = pd.Series(y_res).astype(str).value_counts().to_dict()
    notes.append(f"Strategy '{strategy.name}' ({strategy.family}): {strategy.description}")
    notes.append(strategy.when_to_use)
    if strategy.family == "over" or strategy.family == "synthetic":
        notes.append(
            "Train grew via oversampling/synthesis — evaluate on untouched validation/test only."
        )
    else:
        notes.append("Train shrank via undersampling — monitor majority-class recall on holdout.")

    # Balance ratio tip
    before_ratio = _imbalance_ratio(before_counts)
    after_ratio = _imbalance_ratio(after_counts)
    notes.append(f"Imbalance ratio (max/min class count): {before_ratio:.3f} → {after_ratio:.3f}.")

    plan = ResamplePlan(
        sampler=strategy.name,  # type: ignore[arg-type]
        n_train_before=int(len(train)),
        n_train_after=int(n_train),
        class_counts_before={str(k): int(v) for k, v in before_counts.items()},
        class_counts_after={str(k): int(v) for k, v in after_counts.items()},
        n_validation_unchanged=int(n_valid),
        n_test_unchanged=int(len(test_fingerprint)),
        feature_columns=tuple(feature_cols),
        notes=notes,
    )
    return new_dataset, new_plan, plan


def _imbalance_ratio(counts: dict[Any, int]) -> float:
    vals = [int(v) for v in counts.values() if int(v) > 0]
    if not vals:
        return float("nan")
    return float(max(vals) / max(1, min(vals)))
