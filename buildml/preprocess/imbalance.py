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
    """Describe every resampling strategy available, so you can pick one.

    Use this before :func:`resample_train` to see what exists and what each
    approach is suited to, rather than guessing at a name.

    Returns
    -------
    list of dict
        One entry per strategy, each with ``name`` (the value to pass as
        ``sampler``), ``family`` (whether it adds minority rows, removes
        majority rows, or does both), ``requires_numeric_features`` (synthetic
        samplers interpolate between rows and so cannot work on text or raw
        categories), ``description``, ``when_to_use``, and ``extra`` naming the
        optional dependency group needed.

    Notes
    -----
    The list is static: it describes what the library supports, not what your
    environment has installed. Every strategy here needs
    ``pip install 'buildml[imbalanced]'`` before it can actually run.

    See Also
    --------
    resample_train : Applies one of these strategies.
    """
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
    """What resampling did to the training rows, and what it deliberately left alone.

    Unlike the other plans in this package, this one is a record rather than a
    replayable transform. Resampling changes the training set to help the model
    learn; it is never applied at inference time, when you want the real class
    distribution.

    Attributes
    ----------
    sampler:
        Which strategy ran.
    n_train_before:
        Training rows before resampling.
    n_train_after:
        Training rows after. Compare the two: oversampling to balance a
        1-in-1000 class means roughly a 500-fold increase in rows, which is
        both a memory concern and a sign that the balance you asked for may be
        too aggressive.
    class_counts_before:
        Rows per class before, which is the honest picture of your data.
    class_counts_after:
        Rows per class after, which is what the model will see.
    n_validation_unchanged:
        Validation rows, confirmed untouched.
    n_test_unchanged:
        Test rows, confirmed untouched. These two fields exist so the record
        itself proves the holdout still reflects reality.
    feature_columns:
        The feature columns the sampler operated over.
    notes:
        Observations recorded during resampling, such as a class with too few
        examples to synthesise from safely.
    """

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
        """Return the resampling record as plain JSON-safe values.

        Belongs in a model card: a reader needs to know the training
        distribution was altered before they can interpret the model's
        behaviour.

        Returns
        -------
        dict
            Every attribute in plain-data form, plus ``delta_train_rows``
            giving the net change in training row count.
        """
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
    """Rebalance the training classes, leaving validation and test untouched.

    When one class is rare: fraud, equipment failure, disease: a model can
    reach 99% accuracy by never predicting it, and gradient descent will
    happily settle there because the rare class contributes so little to the
    loss. Resampling changes the training distribution so the rare class
    carries real weight: either by duplicating or synthesising minority rows,
    by discarding majority rows, or by doing both.

    Only the training rows are altered. Validation and test rows keep the true
    distribution, because those are what tell you how the model will behave in
    a world where the rare class really is rare. Balancing them would produce a
    score describing a world that does not exist.

    Because rows are added or removed, positions shift, so a rebuilt split plan
    comes back alongside the new dataset. Use it in place of your old one.

    Parameters
    ----------
    dataset:
        The dataset, which must have a target role assigned: there is no class
        balance without knowing which column holds the class.
    split_plan:
        The split defining the training rows. Required: resampling before
        splitting puts synthesised copies of training rows into your test set,
        which is one of the more spectacular ways to produce a meaningless
        score.
    sampler:
        Which strategy to use, named from :func:`list_resample_strategies`.
        ``'smote'`` is the common default: it synthesises new minority rows by
        interpolating between existing ones, which avoids the exact-duplicate
        overfitting that naive oversampling causes. Undersampling strategies
        discard majority rows instead: faster and lighter, but you are
        throwing away real data.
    random_state:
        Seed for the sampler, so the synthesised rows reproduce.
    sampling_strategy:
        How far to rebalance, forwarded to imbalanced-learn. ``'auto'``
        equalises the classes. A float sets the desired minority-to-majority
        ratio, and partial rebalancing to something like ``0.3`` is often
        better than full equality: it gives the rare class weight without
        flooding the training set with synthetic rows. A dict specifies target
        counts per class.

    Returns
    -------
    tuple of (~buildml.data.dataset.Dataset, ~buildml.data.splits.SplitPlan, ResamplePlan)
        The resampled dataset; the rebuilt split plan you must use from here
        on; and the before-and-after record.

    Raises
    ------
    ~buildml.core.errors.LeakageError
        No split exists, or the fit partition is not the training set.
    ~buildml.core.errors.MissingExtraError
        imbalanced-learn is not installed. Run
        ``pip install 'buildml[imbalanced]'``.
    ~buildml.core.errors.ValidationError
        The strategy name is unknown, a synthetic sampler was given
        non-numeric features, or a class has too few examples to synthesise
        from.

    Notes
    -----
    **Consider the alternatives first.** Most estimators accept
    ``class_weight='balanced'``, which achieves much the same effect by
    reweighting the loss rather than fabricating rows: cheaper, and it does
    not invent data. Adjusting the decision threshold with
    :meth:`~buildml.session.Session.tune_threshold` is often better still,
    since it addresses the real problem, which is usually that the default
    0.5 cutoff is wrong for your cost structure.

    **Synthetic rows are interpolations, not observations.** SMOTE creates a
    row partway between two real minority rows. If the minority class is
    genuinely multi-modal, the midpoint between two clusters may be a
    combination that could never occur.

    **Order matters.** Resample after encoding and imputing, since synthetic
    samplers need complete numeric input, and after splitting, always.

    **Your probabilities will be miscalibrated.** A model trained on a balanced
    set predicts probabilities for a balanced world. Check with
    :meth:`~buildml.session.Session.calibration` before using the numbers as
    probabilities rather than as a ranking.

    Examples
    --------
    >>> data, split, plan = resample_train(  # doctest: +SKIP
    ...     dataset, split_plan, sampler="smote", sampling_strategy=0.3
    ... )
    >>> plan.class_counts_after  # doctest: +SKIP
    {'0': 9500, '1': 2850}

    See Also
    --------
    list_resample_strategies : What is available and when to use each.
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

    # Holdout fingerprints: must remain byte-identical after concat rebuild.
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
            "Train grew via oversampling/synthesis: evaluate on untouched validation/test only."
        )
    else:
        notes.append("Train shrank via undersampling: monitor majority-class recall on holdout.")

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
