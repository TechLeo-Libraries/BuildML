"""Fit Session-facing tabular synthesizers on train only."""

from __future__ import annotations

from typing import Any, Sequence

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.synthetic.features import (
    assert_train_only_fit,
    require_split,
    require_train_frame,
    resolve_columns,
)
from buildml.synthetic.models import (
    BootstrapGenerator,
    GaussianCopulaGenerator,
    SmoteGenerator,
    build_column_specs,
)
from buildml.synthetic.results import SynthesizerFitResult, SynthesizerPlan
from buildml.synthetic.types import SynthesizerConfig, SynthesizerMethod


PRIVACY_DISCLOSURE = (
    "Privacy honesty: synthetic generators are not a differential-privacy "
    "product. Bootstrap can emit near-duplicates of train rows; Gaussian "
    "copula and SMOTE can still memorize or leak training structure. Do not "
    "treat samples as anonymized releases without a dedicated privacy review."
)

RESAMPLE_CROSSLINK = (
    "Cross-link: Session.resample (buildml[imbalanced]) is class-balance "
    "preprocessing that mutates train membership in-place. The synthetic "
    "path (fit_synthesizer / sample_synthetic) fits a reusable generator "
    "and only merges when merge_mode='extend_train' is requested explicitly."
)


def fit_synthesizer(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    method: SynthesizerMethod = "gaussian_copula",
    columns: Sequence[str] | None = None,
    random_state: int = 42,
    smooth_sigma: float = 0.0,
    correlation_ridge: float = 1e-3,
    target_column: str | None = None,
    k_neighbors: int = 5,
    sampling_strategy: str | float | dict[str, float] = "auto",
) -> tuple[SynthesizerPlan, SynthesizerFitResult]:
    """Fit a tabular synthesizer on the Session **train** partition only.

    Methods
    -------
    bootstrap
        Row resampling with replacement; optional Gaussian smoothing
        (``smooth_sigma`` × column std) on continuous/integer columns.
    gaussian_copula
        Mixed-type Gaussian copula (empirical CDF + correlation); categoricals
        participate via frequency-bin latent scores.
    smote
        Reusable SMOTE wrapper (requires ``buildml[imbalanced]``). Distinct
        from ``Session.resample`` — does not mutate Session until merge.

    Honesty
    -------
    Never fits on validation/test. Not a differential-privacy product.
    """
    if method not in {"bootstrap", "gaussian_copula", "smote"}:
        raise ValidationError(
            f"Unknown synthesizer method: {method!r}. "
            "Expected bootstrap | gaussian_copula | smote."
        )
    split = require_split(split_plan)
    assert_train_only_fit("train")
    train = require_train_frame(dataset, split)

    # Resolve target for SMOTE
    tgt = target_column
    if tgt is None:
        for name, role in dataset.roles.items():
            if role == ColumnRole.TARGET:
                tgt = name
                break

    cols = resolve_columns(
        dataset,
        train,
        columns=columns,
        target_column=tgt,
        method=method,
    )
    train_sub = train[cols].copy()
    specs = build_column_specs(train_sub)
    config = SynthesizerConfig(
        method=method,
        partition="train",
        columns=list(cols),
        random_state=int(random_state),
        smooth_sigma=float(smooth_sigma),
        correlation_ridge=float(correlation_ridge),
        target_column=tgt,
        k_neighbors=int(k_neighbors),
        sampling_strategy=sampling_strategy,
    )

    disclosures = [
        f"method={method!r} fitted on partition='train' only (n={len(train_sub)}).",
        PRIVACY_DISCLOSURE,
        RESAMPLE_CROSSLINK,
        "Holdout partitions are never used to fit the generator.",
    ]
    warnings: list[str] = []

    generator: Any
    if method == "bootstrap":
        if smooth_sigma < 0:
            raise ValidationError("smooth_sigma must be >= 0.")
        generator = BootstrapGenerator.fit(
            train_sub,
            specs,
            smooth_sigma=smooth_sigma,
            random_state=random_state,
        )
        if smooth_sigma > 0:
            disclosures.append(
                f"Smoothed bootstrap: Gaussian noise with smooth_sigma={smooth_sigma} "
                "× train column std on continuous/integer columns."
            )
        else:
            disclosures.append(
                "Plain bootstrap: samples are resampled train rows (with replacement)."
            )
    elif method == "gaussian_copula":
        generator = GaussianCopulaGenerator.fit(
            train_sub,
            specs,
            correlation_ridge=correlation_ridge,
            random_state=random_state,
        )
        disclosures.append(
            "Gaussian copula models rank correlations via a multivariate normal "
            "latent; marginals use empirical CDFs (not a deep generative model / CTGAN)."
        )
    else:
        generator = SmoteGenerator.fit(
            train_sub,
            specs,
            target_column=str(tgt),
            feature_columns=tuple(
                s.name for s in specs if s.name != tgt and s.kind != "categorical"
            ),
            k_neighbors=k_neighbors,
            sampling_strategy=sampling_strategy,
            random_state=random_state,
        )
        disclosures.append(
            "SMOTE synthesizer uses imbalanced-learn (extra 'imbalanced'); "
            "prefer Session.resample when the only goal is class rebalancing."
        )
        warnings.append(
            "SMOTE interpolates in feature space — poor fit for pure categoricals "
            "or non-metric encodings."
        )

    roles_snapshot = {k: v.value for k, v in dataset.roles.items()}
    plan = SynthesizerPlan(
        method=method,
        partition_fitted="train",
        columns=tuple(cols),
        column_specs=specs,
        n_rows_fitted=int(len(train_sub)),
        random_state=int(random_state),
        config=config.to_dict(),
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
        generator_=generator,
        target_column=tgt,
        roles_snapshot=roles_snapshot,
    )
    kinds = {s.name: s.kind for s in specs}
    fit_result = SynthesizerFitResult(
        method=method,
        partition="train",
        n_rows=int(len(train_sub)),
        n_columns=int(len(cols)),
        column_kinds=kinds,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
        metrics={
            "n_rows_fitted": float(len(train_sub)),
            "n_columns": float(len(cols)),
            "n_continuous": float(sum(1 for k in kinds.values() if k == "continuous")),
            "n_categorical": float(sum(1 for k in kinds.values() if k == "categorical")),
            "n_integer": float(sum(1 for k in kinds.values() if k == "integer")),
        },
    )
    return plan, fit_result
