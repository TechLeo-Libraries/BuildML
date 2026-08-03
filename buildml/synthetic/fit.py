"""Fit Session-facing tabular synthesizers on train only."""

from __future__ import annotations

from typing import Any, Sequence

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.synthetic.adapters.sdv import SdvTabularGenerator
from buildml.synthetic.catalog import resolve_backend_method
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
from buildml.synthetic.types import SyntheticBackend, SynthesizerConfig, SynthesizerMethod
from buildml.synthetic.validation import enrich_specs_with_train_stats


PRIVACY_DISCLOSURE = (
    "Privacy honesty: synthetic generators are not a differential-privacy "
    "product. Bootstrap can emit near-duplicates of train rows; Gaussian "
    "copula, SMOTE, and SDV deep models can still memorize or leak training "
    "structure. Do not treat samples as anonymized releases without a dedicated "
    "privacy review."
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
    backend: SyntheticBackend | None = None,
    method: SynthesizerMethod = "gaussian_copula",
    columns: Sequence[str] | None = None,
    random_state: int = 42,
    smooth_sigma: float = 0.0,
    correlation_ridge: float = 1e-3,
    target_column: str | None = None,
    k_neighbors: int = 5,
    sampling_strategy: str | float | dict[str, float] = "auto",
    epochs: int = 300,
    batch_size: int = 500,
) -> tuple[SynthesizerPlan, SynthesizerFitResult]:
    """Fit a tabular synthesizer on the Session **train** partition only.

Backends
--------
native (default when SDV absent):
    bootstrap, gaussian_copula, smote (``buildml[imbalanced]`` for smote).
sdv (``buildml[synthetic-industry]`` when installed):
    ctgan, tvae, copulagan via SDV single-table synthesizers.
Honesty
-------
Never fits on validation/test. Not a differential-privacy product.

Parameters
----------
dataset:
    BuildML dataset with features, target, and role metadata.
split_plan:
    Train/validation/test split; fit uses train partition only.
backend:
    Optional backend override (see capability matrix for identifiers).
method:
    Method or strategy identifier for the resolved backend.
columns:
    Optional explicit feature column list; ``None`` auto-selects numerics.
random_state:
    Seed for stochastic steps (sampling, initialization, bagging).
smooth_sigma:
    smooth sigma (float).
correlation_ridge:
    correlation ridge (float).
target_column:
    Name of the supervised target column.
k_neighbors:
    k neighbors (int).
sampling_strategy:
    sampling strategy (str | float | dict[str, float]).
epochs:
    Training epochs for torch-backed estimators.
batch_size:
    Number of rows to select per query or training minibatch.

Returns
-------
tuple[SynthesizerPlan, SynthesizerFitResult]
    Tuple of results (tuple[SynthesizerPlan, SynthesizerFitResult]) for downstream Session steps.

Raises
------
ValidationError
    When preconditions for this operation are not met.
    """
    resolved_backend, resolved_method = resolve_backend_method(
        backend=backend,
        method=method,
    )
    split = require_split(split_plan)
    assert_train_only_fit("train")
    train = require_train_frame(dataset, split)

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
        method=resolved_method,
    )
    train_sub = train[cols].copy()
    specs = build_column_specs(train_sub)
    specs = enrich_specs_with_train_stats(train_sub, specs)
    config = SynthesizerConfig(
        method=resolved_method,
        backend=resolved_backend,
        partition="train",
        columns=list(cols),
        random_state=int(random_state),
        smooth_sigma=float(smooth_sigma),
        correlation_ridge=float(correlation_ridge),
        target_column=tgt,
        k_neighbors=int(k_neighbors),
        sampling_strategy=sampling_strategy,
        epochs=int(epochs),
        batch_size=int(batch_size),
    )

    disclosures = [
        f"backend={resolved_backend!r} method={resolved_method!r} "
        f"fitted on partition='train' only (n={len(train_sub)}).",
        PRIVACY_DISCLOSURE,
        RESAMPLE_CROSSLINK,
        "Holdout partitions are never used to fit the generator.",
    ]
    warnings: list[str] = []

    generator: Any
    if resolved_backend == "sdv":
        generator = SdvTabularGenerator.fit(
            train_sub,
            specs,
            method=resolved_method,  # type: ignore[arg-type]
            epochs=epochs,
            batch_size=batch_size,
            random_state=random_state,
        )
        disclosures.append(
            f"SDV {resolved_method.upper()} synthesizer (buildml[synthetic-industry]); "
            f"epochs={epochs}, batch_size={batch_size}. Deep generative model — "
            "not differential privacy."
        )
        if len(train_sub) < 100:
            warnings.append(
                "SDV deep synthesizers are data-hungry; small train sets may "
                "underfit or overfit — prefer native gaussian_copula for n<100."
            )
    elif resolved_method == "bootstrap":
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
    elif resolved_method == "gaussian_copula":
        generator = GaussianCopulaGenerator.fit(
            train_sub,
            specs,
            correlation_ridge=correlation_ridge,
            random_state=random_state,
        )
        disclosures.append(
            "Gaussian copula models rank correlations via a multivariate normal "
            "latent; marginals use empirical CDFs (native fallback — not SDV CTGAN)."
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
        method=resolved_method,
        backend=resolved_backend,
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
        method=resolved_method,
        backend=resolved_backend,
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
