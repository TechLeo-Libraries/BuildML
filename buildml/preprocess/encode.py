"""Categorical encoding with train-only fit semantics.

Methods
-------
onehot / ordinal
    Standard sklearn encoders fit on train categories.
infrequent
    Collapse rare train levels to ``__infrequent__``, then one-hot encode.
target
    Mean target encoding with out-of-fold values on train rows and full-train
    means on holdout rows (leakage-safe when the split plan is supplied).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.core.validation import validate_column_names
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition, frame_for_partition
from buildml.explain.schemas import (
    Action,
    ActionPriority,
    Evidence,
    EvidenceKind,
    Finding,
    FindingSeverity,
    Recommendation,
)
from buildml.ingest.detect import schema_from_dataframe
from buildml.preprocess.result import PreprocessResult

EncodeMethod = Literal["onehot", "ordinal", "infrequent", "target"]
INFREQUENT_LABEL = "__infrequent__"


@dataclass(slots=True)
class EncodePlan:
    """Fitted categorical encoding plan."""

    columns: tuple[str, ...]
    method: EncodeMethod
    feature_names_: tuple[str, ...]
    encoder: Any = None
    infrequent_maps_: dict[str, list[str]] = field(default_factory=dict)
    min_frequency: float | int | None = None
    target_maps_: dict[str, dict[str, float]] = field(default_factory=dict)
    target_prior_: float | None = None
    n_folds: int = 5
    random_state: int = 0
    smoothing: float = 10.0

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "columns": list(self.columns),
            "method": self.method,
            "feature_names_": list(self.feature_names_),
        }
        if self.method == "infrequent":
            payload["infrequent_maps_"] = {
                key: list(values) for key, values in self.infrequent_maps_.items()
            }
            payload["min_frequency"] = self.min_frequency
        if self.method == "target":
            payload["target_maps_"] = {
                col: dict(mapping) for col, mapping in self.target_maps_.items()
            }
            payload["target_prior_"] = self.target_prior_
            payload["n_folds"] = self.n_folds
            payload["random_state"] = self.random_state
            payload["smoothing"] = self.smoothing
        return payload


def fit_encoder(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    columns: list[str] | None = None,
    method: EncodeMethod = "onehot",
    min_frequency: float | int = 0.05,
    n_folds: int = 5,
    random_state: int = 0,
    smoothing: float = 10.0,
) -> EncodePlan:
    """Fit an encoder on the train partition only."""
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None

    train = frame_for_partition(dataset, split_plan, "train")
    cols = _resolve_categorical_columns(dataset, train, columns)

    if method in {"onehot", "ordinal"}:
        if method == "onehot":
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        else:
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
        encoder.fit(train[list(cols)].astype(str))
        feature_names = (
            tuple(str(n) for n in encoder.get_feature_names_out(cols))
            if method == "onehot"
            else tuple(cols)
        )
        return EncodePlan(
            columns=tuple(cols),
            method=method,
            feature_names_=feature_names,
            encoder=encoder,
        )

    if method == "infrequent":
        maps: dict[str, list[str]] = {}
        collapsed = train[list(cols)].astype(str).copy()
        for column in cols:
            counts = collapsed[column].value_counts(dropna=False)
            if isinstance(min_frequency, float):
                if not 0.0 < min_frequency < 1.0:
                    raise ValidationError("float min_frequency must be in (0, 1)")
                threshold = min_frequency * len(collapsed)
            else:
                if int(min_frequency) < 1:
                    raise ValidationError("integer min_frequency must be >= 1")
                threshold = float(min_frequency)
            rare = [str(level) for level, count in counts.items() if float(count) < threshold]
            maps[column] = rare
            collapsed[column] = collapsed[column].where(
                ~collapsed[column].isin(rare),
                INFREQUENT_LABEL,
            )
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoder.fit(collapsed)
        feature_names = tuple(str(n) for n in encoder.get_feature_names_out(cols))
        return EncodePlan(
            columns=tuple(cols),
            method=method,
            feature_names_=feature_names,
            encoder=encoder,
            infrequent_maps_=maps,
            min_frequency=min_frequency,
        )

    if method == "target":
        if n_folds < 2:
            raise ValidationError("target encoding n_folds must be at least 2")
        if smoothing < 0:
            raise ValidationError("smoothing must be non-negative")
        target_name = dataset.require_target()
        y_raw = train[target_name]
        y = _numeric_target(y_raw)
        prior = float(np.mean(y))
        maps_target: dict[str, dict[str, float]] = {}
        for column in cols:
            maps_target[column] = _smoothed_means(
                train[column].astype(str),
                y,
                prior=prior,
                smoothing=smoothing,
            )
        return EncodePlan(
            columns=tuple(cols),
            method=method,
            feature_names_=tuple(f"{c}_target" for c in cols),
            target_maps_=maps_target,
            target_prior_=prior,
            n_folds=n_folds,
            random_state=random_state,
            smoothing=smoothing,
        )

    raise ValidationError(f"Unsupported encode method '{method}'")


def transform_encoder(
    dataset: Dataset,
    plan: EncodePlan,
    split_plan: SplitPlan | None = None,
) -> tuple[Dataset, PreprocessResult]:
    """Apply a fitted encode plan to the full dataset."""
    missing = [c for c in plan.columns if c not in dataset.columns]
    if missing:
        raise ValidationError(f"Encode plan columns missing from dataset: {missing}")

    if plan.method == "target":
        if split_plan is None:
            raise ValidationError(
                "Target encoding transform requires the SplitPlan so train rows "
                "can receive out-of-fold values."
            )
        return _transform_target(dataset, plan, split_plan)

    frame = dataset._ensure_pandas().copy()
    work = frame[list(plan.columns)].astype(str)
    if plan.method == "infrequent":
        for column in plan.columns:
            rare = set(plan.infrequent_maps_.get(column, ()))
            work[column] = work[column].where(~work[column].isin(rare), INFREQUENT_LABEL)

    encoded = plan.encoder.transform(work)
    encoded_df = pd.DataFrame(encoded, columns=list(plan.feature_names_), index=frame.index)
    remaining = frame.drop(columns=list(plan.columns))
    out = pd.concat([remaining, encoded_df], axis=1)

    roles = {k: v for k, v in dataset.roles.items() if k not in plan.columns}
    for name in plan.feature_names_:
        roles.setdefault(name, ColumnRole.FEATURE)

    new_dataset = Dataset.from_transformed(
        dataset,
        out,
        schema=schema_from_dataframe(out),
        roles=roles,
    )
    return new_dataset, _build_result(plan)


def _transform_target(
    dataset: Dataset,
    plan: EncodePlan,
    split_plan: SplitPlan,
) -> tuple[Dataset, PreprocessResult]:
    assert_fit_partition(split_plan, "train")
    frame = dataset._ensure_pandas().copy()
    target_name = dataset.require_target()
    train_idx = list(split_plan.train_indices)
    train = frame.iloc[train_idx]
    y = _numeric_target(train[target_name])
    prior = float(plan.target_prior_ if plan.target_prior_ is not None else np.mean(y))

    # Stratify folds when the target looks discrete with few levels.
    unique = pd.unique(y)
    if len(unique) <= 20 and np.all(np.equal(np.mod(unique, 1), 0)):
        splitter: Any = StratifiedKFold(
            n_splits=min(plan.n_folds, max(2, int(pd.Series(y).value_counts().min()))),
            shuffle=True,
            random_state=plan.random_state,
        )
        try:
            fold_iter = list(splitter.split(np.zeros(len(y)), y))
        except ValueError:
            splitter = KFold(
                n_splits=min(plan.n_folds, len(y)),
                shuffle=True,
                random_state=plan.random_state,
            )
            fold_iter = list(splitter.split(np.zeros(len(y))))
    else:
        splitter = KFold(
            n_splits=min(plan.n_folds, len(y)),
            shuffle=True,
            random_state=plan.random_state,
        )
        fold_iter = list(splitter.split(np.zeros(len(y))))

    roles = {k: v for k, v in dataset.roles.items() if k not in plan.columns}
    for column, out_name in zip(plan.columns, plan.feature_names_, strict=True):
        oof = np.full(len(frame), np.nan, dtype=float)
        col_train = train[column].astype(str).to_numpy()
        for fit_pos, pred_pos in fold_iter:
            means = _smoothed_means(
                pd.Series(col_train[fit_pos]),
                y[fit_pos],
                prior=prior,
                smoothing=plan.smoothing,
            )
            mapped = pd.Series(col_train[pred_pos]).map(means).fillna(prior).to_numpy(dtype=float)
            global_positions = [train_idx[i] for i in pred_pos]
            oof[global_positions] = mapped

        global_map = plan.target_maps_[column]
        holdout_positions = [i for i in range(len(frame)) if i not in set(train_idx)]
        if holdout_positions:
            holdout_values = (
                frame.iloc[holdout_positions][column]
                .astype(str)
                .map(global_map)
                .fillna(prior)
                .to_numpy(dtype=float)
            )
            oof[holdout_positions] = holdout_values

        # Any unresolved train rows (edge cases) fall back to prior.
        unresolved = np.isnan(oof)
        if unresolved.any():
            oof[unresolved] = prior
        frame[out_name] = oof
        roles[out_name] = ColumnRole.FEATURE

    frame = frame.drop(columns=list(plan.columns))
    new_dataset = Dataset.from_transformed(
        dataset,
        frame,
        schema=schema_from_dataframe(frame),
        roles=roles,
    )
    warnings = [
        "Target encoding used out-of-fold values on train and full-train means on holdouts.",
        "Prefer fold-local recipes inside cv_score/grid_search when selection itself uses CV.",
    ]
    result = _build_result(plan)
    result.warnings.extend(warnings)
    return new_dataset, result


def _smoothed_means(
    categories: pd.Series,
    y: np.ndarray,
    *,
    prior: float,
    smoothing: float,
) -> dict[str, float]:
    frame = pd.DataFrame({"cat": categories.astype(str).to_numpy(), "y": y})
    grouped = frame.groupby("cat", sort=False)["y"]
    stats = grouped.agg(["mean", "count"])
    means: dict[str, float] = {}
    for cat, row in stats.iterrows():
        count = float(row["count"])
        average = float(row["mean"])
        weight = count / (count + smoothing) if (count + smoothing) else 0.0
        means[str(cat)] = weight * average + (1.0 - weight) * prior
    return means


def _numeric_target(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")
        if values.isna().any():
            raise ValidationError("Target encoding requires a non-null numeric or label target")
        return values.to_numpy(dtype=float)
    codes, _ = pd.factorize(series.astype(str), sort=True)
    if (codes < 0).any():
        raise ValidationError("Target encoding cannot proceed with null target labels")
    return codes.astype(float)


def _resolve_categorical_columns(
    dataset: Dataset,
    train: pd.DataFrame,
    columns: list[str] | None,
) -> list[str]:
    if columns is not None:
        return validate_column_names(columns, dataset.columns)

    target_cols = set(dataset.role_columns("target"))
    cats = [
        str(c)
        for c in train.columns
        if c not in target_cols
        and (
            pd.api.types.is_object_dtype(train[c])
            or isinstance(train[c].dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(train[c])
        )
    ]
    if not cats:
        raise ValidationError("No categorical columns available for encoding")
    return cats


def _build_result(plan: EncodePlan) -> PreprocessResult:
    evidence = [
        Evidence(
            key="encode.method",
            kind=EvidenceKind.CONFIGURATION,
            summary="Train-fitted categorical encoding method and output schema.",
            value={
                "method": plan.method,
                "columns": list(plan.columns),
                "feature_names": list(plan.feature_names_),
            },
            source="train.encode_plan",
            limitations=("Unknown levels follow the method's declared policy.",),
        )
    ]
    findings = [
        Finding(
            key="encode.applied",
            title="Categorical encoding applied",
            detail=(
                f"Method '{plan.method}' remapped {len(plan.columns)} column(s) into "
                f"{len(plan.feature_names_)} feature column(s)."
            ),
            severity=FindingSeverity.INFO,
            evidence=tuple(evidence),
            affected_columns=plan.columns,
        )
    ]
    recommendations: list[Recommendation] = []
    if plan.method == "onehot" and len(plan.feature_names_) > max(20, 5 * len(plan.columns)):
        recommendations.append(
            Recommendation(
                key="encode.consider-infrequent",
                title="Consider infrequent-level pooling before one-hot",
                rationale=(
                    "Wide one-hot schemas often come from rare levels; pooling rares "
                    "reduces width without target leakage."
                ),
                priority=ActionPriority.NEXT,
                action=Action(
                    key="encode.infrequent-action",
                    label="Session.encode(method='infrequent')",
                    operation="encode",
                    parameters={"method": "infrequent"},
                ),
                based_on=("encode.applied",),
                caveats=("Pooling changes level semantics; review domain meaning of rare labels.",),
            )
        )
    if plan.method == "target":
        recommendations.append(
            Recommendation(
                key="encode.target-cv-note",
                title="Keep target encoding inside CV when selecting models",
                rationale=(
                    "Session target encoding is OOF on the current train partition. "
                    "Model selection that reshuffles folds should use PreprocessRecipe "
                    "fold-local preparation instead of a frozen Session plan."
                ),
                priority=ActionPriority.BEFORE_MODELING,
                action=Action(
                    key="encode.target-cv-action",
                    label="Session.cv_score(..., preprocess=PreprocessRecipe(...))",
                    operation="cv_score",
                    parameters={},
                ),
                based_on=("encode.applied",),
                caveats=("OOF encoding still uses the target; never apply it before split.",),
            )
        )
    limitations = [
        "Encoding vocabularies and target means are train-fitted only.",
        "Ordinal codes invent numeric order unless the domain supplies one.",
    ]
    if plan.method == "target":
        limitations.append(
            "Target encoding without out-of-fold discipline leaks labels into train features."
        )
    return PreprocessResult(
        operation="encode",
        plan=plan.to_dict(),
        evidence=evidence,
        findings=findings,
        interpretation=[
            f"Applied train-fitted '{plan.method}' encoding.",
            f"Output feature columns: {len(plan.feature_names_)}.",
        ],
        limitations=limitations,
        recommendations=recommendations,
        methods=[
            f"Encode method={plan.method}.",
            (
                f"Infrequent min_frequency={plan.min_frequency}."
                if plan.method == "infrequent"
                else (
                    f"Target OOF folds={plan.n_folds}, smoothing={plan.smoothing}."
                    if plan.method == "target"
                    else "Unknown levels use the encoder's configured policy."
                )
            ),
        ],
    )
