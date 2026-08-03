"""Turn category labels into numbers, without smuggling the answer in.

Estimators do arithmetic, and ``"Dublin"`` is not a number. Encoding bridges
that gap, and the choice of bridge matters more than people expect: the wrong
one either invents a relationship that does not exist or leaks the target into
the features.

The four methods here trade off along two axes: how much width they add, and
how much they risk. **One-hot** creates an indicator column per category, which
is unambiguous and safe but adds a column for every distinct value.
**Ordinal** assigns 1, 2, 3, keeping the frame narrow but asserting an order :
harmless for a tree, actively wrong for a linear model unless the categories
genuinely are ordered, since it claims "Cork" sits exactly halfway between
"Dublin" and "Galway". **Infrequent** collapses rare categories into a single
bucket before one-hot encoding, which controls the width explosion and stops
the model memorising categories it saw twice. **Target** replaces each category
with the average target value for that category: very compact, often very
predictive, and the most dangerous of the four.

Target encoding is dangerous because the feature is built from the label. If a
category's mean is computed over the same rows the model trains on, each row's
feature partly contains its own answer, and the model will look excellent right
up until it meets new data. The implementation here defends against that: the
training rows get out-of-fold values, so a row's encoding is computed from
*other* rows only, while holdout rows get the full-training mean. That is why
target encoding requires a split plan at transform time and the other methods
do not.

Every method learns its vocabulary from training rows alone. A category that
appears only in test data has no encoding to receive, which is handled
explicitly rather than silently.
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
from buildml.preprocess.columns import resolve_transform_columns
from buildml.preprocess.result import PreprocessResult

EncodeMethod = Literal["onehot", "ordinal", "infrequent", "target"]
INFREQUENT_LABEL = "__infrequent__"


@dataclass(slots=True)
class EncodePlan:
    """The category vocabulary learned from training rows, ready to replay.

    The critical thing this preserves is the *column layout*. A model trained
    on a frame with ``city_dublin`` and ``city_cork`` expects exactly those
    columns in that order forever after. Re-deriving them from a new batch that
    happens to contain only Dublin would produce a frame the model cannot
    consume: or worse, one it consumes while interpreting the wrong column.

    Which fields are populated depends on ``method``; the unused ones stay at
    their defaults.

    Attributes
    ----------
    columns:
        The source categorical columns this plan encodes.
    method:
        Which encoding was fitted.
    feature_names_:
        The output columns, in order. This is the contract with the model, and
        the thing to check when an inference frame does not line up.
    encoder:
        The fitted scikit-learn encoder for the ``'onehot'``, ``'ordinal'``,
        and ``'infrequent'`` methods; ``None`` for ``'target'``, which needs no
        estimator.
    infrequent_maps_:
        For ``'infrequent'``, the training categories judged rare per column
        and therefore folded into the shared bucket. Worth reading: if most of
        a column ended up here, the column is mostly noise.
    min_frequency:
        The rarity threshold that produced those maps.
    target_maps_:
        For ``'target'``, the smoothed mean target per category per column.
        These are model parameters in all but name; treat them as sensitive if
        the target is.
    target_prior_:
        The overall training target mean, used for categories with too little
        support and for anything unseen.
    n_folds:
        Out-of-fold count used when computing training-row target encodings.
    random_state:
        Seed for that fold assignment, so the encoding reproduces.
    smoothing:
        How strongly a category's mean is pulled toward the prior. Higher means
        a category needs more rows before its own mean is trusted.
    """

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
        """Return the plan as plain JSON-safe values.

        Only the fields relevant to the fitted method are included, so a
        one-hot plan does not carry empty target-encoding keys.

        Returns
        -------
        dict
            ``columns``, ``method``, and ``feature_names_`` always. An
            ``'infrequent'`` plan adds ``infrequent_maps_`` and
            ``min_frequency``; a ``'target'`` plan adds ``target_maps_``,
            ``target_prior_``, ``n_folds``, ``random_state``, and
            ``smoothing``. The fitted encoder object is omitted, since it does
            not serialise to JSON: use a saved pipeline to round-trip it.
        """
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
    """Learn the category vocabulary and output layout from the training rows.

    Nothing is transformed here: pass the returned plan to
    :func:`transform_encoder` to apply it.

    Parameters
    ----------
    dataset:
        The full dataset. Only rows the split assigns to ``train`` are read.
    split_plan:
        The split defining the training rows. Required, because a vocabulary
        built from all rows tells the model which categories exist in the test
        set: and for target encoding it would hand over the labels outright.
    columns:
        Which columns to encode. Defaults to categorical ``feature`` columns,
        skipping the protected roles. Pass an explicit list to encode something
        stored as a number that is really a category, such as a postal code.
    method:
        ``'onehot'`` for the safe default; ``'ordinal'`` when the frame must
        stay narrow and the model is tree-based; ``'infrequent'`` when a column
        has a long tail of rare values; ``'target'`` when a
        high-cardinality column carries real signal and you are prepared to
        manage the risk. See the module docstring for the trade-offs.
    min_frequency:
        For ``'infrequent'``, how common a category must be to keep its own
        column. A float is a proportion of training rows, so ``0.05`` means
        anything under five percent is folded into the shared bucket; an
        integer is a raw row count. Raise it to compress harder, lower it to
        preserve more distinctions. Ignored by the other methods.
    n_folds:
        For ``'target'``, how many out-of-fold splits are used when encoding
        the training rows. More folds means each row's encoding is computed
        from more data and is less noisy, at proportionally more work. Five is
        a reasonable default.
    random_state:
        Seed for the out-of-fold assignment, so target encoding reproduces.
    smoothing:
        For ``'target'``, how much a category's own mean is pulled toward the
        overall mean. This is the guard against trusting a category seen three
        times: at the default of 10, a category needs roughly ten rows before
        its own average outweighs the prior. Raise it when categories are
        sparse, lower it when every category has plenty of support.

    Returns
    -------
    EncodePlan
        The learned vocabulary and output column layout, ready to apply.

    Raises
    ------
    ~buildml.core.errors.LeakageError
        No split plan was supplied.
    ~buildml.core.errors.ValidationError
        ``method`` is unrecognised, no categorical columns resolved, or
        ``'target'`` was requested without a target column having a role
        assigned.

    Notes
    -----
    **Watch the width.** One-hot encoding a column with 5,000 distinct values
    produces 5,000 columns, which will exhaust memory long before it helps.
    Check cardinality first and reach for ``'infrequent'`` or ``'target'`` when
    it is high.

    **Values are compared as strings.** The integer ``1`` and the string
    ``"1"`` are the same category here, which is usually what you want from
    mixed-type data but can merge distinctions you meant to keep.

    Examples
    --------
    >>> plan = fit_encoder(dataset, split_plan, method="onehot")  # doctest: +SKIP
    >>> plan.feature_names_[:2]  # doctest: +SKIP
    ('city_cork', 'city_dublin')

    See Also
    --------
    transform_encoder : Applies the plan produced here.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None

    train = frame_for_partition(dataset, split_plan, "train")
    cols = resolve_transform_columns(
        dataset,
        train,
        columns,
        kind="categorical",
        empty_message=(
            "No categorical feature columns available for encoding. "
            "Pass columns=... explicitly to include ignore/id roles."
        ),
    )

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
    """Replace category labels with numbers using an already-learned plan.

    The source columns are removed and the plan's output columns take their
    place, in the plan's own order: which is what keeps the frame consistent
    with what the model was trained on even if the incoming data is missing a
    category or arrives in a different column order.

    Parameters
    ----------
    dataset:
        The dataset to encode. Every column the plan names must be present.
    plan:
        A plan from :func:`fit_encoder`, or one restored from a saved pipeline.
    split_plan:
        Required only for ``'target'`` plans, and genuinely required there:
        training rows must receive out-of-fold values while holdout rows
        receive the full-training mean, and without the split there is no way
        to tell which row is which. Ignored by the other methods.

    Returns
    -------
    tuple of (~buildml.data.dataset.Dataset, ~buildml.preprocess.result.PreprocessResult)
        The encoded dataset, and a narrated record covering how many columns
        were produced, which categories went unseen, and how much the frame
        widened.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        A column the plan expects is missing, or a ``'target'`` plan was passed
        without a split plan.

    Notes
    -----
    **Unseen categories are handled, not rejected.** With one-hot they produce
    all zeros; with ordinal they take the encoder's unknown value; with
    infrequent they join the rare bucket; with target they fall back to the
    training prior. All four are reasonable, and none of them is a signal you
    will notice unless you look: the returned result reports how often it
    happened, and a high rate means training and serving data have diverged.

    **Missing values are encoded as the literal string** ``"nan"``, so they
    become a category rather than propagating. Impute first if you want
    different behaviour.

    See Also
    --------
    fit_encoder : Produces the plan this consumes.
    """
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
