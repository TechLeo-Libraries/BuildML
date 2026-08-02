"""Initial warm-start fit for online / continual learning (train chunk only)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import (
    PassiveAggressiveClassifier,
    PassiveAggressiveRegressor,
    Perceptron,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition
from buildml.online.features import (
    carve_train_chunk,
    encode_classification_targets,
    matrix_from_frame,
    regression_targets,
    resolve_online_columns,
    train_partition_frame,
)
from buildml.online.results import OnlineFitResult, OnlinePlan
from buildml.online.types import OnlineConfig, OnlineEstimator, OnlineTask

_CLASSIFIERS = {
    "sgd_classifier",
    "passive_aggressive_classifier",
    "perceptron",
    "multinomial_nb",
    "bernoulli_nb",
}
_REGRESSORS = {
    "sgd_regressor",
    "passive_aggressive_regressor",
}


def fit_online(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    estimator: OnlineEstimator = "sgd_classifier",
    task: OnlineTask | None = None,
    columns: list[str] | None = None,
    random_state: int | None = 0,
    chunk_size: int = 50,
    n_init: int | None = None,
    indices: Sequence[Any] | None = None,
    classes: Sequence[Any] | None = None,
    prefer_reduce_components: bool = True,
    allow_refit_fallback: bool = False,
    drift_disclose: bool = True,
    reduce_plan: Any | None = None,
) -> tuple[OnlinePlan, OnlineFitResult]:
    """Warm-start an incremental estimator on the first train chunk.

    Class discovery (classifiers)
    -----------------------------
    On the first call, ``classes`` should cover the full label vocabulary the
    stream may emit. When omitted, BuildML discovers classes from the **entire
    train target column** (labels only — features from unseen chunks are not
    used until ``partial_fit_online``). This matches sklearn's
    ``partial_fit(..., classes=...)`` contract and is disclosed on the plan.

    Honesty
    -------
    Updates are batch/stream-chunk ``partial_fit`` calls on Session train data.
    This is not a distributed streaming platform or lifelong-learning research
    suite. Validation/test are never used for updates.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None

    est_key = str(estimator).lower().replace("-", "_")
    resolved_task = _resolve_task(est_key, task)
    target = dataset.require_target()
    train = train_partition_frame(dataset, split_plan)
    cols, used_reduce, disclosures = resolve_online_columns(
        dataset,
        train,
        columns,
        reduce_plan=reduce_plan,
        prefer_reduce_components=prefer_reduce_components,
        target_column=target,
    )
    n_train = int(len(split_plan.train_indices))
    init_n = int(n_init) if n_init is not None else int(chunk_size)
    if init_n < 1:
        raise ValidationError("n_init / chunk_size for fit_online must be >= 1.")

    chunk, chunk_indices, cursor = carve_train_chunk(
        dataset,
        split_plan,
        cursor=0,
        n_rows=init_n,
        indices=indices,
    )
    x = matrix_from_frame(chunk, cols)
    if est_key in {"multinomial_nb"} and (x < 0).any():
        raise ValidationError(
            "multinomial_nb requires non-negative features. "
            "Use count-like features, or choose sgd_classifier / bernoulli_nb."
        )

    warnings: list[str] = []
    label_encoder = None
    classes_tuple: tuple[Any, ...] | None = None
    y: np.ndarray

    if resolved_task == "classification":
        class_vocab = classes
        if class_vocab is None:
            train_y = train[target]
            if train_y.isna().any():
                raise ValidationError(
                    "Online classification needs non-null train targets to "
                    "discover classes= (or pass classes= explicitly). "
                    "NaN targets belong to active/semi-supervised pools."
                )
            class_vocab = tuple(sorted(train_y.astype(str).unique().tolist()))
            disclosures.append(
                "Classifier classes_ discovered from the full train target "
                "vocabulary before streaming chunks (labels only; features from "
                "unseen train rows were not used in this init fit)."
            )
        else:
            class_vocab = tuple(classes)
            disclosures.append(
                "Classifier classes_ taken from the explicit classes= argument "
                "on fit_online."
            )
        if len(class_vocab) < 2:
            raise ValidationError(
                "Online classification requires at least 2 classes "
                f"(found {class_vocab!r})."
            )
        y, label_encoder, classes_tuple = encode_classification_targets(
            chunk[target],
            classes=class_vocab,
        )
    else:
        y = regression_targets(chunk[target])

    estimator_obj = _build_estimator(est_key, random_state)
    used_refit = False
    update_mode = "partial_fit"
    try:
        if hasattr(estimator_obj, "partial_fit"):
            if resolved_task == "classification":
                # sklearn expects the original class labels or integer codes
                # matching classes=. Pass the encoder's class codes range.
                class_codes = np.arange(len(classes_tuple or ()))
                estimator_obj.partial_fit(x, y, classes=class_codes)
            else:
                estimator_obj.partial_fit(x, y)
        else:
            used_refit, update_mode = _maybe_refit_fallback(
                estimator_obj,
                x,
                y,
                allow_refit_fallback=allow_refit_fallback,
                estimator_name=est_key,
                warnings=warnings,
                disclosures=disclosures,
            )
    except ValidationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(
            f"Online init fit failed for estimator={est_key!r}: {exc}"
        ) from exc

    init_means = tuple(float(v) for v in x.mean(axis=0))
    disclosures.extend(
        [
            "Online / continual learning uses sklearn partial_fit on train "
            "chunks carved from the Session train partition (or role-aligned "
            "user frames).",
            "Validation/test partitions are never used for incremental updates.",
            "This is batch/stream-chunk updating — not a distributed streaming "
            "platform or full lifelong-learning research suite.",
            f"Init chunk: n_init_rows={len(chunk_indices)} of n_train={n_train}; "
            f"cursor advanced to {cursor}.",
            f"Update mode for init: {update_mode}.",
        ]
    )
    if allow_refit_fallback:
        disclosures.append(
            "allow_refit_fallback=True: if an estimator lacks partial_fit, "
            "BuildML may full-refit on cumulative seen rows and will disclose it "
            "(never silently)."
        )
    else:
        disclosures.append(
            "allow_refit_fallback=False: estimators without partial_fit are "
            "rejected rather than silently full-refit."
        )

    config = OnlineConfig(
        estimator=est_key,  # type: ignore[arg-type]
        task=resolved_task,
        columns=tuple(cols),
        random_state=random_state,
        chunk_size=int(chunk_size),
        n_init=init_n,
        classes=classes_tuple,
        prefer_reduce_components=prefer_reduce_components,
        allow_refit_fallback=allow_refit_fallback,
        drift_disclose=drift_disclose,
    )
    history = (
        {
            "round": 0,
            "kind": "init",
            "n_rows": len(chunk_indices),
            "indices": list(chunk_indices),
            "update_mode": update_mode,
            "used_refit_fallback": used_refit,
        },
    )
    plan = OnlinePlan(
        estimator_name=est_key,
        task=resolved_task,
        columns=tuple(cols),
        target_column=target,
        n_train_rows=n_train,
        n_seen_rows=len(chunk_indices),
        n_updates=0,
        cursor=cursor,
        chunk_size=int(chunk_size),
        classes_=classes_tuple,
        seen_train_indices=tuple(chunk_indices),
        update_history=history,
        estimator_=estimator_obj,
        label_encoder_=label_encoder,
        init_feature_means_=init_means,
        used_refit_fallback=used_refit,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
        used_reduce_components=used_reduce,
        config=config.to_dict(),
    )
    result = OnlineFitResult(
        estimator_name=est_key,
        task=resolved_task,
        n_init_rows=len(chunk_indices),
        n_train_rows=n_train,
        n_remaining_train=max(0, n_train - cursor),
        columns=tuple(cols),
        target_column=target,
        classes=classes_tuple,
        used_reduce_components=used_reduce,
        used_refit_fallback=used_refit,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
    return plan, result


def _resolve_task(estimator: str, task: OnlineTask | None) -> OnlineTask:
    if estimator in _CLASSIFIERS:
        if task == "regression":
            raise ValidationError(
                f"Estimator {estimator!r} is a classifier; task cannot be "
                "'regression'."
            )
        return "classification"
    if estimator in _REGRESSORS:
        if task == "classification":
            raise ValidationError(
                f"Estimator {estimator!r} is a regressor; task cannot be "
                "'classification'."
            )
        return "regression"
    raise ValidationError(
        f"Unknown online estimator={estimator!r}. "
        f"Supported: {sorted(_CLASSIFIERS | _REGRESSORS)}"
    )


def _build_estimator(name: str, random_state: int | None) -> Any:
    if name == "sgd_classifier":
        return SGDClassifier(loss="log_loss", random_state=random_state)
    if name == "sgd_regressor":
        return SGDRegressor(random_state=random_state)
    if name == "passive_aggressive_classifier":
        return PassiveAggressiveClassifier(random_state=random_state)
    if name == "passive_aggressive_regressor":
        return PassiveAggressiveRegressor(random_state=random_state)
    if name == "perceptron":
        return Perceptron(random_state=random_state)
    if name == "multinomial_nb":
        return MultinomialNB()
    if name == "bernoulli_nb":
        return BernoulliNB()
    raise ValidationError(f"Unsupported online estimator '{name}'")


def _maybe_refit_fallback(
    estimator_obj: Any,
    x: np.ndarray,
    y: np.ndarray,
    *,
    allow_refit_fallback: bool,
    estimator_name: str,
    warnings: list[str],
    disclosures: list[str],
) -> tuple[bool, str]:
    if not allow_refit_fallback:
        raise ValidationError(
            f"Estimator {estimator_name!r} does not support partial_fit. "
            "Online / continual learning refuses silent full refits. Pass "
            "allow_refit_fallback=True to permit an explicit disclosed full "
            "refit on cumulative seen rows, or choose an estimator from the "
            "sklearn partial_fit family (SGD*, PassiveAggressive*, Perceptron, "
            "MultinomialNB, BernoulliNB)."
        )
    estimator_obj.fit(x, y)
    msg = (
        f"REFIT FALLBACK (disclosed): estimator={estimator_name!r} lacks "
        "partial_fit; fitted with a full .fit on the current cumulative chunk "
        "instead. This is not incremental online learning."
    )
    warnings.append(msg)
    disclosures.append(msg)
    return True, "refit_fallback"
