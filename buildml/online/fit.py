"""Initial warm-start fit for online / continual learning (train chunk only)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition
from buildml.online.adapters import build_online_estimator, resolve_online_task
from buildml.online.catalog import resolve_backend_estimator, resolve_drift_detector
from buildml.online.features import (
    carve_train_chunk,
    encode_classification_targets,
    matrix_from_frame,
    regression_targets,
    resolve_online_columns,
    train_partition_frame,
)
from buildml.online.results import OnlineFitResult, OnlinePlan
from buildml.online.types import OnlineBackend, OnlineConfig, OnlineDriftDetector, OnlineTask


def fit_online(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    backend: OnlineBackend | None = None,
    estimator: str = "sgd_classifier",
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
    drift_detector: OnlineDriftDetector | None = None,
    buffer_size: int = 512,
    epochs_per_update: int = 5,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    ewc_lambda: float = 100.0,
    hidden_dim: int = 64,
    device: str = "cpu",
    reduce_plan: Any | None = None,
) -> tuple[OnlinePlan, OnlineFitResult]:
    """Warm-start an incremental estimator on the first train chunk.

    Backends
    --------
    sklearn (default):
        Sklearn ``partial_fit`` family — always available.
    industry (``buildml[online-industry]``):
        River streaming models with ADWIN / Page-Hinkley drift hooks.
    torch (``buildml[torch]``):
        Lite replay-buffer or EWC tabular MLP continual learner.

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

    resolved_backend, est_key = resolve_backend_estimator(
        backend=backend,
        estimator=estimator,
    )
    resolved_drift = resolve_drift_detector(
        backend=resolved_backend,
        drift_detector=drift_detector,
    )
    resolved_task = resolve_online_task(resolved_backend, est_key, task)
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

    estimator_obj = build_online_estimator(
        resolved_backend,
        est_key,
        random_state=random_state,
        drift_detector=resolved_drift,
        n_features=x.shape[1],
        buffer_size=buffer_size,
        epochs_per_update=epochs_per_update,
        batch_size=batch_size,
        learning_rate=learning_rate,
        ewc_lambda=ewc_lambda,
        hidden_dim=hidden_dim,
        device=device,
    )
    used_refit = False
    update_mode = "partial_fit"
    try:
        if hasattr(estimator_obj, "partial_fit"):
            if resolved_task == "classification":
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
            f"Backend={resolved_backend}, estimator={est_key}, "
            f"drift_detector={resolved_drift}.",
            "Online / continual learning uses partial_fit on train "
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
    if resolved_backend == "torch":
        disclosures.append(
            "Torch continual path is a lite replay/EWC tabular MLP — not a "
            "full lifelong-learning research implementation."
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
        estimator=est_key,
        backend=resolved_backend,
        task=resolved_task,
        columns=tuple(cols),
        random_state=random_state,
        chunk_size=int(chunk_size),
        n_init=init_n,
        classes=classes_tuple,
        prefer_reduce_components=prefer_reduce_components,
        allow_refit_fallback=allow_refit_fallback,
        drift_disclose=drift_disclose,
        drift_detector=resolved_drift,  # type: ignore[arg-type]
        buffer_size=int(buffer_size),
        epochs_per_update=int(epochs_per_update),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        ewc_lambda=float(ewc_lambda),
        hidden_dim=int(hidden_dim),
        device=device,
    )
    history = (
        {
            "round": 0,
            "kind": "init",
            "n_rows": len(chunk_indices),
            "indices": list(chunk_indices),
            "update_mode": update_mode,
            "used_refit_fallback": used_refit,
            "backend": resolved_backend,
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
        backend=resolved_backend,
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
        backend=resolved_backend,
    )
    return plan, result


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
