"""Train-only semi-supervised fit with sklearn/industry/torch/HF backends."""

from __future__ import annotations

from typing import Any

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition, frame_for_partition
from buildml.semisupervised.adapters.sklearn import build_sklearn_estimator
from buildml.semisupervised.adapters.text_hf import build_text_estimator
from buildml.semisupervised.adapters.torch_consistency import build_torch_estimator
from buildml.semisupervised.adapters.xgb_pseudo import build_industry_estimator
from buildml.semisupervised.catalog import resolve_backend_method
from buildml.semisupervised.features import (
    encode_targets_for_sklearn,
    matrix_from_frame,
    resolve_semisupervised_columns,
)
from buildml.semisupervised.results import SemiSupervisedFitResult, SemiSupervisedPlan
from buildml.semisupervised.types import (
    SemiSupervisedBackend,
    SemiSupervisedConfig,
    SemiSupervisedMethod,
)


def fit_semisupervised(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    backend: SemiSupervisedBackend | None = None,
    method: SemiSupervisedMethod = "label_propagation",
    columns: list[str] | None = None,
    random_state: int | None = 0,
    kernel: str = "knn",
    n_neighbors: int = 7,
    max_iter: int = 1000,
    alpha: float = 0.2,
    base_estimator: str = "logistic_regression",
    threshold: float = 0.75,
    criterion: str = "threshold",
    k_best: int = 10,
    max_self_train_iter: int = 10,
    epochs: int = 40,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    consistency_weight: float = 1.0,
    mixup_alpha: float = 0.75,
    device: str = "cpu",
    text_column: str | None = None,
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    unlabeled_marker: Any = None,
    prefer_reduce_components: bool = True,
    reduce_plan: Any | None = None,
) -> tuple[SemiSupervisedPlan, SemiSupervisedFitResult]:
    """Fit a semi-supervised classifier on the train partition only.

    Backends
    --------
    sklearn (default):
        LabelPropagation, LabelSpreading, SelfTrainingClassifier — core sklearn.
    industry (``buildml[semisupervised-industry]``):
        XGBoost/LightGBM iterative pseudo-labeling when installed.
    torch (``buildml[torch]``):
        FixMatch/MixMatch-style tabular consistency training.
    hf (``buildml[ssl]``):
        Sentence-transformer text embeddings + pseudo-label self-training.

    Label missingness
    -----------------
    Rows with missing targets (NaN by default) are unlabeled. Sklearn's ``-1``
    convention is applied internally. Validation/test partitions are never used
    to invent labels or to select the model during this fit.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None

    resolved_backend, resolved_method = resolve_backend_method(
        backend=backend, method=method
    )
    target = dataset.require_target()
    train = frame_for_partition(dataset, split_plan, "train")
    modality = "text" if resolved_backend == "hf" else "tabular"
    warnings: list[str] = []
    disclosures: list[str] = [
        "Semi-supervised fit uses the train partition only. "
        "Unlabeled rows are target missingness (NaN by default) mapped to "
        "sklearn's -1 convention internally.",
        "Validation/test rows are never used to invent labels or to select "
        "the estimator during fit_semisupervised.",
        f"Backend={resolved_backend}, method={resolved_method}, modality={modality}.",
    ]

    if resolved_backend == "hf":
        col = text_column or (columns[0] if columns else None)
        if col is None:
            from buildml.core.types import ColumnRole

            feature_cols = dataset.role_columns(ColumnRole.FEATURE) or [
                c for c in train.columns if c != target
            ]
            text_candidates = [
                str(c)
                for c in feature_cols
                if str(c) in train.columns and train[str(c)].dtype == object
            ]
            col = text_candidates[0] if text_candidates else None
        if col is None or col not in train.columns:
            raise ValidationError(
                "text_pseudo_label requires a text feature column via text_column= "
                "or a single object-dtype feature column."
            )
        cols = [str(col)]
        used_reduce = False
        col_disclosures: list[str] = [
            f"HF text pseudo-label uses text column {col!r} with model {text_model_name!r}."
        ]
        y_sk, encoder, classes, n_labeled, n_unlabeled = encode_targets_for_sklearn(
            train[target],
            unlabeled_marker=unlabeled_marker,
        )
        texts = train[col].astype(str).tolist()
        estimator = build_text_estimator(
            model_name=text_model_name,
            threshold=threshold,
            max_iter=max_self_train_iter,
            random_state=random_state,
        )
        estimator.text_column_ = col
        try:
            estimator.fit(texts, y_sk)
        except Exception as exc:  # noqa: BLE001
            raise ValidationError(
                f"Semi-supervised fit failed for method={resolved_method!r}: {exc}"
            ) from exc
        x = None
    else:
        cols, used_reduce, col_disclosures = resolve_semisupervised_columns(
            dataset,
            train,
            columns,
            reduce_plan=reduce_plan,
            prefer_reduce_components=prefer_reduce_components,
            target_column=target,
        )
        x = matrix_from_frame(train, cols)
        y_sk, encoder, classes, n_labeled, n_unlabeled = encode_targets_for_sklearn(
            train[target],
            unlabeled_marker=unlabeled_marker,
        )
        estimator = _build_estimator(
            backend=resolved_backend,
            method=resolved_method,
            random_state=random_state,
            kernel=kernel,
            n_neighbors=n_neighbors,
            max_iter=max_iter,
            alpha=alpha,
            base_estimator=base_estimator,
            threshold=threshold,
            criterion=criterion,
            k_best=k_best,
            max_self_train_iter=max_self_train_iter,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            consistency_weight=consistency_weight,
            mixup_alpha=mixup_alpha,
            device=device,
        )
        try:
            estimator.fit(x, y_sk)
        except Exception as exc:  # noqa: BLE001
            raise ValidationError(
                f"Semi-supervised fit failed for method={resolved_method!r}: {exc}"
            ) from exc

    disclosures.extend(col_disclosures)
    n_train = int(len(train))
    disclosures.extend(
        [
            f"Train label mix: n_labeled={n_labeled}, n_unlabeled={n_unlabeled} "
            f"of n_train={n_train}.",
        ]
    )
    if n_unlabeled == 0:
        warnings.append(
            "No unlabeled train rows detected; semi-supervised methods reduce to "
            "supervised fit on the labeled train set (disclosed)."
        )
    disclosures.extend(_backend_disclosures(resolved_backend, resolved_method, estimator))

    config = SemiSupervisedConfig(
        method=resolved_method,
        backend=resolved_backend,
        columns=tuple(cols),
        random_state=random_state,
        kernel=kernel,
        n_neighbors=n_neighbors,
        max_iter=max_iter,
        alpha=alpha,
        base_estimator=base_estimator,
        threshold=threshold,
        criterion=criterion,
        k_best=k_best,
        max_self_train_iter=max_self_train_iter,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        consistency_weight=consistency_weight,
        mixup_alpha=mixup_alpha,
        device=device,
        text_column=cols[0] if resolved_backend == "hf" else None,
        text_model_name=text_model_name,
        unlabeled_marker=unlabeled_marker,
        prefer_reduce_components=prefer_reduce_components,
        modality=modality,
    )
    plan = SemiSupervisedPlan(
        method=resolved_method,
        backend=resolved_backend,
        modality=modality,
        columns=tuple(cols),
        target_column=target,
        n_train_rows=n_train,
        n_labeled_train=n_labeled,
        n_unlabeled_train=n_unlabeled,
        classes_=classes,
        estimator_=estimator,
        label_encoder_=encoder,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
        used_reduce_components=used_reduce,
        config=config.to_dict(),
    )
    result = SemiSupervisedFitResult(
        method=resolved_method,
        backend=resolved_backend,
        modality=modality,
        n_train_rows=n_train,
        n_labeled_train=n_labeled,
        n_unlabeled_train=n_unlabeled,
        columns=tuple(cols),
        target_column=target,
        classes=classes,
        used_reduce_components=used_reduce,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
    return plan, result


def _build_estimator(
    *,
    backend: SemiSupervisedBackend,
    method: str,
    random_state: int | None,
    kernel: str,
    n_neighbors: int,
    max_iter: int,
    alpha: float,
    base_estimator: str,
    threshold: float,
    criterion: str,
    k_best: int,
    max_self_train_iter: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    consistency_weight: float,
    mixup_alpha: float,
    device: str,
) -> Any:
    if backend == "sklearn":
        return build_sklearn_estimator(
            method=method,  # type: ignore[arg-type]
            random_state=random_state,
            kernel=kernel,
            n_neighbors=n_neighbors,
            max_iter=max_iter,
            alpha=alpha,
            base_estimator=base_estimator,
            threshold=threshold,
            criterion=criterion,
            k_best=k_best,
            max_self_train_iter=max_self_train_iter,
        )
    if backend == "industry":
        return build_industry_estimator(
            method=method,  # type: ignore[arg-type]
            threshold=threshold,
            max_iter=max_self_train_iter,
            random_state=random_state,
        )
    if backend == "torch":
        return build_torch_estimator(
            method=method,  # type: ignore[arg-type]
            threshold=threshold,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            consistency_weight=consistency_weight,
            mixup_alpha=mixup_alpha,
            random_state=random_state,
            device=device,
        )
    raise ValidationError(f"Unsupported semi-supervised backend '{backend}'")


def _backend_disclosures(backend: str, method: str, estimator: Any) -> list[str]:
    notes: list[str] = []
    if backend == "industry":
        n_pseudo = getattr(estimator, "n_pseudo_labels_", None)
        iters = getattr(estimator, "iterations_run_", None)
        notes.append(
            f"Industry pseudo-label ({method}): iterations_run={iters}, "
            f"n_pseudo_labels_accepted={n_pseudo} (train-only, disclosed)."
        )
    elif backend == "torch":
        n_pseudo = getattr(estimator, "n_pseudo_labels_", None)
        notes.append(
            f"Torch consistency ({method}): n_consistency_pseudo_steps={n_pseudo} "
            "(train-only pseudo-label/consistency steps, disclosed)."
        )
    elif backend == "hf":
        notes.append(
            "HF text path embeds train text only; holdout text is encoded at predict "
            "without label leakage."
        )
    return notes
