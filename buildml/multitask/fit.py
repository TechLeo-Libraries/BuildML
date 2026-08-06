"""Train-only multi-task fit with sklearn / industry / torch backends."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition, frame_for_partition
from buildml.multitask.adapters.gbdt_multioutput import build_gbdt_estimator
from buildml.multitask.adapters.sklearn import build_sklearn_estimator
from buildml.multitask.adapters.torch_multihead import build_torch_estimator
from buildml.multitask.catalog import resolve_backend_method
from buildml.multitask.features import (
    encode_multitask_y,
    infer_task_kinds,
    infer_task_type,
    matrix_from_frame,
    resolve_multitask_columns,
    resolve_target_columns,
)
from buildml.multitask.results import MultiTaskFitResult, MultiTaskPlan
from buildml.multitask.types import (
    MultiTaskBackend,
    MultiTaskBaseEstimator,
    MultiTaskConfig,
    MultiTaskMethod,
    MultiTaskTask,
)


def fit_multitask(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    backend: MultiTaskBackend | None = None,
    method: MultiTaskMethod = "multi_output",
    task: MultiTaskTask = "auto",
    targets: Sequence[str] | None = None,
    columns: list[str] | None = None,
    base_estimator: MultiTaskBaseEstimator | str = "logistic_regression",
    random_state: int | None = 0,
    order: Sequence[str] | None = None,
    prefer_reduce_components: bool = True,
    prediction_prefix: str = "multitask_pred",
    reduce_plan: Any | None = None,
    epochs: int = 60,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    device: str = "cpu",
) -> tuple[MultiTaskPlan, MultiTaskFitResult]:
    """Fit a multi-target estimator on the train partition only.

    Backends
    --------
    sklearn (default when no extras):
        MultiOutput / Chain façades on shared features.
    industry (``buildml[multitask-industry]``):
        XGBoost/LightGBM/CatBoost multi-target when installed.
    torch (``buildml[torch]``):
        Shared-trunk multi-head joint training; supports mixed cls+reg targets.

    Honesty
    -------
    Train-only fit; validation/test are evaluation-only. Classical
    ``Session.fit`` remains single-target. Not a deep MTL research platform.

    Parameters
    ----------
    dataset:
        BuildML dataset with features and multi-task target columns.
    split_plan:
        Train/validation/test split; train partition is used for fitting.
    backend:
        Optional backend override (``sklearn``, ``industry``, ``torch``).
    method:
        Multi-task method (``multi_output``, ``shared_trunk_multihead``, etc.).
    task:
        ``classification``, ``regression``, ``auto``, or ``mixed`` (torch only).
    targets:
        Optional explicit target column names (>= 2 required).
    columns:
        Optional explicit feature columns.
    base_estimator:
        Sklearn base estimator for MultiOutput/Chain backends.
    random_state:
        Seed for sklearn and torch backends.
    order:
        Optional chain order for ClassifierChain/RegressorChain targets.
    prefer_reduce_components:
        Prefer reduced component columns when a reduce plan exists.
    prediction_prefix:
        Column prefix used when predictions are attached to the dataset.
    reduce_plan:
        Optional dimensionality-reduction plan from an upstream Session step.
    epochs:
        Training epochs for torch shared-trunk backend.
    batch_size:
        Minibatch size for torch shared-trunk backend.
    learning_rate:
        AdamW learning rate for torch shared-trunk backend.
    device:
        Torch device string (e.g. ``cpu``, ``cuda``).

    Returns
    -------
    tuple[MultiTaskPlan, MultiTaskFitResult]
        Fitted plan with estimator and encoders, plus fit summary for history.

    Raises
    ------
    ValidationError
        When targets, task type, backend/method pairing, or partitions are invalid.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None

    resolved_backend, resolved_method = resolve_backend_method(
        backend=backend, method=str(method)
    )
    method_key = str(resolved_method).lower().replace("-", "_")

    target_cols, disclosures = resolve_target_columns(dataset, targets)
    train = frame_for_partition(dataset, split_plan, "train")
    allow_mixed = resolved_backend == "torch" and method_key == "shared_trunk_multihead"
    resolved_task, task_notes = infer_task_type(
        train,
        target_cols,
        task,
        allow_mixed=allow_mixed,
    )
    disclosures.extend(task_notes)
    task_kinds = infer_task_kinds(train, target_cols)

    if method_key == "classifier_chain" and resolved_task != "classification":
        raise ValidationError(
            "method='classifier_chain' requires classification targets "
            f"(resolved task={resolved_task!r})."
        )
    if method_key == "regressor_chain" and resolved_task != "regression":
        raise ValidationError(
            "method='regressor_chain' requires regression targets "
            f"(resolved task={resolved_task!r})."
        )
    if resolved_backend == "industry" and resolved_task == "mixed":
        raise ValidationError(
            "Industry GBDT multi-task supports same-type targets only."
        )
    if resolved_backend == "sklearn" and resolved_task == "mixed":
        raise ValidationError(
            "Sklearn multi-task supports same-type targets only. "
            "Use backend='torch', method='shared_trunk_multihead' for mixed "
            "classification+regression."
        )

    cols, used_reduce, col_notes = resolve_multitask_columns(
        dataset,
        train,
        columns,
        reduce_plan=reduce_plan,
        prefer_reduce_components=prefer_reduce_components,
        target_columns=target_cols,
    )
    disclosures.extend(col_notes)

    x = matrix_from_frame(train, cols)
    y, encoders, classes = encode_multitask_y(
        train,
        target_cols,
        task=resolved_task,
        task_kinds=task_kinds,
    )

    chain_order = _resolve_chain_order(target_cols, order) if order is not None else None
    if order is not None:
        disclosures.append(
            f"Chain order (target names): {[target_cols[i] for i in chain_order]}."
            if chain_order is not None
            else f"Chain order requested: {list(order)}."
        )

    estimator = _build_estimator(
        resolved_backend,
        method_key,
        resolved_task,
        str(base_estimator),
        random_state,
        order_indices=chain_order,
        target_columns=target_cols,
        task_kinds=task_kinds,
        classes_per_task=classes,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
    )
    try:
        estimator.fit(x, y)
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(
            f"Multi-task fit failed for backend={resolved_backend!r}, "
            f"method={method_key!r}, base_estimator={base_estimator!r}: {exc}"
        ) from exc

    backend_notes = {
        "sklearn": (
            "Sklearn MultiOutput / Chain on shared features (train partition only)."
        ),
        "industry": (
            "Industry GBDT multi-target estimators on shared features "
            "(train partition only)."
        ),
        "torch": (
            "Torch shared-trunk multi-head joint training on shared features "
            "(train partition only)."
        ),
    }
    disclosures.extend(
        [
            backend_notes[resolved_backend],
            "Validation/test partitions are never used for fitting.",
            (
                "Mixed classification+regression supported only on torch "
                "shared_trunk_multihead; other backends require same-type targets."
            ),
            "Not a deep multi-head MTL research platform or multi-label binary "
            "relevance zoo.",
            f"n_tasks={len(target_cols)}, backend={resolved_backend}, "
            f"method={method_key}, task={resolved_task}, n_train_rows={len(train)}.",
        ]
    )

    config = MultiTaskConfig(
        method=method_key,  # type: ignore[arg-type]
        backend=resolved_backend,
        task=resolved_task,  # type: ignore[arg-type]
        targets=tuple(target_cols),
        columns=tuple(cols),
        base_estimator=str(base_estimator),  # type: ignore[arg-type]
        random_state=random_state,
        order=None if order is None else tuple(order),
        prefer_reduce_components=prefer_reduce_components,
        prediction_prefix=prediction_prefix,
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        device=str(device),
    )
    plan = MultiTaskPlan(
        method=method_key,
        backend=resolved_backend,
        task=resolved_task,
        columns=tuple(cols),
        target_columns=tuple(target_cols),
        n_train_rows=int(len(train)),
        classes_per_task_=classes,
        task_kinds_=task_kinds,
        estimator_=estimator,
        label_encoders_=encoders,
        disclosures=tuple(disclosures),
        warnings=(),
        used_reduce_components=used_reduce,
        config=config.to_dict(),
    )
    result = MultiTaskFitResult(
        method=method_key,
        backend=resolved_backend,
        task=resolved_task,
        n_train_rows=int(len(train)),
        columns=tuple(cols),
        target_columns=tuple(target_cols),
        n_tasks=len(target_cols),
        used_reduce_components=used_reduce,
        disclosures=tuple(disclosures),
        warnings=(),
    )
    return plan, result


def _resolve_chain_order(
    target_columns: list[str], order: Sequence[str]
) -> list[int]:
    names = list(order)
    if sorted(names) != sorted(target_columns):
        raise ValidationError(
            f"order= must be a permutation of target columns {target_columns}; "
            f"got {names}."
        )
    index = {name: i for i, name in enumerate(target_columns)}
    return [index[name] for name in names]


def _build_estimator(
    backend: str,
    method: str,
    task: str,
    base_estimator: str,
    random_state: int | None,
    *,
    order_indices: list[int] | None,
    target_columns: list[str],
    task_kinds: dict[str, str],
    classes_per_task: dict[str, tuple[Any, ...]],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
) -> Any:
    if backend == "sklearn":
        return build_sklearn_estimator(
            method=method,
            task=task,
            base_estimator=base_estimator,
            random_state=random_state,
            order_indices=order_indices,
        )
    if backend == "industry":
        return build_gbdt_estimator(
            method=method,  # type: ignore[arg-type]
            task=task,
            random_state=random_state,
        )
    if backend == "torch":
        return build_torch_estimator(
            target_columns=target_columns,
            task_kinds=task_kinds,
            classes_per_task=classes_per_task,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            random_state=random_state,
            device=device,
        )
    raise ValidationError(f"Unknown multi-task backend={backend!r}.")
