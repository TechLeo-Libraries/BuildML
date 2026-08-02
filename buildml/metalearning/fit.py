"""Train-only meta-learning fit (episodic few-shot / warm-start init)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition, frame_for_partition
from buildml.metalearning.catalog import resolve_backend_method
from buildml.metalearning.features import (
    compute_prototypes,
    encode_labels,
    frame_for_task,
    matrix_from_frame,
    nearest_prototype_predict,
    resolve_metalearning_columns,
    resolve_target_column,
    resolve_task_column,
    sample_support_query,
    task_ids_in_frame,
)
from buildml.metalearning.results import MetaLearningFitResult, MetaLearningPlan
from buildml.metalearning.types import (
    MetaLearningBackend,
    MetaLearningBaseEstimator,
    MetaLearningConfig,
    MetaLearningMethod,
)

_BASES = {
    "logistic_regression": lambda rs: LogisticRegression(
        max_iter=500, random_state=rs
    ),
    "sgd_classifier": lambda rs: SGDClassifier(
        loss="log_loss", max_iter=500, random_state=rs, tol=1e-3
    ),
}


def fit_metalearning(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    backend: MetaLearningBackend | None = None,
    method: MetaLearningMethod = "prototypical",
    task_column: str | None = None,
    columns: list[str] | None = None,
    n_way: int | None = None,
    k_shot: int = 5,
    n_query: int = 10,
    n_episodes: int = 20,
    base_estimator: MetaLearningBaseEstimator | str = "logistic_regression",
    random_state: int | None = 0,
    prefer_reduce_components: bool = True,
    task_holdout_fraction: float = 0.25,
    meta_epochs: int = 40,
    inner_lr: float = 0.05,
    inner_steps: int = 5,
    meta_lr: float = 1e-3,
    embed_dim: int = 32,
    hidden_dim: int = 64,
    device: str = "cpu",
    reduce_plan: Any | None = None,
) -> tuple[MetaLearningPlan, MetaLearningFitResult]:
    """Meta-train on episodic tasks carved from the train partition only.

    Backends
    --------
    sklearn (default):
        ``prototypical`` nearest-centroid and ``warm_start`` pooled sklearn adapt.
    torch (``buildml[torch]``):
        ``prototypical_torch`` — MLP encoder + episodic prototype loss.
    industry (``buildml[metalearning-industry,torch]``):
        ``maml`` / ``reptile`` — first-order tabular task adaptation via learn2learn
        when installed.

    Honesty
    -------
    Practical tabular few-shot / episodic protocols — not foundation-model
    meta-learning, not MAML-at-scale, not EconML-style causal meta.
    Validation/test partitions are never used for meta-training.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None

    resolved_backend, method_key = resolve_backend_method(
        backend=backend,
        method=str(method),
    )
    if int(k_shot) < 1:
        raise ValidationError("k_shot must be >= 1.")
    if int(n_query) < 1:
        raise ValidationError("n_query must be >= 1.")
    if int(n_episodes) < 1:
        raise ValidationError("n_episodes must be >= 1.")
    if not (0.0 <= float(task_holdout_fraction) < 1.0):
        raise ValidationError(
            "task_holdout_fraction must be in [0, 1)."
        )

    disclosures: list[str] = []
    warnings: list[str] = []

    task_col, task_notes = resolve_task_column(dataset, task_column)
    disclosures.extend(task_notes)
    target_col, target_notes = resolve_target_column(dataset)
    disclosures.extend(target_notes)

    train = frame_for_partition(dataset, split_plan, "train")
    cols, used_reduce, col_notes = resolve_metalearning_columns(
        dataset,
        train,
        columns,
        reduce_plan=reduce_plan,
        prefer_reduce_components=prefer_reduce_components,
        target_column=target_col,
        task_column=task_col,
    )
    disclosures.extend(col_notes)

    all_task_ids = task_ids_in_frame(train, task_col)
    if len(all_task_ids) < 2:
        raise ValidationError(
            "Meta-learning needs at least 2 distinct task ids on the train "
            f"partition (found {all_task_ids!r} via column {task_col!r})."
        )

    rng = np.random.default_rng(random_state)
    held_out: list[Any] = []
    meta_train_ids = list(all_task_ids)
    if float(task_holdout_fraction) > 0.0 and len(all_task_ids) >= 3:
        n_hold = max(1, int(round(len(all_task_ids) * float(task_holdout_fraction))))
        n_hold = min(n_hold, len(all_task_ids) - 2)
        perm = list(rng.permutation(len(all_task_ids)))
        held_idx = perm[:n_hold]
        train_idx = perm[n_hold:]
        held_out = [all_task_ids[i] for i in held_idx]
        meta_train_ids = [all_task_ids[i] for i in train_idx]
        disclosures.append(
            f"Held out {len(held_out)} train task id(s) for internal episodic "
            f"checks (task_holdout_fraction={task_holdout_fraction}): {held_out}."
        )
    else:
        disclosures.append(
            "No internal task holdout "
            f"(task_holdout_fraction={task_holdout_fraction}, "
            f"n_tasks={len(all_task_ids)}); all train tasks used for meta-train."
        )

    # Fit global label encoder on meta-train rows only (leakage-safe).
    meta_train_frame = train.loc[train[task_col].isin(meta_train_ids)]
    _, label_encoder, classes = encode_labels(meta_train_frame[target_col])

    resolved_n_way = int(n_way) if n_way is not None else len(classes)
    if resolved_n_way < 2:
        raise ValidationError("n_way must be >= 2.")
    if resolved_n_way > len(classes):
        raise ValidationError(
            f"n_way={resolved_n_way} exceeds n_classes={len(classes)} "
            f"({list(classes)})."
        )

    init_estimator: Any = None
    meta_learner: Any = None
    meta_acc: float | None = None

    if method_key == "prototypical":
        meta_acc, episode_notes, episode_warns = _meta_train_prototypical(
            train,
            task_column=task_col,
            target_column=target_col,
            columns=cols,
            label_encoder=label_encoder,
            meta_train_ids=meta_train_ids,
            n_way=resolved_n_way,
            k_shot=int(k_shot),
            n_query=int(n_query),
            n_episodes=int(n_episodes),
            rng=rng,
        )
        disclosures.extend(episode_notes)
        warnings.extend(episode_warns)
    elif method_key == "prototypical_torch":
        from buildml.metalearning.adapters.torch_prototypical import (
            meta_train_prototypical_torch,
        )

        meta_learner, meta_acc, episode_notes, episode_warns = (
            meta_train_prototypical_torch(
                train,
                task_column=task_col,
                target_column=target_col,
                columns=cols,
                label_encoder=label_encoder,
                meta_train_ids=meta_train_ids,
                n_way=resolved_n_way,
                k_shot=int(k_shot),
                n_query=int(n_query),
                n_episodes=int(n_episodes),
                rng=rng,
                meta_epochs=int(meta_epochs),
                embed_dim=int(embed_dim),
                hidden_dim=int(hidden_dim),
                meta_lr=float(meta_lr),
                random_state=random_state,
                device=device,
            )
        )
        disclosures.extend(episode_notes)
        warnings.extend(episode_warns)
    elif method_key in {"maml", "reptile"}:
        from buildml.metalearning.adapters.industry_maml import (
            meta_train_maml,
            meta_train_reptile,
        )

        trainer = meta_train_maml if method_key == "maml" else meta_train_reptile
        meta_learner, meta_acc, episode_notes, episode_warns = trainer(
            train,
            task_column=task_col,
            target_column=target_col,
            columns=cols,
            label_encoder=label_encoder,
            meta_train_ids=meta_train_ids,
            n_way=resolved_n_way,
            k_shot=int(k_shot),
            n_query=int(n_query),
            n_episodes=int(n_episodes),
            rng=rng,
            n_classes=len(classes),
            meta_epochs=int(meta_epochs),
            inner_lr=float(inner_lr),
            inner_steps=int(inner_steps),
            meta_lr=float(meta_lr),
            hidden_dim=int(hidden_dim),
            random_state=random_state,
            device=device,
            frame_for_task=frame_for_task,
            sample_support_query=sample_support_query,
            matrix_from_frame=matrix_from_frame,
            encode_labels=encode_labels,
        )
        disclosures.extend(episode_notes)
        warnings.extend(episode_warns)
    else:
        init_estimator, warm_notes = _meta_train_warm_start(
            meta_train_frame,
            columns=cols,
            target_column=target_col,
            label_encoder=label_encoder,
            base_estimator=str(base_estimator),
            random_state=random_state,
        )
        disclosures.extend(warm_notes)
        meta_acc, episode_notes, episode_warns = _meta_eval_warm_start_episodes(
            train,
            init_estimator=init_estimator,
            task_column=task_col,
            target_column=target_col,
            columns=cols,
            label_encoder=label_encoder,
            meta_train_ids=meta_train_ids,
            n_way=resolved_n_way,
            k_shot=int(k_shot),
            n_query=int(n_query),
            n_episodes=int(n_episodes),
            rng=rng,
        )
        disclosures.extend(episode_notes)
        warnings.extend(episode_warns)

    disclosures.extend(
        [
            "Meta-learning fits on the train partition only; validation/test "
            "are never used for meta-training.",
            "Task identity comes from a task/group column; that column is "
            "excluded from features.",
            "Honesty: practical tabular few-shot / episodic Session protocol "
            "— not foundation-model meta-learning or MAML-at-scale.",
            f"backend={resolved_backend}, method={method_key}, "
            f"n_meta_train_tasks={len(meta_train_ids)}, "
            f"n_way={resolved_n_way}, k_shot={k_shot}, n_episodes={n_episodes}, "
            f"n_train_rows={len(train)}.",
        ]
    )

    config = MetaLearningConfig(
        backend=resolved_backend,  # type: ignore[arg-type]
        method=method_key,  # type: ignore[arg-type]
        task_column=task_col,
        columns=tuple(cols),
        n_way=resolved_n_way,
        k_shot=int(k_shot),
        n_query=int(n_query),
        n_episodes=int(n_episodes),
        base_estimator=str(base_estimator),  # type: ignore[arg-type]
        random_state=random_state,
        prefer_reduce_components=prefer_reduce_components,
        task_holdout_fraction=float(task_holdout_fraction),
        meta_epochs=int(meta_epochs),
        inner_lr=float(inner_lr),
        inner_steps=int(inner_steps),
        meta_lr=float(meta_lr),
        embed_dim=int(embed_dim),
        hidden_dim=int(hidden_dim),
        device=str(device),
    )
    plan = MetaLearningPlan(
        backend=resolved_backend,
        method=method_key,
        columns=tuple(cols),
        target_column=target_col,
        task_column=task_col,
        train_task_ids=tuple(meta_train_ids),
        held_out_task_ids=tuple(held_out),
        classes_=classes,
        n_train_rows=int(len(train)),
        n_way=resolved_n_way,
        k_shot=int(k_shot),
        n_query=int(n_query),
        n_episodes=int(n_episodes),
        meta_train_accuracy=meta_acc,
        label_encoder_=label_encoder,
        init_estimator_=init_estimator,
        meta_learner_=meta_learner,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
        used_reduce_components=used_reduce,
        config=config.to_dict(),
    )
    result = MetaLearningFitResult(
        backend=resolved_backend,
        method=method_key,
        n_train_rows=int(len(train)),
        columns=tuple(cols),
        target_column=target_col,
        task_column=task_col,
        n_meta_train_tasks=len(meta_train_ids),
        n_held_out_tasks=len(held_out),
        n_way=resolved_n_way,
        k_shot=int(k_shot),
        n_episodes=int(n_episodes),
        meta_train_accuracy=meta_acc,
        used_reduce_components=used_reduce,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
    return plan, result


def _meta_train_prototypical(
    train: Any,
    *,
    task_column: str,
    target_column: str,
    columns: list[str],
    label_encoder: Any,
    meta_train_ids: list[Any],
    n_way: int,
    k_shot: int,
    n_query: int,
    n_episodes: int,
    rng: np.random.Generator,
) -> tuple[float | None, list[str], list[str]]:
    scores: list[float] = []
    skipped = 0
    for _ in range(n_episodes):
        task_id = meta_train_ids[int(rng.integers(0, len(meta_train_ids)))]
        task_frame = frame_for_task(train, task_column, task_id)
        sampled = sample_support_query(
            task_frame,
            target_column=target_column,
            columns=columns,
            label_encoder=label_encoder,
            k_shot=k_shot,
            n_query=n_query,
            n_way=n_way,
            rng=rng,
        )
        if sampled is None:
            skipped += 1
            continue
        support, query, _ = sampled
        x_s = matrix_from_frame(support, columns)
        y_s, _, _ = encode_labels(
            support[target_column], label_encoder=label_encoder
        )
        x_q = matrix_from_frame(query, columns)
        y_q, _, _ = encode_labels(query[target_column], label_encoder=label_encoder)
        protos = compute_prototypes(x_s, y_s)
        pred = nearest_prototype_predict(x_q, protos)
        scores.append(float(accuracy_score(y_q, pred)))

    notes = [
        "Prototypical meta-train: episodic nearest-centroid on tabular "
        "features (no learned neural embedding)."
    ]
    warns: list[str] = []
    if not scores:
        warns.append(
            "No successful prototypical episodes during meta-train; "
            "increase rows-per-task or reduce k_shot/n_way."
        )
        return None, notes, warns
    if skipped:
        warns.append(
            f"Skipped {skipped}/{n_episodes} episodes (insufficient per-class "
            "rows for support/query)."
        )
    acc = float(np.mean(scores))
    notes.append(
        f"Meta-train episodic mean query accuracy={acc:.4f} over "
        f"{len(scores)} episode(s)."
    )
    return acc, notes, warns


def _meta_train_warm_start(
    meta_train_frame: Any,
    *,
    columns: list[str],
    target_column: str,
    label_encoder: Any,
    base_estimator: str,
    random_state: int | None,
) -> tuple[Any, list[str]]:
    base_key = str(base_estimator).lower().replace("-", "_")
    if base_key not in _BASES:
        raise ValidationError(
            f"Unknown warm_start base_estimator={base_estimator!r}. "
            f"Supported: {sorted(_BASES)}"
        )
    x = matrix_from_frame(meta_train_frame, columns)
    y, _, _ = encode_labels(
        meta_train_frame[target_column], label_encoder=label_encoder
    )
    estimator = _BASES[base_key](random_state)
    try:
        estimator.fit(x, y)
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(
            f"Warm-start meta-init fit failed for "
            f"base_estimator={base_estimator!r}: {exc}"
        ) from exc
    notes = [
        f"Warm-start meta-init: pooled {base_key} fit on meta-train rows "
        f"(n={len(meta_train_frame)}); adapt_to_task clones and refits on "
        "a support set.",
    ]
    return estimator, notes


def _meta_eval_warm_start_episodes(
    train: Any,
    *,
    init_estimator: Any,
    task_column: str,
    target_column: str,
    columns: list[str],
    label_encoder: Any,
    meta_train_ids: list[Any],
    n_way: int,
    k_shot: int,
    n_query: int,
    n_episodes: int,
    rng: np.random.Generator,
) -> tuple[float | None, list[str], list[str]]:
    scores: list[float] = []
    skipped = 0
    for _ in range(n_episodes):
        task_id = meta_train_ids[int(rng.integers(0, len(meta_train_ids)))]
        task_frame = frame_for_task(train, task_column, task_id)
        sampled = sample_support_query(
            task_frame,
            target_column=target_column,
            columns=columns,
            label_encoder=label_encoder,
            k_shot=k_shot,
            n_query=n_query,
            n_way=n_way,
            rng=rng,
        )
        if sampled is None:
            skipped += 1
            continue
        support, query, _ = sampled
        x_s = matrix_from_frame(support, columns)
        y_s, _, _ = encode_labels(
            support[target_column], label_encoder=label_encoder
        )
        x_q = matrix_from_frame(query, columns)
        y_q, _, _ = encode_labels(query[target_column], label_encoder=label_encoder)
        adapted = deepcopy(init_estimator)
        try:
            adapted.fit(x_s, y_s)
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        pred = adapted.predict(x_q)
        scores.append(float(accuracy_score(y_q, pred)))

    notes = [
        "Warm-start meta-train check: episodic support adapt + query score "
        "on meta-train tasks (disclosure metric, not holdout)."
    ]
    warns: list[str] = []
    if not scores:
        warns.append(
            "No successful warm-start episodes during meta-train; "
            "increase rows-per-task or reduce k_shot/n_way."
        )
        return None, notes, warns
    if skipped:
        warns.append(
            f"Skipped {skipped}/{n_episodes} warm-start episodes "
            "(insufficient rows or adapt failure)."
        )
    acc = float(np.mean(scores))
    notes.append(
        f"Meta-train episodic mean query accuracy={acc:.4f} over "
        f"{len(scores)} episode(s)."
    )
    return acc, notes, warns
