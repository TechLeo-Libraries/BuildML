"""Industry tabular MAML/Reptile adapters (learn2learn when installed)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from buildml.core.errors import ValidationError
from buildml.metalearning.extras import (
    learn2learn_available,
    require_torch_metalearning,
)

IndustryMetaMethod = Literal["maml", "reptile"]


def build_tabular_classifier(
    n_features: int,
    n_classes: int,
    *,
    hidden_dim: int = 64,
) -> Any:
    """Build a small tabular MLP classifier for MAML/Reptile meta-learning.

    Requires ``buildml[torch]``; used as the base network for industry
    first-order meta-training loops.

    Parameters
    ----------
    n_features:
        Input feature dimensionality.
    n_classes:
        Number of classification classes.
    hidden_dim:
        Width of hidden ReLU layers.

    Returns
    -------
    torch.nn.Sequential
        Three-layer MLP classifier moved to the active torch device by callers.
    """
    torch = require_torch_metalearning(feature="Industry MAML/Reptile meta-learning")
    return torch.nn.Sequential(
        torch.nn.Linear(n_features, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, n_classes),
    )


@dataclass
class TabularMetaLearner:
    """Frozen meta-trained tabular classifier for fast inner-loop adapt."""

    method: IndustryMetaMethod = "maml"
    hidden_dim: int = 64
    inner_lr: float = 0.05
    inner_steps: int = 5
    meta_epochs: int = 30
    meta_lr: float = 1e-3
    random_state: int | None = 0
    device: str = "cpu"
    n_features_: int = 0
    n_classes_: int = 0
    module_: Any = field(default=None, repr=False)
    maml_wrapper_: Any = field(default=None, repr=False)

    def adapt_predict(
        self,
        x_support: np.ndarray,
        y_support: np.ndarray,
        x_query: np.ndarray,
    ) -> np.ndarray:
        """Run inner-loop adaptation on support and predict query class codes.

        Uses learn2learn MAML cloning when available; otherwise applies native
        first-order SGD steps on a cloned module (Reptile-style path).

        Parameters
        ----------
        x_support:
            Support feature matrix.
        y_support:
            Support integer class codes.
        x_query:
            Query feature matrix.

        Returns
        -------
        numpy.ndarray
            Predicted integer class codes for query rows.

        Raises
        ------
        ValidationError
            When the meta-learned module is not initialized.
        """
        torch = require_torch_metalearning()
        if self.module_ is None:
            raise ValidationError("TabularMetaLearner module is not initialized.")
        device = torch.device(self.device)
        x_s = torch.as_tensor(x_support, dtype=torch.float32, device=device)
        y_s = torch.as_tensor(y_support, dtype=torch.long, device=device)
        x_q = torch.as_tensor(x_query, dtype=torch.float32, device=device)

        if self.method == "maml" and self.maml_wrapper_ is not None:
            learner = self.maml_wrapper_.clone()
            adapt_opt = torch.optim.SGD(learner.parameters(), lr=float(self.inner_lr))
            for _ in range(int(self.inner_steps)):
                logits = learner(x_s)
                loss = torch.nn.functional.cross_entropy(logits, y_s)
                adapt_opt.zero_grad()
                loss.backward()
                adapt_opt.step()
            with torch.no_grad():
                pred = learner(x_q).argmax(dim=1)
            return pred.cpu().numpy().astype(int)

        # Reptile / native first-order: clone weights and SGD steps on support.
        model = _clone_module(self.module_)
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=float(self.inner_lr))
        for _ in range(int(self.inner_steps)):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x_s), y_s)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(x_q).argmax(dim=1)
        return pred.cpu().numpy().astype(int)


def _clone_module(module: Any) -> Any:
    torch = require_torch_metalearning()
    import copy

    return copy.deepcopy(module)


def _episode_tensors(
    support: Any,
    query: Any,
    columns: list[str],
    target_column: str,
    label_encoder: Any,
    matrix_from_frame: Any,
    encode_labels: Any,
    device: Any,
) -> tuple[Any, Any, Any, Any] | None:
    torch = require_torch_metalearning()
    x_s = matrix_from_frame(support, columns)
    y_s, _, _ = encode_labels(support[target_column], label_encoder=label_encoder)
    x_q = matrix_from_frame(query, columns)
    y_q, _, _ = encode_labels(query[target_column], label_encoder=label_encoder)
    if len(np.unique(y_s)) < 2:
        return None
    return (
        torch.as_tensor(x_s, dtype=torch.float32, device=device),
        torch.as_tensor(y_s, dtype=torch.long, device=device),
        torch.as_tensor(x_q, dtype=torch.float32, device=device),
        torch.as_tensor(y_q, dtype=torch.long, device=device),
    )


def meta_train_maml(
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
    n_classes: int,
    meta_epochs: int = 30,
    inner_lr: float = 0.05,
    inner_steps: int = 5,
    meta_lr: float = 1e-3,
    hidden_dim: int = 64,
    random_state: int | None = 0,
    device: str = "cpu",
    frame_for_task: Any,
    sample_support_query: Any,
    matrix_from_frame: Any,
    encode_labels: Any,
) -> tuple[TabularMetaLearner, float | None, list[str], list[str]]:
    """Meta-train tabular MAML on episodic support/query tasks.

    Uses learn2learn first-order MAML when installed; otherwise an honest native
    SGD meta-loop. Small-scale tabular protocol: not second-order MAML-at-scale.

    Parameters
    ----------
    train:
        Train-partition DataFrame with task and target columns.
    task_column:
        Episodic task identifier column.
    target_column:
        Classification target column.
    columns:
        Feature columns for matrix extraction.
    label_encoder:
        Fitted sklearn label encoder for consistent class codes.
    meta_train_ids:
        Task ids used for meta-training episodes.
    n_way:
        Classes sampled per episode.
    k_shot:
        Support examples per class.
    n_query:
        Query examples per class.
    n_episodes:
        Episodes per meta-epoch.
    rng:
        NumPy generator for task/episode sampling (reserved for callers).
    n_classes:
        Total number of classes in the label encoder.
    meta_epochs:
        Number of outer meta-training epochs.
    inner_lr:
        Inner-loop adaptation learning rate.
    inner_steps:
        Inner-loop SGD steps per episode.
    meta_lr:
        Outer meta-optimizer learning rate.
    hidden_dim:
        MLP hidden width.
    random_state:
        Seed for episodic sampling.
    device:
        Torch device string.
    frame_for_task:
        Callable to slice rows for one task id.
    sample_support_query:
        Callable to draw support/query splits for one task.
    matrix_from_frame:
        Callable to build float feature matrices.
    encode_labels:
        Callable to encode target labels.

    Returns
    -------
    tuple[TabularMetaLearner, float | None, list[str], list[str]]
        Meta-trained learner, mean episodic query accuracy (or ``None``), notes,
        and warnings.
    """
    from sklearn.metrics import accuracy_score

    torch = require_torch_metalearning(feature="Industry MAML meta-learning")

    peek = train.loc[train[task_column].isin(meta_train_ids)]
    n_features = int(matrix_from_frame(peek.head(1), columns).shape[1])
    dev = torch.device(device)
    base = build_tabular_classifier(
        n_features, int(n_classes), hidden_dim=int(hidden_dim)
    ).to(dev)

    learner_obj = TabularMetaLearner(
        method="maml",
        hidden_dim=int(hidden_dim),
        inner_lr=float(inner_lr),
        inner_steps=int(inner_steps),
        meta_epochs=int(meta_epochs),
        meta_lr=float(meta_lr),
        random_state=random_state,
        device=device,
        n_features_=n_features,
        n_classes_=int(n_classes),
        module_=base,
    )

    notes = [
        "Industry MAML meta-train: first-order tabular MAML on episodic "
        "support/query tasks (honest small-scale: not second-order MAML-at-scale)."
    ]
    if learn2learn_available():
        import learn2learn as l2l

        maml = l2l.algorithms.MAML(base, lr=float(inner_lr), first_order=True)
        learner_obj.maml_wrapper_ = maml
        meta_opt = torch.optim.Adam(maml.parameters(), lr=float(meta_lr))
        notes.append("MAML wrapper: learn2learn first_order=True.")
    else:
        meta_opt = torch.optim.Adam(base.parameters(), lr=float(meta_lr))
        notes.append("MAML wrapper unavailable; native first-order SGD meta-loop.")

    warns: list[str] = []
    scores: list[float] = []
    episode_rng = np.random.default_rng(random_state)

    for _epoch in range(int(meta_epochs)):
        for _ in range(int(n_episodes)):
            task_id = meta_train_ids[int(episode_rng.integers(0, len(meta_train_ids)))]
            task_frame = frame_for_task(train, task_column, task_id)
            sampled = sample_support_query(
                task_frame,
                target_column=target_column,
                columns=columns,
                label_encoder=label_encoder,
                k_shot=k_shot,
                n_query=n_query,
                n_way=n_way,
                rng=episode_rng,
            )
            if sampled is None:
                continue
            support, query, _ = sampled
            tensors = _episode_tensors(
                support,
                query,
                columns,
                target_column,
                label_encoder,
                matrix_from_frame,
                encode_labels,
                dev,
            )
            if tensors is None:
                continue
            x_s, y_s, x_q, y_q = tensors

            meta_opt.zero_grad()
            if learner_obj.maml_wrapper_ is not None:
                learner = learner_obj.maml_wrapper_.clone()
                adapt_opt = torch.optim.SGD(learner.parameters(), lr=float(inner_lr))
                for _ in range(int(inner_steps)):
                    logits = learner(x_s)
                    loss = torch.nn.functional.cross_entropy(logits, y_s)
                    adapt_opt.zero_grad()
                    loss.backward()
                    adapt_opt.step()
                query_loss = torch.nn.functional.cross_entropy(learner(x_q), y_q)
                query_loss.backward()
                meta_opt.step()
                pred = learner(x_q).argmax(dim=1).detach().cpu().numpy()
            else:
                model = _clone_module(base)
                opt = torch.optim.SGD(model.parameters(), lr=float(inner_lr))
                for _ in range(int(inner_steps)):
                    opt.zero_grad()
                    loss = torch.nn.functional.cross_entropy(model(x_s), y_s)
                    loss.backward()
                    opt.step()
                query_loss = torch.nn.functional.cross_entropy(model(x_q), y_q)
                meta_opt.zero_grad()
                query_loss.backward()
                meta_opt.step()
                pred = model(x_q).argmax(dim=1).detach().cpu().numpy()

            scores.append(float(accuracy_score(y_q.cpu().numpy(), pred)))

    learner_obj.module_ = base
    if not scores:
        warns.append(
            "No successful MAML episodes during meta-train; increase rows-per-task "
            "or reduce k_shot/n_way."
        )
        return learner_obj, None, notes, warns
    acc = float(np.mean(scores[-min(len(scores), n_episodes) :]))
    notes.append(
        f"Meta-train episodic mean query accuracy={acc:.4f} over "
        f"{min(len(scores), n_episodes)} episode(s)."
    )
    return learner_obj, acc, notes, warns


def meta_train_reptile(
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
    n_classes: int,
    meta_epochs: int = 30,
    inner_lr: float = 0.05,
    inner_steps: int = 5,
    meta_lr: float = 1e-3,
    hidden_dim: int = 64,
    random_state: int | None = 0,
    device: str = "cpu",
    frame_for_task: Any,
    sample_support_query: Any,
    matrix_from_frame: Any,
    encode_labels: Any,
) -> tuple[TabularMetaLearner, float | None, list[str], list[str]]:
    """Meta-train tabular Reptile by interpolating toward task-adapted weights.

    Performs inner-loop SGD on support then moves base weights toward the adapted
    network each episode. First-order, small-scale tabular protocol.

    Parameters
    ----------
    train:
        Train-partition DataFrame with task and target columns.
    task_column:
        Episodic task identifier column.
    target_column:
        Classification target column.
    columns:
        Feature columns for matrix extraction.
    label_encoder:
        Fitted sklearn label encoder for consistent class codes.
    meta_train_ids:
        Task ids used for meta-training episodes.
    n_way:
        Classes sampled per episode.
    k_shot:
        Support examples per class.
    n_query:
        Query examples per class.
    n_episodes:
        Episodes per meta-epoch.
    rng:
        NumPy generator for task/episode sampling (reserved for callers).
    n_classes:
        Total number of classes in the label encoder.
    meta_epochs:
        Number of outer meta-training epochs.
    inner_lr:
        Inner-loop adaptation learning rate.
    inner_steps:
        Inner-loop SGD steps per episode.
    meta_lr:
        Reptile interpolation coefficient toward task-adapted weights.
    hidden_dim:
        MLP hidden width.
    random_state:
        Seed for episodic sampling.
    device:
        Torch device string.
    frame_for_task:
        Callable to slice rows for one task id.
    sample_support_query:
        Callable to draw support/query splits for one task.
    matrix_from_frame:
        Callable to build float feature matrices.
    encode_labels:
        Callable to encode target labels.

    Returns
    -------
    tuple[TabularMetaLearner, float | None, list[str], list[str]]
        Meta-trained learner, mean episodic query accuracy (or ``None``), notes,
        and warnings.
    """
    from sklearn.metrics import accuracy_score

    torch = require_torch_metalearning(feature="Industry Reptile meta-learning")
    peek = train.loc[train[task_column].isin(meta_train_ids)]
    n_features = int(matrix_from_frame(peek.head(1), columns).shape[1])
    dev = torch.device(device)
    base = build_tabular_classifier(
        n_features, int(n_classes), hidden_dim=int(hidden_dim)
    ).to(dev)

    learner_obj = TabularMetaLearner(
        method="reptile",
        hidden_dim=int(hidden_dim),
        inner_lr=float(inner_lr),
        inner_steps=int(inner_steps),
        meta_epochs=int(meta_epochs),
        meta_lr=float(meta_lr),
        random_state=random_state,
        device=device,
        n_features_=n_features,
        n_classes_=int(n_classes),
        module_=base,
    )

    notes = [
        "Industry Reptile meta-train: interpolate meta-weights toward "
        "support-adapted tabular nets (first-order, small-scale)."
    ]
    warns: list[str] = []
    scores: list[float] = []
    episode_rng = np.random.default_rng(random_state)

    for _epoch in range(int(meta_epochs)):
        for _ in range(int(n_episodes)):
            task_id = meta_train_ids[int(episode_rng.integers(0, len(meta_train_ids)))]
            task_frame = frame_for_task(train, task_column, task_id)
            sampled = sample_support_query(
                task_frame,
                target_column=target_column,
                columns=columns,
                label_encoder=label_encoder,
                k_shot=k_shot,
                n_query=n_query,
                n_way=n_way,
                rng=episode_rng,
            )
            if sampled is None:
                continue
            support, query, _ = sampled
            tensors = _episode_tensors(
                support,
                query,
                columns,
                target_column,
                label_encoder,
                matrix_from_frame,
                encode_labels,
                dev,
            )
            if tensors is None:
                continue
            x_s, y_s, x_q, y_q = tensors

            model = _clone_module(base)
            opt = torch.optim.SGD(model.parameters(), lr=float(inner_lr))
            for _ in range(int(inner_steps)):
                opt.zero_grad()
                loss = torch.nn.functional.cross_entropy(model(x_s), y_s)
                loss.backward()
                opt.step()

            # Reptile meta-update: move base toward adapted weights.
            with torch.no_grad():
                for p_base, p_task in zip(base.parameters(), model.parameters()):
                    p_base.add_(p_task - p_base, alpha=float(meta_lr))

            pred = model(x_q).argmax(dim=1).detach().cpu().numpy()
            scores.append(float(accuracy_score(y_q.cpu().numpy(), pred)))

    learner_obj.module_ = base
    if not scores:
        warns.append(
            "No successful Reptile episodes during meta-train; increase rows-per-task "
            "or reduce k_shot/n_way."
        )
        return learner_obj, None, notes, warns
    acc = float(np.mean(scores[-min(len(scores), n_episodes) :]))
    notes.append(
        f"Meta-train episodic mean query accuracy={acc:.4f} over "
        f"{min(len(scores), n_episodes)} episode(s)."
    )
    return learner_obj, acc, notes, warns
