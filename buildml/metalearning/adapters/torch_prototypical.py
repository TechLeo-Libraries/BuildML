"""PyTorch tabular ProtoNet encoder (deep prototypical few-shot)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from buildml.core.errors import ValidationError
from buildml.metalearning.extras import require_torch_metalearning
from buildml.metalearning.features import compute_prototypes, nearest_prototype_predict


def _build_encoder(n_features: int, embed_dim: int, hidden_dim: int) -> Any:
    torch = require_torch_metalearning(feature="Torch prototypical meta-learning")
    return torch.nn.Sequential(
        torch.nn.Linear(n_features, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, embed_dim),
    )


@dataclass
class TabularProtoNet:
    """Small MLP encoder + episodic prototype loss for tabular few-shot."""

    embed_dim: int = 32
    hidden_dim: int = 64
    meta_epochs: int = 40
    inner_lr: float = 1e-2
    meta_lr: float = 1e-3
    weight_decay: float = 1e-5
    random_state: int | None = 0
    device: str = "cpu"
    n_features_: int = 0
    n_classes_: int = 0
    encoder_: Any = field(default=None, repr=False)
    classes_: np.ndarray = field(default_factory=lambda: np.array([]))

    def fit_episode(
        self,
        x_support: np.ndarray,
        y_support: np.ndarray,
        x_query: np.ndarray,
        y_query: np.ndarray,
    ) -> float:
        """One episodic forward pass; returns query accuracy (for meta-train scoring)."""
        torch = require_torch_metalearning()
        device = torch.device(self.device)
        if self.encoder_ is None:
            raise ValidationError("TabularProtoNet encoder is not initialized.")
        self.encoder_.train()
        x_s = torch.as_tensor(x_support, dtype=torch.float32, device=device)
        y_s = torch.as_tensor(y_support, dtype=torch.long, device=device)
        x_q = torch.as_tensor(x_query, dtype=torch.float32, device=device)
        y_q = torch.as_tensor(y_query, dtype=torch.long, device=device)
        emb_s = self.encoder_(x_s)
        emb_q = self.encoder_(x_q)
        protos = {}
        for code in torch.unique(y_s):
            mask = y_s == code
            protos[int(code.item())] = emb_s[mask].mean(dim=0)
        if not protos:
            return 0.0
        codes = sorted(protos)
        proto_mat = torch.stack([protos[c] for c in codes])
        d2 = (
            (emb_q**2).sum(dim=1, keepdim=True)
            + (proto_mat**2).sum(dim=1)
            - 2.0 * emb_q @ proto_mat.T
        )
        nearest = torch.argmin(d2, dim=1)
        pred_codes = torch.tensor([codes[i] for i in nearest.tolist()], device=device)
        return float((pred_codes == y_q).float().mean().item())

    def embed(self, x: np.ndarray) -> np.ndarray:
        torch = require_torch_metalearning()
        if self.encoder_ is None:
            raise ValidationError("TabularProtoNet encoder is not initialized.")
        device = torch.device(self.device)
        self.encoder_.eval()
        with torch.no_grad():
            tensor_x = torch.as_tensor(x, dtype=torch.float32, device=device)
            emb = self.encoder_(tensor_x)
        return emb.cpu().numpy()

    def predict_from_support(
        self,
        x_support: np.ndarray,
        y_support: np.ndarray,
        x_query: np.ndarray,
    ) -> np.ndarray:
        emb_s = self.embed(x_support)
        emb_q = self.embed(x_query)
        protos = compute_prototypes(emb_s, y_support)
        return nearest_prototype_predict(emb_q, protos)


def meta_train_prototypical_torch(
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
    meta_epochs: int = 40,
    embed_dim: int = 32,
    hidden_dim: int = 64,
    meta_lr: float = 1e-3,
    random_state: int | None = 0,
    device: str = "cpu",
) -> tuple[TabularProtoNet, float | None, list[str], list[str]]:
    """Meta-train a tabular ProtoNet encoder on episodic tasks."""
    from buildml.metalearning.features import (
        encode_labels,
        frame_for_task,
        matrix_from_frame,
        sample_support_query,
    )

    torch = require_torch_metalearning(feature="Torch prototypical meta-learning")

    peek = train.loc[train[task_column].isin(meta_train_ids)]
    n_features = int(matrix_from_frame(peek.head(1), columns).shape[1])

    model = TabularProtoNet(
        embed_dim=int(embed_dim),
        hidden_dim=int(hidden_dim),
        meta_epochs=int(meta_epochs),
        meta_lr=float(meta_lr),
        random_state=random_state,
        device=device,
        n_features_=n_features,
    )
    dev = torch.device(device)
    encoder = _build_encoder(n_features, int(embed_dim), int(hidden_dim)).to(dev)
    optimizer = torch.optim.AdamW(
        encoder.parameters(), lr=float(meta_lr), weight_decay=model.weight_decay
    )
    criterion = torch.nn.CrossEntropyLoss()
    episode_rng = np.random.default_rng(random_state)

    notes = [
        "Torch prototypical meta-train: MLP encoder + episodic prototype "
        "cross-entropy on tabular features (small-scale ProtoNet, not vision)."
    ]
    warns: list[str] = []
    meta_scores: list[float] = []

    for epoch in range(int(meta_epochs)):
        epoch_losses: list[float] = []
        skipped = 0
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
                skipped += 1
                continue
            support, query, _ = sampled
            x_s = matrix_from_frame(support, columns)
            y_s, _, _ = encode_labels(support[target_column], label_encoder=label_encoder)
            x_q = matrix_from_frame(query, columns)
            y_q, _, _ = encode_labels(query[target_column], label_encoder=label_encoder)

            encoder.train()
            optimizer.zero_grad()
            x_s_t = torch.as_tensor(x_s, dtype=torch.float32, device=dev)
            y_s_t = torch.as_tensor(y_s, dtype=torch.long, device=dev)
            x_q_t = torch.as_tensor(x_q, dtype=torch.float32, device=dev)
            y_q_t = torch.as_tensor(y_q, dtype=torch.long, device=dev)
            emb_s = encoder(x_s_t)
            emb_q = encoder(x_q_t)
            protos: dict[int, Any] = {}
            for code in torch.unique(y_s_t):
                mask = y_s_t == code
                protos[int(code.item())] = emb_s[mask].mean(dim=0)
            if not protos:
                skipped += 1
                continue
            codes = sorted(protos)
            proto_mat = torch.stack([protos[c] for c in codes])
            logits = -(
                (emb_q.unsqueeze(1) - proto_mat.unsqueeze(0)) ** 2
            ).sum(dim=2)
            code_map = {c: i for i, c in enumerate(codes)}
            y_mapped = torch.tensor(
                [code_map[int(c)] for c in y_q_t.tolist()],
                device=dev,
                dtype=torch.long,
            )
            loss = criterion(logits, y_mapped)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

            model.encoder_ = encoder
            acc = model.fit_episode(x_s, y_s, x_q, y_q)
            meta_scores.append(acc)
            _ = skipped

        if epoch == int(meta_epochs) - 1 and epoch_losses:
            notes.append(
                f"Final meta-epoch mean episodic loss={float(np.mean(epoch_losses)):.4f}."
            )

    model.encoder_ = encoder
    model.n_features_ = n_features
    if meta_scores:
        acc = float(np.mean(meta_scores[-min(len(meta_scores), n_episodes) :]))
        notes.append(
            f"Meta-train episodic mean query accuracy={acc:.4f} over "
            f"{min(len(meta_scores), n_episodes)} last-epoch episode(s)."
        )
        return model, acc, notes, warns

    warns.append(
        "No successful torch prototypical episodes during meta-train; "
        "increase rows-per-task or reduce k_shot/n_way."
    )
    return model, None, notes, warns
