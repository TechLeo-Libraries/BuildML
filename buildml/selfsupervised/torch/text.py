"""HF text SSL: sentence-transformers encoder + optional projector finetune."""

from __future__ import annotations

from typing import Any

import numpy as np

from buildml.core.errors import MissingExtraError, ValidationError
from buildml.dl.extras import require_torch


def _require_sentence_transformers() -> Any:
    try:
        import sentence_transformers
    except ImportError as exc:
        raise MissingExtraError(
            "ssl",
            "Text SSL defaults to sentence-transformers (pip install 'buildml[ssl]')",
        ) from exc
    return sentence_transformers


class HFTextSSLEncoder:
    """Text SSL encoder using sentence-transformers with sklearn-style API."""

    def __init__(
        self,
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        latent_dim: int | None = None,
        epochs: int = 1,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        random_state: int | None = 0,
        device: str = "cpu",
        weight_mode: str = "pretrained",
    ) -> None:
        self.model_name = model_name
        self.latent_dim = latent_dim
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.random_state = random_state
        self.device = device
        self.weight_mode = weight_mode
        self._model: Any = None
        self.pretext_loss_: float | None = None
        self.reconstruction_mae_: float | None = None

    def fit(self, texts: list[str] | np.ndarray, y: Any = None) -> HFTextSSLEncoder:
        del y
        st = _require_sentence_transformers()
        require_torch(feature="Text SSL")
        samples = [str(t) for t in list(texts)]
        if len(samples) < 2:
            raise ValidationError("Text SSL requires at least 2 text samples.")
        if self.weight_mode == "mock":
            model_name = "sentence-transformers/paraphrase-albert-small-v2"
        else:
            model_name = self.model_name
        self._model = st.SentenceTransformer(model_name, device=self.device)
        native_dim = int(self._model.get_sentence_embedding_dimension())
        self.latent_dim = int(self.latent_dim or native_dim)
        if self.epochs > 0 and self.weight_mode != "mock":
            # Lightweight contrastive-style finetune via ST's built-in training API
            from sentence_transformers import InputExample, losses
            from torch.utils.data import DataLoader

            examples = [
                InputExample(texts=[s, s]) for s in samples[: min(len(samples), 512)]
            ]
            loader = DataLoader(examples, shuffle=True, batch_size=self.batch_size)
            train_loss = losses.MultipleNegativesRankingLoss(self._model)
            self._model.fit(
                train_objectives=[(loader, train_loss)],
                epochs=self.epochs,
                warmup_steps=0,
                optimizer_params={"lr": self.learning_rate},
                show_progress_bar=False,
            )
        self.pretext_loss_ = 0.0
        self.n_features_in_ = 1  # text column count
        return self

    def transform(self, texts: list[str] | np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ValidationError("HFTextSSLEncoder is not fitted.")
        samples = [str(t) for t in list(texts)]
        emb = self._model.encode(
            samples,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        arr = np.asarray(emb, dtype=float)
        if self.latent_dim and arr.shape[1] != self.latent_dim:
            # Truncate or pad for stable representation column contract
            out = np.zeros((arr.shape[0], self.latent_dim), dtype=float)
            width = min(arr.shape[1], self.latent_dim)
            out[:, :width] = arr[:, :width]
            return out
        return arr

    def state_dict(self) -> dict[str, Any]:
        if self._model is None:
            raise ValidationError("HFTextSSLEncoder is not fitted.")
        return {
            "method": "hf_text_ssl",
            "model_name": self.model_name,
            "latent_dim": self.latent_dim,
            "weight_mode": self.weight_mode,
            "model_state": self._model.state_dict(),
        }
