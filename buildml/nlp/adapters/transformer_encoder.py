"""Mean-pooled Hugging Face encoder vectors for the NLP transformer backend.

The encoder stays **frozen**: BuildML pools its last hidden states and fits a
linear head on top. Full fine-tuning of a transformer lives in the Torch path
(:mod:`buildml.dl`), not here: pooling a frozen encoder is cheap, reproducible,
and honest about what it does.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from buildml.core.errors import ValidationError
from buildml.nlp.extras import require_transformers


class TransformerEncoderVectorizer:
    """Turn documents into vectors by averaging a frozen transformer's outputs.

    A transformer produces one vector per token. To get one vector per
    document, those are averaged: weighted by the attention mask, so padding
    contributes nothing. Crude compared to a model trained to produce document
    vectors, and it works well enough to be worth having when
    sentence-transformers is not an option or you want a specific encoder.

    The encoder's weights are never updated. Only the classifier head on top is
    fitted, which keeps this cheap and reproducible. Fine-tuning lives in
    :mod:`buildml.dl`.

    The model is dropped on pickling and reloaded by name afterwards, so a
    saved bundle stays small and portable rather than carrying hundreds of
    megabytes of weights.

    Attributes
    ----------
    model_name:
        Which pretrained encoder is loaded.
    max_seq_tokens:
        Truncation limit in tokens.
    embedding_dim_:
        The output width, known once the encoder has run.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        max_seq_tokens: int = 256,
        batch_size: int = 16,
        device: str = "cpu",
        normalize_embeddings: bool = True,
    ) -> None:
        """Configure the encoder without loading it.

        Construction is cheap; the model and tokenizer load on first use.

        Parameters
        ----------
        model_name:
            Any Hugging Face encoder identifier. Downloaded on first use.
        max_seq_tokens:
            How many tokens of each document to read. Text beyond this is
            truncated and cannot influence the vector, so raise it for long
            documents whose signal is not front-loaded: attention cost grows
            quadratically with this number.
        batch_size:
            How many documents to encode at once. Smaller than the
            sentence-transformer default, since raw encoders use more memory
            per document.
        device:
            Where to run. ``'cuda'`` is dramatically faster where available.
        normalize_embeddings:
            Scale each vector to unit length, so document length does not
            affect magnitude and dot products equal cosine similarity.
        """
        self.model_name = str(model_name)
        self.max_seq_tokens = int(max_seq_tokens)
        self.batch_size = int(batch_size)
        self.device = str(device)
        self.normalize_embeddings = bool(normalize_embeddings)
        self.embedding_dim_: int | None = None
        self._tokenizer: Any = None
        self._model: Any = None

    def _ensure_model(self) -> tuple[Any, Any]:
        if self._model is None or self._tokenizer is None:
            transformers = require_transformers(
                feature="NLP transformer backend (frozen encoder pooling)"
            )
            try:
                self._tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
                self._model = transformers.AutoModel.from_pretrained(self.model_name)
                self._model.eval()
                self._model.to(self.device)
            except Exception as exc:  # pragma: no cover - network / model errors
                raise ValidationError(
                    f"Could not load transformer encoder {self.model_name!r}: {exc}"
                ) from exc
        return self._tokenizer, self._model

    def fit(self, documents: list[str], y: Any = None) -> TransformerEncoderVectorizer:
        """Record the output width by encoding the training documents.

        The encoder is frozen, so nothing is learned. The only state acquired
        is the pooled width.

        Parameters
        ----------
        documents:
            Training documents.
        y:
            Ignored. Present for scikit-learn compatibility.

        Returns
        -------
        TransformerEncoderVectorizer
            Self, following the scikit-learn convention.

        Raises
        ------
        ~buildml.core.errors.MissingExtraError
            transformers or PyTorch is not installed.
        ~buildml.core.errors.ValidationError
            The named encoder could not be loaded or downloaded.
        """
        self.fit_transform(documents)
        return self

    def fit_transform(self, documents: list[str], y: Any = None) -> np.ndarray:
        """Encode the training documents and record the output width.

        Equivalent to fit then transform, and cheaper: since the encoder is
        frozen, doing it in one step avoids running the model twice over the
        same documents, which is the expensive part.

        Parameters
        ----------
        documents:
            Training documents.
        y:
            Ignored. Present for scikit-learn compatibility.

        Returns
        -------
        ~numpy.ndarray
            Pooled float32 vectors, one row per document.

        Raises
        ------
        ~buildml.core.errors.MissingExtraError
            transformers or PyTorch is not installed.
        ~buildml.core.errors.ValidationError
            The named encoder could not be loaded or downloaded.
        """
        matrix = self.transform(documents)
        self.embedding_dim_ = int(matrix.shape[1])
        return matrix

    def transform(self, documents: list[str]) -> np.ndarray:
        """Encode documents by running the transformer and pooling its outputs.

        Each document is tokenised, passed through the encoder, and its
        token vectors averaged: masked so padding contributes nothing, which
        is what keeps a short document in a padded batch from being diluted.

        Parameters
        ----------
        documents:
            Raw strings. ``None`` becomes an empty string.

        Returns
        -------
        ~numpy.ndarray
            Pooled float32 vectors, one row per document. An empty input gives
            a correctly shaped empty array.

        Raises
        ------
        ~buildml.core.errors.MissingExtraError
            transformers or PyTorch is not installed.
        ~buildml.core.errors.ValidationError
            The named encoder could not be loaded or downloaded.

        Notes
        -----
        Runs under ``torch.no_grad()`` in evaluation mode: no gradients, no
        dropout, and the same input always gives the same vector.

        **Documents longer than ``max_seq_tokens`` are silently truncated.**
        Nothing warns, and the discarded text simply does not influence the
        result. On long documents this is the setting most likely to be
        quietly costing you accuracy.

        Mean pooling weights every token equally, which dilutes a single
        important sentence in a long document. A model trained to produce
        sentence embeddings generally does better; this is the fallback that
        works with any encoder.
        """
        from buildml.dl.extras import require_torch

        torch = require_torch(feature="NLP transformer backend")
        tokenizer, model = self._ensure_model()
        texts = [("" if item is None else str(item)) for item in documents]
        if not texts:
            return np.zeros((0, self.embedding_dim_ or 0), dtype=np.float32)

        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                encoded = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_tokens,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                output = model(**encoded)
                hidden = output.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                if self.normalize_embeddings:
                    pooled = pooled / pooled.norm(dim=1, keepdim=True).clamp(min=1e-12)
                chunks.append(pooled.cpu().numpy().astype(np.float32))
        return np.vstack(chunks)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return positional names for the pooled dimensions.

        Names like ``pooled_0`` are placeholders. A pooled transformer
        dimension is an axis of a learned space, corresponding to no word and
        no nameable concept, so token attributions are refused for this backend
        rather than reported against these names.

        Parameters
        ----------
        input_features:
            Ignored. Present for scikit-learn compatibility.

        Returns
        -------
        ~numpy.ndarray
            Positional names, one per dimension. Empty before the width is
            known.
        """
        width = self.embedding_dim_ or 0
        return np.asarray([f"pooled_{index}" for index in range(width)], dtype=object)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_model"] = None
        state["_tokenizer"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._model = None
        self._tokenizer = None


__all__ = ["TransformerEncoderVectorizer"]
