"""Sentence-transformer document vectors for the NLP embedding backend."""

from __future__ import annotations

from typing import Any

import numpy as np

from buildml.core.errors import ValidationError
from buildml.nlp.extras import require_sentence_transformers


class SentenceEmbeddingVectorizer:
    """Encode documents with a sentence-transformer model.

    The heavy model object is never pickled: :meth:`__getstate__` drops it and it
    is reloaded by name on first use after a bundle round-trip, so
    ``buildml.nlp_bundle.v1`` stays small and portable.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        device: str | None = None,
    ) -> None:
        """Configure the encoder without loading it.

        Construction is deliberately cheap: the model is loaded on first use,
        so building a vectorizer costs nothing and a plan that is never applied
        never downloads weights.

        Parameters
        ----------
        model_name:
            Which sentence-transformer to use. The default is small and fast;
            larger models capture more nuance at proportional cost. Downloaded
            on first use and cached thereafter.
        batch_size:
            How many documents to encode at once. Larger is faster and uses
            more memory — lower it if you run out on a GPU.
        normalize_embeddings:
            Scale each vector to unit length, which makes dot products equal
            cosine similarity and keeps document length from affecting
            magnitude. Almost always what you want.
        device:
            Where to run. ``None`` lets the library choose, which usually means
            a GPU if one is visible.
        """
        self.model_name = str(model_name)
        self.batch_size = int(batch_size)
        self.normalize_embeddings = bool(normalize_embeddings)
        self.device = device
        self.embedding_dim_: int | None = None
        self._model: Any = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            module = require_sentence_transformers(
                feature="NLP embedding backend (sentence-transformer document vectors)"
            )
            try:
                self._model = module.SentenceTransformer(self.model_name, device=self.device)
            except Exception as exc:  # pragma: no cover - network / model errors
                raise ValidationError(
                    f"Could not load sentence-transformer model {self.model_name!r}: {exc}"
                ) from exc
        return self._model

    def fit(self, documents: list[str], y: Any = None) -> SentenceEmbeddingVectorizer:
        """Record the embedding width by encoding the training documents.

        Nothing is learned here, unlike a bag-of-words fit. The encoder's
        weights are frozen and no vocabulary is built — the only state acquired
        is the output width, needed so an empty input can still return a
        correctly shaped array.

        Parameters
        ----------
        documents:
            Training documents.
        y:
            Ignored. Present for scikit-learn compatibility.

        Returns
        -------
        SentenceEmbeddingVectorizer
            Self, following the scikit-learn convention.

        Raises
        ------
        ~buildml.core.errors.MissingExtraError
            sentence-transformers is not installed.
        ~buildml.core.errors.ValidationError
            The named model could not be loaded or downloaded.
        """
        matrix = self.transform(documents)
        self.embedding_dim_ = int(matrix.shape[1])
        return self

    def fit_transform(self, documents: list[str], y: Any = None) -> np.ndarray:
        """Encode the training documents and record the embedding width.

        Genuinely equivalent to fit followed by transform here, and cheaper —
        because nothing is learned, running it as one step avoids encoding the
        same documents twice.

        Parameters
        ----------
        documents:
            Training documents.
        y:
            Ignored. Present for scikit-learn compatibility.

        Returns
        -------
        ~numpy.ndarray
            Dense float32 vectors, one row per document.

        Raises
        ------
        ~buildml.core.errors.MissingExtraError
            sentence-transformers is not installed.
        ~buildml.core.errors.ValidationError
            The named model could not be loaded or downloaded.
        """
        matrix = self.transform(documents)
        self.embedding_dim_ = int(matrix.shape[1])
        return matrix

    def transform(self, documents: list[str]) -> np.ndarray:
        """Encode documents as dense vectors positioned by meaning.

        Two documents that say the same thing in different words land close
        together, which is the entire reason to prefer this over
        bag-of-n-grams. The cost is that no dimension corresponds to any
        particular word, so token attributions become impossible.

        Parameters
        ----------
        documents:
            Raw strings. ``None`` becomes an empty string.

        Returns
        -------
        ~numpy.ndarray
            Dense float32 vectors, one row per document. An empty input gives a
            correctly shaped empty array so downstream code does not need a
            special case.

        Raises
        ------
        ~buildml.core.errors.MissingExtraError
            sentence-transformers is not installed.
        ~buildml.core.errors.ValidationError
            The named model could not be loaded or downloaded.

        Notes
        -----
        The model is loaded on first call, which may download weights. Later
        calls reuse it.

        Long documents are truncated at the model's own sequence limit, quietly
        — text past the cut-off does not influence the vector.
        """
        model = self._ensure_model()
        texts = [("" if item is None else str(item)) for item in documents]
        if not texts:
            width = self.embedding_dim_ or 0
            return np.zeros((0, width), dtype=np.float32)
        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return positional names for the embedding dimensions.

        Names like ``embed_0`` are placeholders, and honestly so. An embedding
        dimension does not correspond to a word or to any nameable concept —
        it is one axis of a learned space. This exists for scikit-learn
        compatibility, not for interpretation, which is why
        :func:`~buildml.nlp.interpret.interpret_text_prediction` refuses this
        backend rather than reporting attributions against these names.

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
        return np.asarray([f"embed_{index}" for index in range(width)], dtype=object)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_model"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._model = None


__all__ = ["SentenceEmbeddingVectorizer"]
