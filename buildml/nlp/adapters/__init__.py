"""Optional-extra adapters for the NLP domain.

Every adapter is imported lazily so ``import buildml.nlp`` stays on the core
numpy / pandas / scikit-learn stack and never pulls a transformer or spaCy
runtime into a Session that does not ask for one.

* :mod:`buildml.nlp.adapters.sentence_embedding`: sentence-transformer document
  vectors (``buildml[nlp]``).
* :mod:`buildml.nlp.adapters.transformer_encoder`: mean-pooled Hugging Face
  encoder vectors (``buildml[nlp]``).
* :mod:`buildml.nlp.adapters.spacy_pipeline`: statistical entity extraction
  (``buildml[nlp-industry]``, plus a downloaded model).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "DEFAULT_SPACY_MODEL",
    "SPACY_LABEL_ALIASES",
    "SentenceEmbeddingVectorizer",
    "TransformerEncoderVectorizer",
    "extract_spacy_entities",
    "load_spacy_pipeline",
]

_SPACY = {
    "DEFAULT_SPACY_MODEL",
    "SPACY_LABEL_ALIASES",
    "extract_spacy_entities",
    "load_spacy_pipeline",
}


def __getattr__(name: str) -> Any:
    if name == "SentenceEmbeddingVectorizer":
        from buildml.nlp.adapters.sentence_embedding import SentenceEmbeddingVectorizer

        return SentenceEmbeddingVectorizer
    if name == "TransformerEncoderVectorizer":
        from buildml.nlp.adapters.transformer_encoder import TransformerEncoderVectorizer

        return TransformerEncoderVectorizer
    if name in _SPACY:
        from buildml.nlp.adapters import spacy_pipeline

        return getattr(spacy_pipeline, name)
    raise AttributeError(f"module 'buildml.nlp.adapters' has no attribute {name!r}")
