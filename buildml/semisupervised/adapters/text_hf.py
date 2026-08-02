"""HF text pseudo-label semi-supervised adapter (sentence-transformers)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier

from buildml.core.errors import ValidationError
from buildml.semisupervised.extras import require_sentence_transformers
from buildml.semisupervised.types import SKLEARN_UNLABELED


@dataclass
class TextPseudoLabelClassifier:
    """Embed text with sentence-transformers; self-train on partial labels."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    threshold: float = 0.75
    max_iter: int = 10
    random_state: int | None = 0
    text_column_: str = ""
    embedder_: Any = field(default=None, repr=False)
    estimator_: Any = field(default=None, repr=False)
    classes_: np.ndarray = field(default_factory=lambda: np.array([]))

    def fit(self, texts: list[str], y: np.ndarray) -> TextPseudoLabelClassifier:
        st = require_sentence_transformers()
        y_arr = np.asarray(y, dtype=int)
        labeled = y_arr != SKLEARN_UNLABELED
        if labeled.sum() < 2:
            raise ValidationError(
                "Text pseudo-label needs at least 2 labeled train rows."
            )
        classes = np.unique(y_arr[labeled])
        if len(classes) < 2:
            raise ValidationError(
                "Text pseudo-label needs at least 2 classes among labeled rows."
            )
        self.classes_ = classes
        self.embedder_ = st.SentenceTransformer(self.model_name)
        embeddings = np.asarray(self.embedder_.encode(texts, show_progress_bar=False), dtype=float)
        base = LogisticRegression(max_iter=500, random_state=self.random_state)
        self.estimator_ = SelfTrainingClassifier(
            base,
            threshold=float(self.threshold),
            max_iter=int(self.max_iter),
        )
        self.estimator_.fit(embeddings, y_arr)
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        if self.embedder_ is None or self.estimator_ is None:
            raise ValidationError("TextPseudoLabelClassifier is not fitted.")
        embeddings = np.asarray(
            self.embedder_.encode(texts, show_progress_bar=False), dtype=float
        )
        return np.asarray(self.estimator_.predict(embeddings), dtype=int)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        if self.embedder_ is None or self.estimator_ is None:
            raise ValidationError("TextPseudoLabelClassifier is not fitted.")
        embeddings = np.asarray(
            self.embedder_.encode(texts, show_progress_bar=False), dtype=float
        )
        return np.asarray(self.estimator_.predict_proba(embeddings), dtype=float)


def build_text_estimator(
    *,
    model_name: str,
    threshold: float,
    max_iter: int,
    random_state: int | None,
) -> TextPseudoLabelClassifier:
    return TextPseudoLabelClassifier(
        model_name=model_name,
        threshold=float(threshold),
        max_iter=int(max_iter),
        random_state=random_state,
    )
