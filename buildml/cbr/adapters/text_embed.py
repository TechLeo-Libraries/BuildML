"""Sentence-transformer case embedding for text/hybrid CBR."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from buildml.cbr.extras import require_sentence_transformers
from buildml.core.errors import ValidationError


def embed_text_cases(
    frame: pd.DataFrame,
    text_columns: Sequence[str],
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    numeric_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """Embed train cases; optionally concatenate standardized numerics."""
    st = require_sentence_transformers(feature="CBR text case embedding")
    cols = list(text_columns)
    if not cols:
        raise ValidationError("text_columns must be non-empty for embedding backend.")
    for c in cols:
        if c not in frame.columns:
            raise ValidationError(f"Text column {c!r} missing from frame.")
    texts = _concat_text_rows(frame, cols)
    model = st.SentenceTransformer(model_name)
    emb = np.asarray(model.encode(texts, show_progress_bar=False), dtype=float)
    if numeric_matrix is not None and numeric_matrix.shape[1] > 0:
        num = np.asarray(numeric_matrix, dtype=float)
        if num.shape[0] != emb.shape[0]:
            raise ValidationError("numeric_matrix row count must match text rows.")
        emb = np.hstack([emb, num])
    embedder_id = f"sentence-transformers:{model_name}"
    return emb, embedder_id


def embed_text_queries(
    frame: pd.DataFrame,
    text_columns: Sequence[str],
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    numeric_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Embed query rows with the same text columns / optional numeric concat."""
    emb, _ = embed_text_cases(
        frame,
        text_columns,
        model_name=model_name,
        numeric_matrix=numeric_matrix,
    )
    return emb


def _concat_text_rows(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    if len(columns) == 1:
        return frame[columns[0]].astype(str).tolist()
    parts = [frame[c].astype(str) for c in columns]
    merged = parts[0]
    for part in parts[1:]:
        merged = merged + " [SEP] " + part
    return merged.tolist()
