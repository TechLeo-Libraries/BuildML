"""Build retrieval artifacts (search matrix, ANN index, encoders) at fit time."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from buildml.cbr.adapters.industry_ann import build_ann_index
from buildml.cbr.adapters.text_embed import embed_text_cases
from buildml.cbr.adapters.torch_metric import (
    build_torch_encoder,
    encode_with_torch,
    fit_torch_encoder,
)
from buildml.cbr.extras import cbr_industry_available
from buildml.core.errors import ValidationError


def build_search_artifacts(
    *,
    backend: str,
    train_frame: pd.DataFrame,
    numeric_matrix: np.ndarray,
    task: str,
    metric: str,
    text_columns: tuple[str, ...],
    text_model_name: str,
    y_fit: np.ndarray,
    torch_epochs: int,
    torch_learning_rate: float,
    torch_hidden_dim: int,
    torch_embed_dim: int,
    device: str,
    random_state: int | None,
    n_classes: int,
) -> tuple[np.ndarray, Any, str | None, str | None, Any, list[str]]:
    """Return search_matrix, ann_index, ann_library, embedder_id, torch_encoder, notes."""
    notes: list[str] = []
    backend_key = str(backend).lower()
    search = np.asarray(numeric_matrix, dtype=float)
    ann_index = None
    ann_library: str | None = None
    embedder_id: str | None = None
    torch_encoder = None

    if backend_key == "embedding":
        if not text_columns:
            raise ValidationError(
                "backend='embedding' requires text_columns with at least one column."
            )
        search, embedder_id = embed_text_cases(
            train_frame,
            text_columns,
            model_name=text_model_name,
            numeric_matrix=numeric_matrix if numeric_matrix.shape[1] else None,
        )
        notes.append(
            f"Text case embedding via {embedder_id}; hybrid numeric concat when present."
        )
    elif backend_key == "torch":
        if search.shape[1] == 0:
            raise ValidationError("torch backend requires numeric feature columns.")
        encoder = build_torch_encoder(
            int(search.shape[1]),
            n_classes=max(int(n_classes), 2),
            task=task,
            hidden_dim=int(torch_hidden_dim),
            embed_dim=int(torch_embed_dim),
            device=device,
        )
        torch_encoder = fit_torch_encoder(
            encoder,
            search,
            y_fit,
            task=task,
            epochs=int(torch_epochs),
            learning_rate=float(torch_learning_rate),
            device=device,
            random_state=random_state,
        )
        search = encode_with_torch(torch_encoder, search, device=device)
        notes.append(
            "Learned metric encoder (lite supervised MLP trunk) for kNN retrieval."
        )
    elif backend_key == "industry":
        if search.shape[1] == 0:
            raise ValidationError("industry backend requires numeric feature columns.")
        notes.append("Industry ANN retrieval on standardized numeric features.")

    if backend_key in {"industry", "embedding", "torch"} and cbr_industry_available():
        ann_index, ann_library = build_ann_index(search, metric=metric)
        notes.append(f"Approximate NN index built with {ann_library} (metric={metric}).")
    elif backend_key == "industry":
        notes.append(
            "cbr-industry extra missing — falling back to exact kNN on search matrix."
        )

    return search, ann_index, ann_library, embedder_id, torch_encoder, notes
