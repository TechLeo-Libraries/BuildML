"""Build, once at fit time, whatever the chosen backend needs to search.

Exact search over raw features needs nothing; the numeric matrix is already the
search space. The other backends need something constructed first — text
embedded into vectors, a metric encoder trained, an approximate index built —
and all of it belongs here, at fit time, on training rows.

Doing it once is not only an optimisation. These artefacts define the space
queries are compared in, so building them per query would mean each query was
compared in a slightly different space. Fitting once and reusing is what makes
distances comparable at all.

The build degrades rather than fails. If an approximate index cannot be built,
the search matrix is still produced and retrieval falls back to an exact scan
over it — same answers, less speed — and the fallback is recorded in the notes.

See Also
--------
buildml.cbr.retrieval_engine.retrieve_neighbor_batches : Using these artefacts.
buildml.cbr.catalog.cbr_capability_matrix : Which backends are available.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from buildml.cbr.adapters.industry_ann import build_ann_index
from buildml.cbr.adapters.text_embed import embed_text_cases
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
    """Construct the search space and index the chosen backend needs.

    Produces the matrix that will actually be searched — raw features, text
    embeddings, or a learned encoding depending on the backend — and, where
    possible, an approximate index over it.

    Parameters
    ----------
    backend:
        ``'sklearn'``, ``'industry'``, ``'embedding'``, or ``'torch'``.
    train_frame:
        The training rows, needed by the embedding backend for text columns.
    numeric_matrix:
        Train numeric features, already standardised.
    task:
        ``'classification'`` or ``'regression'``, shaping the torch encoder's
        objective.
    metric:
        The distance function, which the approximate index must be built for.
    text_columns:
        Columns to embed. Required by the embedding backend.
    text_model_name:
        The sentence-transformer to use.
    y_fit:
        Encoded targets, supervising the torch encoder.
    torch_epochs:
        Training passes for the metric encoder. Ignored by other backends.
    torch_learning_rate:
        Adam step size for the metric encoder.
    torch_hidden_dim:
        Hidden layer width of the metric encoder.
    torch_embed_dim:
        Width of the learned space that retrieval will search.
    device:
        Where the torch encoder trains.
    random_state:
        Seed for reproducible index construction and training.
    n_classes:
        Class count for the torch encoder's output layer.

    Returns
    -------
    tuple
        ``(search_matrix, ann_index, ann_library, embedder_id, torch_encoder,
        notes)``. Everything but the matrix and notes may be ``None``, depending
        on the backend and what is installed.

    Raises
    ------
    ValidationError
        If the embedding backend was requested with no text columns, or an
        artefact could not be built for the chosen backend.
    MissingExtraError
        If a backend's dependency is absent.

    Notes
    -----
    **A failed index build is a warning, not an error.** The search matrix
    stands on its own and retrieval scans it exactly, which returns the same
    neighbours more slowly. The note records what happened.

    **This is the expensive part of fitting.** Embedding a corpus or training an
    encoder can take minutes, against near-zero for exact search — which is why
    persisting the plan is worth doing.

    **The learned backends produce a space nobody can read.** Neighbours are
    still correct in that space, but "why is this case similar?" no longer has
    an answer in terms of your columns.
    """
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
        # Lazy import — keep sklearn/industry paths free of torch DLL init.
        from buildml.cbr.adapters.torch_metric import (
            build_torch_encoder,
            encode_with_torch,
            fit_torch_encoder,
        )

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
