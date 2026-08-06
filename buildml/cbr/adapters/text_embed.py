"""Embed text case fields so similarity can survive a change of wording.

Treating a text column as categorical asks only whether two strings are
identical, which for anything longer than a label means the answer is always no.
"Laptop will not power on" and "machine won't turn on" are the same case
described twice, and a categorical encoding sees two unrelated values.

Embedding fixes that by mapping text into a vector space where meaning drives
position. The two descriptions land close together, distance registers them as
similar, and case-based reasoning can work over free-text fields the way it
works over numbers.

Numeric features can be concatenated onto the embedding to give a hybrid space
where both contribute. Note that the numeric side is then a handful of
dimensions against several hundred embedding dimensions, so it contributes far
less to distance than a column count would suggest.

See Also
--------
buildml.cbr.extras.text_embedding_available : Whether this backend is usable.
buildml.rag.embed : Embedding for document retrieval.
"""

from __future__ import annotations

from collections.abc import Sequence

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
    """Turn each case's text fields into a vector, optionally with numerics attached.

    Joins the text columns per row, encodes them with a sentence-transformer,
    and concatenates the numeric features if any were supplied. The result is
    the matrix retrieval will search.

    Parameters
    ----------
    frame:
        The rows to embed.
    text_columns:
        Which columns hold text. Multiple columns are joined with a separator
        before encoding, so the model sees one passage per case.
    model_name:
        The sentence-transformer to use. The default is small and fast enough
        for interactive work.
    numeric_matrix:
        Standardised numeric features to concatenate, or ``None``.

    Returns
    -------
    tuple
        ``(embeddings, embedder_id)``: the vectors and an identifier recording
        which model produced them.

    Raises
    ------
    ValidationError
        If ``text_columns`` is empty, a column is missing, or the numeric matrix
        has a different row count.
    MissingExtraError
        If sentence-transformers is not installed.

    Notes
    -----
    **The model loads on every call.** Sentence-transformers caches weights on
    disk but reconstructs the model object, so embedding a corpus in one call is
    much faster than many small ones.

    **Numeric features are outnumbered in the hybrid space.** A few numeric
    dimensions beside several hundred embedding dimensions contribute
    proportionally little to Euclidean distance, whatever their importance.

    **Long text is truncated by the model**, typically past a few hundred
    tokens. A case whose distinguishing detail sits at the end of a long field
    may embed as though that detail were absent.

    **The first call may need network access** to download the model.

    See Also
    --------
    embed_text_queries : The query-side counterpart.
    """
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
    """Embed query rows the same way the cases were embedded.

    Delegates to :func:`embed_text_cases` and drops the identifier, since a
    query does not need to record which model encoded it. Sharing the
    implementation is the point: query and case vectors must come from the same
    model and the same column joining, or the distances between them mean
    nothing.

    Parameters
    ----------
    frame:
        The query rows.
    text_columns:
        The same columns, in the same order, used when the cases were embedded.
    model_name:
        The same model. A different one produces an unrelated space.
    numeric_matrix:
        Standardised numeric features to concatenate, matching what the cases
        carry.

    Returns
    -------
    numpy.ndarray
        Query vectors, one row each.

    Raises
    ------
    ValidationError
        If ``text_columns`` is empty, a column is missing, or the numeric matrix
        has a different row count.

    Notes
    -----
    **A mismatched model or column order is not detected.** The dimensions may
    line up perfectly while the spaces are unrelated, producing confidently
    ranked and meaningless neighbours. The plan carries both, so the normal path
    cannot get this wrong.

    See Also
    --------
    embed_text_cases : The case-side function this reuses.
    """
    emb, _ = embed_text_cases(
        frame,
        text_columns,
        model_name=model_name,
        numeric_matrix=numeric_matrix,
    )
    return emb


def _concat_text_rows(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    """Join several text columns into one passage per row.

    A sentence-transformer encodes one passage at a time, so multiple text
    fields have to be combined. They are joined with an explicit ``[SEP]``
    marker rather than a space, which keeps the boundary visible to the model
    instead of running the end of one field into the start of the next.

    Parameters
    ----------
    frame:
        The rows.
    columns:
        Text columns, in the order they should appear.

    Returns
    -------
    list of str
        One joined passage per row.

    Notes
    -----
    **Values are stringified, so nulls become the literal ``"nan"``.** That text
    is embedded like any other and quietly pulls rows with missing fields toward
    each other. Fill text nulls with something meaningful before embedding.

    **Column order affects the embedding.** The same fields joined in a
    different order produce a different vector, which is why queries must use
    the order the cases used.
    """
    if len(columns) == 1:
        return frame[columns[0]].astype(str).tolist()
    parts = [frame[c].astype(str) for c in columns]
    merged = parts[0]
    for part in parts[1:]:
        merged = merged + " [SEP] " + part
    return merged.tolist()
