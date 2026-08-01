"""Corpus ingest for the RAG path."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from buildml.core.errors import LeakageError, ValidationError
from buildml.rag.results import CorpusHandle, Document


def _as_documents(
    documents: Sequence[Document | Mapping[str, Any] | str],
    *,
    default_role: str = "index",
) -> list[Document]:
    out: list[Document] = []
    for i, item in enumerate(documents):
        if isinstance(item, Document):
            out.append(item)
            continue
        if isinstance(item, str):
            out.append(Document(doc_id=f"doc-{i}", text=item, role=default_role))
            continue
        if not isinstance(item, Mapping):
            raise ValidationError(
                f"Unsupported document type at index {i}: {type(item).__name__}"
            )
        doc_id = str(item.get("doc_id") or item.get("id") or f"doc-{i}")
        text = item.get("text")
        if text is None:
            raise ValidationError(f"Document {doc_id!r} is missing a 'text' field.")
        role = str(item.get("role") or default_role)
        if role not in {"index", "eval_only"}:
            raise ValidationError(
                f"Document {doc_id!r} has invalid role {role!r}; "
                "expected 'index' or 'eval_only'."
            )
        meta = dict(item.get("metadata") or {})
        out.append(Document(doc_id=doc_id, text=str(text), metadata=meta, role=role))
    if not out:
        raise ValidationError("Corpus is empty; provide at least one document.")
    return out


def load_text_corpus(
    path: str | Path,
    *,
    glob: str = "*.txt",
    encoding: str = "utf-8",
    role: str = "index",
) -> CorpusHandle:
    """Load UTF-8 text files from a path or directory into a :class:`CorpusHandle`.

    Parameters
    ----------
    path:
        File or directory. Directories are scanned with ``glob`` (non-recursive).
    glob:
        Filename pattern when ``path`` is a directory.
    encoding:
        Text encoding (default UTF-8). Decode errors raise :class:`ValidationError`.
    role:
        Corpus role for all loaded files (``index`` or ``eval_only``).
    """
    root = Path(path)
    if root.is_file():
        files = [root]
        source = str(root)
    elif root.is_dir():
        files = sorted(root.glob(glob))
        source = f"{root}/{glob}"
    else:
        raise ValidationError(f"Corpus path does not exist: {root}")
    if not files:
        raise ValidationError(f"No files matched {glob!r} under {root}")
    docs: list[Document] = []
    for file_path in files:
        try:
            text = file_path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            raise ValidationError(
                f"Failed to decode {file_path} as {encoding}: {exc}"
            ) from exc
        docs.append(
            Document(
                doc_id=file_path.stem,
                text=text,
                metadata={"path": str(file_path)},
                role=role,
            )
        )
    return CorpusHandle(documents=tuple(docs), source=source)


def corpus_from_documents(
    documents: Sequence[Document | Mapping[str, Any] | str],
    *,
    source: str = "memory",
    default_role: str = "index",
) -> CorpusHandle:
    """Build a :class:`CorpusHandle` from in-memory documents."""
    return CorpusHandle(
        documents=tuple(_as_documents(documents, default_role=default_role)),
        source=source,
    )


def corpus_from_frame(
    frame: pd.DataFrame,
    *,
    text_column: str,
    id_column: str | None = None,
    role: str = "index",
    source: str = "dataframe",
) -> CorpusHandle:
    """Bridge a tabular text column into a :class:`CorpusHandle`.

    Explicit column selection only — never silently indexes every column.
    """
    if text_column not in frame.columns:
        raise ValidationError(
            f"text_column {text_column!r} not found. Available: {list(frame.columns)}"
        )
    if id_column is not None and id_column not in frame.columns:
        raise ValidationError(
            f"id_column {id_column!r} not found. Available: {list(frame.columns)}"
        )
    docs: list[Document] = []
    for i, row in enumerate(frame.itertuples(index=False)):
        mapping = (
            row._asdict()
            if hasattr(row, "_asdict")
            else dict(zip(frame.columns, row, strict=True))
        )
        text = mapping[text_column]
        if text is None or (isinstance(text, float) and pd.isna(text)):
            continue
        doc_id = str(mapping[id_column]) if id_column is not None else f"row-{i}"
        docs.append(Document(doc_id=doc_id, text=str(text), role=role))
    if not docs:
        raise ValidationError(
            f"No non-null text found in column {text_column!r}."
        )
    return CorpusHandle(documents=tuple(docs), source=source)


def indexable_documents(corpus: CorpusHandle) -> tuple[Document, ...]:
    """Return documents allowed in the index; refuse silent eval contamination."""
    index_docs = tuple(d for d in corpus.documents if d.role == "index")
    eval_docs = [d for d in corpus.documents if d.role == "eval_only"]
    if not index_docs and eval_docs:
        raise LeakageError(
            "Corpus contains only eval_only documents. "
            "Refuse to build an index from the evaluation set; "
            "ingest index documents separately."
        )
    if eval_docs:
        # Presence of eval_only docs in the same handle is allowed for storage,
        # but callers must use indexable_documents before indexing.
        pass
    if not index_docs:
        raise ValidationError("No index-role documents available for indexing.")
    return index_docs


def refuse_eval_only_index(corpus: CorpusHandle) -> None:
    """Raise :class:`LeakageError` when any eval_only document would be indexed."""
    bad = [d.doc_id for d in corpus.documents if d.role == "eval_only"]
    if bad:
        raise LeakageError(
            "Refusing to index eval_only documents "
            f"(doc_ids={bad[:5]}{'…' if len(bad) > 5 else ''}). "
            "Keep evaluation answers out of the index corpus."
        )
