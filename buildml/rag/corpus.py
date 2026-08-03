"""Get documents in, and keep the answers out of the index.

Three ways in: a directory of text files, a list of in-memory documents, or a
column of a DataFrame: all producing the same
:class:`~buildml.rag.results.CorpusHandle`.

The part worth understanding is the ``role`` on each document. RAG has its own
version of the leakage problem that splitting solves for supervised learning: if
the documents you evaluate against are also in the index, retrieval finds them
trivially and every metric looks excellent. Marking a document ``'eval_only'``
keeps it out of the index while leaving it available for evaluation, and the
guards here refuse rather than warn when that boundary would be crossed.

See Also
--------
buildml.rag.chunk : What happens to documents next.
buildml.rag.evaluate : Where eval-only documents are used.
"""

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
    """Read text files from disk into a corpus.

    The quickest way to get started: point at a folder of documents and get back
    something indexable. Each file becomes one document, named after the file,
    with its path kept as metadata so a retrieved passage can be traced back to
    where it came from.

    Parameters
    ----------
    path:
        A single file, or a directory to scan.
    glob:
        Which files to take from a directory. **Not recursive**: subdirectories
        are skipped.
    encoding:
        Text encoding. A file that does not decode raises rather than being
        skipped, because a silently missing document is a silently missing
        answer.
    role:
        ``'index'`` for documents to search, ``'eval_only'`` for held-out
        documents. Applied to every file loaded.

    Returns
    -------
    CorpusHandle
        The documents, in sorted filename order.

    Raises
    ------
    ValidationError
        If the path does not exist, nothing matches the pattern, or a file
        cannot be decoded.

    Notes
    -----
    **Whole files become whole documents**, however large. A book-length file is
    one document until :mod:`buildml.rag.chunk` divides it.

    **Everything is read into memory.** A large corpus is held in full.

    **Document IDs are filename stems**, so ``a/notes.txt`` and ``b/notes.txt``
    both become ``notes``. Duplicate IDs make citations ambiguous.

    Examples
    --------
    Load a folder of Markdown files::

        corpus = load_text_corpus("docs/", glob="*.md")

    See Also
    --------
    corpus_from_frame : When the text is in a DataFrame.
    corpus_from_documents : When it is already in memory.
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
    """Build a corpus from documents you already have.

    Accepts three shapes and normalises them: a plain string becomes a document
    with a generated ID, a mapping supplies its own ID, metadata, and role, and
    a :class:`~buildml.rag.results.Document` passes through unchanged. Use this
    when documents come from a database, an API, or a script rather than files.

    Parameters
    ----------
    documents:
        Strings, mappings with a ``'text'`` key, or ``Document`` objects, mixed
        freely.
    source:
        Provenance label recorded on the handle.
    default_role:
        Role for items that do not specify one.

    Returns
    -------
    CorpusHandle
        The documents in the order given.

    Raises
    ------
    ValidationError
        If the sequence is empty, an item is an unsupported type, a mapping has
        no ``'text'``, or a role is neither ``'index'`` nor ``'eval_only'``.

    Notes
    -----
    **Generated IDs are positional**, so a document supplied as a bare string
    gets an ID that changes if the list order changes. Supply real IDs when
    citations need to be stable.

    **Mappings can set their own role**, which is how a mixed batch of index and
    eval-only documents is loaded in one call.

    Examples
    --------
    Mixed roles in one corpus::

        corpus = corpus_from_documents([
            {"doc_id": "faq-1", "text": "..."},
            {"doc_id": "gold-1", "text": "...", "role": "eval_only"},
        ])

    See Also
    --------
    load_text_corpus : When the documents are files.
    """
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
    """Turn one column of a DataFrame into a corpus.

    The bridge between tabular work and retrieval: support tickets, product
    descriptions, review text. You name the column explicitly: the function
    will never guess which column holds the text, because indexing the wrong one
    produces a system that returns results and answers nothing.

    Parameters
    ----------
    frame:
        The data.
    text_column:
        Which column holds the document text.
    id_column:
        Which column holds a stable identifier. Without one, IDs are row
        positions, which change when the frame is re-sorted.
    role:
        ``'index'`` or ``'eval_only'``, applied to every row.
    source:
        Provenance label recorded on the handle.

    Returns
    -------
    CorpusHandle
        One document per row with usable text.

    Raises
    ------
    ValidationError
        If a named column is missing, or every row's text is null. Both
        messages list the available columns.

    Notes
    -----
    **Null text rows are skipped silently.** The result can be shorter than the
    frame; compare the counts if that matters.

    **Only the text column is carried over.** Other columns are not attached as
    metadata, so they cannot be used as retrieval filters: build documents
    through :func:`corpus_from_documents` when you need that.

    **Positional IDs are fragile.** Pass ``id_column`` whenever the frame has a
    key, so citations survive a re-sort.

    Examples
    --------
    Index a ticket description column::

        corpus = corpus_from_frame(
            tickets, text_column="description", id_column="ticket_id",
        )

    See Also
    --------
    corpus_from_documents : When metadata should travel too.
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
    """Select the documents that may be indexed, and refuse if none may.

    The gate between a mixed corpus and the index. Documents marked
    ``'eval_only'`` are dropped rather than indexed, so a handle can carry both
    kinds and the indexing path still cannot reach the held-out ones.

    Parameters
    ----------
    corpus:
        The corpus, possibly mixed.

    Returns
    -------
    tuple of Document
        Only the index-role documents, in corpus order.

    Raises
    ------
    LeakageError
        If the corpus is entirely eval-only. Indexing it would mean evaluating
        retrieval against documents retrieval was built from, which measures
        nothing.
    ValidationError
        If there are no index-role documents for another reason.

    Notes
    -----
    **Filtering is silent by design.** Dropping eval-only documents is the
    intended behaviour, so it does not warn: but the returned tuple can be much
    shorter than the corpus, and that is worth checking when an index seems
    small.

    See Also
    --------
    refuse_eval_only_index : The stricter check.
    """
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
    """Refuse outright if the corpus contains any held-out document.

    The strict counterpart to :func:`indexable_documents`. Where that one
    filters, this one raises: for callers that intend to index a whole corpus
    and want to be told, rather than quietly given a subset, if it is not clean.

    Parameters
    ----------
    corpus:
        The corpus to check.

    Returns
    -------
    None
        Returns nothing when the corpus is clean; the value is the absence of
        an exception.

    Raises
    ------
    LeakageError
        If any document is ``'eval_only'``. The message names up to five of
        them.

    Notes
    -----
    **Roles are declared, not detected.** This checks the label. A document
    duplicated across an index corpus and an evaluation set under different IDs
    passes here and still contaminates the measurement.

    See Also
    --------
    indexable_documents : The filtering alternative.
    """
    bad = [d.doc_id for d in corpus.documents if d.role == "eval_only"]
    if bad:
        raise LeakageError(
            "Refusing to index eval_only documents "
            f"(doc_ids={bad[:5]}{'…' if len(bad) > 5 else ''}). "
            "Keep evaluation answers out of the index corpus."
        )
