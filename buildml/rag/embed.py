"""Turn text into vectors, which is where "similar" gets its meaning.

Dense retrieval works by putting text in a vector space and returning whatever
sits nearest the query. Everything about whether that works — whether "cancel my
plan" finds the passage headed "terminating a subscription" — is a property of
the model that built the space.

Three backends, and the difference between them is not a matter of degree.
:class:`HashingEmbedder` hashes tokens into buckets: it is deterministic,
dependency-free, and **has no notion of meaning at all**, so synonyms are as
distant as unrelated words. :class:`SentenceTransformerEmbedder` runs a real
model and does place paraphrases near each other. :class:`CallableEmbedder`
wraps whatever you already have.

The hashing backend exists so the RAG path runs anywhere, including in CI with
no downloads. It is a working default, not a good one, and every result built
with it says so.

Whatever embeds the index must also embed the query. Vectors from two different
models can be compared arithmetically and mean nothing, which produces ranked,
confident, unrelated results — so the embedder identity is recorded and checked
rather than assumed.

See Also
--------
buildml.rag.types.EmbedConfig : The recorded settings.
buildml.rag.index : Where embeddings become an index.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

from buildml.core.errors import ValidationError
from buildml.rag.types import DEFAULT_EMBED_DIM, HASHING_EMBEDDER_ID, EmbedConfig

EmbedFn = Callable[[Sequence[str]], np.ndarray]


class Embedder(Protocol):
    """What the RAG path needs from anything that produces vectors.

    Three things: a stable identity, a fixed width, and a method that turns
    texts into a matrix. Structural conformance is enough — any object with
    these members works, without inheriting anything.

    Attributes
    ----------
    embedder_id:
        Stable identity, recorded with the index and checked at query time. A
        query embedded by a different model than the index produces confident
        nonsense, so this is the field that prevents it.
    dim:
        Vector width. Fixed for the life of an index.

    Notes
    -----
    **The identity must actually change when the model does.** An embedder that
    reports the same ID after being swapped defeats the check that exists to
    catch exactly that.

    See Also
    --------
    HashingEmbedder : The dependency-free default.
    SentenceTransformerEmbedder : A real semantic model.
    """

    embedder_id: str
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Turn texts into a matrix of vectors.

        The one operation the rest of the RAG path needs, used for both indexing
        passages and embedding queries.

        Parameters
        ----------
        texts:
            The passages or queries to embed.

        Returns
        -------
        numpy.ndarray
            Shape ``(len(texts), dim)``, float32, one row per input in order.

        Notes
        -----
        **Rows should be L2-normalised**, so that a dot product is a cosine
        similarity and the stores can skip normalising per query.

        **An empty input must return an empty matrix with the right width**,
        not a zero-dimensional array.
        """
        ...


class HashingEmbedder:
    """A vector per document, built from word hashes rather than meaning.

    **Not a semantic embedder.** Each word is hashed to a bucket and the counts
    are normalised, so two passages are "similar" when they use the same words.
    "Cancel my subscription" and "terminate my plan" share almost nothing and
    land far apart.

    It is the default because it needs no downloads, no GPU, and no optional
    dependencies, so the RAG path runs everywhere including CI. Every index
    built with it carries a disclosure saying what it is.

    Notes
    -----
    **Judge retrieval quality only after switching to a real model.** Poor
    results from this backend say nothing about your corpus or your chunking.

    **Hash collisions merge unrelated words.** At the default width some
    distinct terms share a bucket; a wider vector reduces this at the cost of
    memory.

    **It is genuinely good at exact terms.** Error codes, part numbers, and
    surnames match reliably here, which is the same strength BM25 has.

    See Also
    --------
    SentenceTransformerEmbedder : The semantic alternative.
    buildml.rag.extras : Installing it.
    """

    def __init__(self, *, n_features: int = DEFAULT_EMBED_DIM) -> None:
        """Build the hashing vectoriser.

        No model is loaded and nothing is fitted — hashing is a fixed function,
        which is what makes it deterministic across processes and machines.

        Parameters
        ----------
        n_features:
            Vector width, which is the number of hash buckets. Wider means
            fewer collisions and more memory.

        Raises
        ------
        ValidationError
            If the width is not positive.
        """
        if n_features <= 0:
            raise ValidationError(f"n_features must be positive; got {n_features}")
        self.dim = int(n_features)
        self.embedder_id = HASHING_EMBEDDER_ID
        self._vectorizer = HashingVectorizer(
            n_features=self.dim,
            alternate_sign=False,
            norm=None,
            lowercase=True,
            analyzer="word",
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Hash each text into a normalised bag-of-words vector.

        Lowercases, splits on words, hashes each into a bucket, counts, then L2
        normalises so that dot products are cosine similarities.

        Parameters
        ----------
        texts:
            The passages or queries.

        Returns
        -------
        numpy.ndarray
            Shape ``(len(texts), dim)``, float32, L2-normalised.

        Notes
        -----
        **Dense output.** The sparse counts are expanded to a full matrix, so
        memory is the text count times the width regardless of how few words
        each text has.

        **A text with no recognised words gives a zero row**, which is
        equidistant from everything and will rank arbitrarily.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        matrix = self._vectorizer.transform(list(texts)).astype(np.float32)
        dense = matrix.toarray()
        return normalize(dense, norm="l2", axis=1).astype(np.float32)


class CallableEmbedder:
    """Use your own embedding function, with its output shape checked.

    For an API-backed model, a fine-tuned local one, or anything else that turns
    a list of strings into an array. The wrapper adds validation: a function
    returning the wrong shape fails here with a clear message rather than
    producing an index whose vectors do not line up with its chunks.

    Notes
    -----
    **Normalise your output.** The stores assume unit-length rows so that dot
    products are cosine similarities; unnormalised vectors make magnitude count
    as relevance, and longer passages win.

    **Give it a meaningful ``embedder_id``.** The default of ``'callable'``
    cannot distinguish two different functions, which defeats the check that
    stops a query being embedded differently from the index.

    See Also
    --------
    Embedder : The contract this satisfies.
    """

    def __init__(
        self,
        fn: EmbedFn,
        *,
        dim: int,
        embedder_id: str = "callable",
    ) -> None:
        """Wrap the function and record what its output must look like.

        The declared width becomes the contract that every later call is checked
        against.

        Parameters
        ----------
        fn:
            Takes a list of strings, returns an array of shape
            ``(n_texts, dim)``.
        dim:
            The width to enforce.
        embedder_id:
            Identity recorded with the index. Make it specific.

        Notes
        -----
        **The function is not called here**, so a broken one is discovered on
        first use rather than at construction.
        """
        self._fn = fn
        self.dim = int(dim)
        self.embedder_id = embedder_id

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Call the function and check that its output is usable.

        Both the row count and the width are verified, because a mismatch in
        either produces an index whose vectors do not correspond to its chunks
        — and nothing downstream would notice.

        Parameters
        ----------
        texts:
            The passages or queries.

        Returns
        -------
        numpy.ndarray
            Shape ``(len(texts), dim)``, float32.

        Raises
        ------
        ValidationError
            If the output is not two-dimensional, has the wrong number of rows,
            or the wrong width.

        Notes
        -----
        **Row order is trusted, not verified.** A function that reorders its
        output silently misaligns every vector with its text.

        **Normalisation is not checked.** Unnormalised vectors are accepted and
        will make retrieval prefer longer passages.
        """
        arr = np.asarray(self._fn(list(texts)), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != len(texts):
            raise ValidationError(
                f"Embed callable must return shape (n_texts, dim); got {arr.shape} "
                f"for n_texts={len(texts)}"
            )
        if arr.shape[1] != self.dim:
            raise ValidationError(
                f"Embed callable dim mismatch: expected {self.dim}, got {arr.shape[1]}"
            )
        return arr


class SentenceTransformerEmbedder:
    """A real semantic embedder, run locally.

    **What makes dense retrieval actually work.** Trained so that text with
    similar meaning lands in similar places, so a question phrased nothing like
    the passage that answers it can still find it. This is the difference
    between a RAG system that handles how people really ask questions and one
    that only matches keywords.

    Runs on your machine — no data leaves it, and no per-query cost. Requires
    the ``buildml[rag]`` extra, and downloads model weights on first use.

    Notes
    -----
    **The first run downloads and is slow.** Subsequent runs load from cache.
    Pre-download in environments without network access at run time.

    **Models have an input length limit**, typically a few hundred tokens.
    Longer passages are truncated, and the truncated part contributes nothing —
    keep chunks within the model's window.

    **A GPU is a large speedup for indexing** and rarely necessary for querying
    a single embedding.

    See Also
    --------
    HashingEmbedder : The dependency-free fallback.
    buildml.rag.extras.require_sentence_transformers : The dependency gate.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str | None = None,
    ) -> None:
        """Load the model and determine its output width.

        The width is discovered by embedding a probe string rather than being
        configured, so it is always the model's true dimension.

        Parameters
        ----------
        model_name:
            A Hugging Face model identifier. The default is small, fast, and a
            reasonable general-purpose starting point.
        device:
            Where to run, such as ``'cuda'`` or ``'cpu'``. Defaults to
            automatic selection.

        Raises
        ------
        MissingExtraError
            If sentence-transformers is not installed.

        Notes
        -----
        **Loading is expensive** — weights are read into memory and possibly
        downloaded first. Construct once and reuse.

        **The recorded identity includes the model name**, so an index built
        with one model cannot be queried with another without the mismatch
        being caught.
        """
        from buildml.rag.extras import require_sentence_transformers

        st = require_sentence_transformers(feature="Sentence-transformer embeddings")
        kwargs: dict[str, Any] = {}
        if device is not None:
            kwargs["device"] = device
        self._model = st.SentenceTransformer(model_name, **kwargs)
        self.model_name = model_name
        self.device = device
        self.embedder_id = f"sentence-transformers:{model_name}"
        # Probe dim with an empty-safe encode of a single token.
        probe = np.asarray(self._model.encode(["probe"], convert_to_numpy=True))
        self.dim = int(probe.shape[-1])

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Embed texts with the loaded model.

        Batched internally by sentence-transformers, then L2-normalised so dot
        products are cosine similarities.

        Parameters
        ----------
        texts:
            The passages or queries.

        Returns
        -------
        numpy.ndarray
            Shape ``(len(texts), dim)``, float32, L2-normalised.

        Notes
        -----
        **Text beyond the model's window is truncated silently.** A chunk
        longer than a few hundred tokens is embedded from its beginning only,
        and the rest may as well not exist.

        **The whole batch is held in memory.** Embed a very large corpus in
        batches rather than one call.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        arr = np.asarray(
            self._model.encode(list(texts), convert_to_numpy=True),
            dtype=np.float32,
        )
        return normalize(arr, norm="l2", axis=1).astype(np.float32)


def resolve_embedder(
    embedder: Embedder | EmbedFn | str | None = None,
    *,
    dim: int = DEFAULT_EMBED_DIM,
    device: str | None = None,
) -> tuple[Any, EmbedConfig]:
    """Turn whatever the caller passed for ``embedder`` into a real embedder.

    Accepts a name, a model identifier, a function, an object, or nothing at
    all, and returns the embedder together with the config that records what
    was chosen. That config is what travels with the index.

    Parameters
    ----------
    embedder:
        ``None`` or ``'auto'`` picks the best available. ``'hashing'`` forces
        the lexical fallback. ``'sentence-transformers'`` or ``'minilm'`` loads
        the default model; any other string is treated as a model identifier. A
        callable is wrapped; an object with ``encode`` is used directly.
    dim:
        Width for the hashing and callable backends. Ignored for
        sentence-transformers, which reports its own.
    device:
        Where to run, for backends that can use one.

    Returns
    -------
    tuple
        The embedder, and the :class:`~buildml.rag.types.EmbedConfig`
        describing it.

    Raises
    ------
    ValidationError
        If the argument is not a recognised name, a callable, or an object with
        an ``encode`` method.
    MissingExtraError
        If a sentence-transformers model is requested without the extra.

    Notes
    -----
    **``'auto'`` resolves by what is installed, not by what is best.** With the
    extra present you get a semantic model; without it you get hashing, which is
    not semantic at all. The choice is recorded in the returned config and
    disclosed on the index — check it rather than assuming.

    **``dim`` is ignored for sentence-transformers.** The model's real
    dimension always wins, so a mismatched request is not honoured and not an
    error.

    Examples
    --------
    Force the semantic path::

        embedder, config = resolve_embedder("minilm")

    See Also
    --------
    buildml.rag.types.EmbedConfig : What comes back alongside.
    buildml.rag.extras.rag_available : The check ``'auto'`` performs.
    """
    if embedder is None:
        embedder = "auto"
    if embedder == "auto":
        from buildml.rag.extras import rag_available

        if rag_available():
            embedder = "sentence-transformers"
        else:
            embedder = "hashing"
    if embedder is None or embedder == "hashing":
        if device is not None:
            # Hashing is CPU-only; record the request without claiming GPU use.
            pass
        resolved = HashingEmbedder(n_features=dim)
        cfg = EmbedConfig(
            embedder_id=resolved.embedder_id,
            dim=resolved.dim,
            backend="hashing",
            device=None,
        )
        return resolved, cfg
    if isinstance(embedder, str):
        if embedder in {"sentence-transformers", "minilm"}:
            model = "sentence-transformers/all-MiniLM-L6-v2"
        else:
            model = embedder
        resolved = SentenceTransformerEmbedder(model_name=model, device=device)
        cfg = EmbedConfig(
            embedder_id=resolved.embedder_id,
            dim=resolved.dim,
            backend="sentence-transformers",
            model_name=resolved.model_name,
            device=device,
        )
        return resolved, cfg
    if callable(embedder) and not hasattr(embedder, "encode"):
        resolved = CallableEmbedder(embedder, dim=dim, embedder_id="callable")
        cfg = EmbedConfig(
            embedder_id=resolved.embedder_id,
            dim=resolved.dim,
            backend="callable",
            device=device,
        )
        return resolved, cfg
    # Protocol / object with encode
    if not hasattr(embedder, "encode"):
        raise ValidationError(
            "embedder must be 'hashing', a sentence-transformers model id, "
            "a callable list[str]->ndarray, or an object with .encode(texts)."
        )
    resolved = embedder
    embedder_id = str(getattr(resolved, "embedder_id", type(resolved).__name__))
    resolved_dim = int(getattr(resolved, "dim", dim))
    cfg = EmbedConfig(
        embedder_id=embedder_id,
        dim=resolved_dim,
        backend="callable",
        device=device or getattr(resolved, "device", None),
    )
    return resolved, cfg
