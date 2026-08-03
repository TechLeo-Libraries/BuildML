"""Re-order retrieved passages with a model that reads query and passage together.

Retrieval is fast because the query and the passages never meet: each was
embedded independently, and search is arithmetic on those vectors. That
independence is the whole reason a corpus of a million chunks can be searched in
milliseconds, and it is also the reason retrieval misses things. A vector
committed to before the question was known cannot emphasise the part of the
passage the question is about.

A cross-encoder gives up the speed to recover the accuracy. It takes the query
and one passage as a single input and produces a relevance score, which means it
can attend to the query while reading: but it must run once per candidate, so
it cannot search a corpus. It can only re-order a shortlist.

The practical shape is: retrieve fifty cheaply, rerank them, keep five. Almost
all of the quality gain and a bounded, predictable cost.

See Also
--------
buildml.rag.retrieve.retrieve : Where reranking is switched on.
buildml.rag.embed : The bi-encoder side that produces the shortlist.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from buildml.core.errors import ValidationError
from buildml.rag.results import Hit

DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def require_cross_encoder(
    *,
    feature: str = "Cross-encoder rerank",
) -> Any:
    """Load the ``CrossEncoder`` class, failing with install guidance if absent.

    Deferred so that importing BuildML never pulls in torch, and so the error
    when the extra is missing names the feature the caller wanted rather than
    surfacing a bare import error.

    Parameters
    ----------
    feature:
        What the caller was trying to do, quoted back in the error message.

    Returns
    -------
    type
        The ``sentence_transformers.CrossEncoder`` class, uninstantiated.

    Raises
    ------
    MissingExtraError
        If ``buildml[rag]`` is not installed, or the installed
        sentence-transformers is too old to expose ``CrossEncoder``.
    """
    from buildml.rag.extras import require_sentence_transformers

    st = require_sentence_transformers(feature=feature)
    if not hasattr(st, "CrossEncoder"):
        from buildml.core.errors import MissingExtraError

        raise MissingExtraError("rag", feature)
    return st.CrossEncoder


class CrossEncoderReranker:
    """A loaded cross-encoder, ready to re-score shortlists.

    Wraps one sentence-transformers model. Loading is the expensive part :
    hundreds of megabytes on first use, downloaded and cached: so build this
    once and reuse it rather than constructing one per query.

    Attributes
    ----------
    model_name:
        The model identifier, recorded so results can say what scored them.
    device:
        Where the model runs, or ``None`` for the library's own choice.

    Notes
    -----
    **Everything runs locally.** No network calls after the initial model
    download, so this works offline and sends no passages anywhere.

    **Not thread-safe for concurrent scoring**, since the underlying model is
    shared. Use one instance per worker.

    Examples
    --------
    Re-order a shortlist::

        reranker = CrossEncoderReranker()
        top = reranker.rerank("how do I cancel?", candidates, k=5)

    See Also
    --------
    resolve_reranker : Building one from a user-supplied flag.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER,
        *,
        device: str | None = None,
    ) -> None:
        """Load the model, downloading it on first use.

        Construction is where the cost lives, so hold onto the instance: the
        weights stay resident and every later ``rerank`` call is just inference.

        Parameters
        ----------
        model_name:
            A sentence-transformers cross-encoder identifier. The default is a
            small MS MARCO model chosen to be fast enough to use by default.
        device:
            ``'cpu'``, ``'cuda'``, or ``None`` to let the library decide.

        Raises
        ------
        MissingExtraError
            If ``buildml[rag]`` is not installed.

        Notes
        -----
        **The first construction may take a while and needs network access**,
        because the weights are fetched and cached. Later ones read the cache.
        """
        CrossEncoder = require_cross_encoder(feature="Cross-encoder rerank")
        kwargs: dict[str, Any] = {}
        if device is not None:
            kwargs["device"] = device
        self._model = CrossEncoder(model_name, **kwargs)
        self.model_name = model_name
        self.device = device

    def rerank(self, query: str, hits: Sequence[Hit], *, k: int) -> list[Hit]:
        """Re-score a shortlist against the query and keep the best ``k``.

        Every hit is scored afresh; the incoming ranks and scores are discarded
        entirely, which is the point: the retriever's opinion is what we are
        trying to improve on.

        Parameters
        ----------
        query:
            The question, paired with each passage.
        hits:
            The shortlist. Cost is linear in its length.
        k:
            How many to keep.

        Returns
        -------
        list of Hit
            The best ``k``, renumbered from 1, carrying cross-encoder scores.
            Empty when ``hits`` is empty.

        Raises
        ------
        ValidationError
            If ``k`` is not positive.

        Notes
        -----
        **The returned scores are on a different scale from what came in.**
        Cross-encoder outputs are model-specific logits, often unbounded and
        sometimes negative. Do not compare them against cosine similarities or
        thresholds tuned on the retriever.

        **Ties break by chunk ID**, so repeated calls give identical output.

        **Cost is one forward pass per hit**, batched internally. A hundred
        candidates on CPU is noticeably slower than ten.
        """
        if k <= 0:
            raise ValidationError(f"k must be positive; got {k}")
        if not hits:
            return []
        pairs = [[query, h.text] for h in hits]
        scores = self._model.predict(pairs)
        ranked = sorted(
            zip(hits, scores, strict=True),
            key=lambda item: (-float(item[1]), item[0].chunk_id),
        )[:k]
        out: list[Hit] = []
        for rank, (hit, score) in enumerate(ranked, start=1):
            out.append(
                Hit(
                    chunk_id=hit.chunk_id,
                    doc_id=hit.doc_id,
                    score=float(score),
                    text=hit.text,
                    rank=rank,
                    metadata=dict(hit.metadata),
                )
            )
        return out


def resolve_reranker(
    rerank: bool | str,
    *,
    model_name: str | None = None,
    device: str | None = None,
) -> CrossEncoderReranker | None:
    """Turn a user-facing rerank setting into a reranker, or nothing.

    Lets callers write ``rerank=True`` for the default model or
    ``rerank="some/model"`` to name one, without knowing which class implements
    it. The falsy path is deliberately taken before any import, so a project
    that never reranks never pays for the optional dependency.

    Parameters
    ----------
    rerank:
        ``False``, ``None``, or ``""`` to disable. ``True`` or
        ``"cross-encoder"`` for the default model. Any other string is treated
        as a model identifier.
    model_name:
        Overrides the model chosen by ``rerank``.
    device:
        ``'cpu'``, ``'cuda'``, or ``None``.

    Returns
    -------
    CrossEncoderReranker or None
        ``None`` when reranking is off; otherwise a loaded reranker.

    Raises
    ------
    ValidationError
        If ``rerank`` is neither a bool nor a string.
    MissingExtraError
        If reranking was requested but ``buildml[rag]`` is not installed.

    Notes
    -----
    **The model loads here, not on first query.** Enabling rerank makes this
    call slow the first time and can raise if the extra is missing: better to
    fail at setup than mid-retrieval.
    """
    if rerank is False or rerank is None or rerank == "":
        return None
    if rerank is True or rerank == "cross-encoder":
        name = model_name or DEFAULT_CROSS_ENCODER
    elif isinstance(rerank, str):
        name = model_name or rerank
    else:
        raise ValidationError(
            "rerank must be False, True, 'cross-encoder', or a model id string."
        )
    return CrossEncoderReranker(model_name=name, device=device)
