"""Answer a question using only the passages that were retrieved for it.

The last stage, and the one that determines whether the whole pipeline was worth
building. A language model asked a question from its own memory will produce
something plausible whether or not it knows the answer. Given a set of retrieved
passages and told to answer from those alone, it can be checked: every claim
either traces to a passage the caller can read or it does not.

Three things make that checkable in practice. The prompt labels each passage
``[source:N]`` and instructs the model to cite those markers. The result carries
the full :class:`~buildml.rag.results.Citation` list, so a marker resolves back
to a chunk, a document, and its text. And a cheap faithfulness pass measures
whether the answer actually used them.

Failures here are loud on purpose. Zero retrieved passages, a provider error, or
an empty completion all raise rather than falling back to an ungrounded answer :
a wrong answer that looks grounded is worse than no answer.

No provider is bundled. Core BuildML never imports an LLM SDK; pass any object
with a ``chat`` method, or let Session supply the one it was configured with.

See Also
--------
buildml.rag.retrieve.retrieve : Producing the passages.
buildml.rag.results.GenerateResult : What comes back.
buildml.rag.evaluate.evaluate_generation : Measuring answer quality.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from buildml.core.errors import ValidationError
from buildml.rag.results import (
    Citation,
    FaithfulnessReport,
    GenerateResult,
    Hit,
    RetrieveResult,
)
from buildml.rag.retrieve import retrieve
from buildml.rag.types import GenerateConfig, RetrieveConfig

_SOURCE_MARKER_RE = re.compile(r"\[source:(\d+)\]")

_SYSTEM_TEMPLATE = """\
You are a retrieval-grounded assistant. Answer ONLY using the CONTEXT passages \
below. If the context is insufficient, say you cannot answer from the provided \
sources. Cite sources using [source:N] markers that match the CONTEXT labels.

CONTEXT:
{context}
"""

_USER_TEMPLATE = """\
Question: {query}

Answer with citations where claims come from CONTEXT."""


@runtime_checkable
class ChatProvider(Protocol):
    """The only thing grounded generation needs from a language model.

    Structural, not nominal: anything with a matching ``chat`` method satisfies
    it, with no base class to inherit and no import of ``buildml.ai``. That
    keeps this module free of any LLM SDK dependency and makes it trivial to
    substitute a fake in tests.

    Notes
    -----
    **Deliberately narrower than a full provider interface.** Streaming, tool
    execution, and multi-turn management are out of scope here, so a wider range
    of objects: including a five-line test double: can stand in.

    See Also
    --------
    EchoGroundedProvider : A deterministic implementation for tests.
    buildml.ai.provider : The full provider abstraction.
    """

    def chat(
        self,
        messages: list[Any],
        tools: list[dict[str, Any]] | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Any:
        """Send messages to the model and return its reply.

        One round trip, no streaming and no state. Grounded generation calls
        this exactly once per answer, with the passages already in the system
        message.

        Parameters
        ----------
        messages:
            Ordered conversation turns. Each has ``role`` and ``content``,
            either as attributes or dict keys.
        tools:
            Tool schemas. Always ``None`` from grounded generation, which wants
            prose rather than tool calls.
        max_tokens:
            Cap on reply length, or ``None`` for the provider's default.
        temperature:
            Sampling randomness. Grounded generation passes something near zero.

        Returns
        -------
        object
            Anything exposing ``content``; ``model`` and ``usage`` are read when
            present and omitted quietly when not.
        """
        ...


@dataclass(slots=True)
class _SimpleMessage:
    """A chat turn, in the shape every provider understands.

    The fallback message type, used when ``buildml.ai`` is unavailable so that
    grounded generation works without it.
    """

    role: str
    content: str

    def to_dict(self) -> dict[str, Any]:
        """Return the ``{"role", "content"}`` mapping providers expect.

        The wire format every chat API accepts, so a provider that does not
        understand this dataclass can still be handed the message.

        Returns
        -------
        dict
            The message as a plain dict, ready to serialise.
        """
        return {"role": self.role, "content": self.content}


def hits_to_citations(hits: Sequence[Hit]) -> tuple[Citation, ...]:
    """Number the retrieved passages so the model can refer to them.

    The step that makes citation possible. Each hit gets a small integer
    ``source_id`` matching its rank, which appears in the prompt as
    ``[source:N]`` and in the model's answer as the same marker. Without stable
    numbering there is nothing for the model to cite and nothing for the reader
    to look up.

    Parameters
    ----------
    hits:
        Ranked retrieval hits, best first.

    Returns
    -------
    tuple of Citation
        One per hit, numbered from 1, carrying the text, scores, and metadata.

    Notes
    -----
    **Numbering follows position, not rank.** They coincide for a normal ranked
    list, and passing a filtered subset renumbers from 1 rather than preserving
    gaps.

    **Full chunk text is copied in**, because a citation the reader cannot read
    is not a citation.

    See Also
    --------
    assemble_context : Rendering these into the prompt.
    """
    citations: list[Citation] = []
    for i, hit in enumerate(hits, start=1):
        citations.append(
            Citation(
                source_id=i,
                chunk_id=hit.chunk_id,
                doc_id=hit.doc_id,
                score=float(hit.score),
                text=hit.text,
                rank=int(hit.rank),
                metadata=dict(hit.metadata),
            )
        )
    return tuple(citations)


def assemble_context(
    citations: Sequence[Citation],
    *,
    max_context_chars: int = 8000,
) -> str:
    """Render the citations into the labelled text block the prompt carries.

    Each passage becomes a block headed by its ``[source:N]`` marker and its
    provenance, followed by the text. Blocks are added in rank order until the
    character budget would be exceeded, so the best passages are the ones that
    survive truncation.

    Parameters
    ----------
    citations:
        Numbered passages, best first.
    max_context_chars:
        Budget for the whole context string. A rough stand-in for a token limit;
        four characters per token is the usual approximation.

    Returns
    -------
    str
        Blocks separated by blank lines, or ``""`` when there are no citations.

    Notes
    -----
    **Dropped passages are dropped silently here.** The answer may then omit
    something the retriever found, with no marker in the text to say so. Compare
    the citation count against the hit count if that matters.

    **The first passage is always included, however long it is.** Returning an
    empty context because the top hit alone exceeded the budget would be worse
    than overshooting it.

    **Characters, not tokens.** Code and non-Latin scripts tokenise far less
    efficiently than English prose, so leave headroom.

    See Also
    --------
    hits_to_citations : Producing the input.
    buildml.rag.types.GenerateConfig : Where the budget is configured.
    """
    if not citations:
        return ""
    parts: list[str] = []
    used = 0
    for cite in citations:
        block = (
            f"[source:{cite.source_id}] doc_id={cite.doc_id} "
            f"chunk_id={cite.chunk_id} score={cite.score:.4f}\n{cite.text}"
        )
        if used and used + len(block) + 2 > max_context_chars:
            break
        parts.append(block)
        used += len(block) + 2
    return "\n\n".join(parts)


def assemble_grounded_messages(
    query: str,
    citations: Sequence[Citation],
    *,
    system_template: str | None = None,
    user_template: str | None = None,
    max_context_chars: int = 8000,
) -> tuple[list[_SimpleMessage], str]:
    """Build the two-message prompt that grounds the model in the passages.

    The default system message does the grounding work: it puts the labelled
    passages in front of the model, restricts it to answering from them, tells
    it to say so when they are insufficient, and asks for ``[source:N]``
    markers. The user message carries the question.

    Parameters
    ----------
    query:
        The question, substituted into the user template.
    citations:
        Numbered passages for the context block.
    system_template:
        Overrides the system message. Must contain ``{context}``.
    user_template:
        Overrides the user message. Must contain ``{query}``.
    max_context_chars:
        Budget passed through to :func:`assemble_context`.

    Returns
    -------
    tuple
        ``(messages, context)``: the system and user messages, and the context
        string, returned separately so faithfulness scoring can measure against
        exactly what the model saw.

    Notes
    -----
    **Replacing the templates replaces the grounding instructions.** A custom
    system template that omits the "answer only from context" constraint gets an
    ungrounded model with passages attached, which is a different and much less
    trustworthy thing.

    **An empty context renders as ``(empty)``** rather than a blank, so the
    model is told plainly that it has nothing rather than being left to guess
    from a gap in the prompt.

    See Also
    --------
    assemble_context : The context block.
    score_faithfulness : What consumes the returned context.
    """
    context = assemble_context(citations, max_context_chars=max_context_chars)
    system = (system_template or _SYSTEM_TEMPLATE).format(context=context or "(empty)")
    user = (user_template or _USER_TEMPLATE).format(query=query)
    return (
        [_SimpleMessage(role="system", content=system), _SimpleMessage(role="user", content=user)],
        context,
    )


def _coerce_provider_messages(messages: Sequence[_SimpleMessage]) -> list[Any]:
    """Prefer buildml.ai.types.Message when available; else keep simple messages."""
    try:
        from buildml.ai.types import Message

        return [Message(role=m.role, content=m.content) for m in messages]
    except Exception:
        return list(messages)


def _response_content(response: Any) -> str:
    if response is None:
        return ""
    content = getattr(response, "content", None)
    if content is None and isinstance(response, dict):
        content = response.get("content")
    return str(content or "")


def _response_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if not isinstance(usage, dict):
        return {}
    return {str(k): int(v) for k, v in usage.items() if isinstance(v, (int, float))}


def _response_model(response: Any) -> str | None:
    model = getattr(response, "model", None)
    if model is None and isinstance(response, dict):
        model = response.get("model")
    return None if model is None else str(model)


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[A-Za-z0-9_]+", str(text).lower()) if t}


def score_faithfulness(
    answer: str,
    citations: Sequence[Citation],
    *,
    context: str = "",
    min_overlap: float = 0.05,
) -> FaithfulnessReport:
    """Check cheaply whether the answer actually used the passages it was given.

    Two signals, both lexical and both fast. Citation coverage asks how many of
    the supplied sources the answer cited. Token overlap asks what fraction of
    the answer's words appear in the context, on the reasoning that an answer
    drawn from the passages reuses their vocabulary while an invented one does
    not.

    Neither measures truth, and it is important to be clear about that. This
    catches the common failure: a model ignoring its context and answering from
    memory: not subtle misstatement. A fluent, well-cited, entirely wrong
    answer scores well here.

    Parameters
    ----------
    answer:
        The generated text, scanned for ``[source:N]`` markers.
    citations:
        The passages that were supplied, defining what could be cited.
    context:
        The assembled context string. Falls back to concatenating citation text
        when empty, which is close but not identical if the context was
        truncated.
    min_overlap:
        Overlap fraction required for ``grounded`` to be true. The default of
        0.05 is deliberately permissive: it flags answers with essentially no
        relationship to the context rather than judging quality.

    Returns
    -------
    FaithfulnessReport
        Coverage, cited and uncited source IDs, overlap, a ``grounded`` verdict,
        and its own statement of what it cannot tell you.

    Notes
    -----
    **Overlap is asymmetric**: it divides by the answer's tokens, so a short
    answer quoting the context scores near 1.0 while a long answer that
    paraphrases correctly scores lower. Do not compare across answer lengths.

    **Stopwords count.** Common English words inflate overlap, which is part of
    why the threshold is set so low.

    **Uncited sources are not a failure.** ``missing_source_ids`` usually just
    means the retriever returned more passages than the answer needed.

    See Also
    --------
    buildml.rag.results.FaithfulnessReport : The fields in full.
    buildml.rag.evaluate.evaluate_generation : Scoring against known answers.
    """
    available = {int(c.source_id) for c in citations}
    cited = tuple(sorted({int(m) for m in _SOURCE_MARKER_RE.findall(answer)}))
    missing = tuple(sorted(available - set(cited)))
    coverage = 0.0 if not available else float(len(set(cited) & available) / len(available))
    if context:
        ctx_tokens = _tokenize(context)
    else:
        ctx_tokens: set[str] = set()
        for cite in citations:
            ctx_tokens |= _tokenize(cite.text)
    ans_tokens = _tokenize(answer)
    overlap = (
        0.0
        if not ans_tokens or not ctx_tokens
        else float(len(ans_tokens & ctx_tokens) / len(ans_tokens))
    )
    return FaithfulnessReport(
        citation_marker_coverage=coverage,
        cited_source_ids=cited,
        missing_source_ids=missing,
        answer_context_token_overlap=overlap,
        grounded=bool(cited) and overlap >= float(min_overlap),
        disclosures=(
            "Faithfulness uses citation-marker coverage + lexical token overlap.",
            "Cheap heuristic: not a learned NLI / LLM-as-judge product.",
        ),
        limitations=("High overlap does not prove factual correctness.",),
    )


def generate_from_retrieve(
    retrieve_result: RetrieveResult,
    provider: ChatProvider,
    *,
    config: GenerateConfig | None = None,
    score_grounding: bool = True,
) -> GenerateResult:
    """Answer from passages that have already been retrieved.

    The generation half on its own, for when retrieval already happened: you
    inspected the hits, reused them across several prompts, or filtered them by
    hand. :func:`generate_grounded` is the same thing with retrieval attached.

    Parameters
    ----------
    retrieve_result:
        The passages to answer from. Must contain at least one hit.
    provider:
        Anything with a ``chat`` method.
    config:
        Templates, context budget, temperature, token cap. Defaults are tuned
        for grounded answering.
    score_grounding:
        Run the faithfulness heuristics. Cheap; leave it on.

    Returns
    -------
    GenerateResult
        The answer, its citations, the retrieval it came from, the exact prompt
        context, provider usage, and the faithfulness report.

    Raises
    ------
    ValidationError
        If retrieval returned no hits, the provider raised, or the completion
        was empty.

    Notes
    -----
    **Every failure is a hard failure.** No hits, a provider error, or an empty
    answer all raise rather than degrading to an ungrounded response, because a
    plausible answer with no basis is the failure mode this whole pipeline
    exists to avoid.

    **The provider is called exactly once.** No retries, no fallback model: add
    those in the provider if you want them, where the policy is visible.

    **Citations record what was supplied, not what was used.** Read
    ``result.faithfulness.cited_source_ids`` for what the answer actually
    referenced.

    Examples
    --------
    Retrieve, inspect, then answer::

        hits = retrieve(index, "what is the refund window?", k=5)
        result = generate_from_retrieve(hits, provider)
        print(result.answer, result.faithfulness.grounded)

    See Also
    --------
    generate_grounded : Retrieval and generation in one call.
    """
    cfg = config or GenerateConfig()
    if not retrieve_result.hits:
        raise ValidationError(
            "Cannot generate: retrieval returned zero hits. "
            "Widen the query, rebuild the index, or lower filters."
        )
    citations = hits_to_citations(retrieve_result.hits)
    messages, context = assemble_grounded_messages(
        retrieve_result.query,
        citations,
        system_template=cfg.system_template,
        user_template=cfg.user_template,
        max_context_chars=cfg.max_context_chars,
    )
    try:
        response = provider.chat(
            _coerce_provider_messages(messages),
            tools=None,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
        )
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError(f"RAG generate provider failed: {exc}") from exc

    answer = _response_content(response)
    if not answer.strip():
        raise ValidationError("RAG generate provider returned an empty answer.")

    faithfulness = (
        score_faithfulness(answer, citations, context=context) if score_grounding else None
    )
    disclosures = (
        "Answer is grounded in retrieved CONTEXT chunks; verify citations before trusting claims.",
        "Provider errors and empty retrieval are hard failures (no silent hallucinated fallback).",
        f"n_citations={len(citations)}, max_context_chars={cfg.max_context_chars}.",
    )
    return GenerateResult(
        query=retrieve_result.query,
        answer=answer,
        citations=citations,
        retrieve_result=retrieve_result,
        provider_model=_response_model(response),
        usage=_response_usage(response),
        prompt_context=context,
        disclosures=disclosures,
        config=cfg.to_dict(),
        faithfulness=faithfulness,
    )


def generate_grounded(
    index: Any,
    query: str,
    provider: ChatProvider,
    *,
    k: int = 5,
    retrieve_config: RetrieveConfig | None = None,
    mode: str | None = None,
    filters: dict[str, Any] | None = None,
    rerank: bool | str | None = None,
    fusion: str | None = None,
    config: GenerateConfig | None = None,
    retrieve_result: RetrieveResult | None = None,
) -> GenerateResult:
    """Ask a question of an indexed corpus and get a cited answer.

    The whole RAG pipeline in one call: retrieve the relevant passages, prompt
    the model with them, return the answer with its citations attached. This is
    the entry point most callers want.

    Parameters
    ----------
    index:
        The index to search. May be ``None`` if ``retrieve_result`` is supplied.
    query:
        The question. Must be non-empty.
    provider:
        Anything with a ``chat`` method.
    k:
        How many passages to retrieve. Ignored when ``config`` sets its own.
    retrieve_config:
        Full retrieval settings, if you need more than ``k``.
    mode:
        ``'dense'``, ``'bm25'``, or ``'hybrid'``.
    filters:
        Metadata equality constraints applied before scoring.
    rerank:
        Run a cross-encoder over the candidates.
    fusion:
        ``'rrf'`` or ``'weighted'``, for hybrid mode.
    config:
        Generation settings. Its ``k`` wins over the ``k`` argument.
    retrieve_result:
        Pre-retrieved passages. Supplying this skips retrieval and ignores every
        retrieval argument above.

    Returns
    -------
    GenerateResult
        The answer with citations, the retrieval behind it, and disclosures.

    Raises
    ------
    ValidationError
        If there is neither an index nor a retrieval result, the query is empty,
        no provider was given, retrieval found nothing, or generation failed.

    Notes
    -----
    **More passages is not reliably better.** Beyond roughly five, the extra
    context dilutes rather than helps, and models attend less well to the middle
    of a long prompt. Raise ``k`` with ``rerank=True`` rather than alone.

    **The answer is only as good as the retrieval.** When answers are wrong,
    inspect ``result.retrieve_result.hits`` before changing the prompt: usually
    the passage needed was never retrieved.

    **Determinism depends on the provider.** BuildML's own steps are
    deterministic and the default temperature is low, but the model is not
    guaranteed to repeat itself.

    Examples
    --------
    A grounded question with reranking::

        result = generate_grounded(
            index, "what is the refund window?", provider, k=5, rerank=True,
        )
        print(result.answer)
        for cite in result.citations:
            print(cite.source_id, cite.doc_id)

    See Also
    --------
    generate_from_retrieve : Generation from passages you already have.
    buildml.rag.retrieve.retrieve : Retrieval on its own.
    """
    if index is None and retrieve_result is None:
        raise ValidationError("No RAG index. Call rag_embed_and_index(...) first.")
    if not str(query or "").strip():
        raise ValidationError("rag_generate requires a non-empty query.")
    if provider is None:
        raise ValidationError(
            "rag_generate requires a chat provider. Pass provider=... or configure "
            "Session.ai_configure(...) and omit provider to reuse the Session AI provider."
        )

    cfg = config or GenerateConfig(k=k)
    if retrieve_result is None:
        result = retrieve(
            index,
            query,
            k=cfg.k,
            config=retrieve_config,
            mode=mode,
            filters=filters,
            rerank=rerank,
            fusion=fusion,
        )
    else:
        result = retrieve_result

    return generate_from_retrieve(result, provider, config=cfg)


@dataclass(slots=True)
class EchoGroundedProvider:
    """A fake model that cites its sources without needing a network.

    Reads the source markers out of the prompt and echoes the first few back in
    a sentence. That is enough to exercise the full grounded-generation path :
    prompt assembly, citation numbering, faithfulness scoring, result
    construction: with no API key, no network, and identical output every run.

    Use it in tests and examples. The answers are structurally correct and
    semantically empty, which is exactly what makes them useful for checking
    plumbing and useless for evaluating quality.

    Attributes
    ----------
    prefix:
        Opening words of the generated sentence.

    Examples
    --------
    Exercise the pipeline offline::

        result = generate_grounded(index, "anything", EchoGroundedProvider())
        assert result.citations

    See Also
    --------
    ChatProvider : The protocol this satisfies.
    """

    prefix: str = "Grounded answer"

    def chat(
        self,
        messages: list[Any],
        tools: list[dict[str, Any]] | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Any:
        """Return a fixed sentence citing the first few sources in the prompt.

        Parses ``[source:N]`` markers out of the system message and names them
        in a one-line answer, which is enough to make citation handling and
        faithfulness scoring behave as they would with a real model.

        Parameters
        ----------
        messages:
            Conversation turns. Only the system message is read, for its
            ``[source:N]`` markers.
        tools:
            Ignored.
        max_tokens:
            Ignored; the reply is one short sentence.
        temperature:
            Ignored; output is deterministic.

        Returns
        -------
        object
            A response object with ``content``, ``usage``, ``model``, and the
            other fields a real provider returns.

        Notes
        -----
        **At most three sources are cited**, so faithfulness scoring sees both
        cited and uncited sources when retrieval returned more.

        **Usage figures are fabricated constants.** They exist so code that
        reads token counts has something to read, and mean nothing.
        """
        _ = tools, max_tokens, temperature
        system = ""
        for msg in messages:
            role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else "")
            content = getattr(msg, "content", None) or (
                msg.get("content") if isinstance(msg, dict) else ""
            )
            if role == "system":
                system = str(content)
        source_ids = []
        for line in system.splitlines():
            if line.startswith("[source:"):
                try:
                    sid = int(line.split("]", 1)[0].split(":", 1)[1])
                    source_ids.append(sid)
                except (IndexError, ValueError):
                    continue
        cites = ", ".join(f"[source:{i}]" for i in source_ids[:3]) or "(no sources)"
        content = f"{self.prefix} based on {cites}."

        @dataclass(slots=True)
        class _Resp:
            content: str
            tool_calls: tuple[Any, ...] = ()
            finish_reason: str = "stop"
            usage: dict[str, int] = field(default_factory=dict)
            model: str = "echo-grounded"

        return _Resp(
            content=content,
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        )
