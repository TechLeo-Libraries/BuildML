"""Grounded generation over retrieved RAG context.

Uses a pluggable chat provider (typically :class:`buildml.ai.provider.ProviderProtocol`
or any object with ``chat(messages, ...)``). Core BuildML never imports OpenAI;
pass a provider explicitly or reuse the Session AI provider.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import re

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
    """Minimal chat provider used by grounded generate (avoids importing buildml.ai)."""

    def chat(
        self,
        messages: list[Any],
        tools: list[dict[str, Any]] | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Any: ...


@dataclass(slots=True)
class _SimpleMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}


def hits_to_citations(hits: Sequence[Hit]) -> tuple[Citation, ...]:
    """Build ordered citations from ranked retrieval hits."""
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
    """Format citations as labeled CONTEXT blocks with a character budget."""
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
    """Build system/user messages and the assembled context string."""
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
    """Cheap grounding heuristics: citation markers + answer↔context token overlap."""
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
            "Cheap heuristic — not a learned NLI / LLM-as-judge product.",
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
    """Generate a grounded answer from an existing :class:`RetrieveResult`."""
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
    """Retrieve (unless ``retrieve_result`` is provided) then generate with citations."""
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
    """Deterministic offline provider for CI: echoes top citation ids into an answer."""

    prefix: str = "Grounded answer"

    def chat(
        self,
        messages: list[Any],
        tools: list[dict[str, Any]] | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Any:
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
