"""Optional LangChain retrieval/generation adapter behind ``buildml[rag-advanced]``."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from buildml.core.errors import ValidationError
from buildml.rag.extras import require_langchain_community
from buildml.rag.generate import ChatProvider, hits_to_citations
from buildml.rag.results import GenerateResult, Hit, RetrieveResult
from buildml.rag.types import GenerateConfig


class LangChainGroundedAdapter:
    """Bridge BuildML retrieve hits to LangChain ``RetrievalQA``-style generation.

    Requires ``buildml[rag-advanced]`` (``langchain-community``). Does not replace
    the core :func:`buildml.rag.generate.generate_grounded` path — use when you
    already run LangChain LLMs and want BuildML retrieval + citations.
    """

    def __init__(
        self,
        llm: Any,
        *,
        chain_type: str = "stuff",
    ) -> None:
        require_langchain_community(feature="LangChain grounded RAG adapter")
        self._llm = llm
        self.chain_type = chain_type

    def generate_from_hits(
        self,
        query: str,
        hits: Sequence[Hit],
        *,
        config: GenerateConfig | None = None,
    ) -> GenerateResult:
        if not hits:
            raise ValidationError("LangChain adapter requires at least one retrieval hit.")
        cfg = config or GenerateConfig()
        citations = hits_to_citations(hits)
        try:
            from langchain.chains.question_answering import load_qa_chain
            from langchain_core.documents import Document
        except ImportError as exc:
            raise ValidationError(
                "LangChain QA chain imports failed. Install buildml[rag-advanced]."
            ) from exc

        docs = [
            Document(
                page_content=h.text,
                metadata={"chunk_id": h.chunk_id, "doc_id": h.doc_id, "rank": h.rank},
            )
            for h in hits
        ]
        chain = load_qa_chain(self._llm, chain_type=self.chain_type)
        raw = chain.run(input_documents=docs, question=query)
        answer = str(raw or "").strip()
        if not answer:
            raise ValidationError("LangChain QA chain returned an empty answer.")
        retrieve_result = RetrieveResult(
            query=query,
            k=len(hits),
            hits=tuple(hits),
            embedder_id="langchain-adapter",
            mode="adapter",
            disclosures=("generation_backend=langchain", f"chain_type={self.chain_type}"),
        )
        return GenerateResult(
            query=query,
            answer=answer,
            citations=citations,
            retrieve_result=retrieve_result,
            provider_model=getattr(self._llm, "model_name", type(self._llm).__name__),
            usage={},
            prompt_context="\n\n".join(h.text for h in hits[: cfg.k]),
            disclosures=(
                "Answer produced via LangChain load_qa_chain over BuildML retrieval hits.",
                "Verify citations against source text; LangChain does not enforce [source:N] markers.",
            ),
            config=cfg.to_dict(),
            faithfulness=None,
        )

    def as_chat_provider(self) -> ChatProvider:
        """Minimal provider shim — expects pre-retrieved context in system message."""

        class _Provider:
            def __init__(self, outer: LangChainGroundedAdapter) -> None:
                self._outer = outer

            def chat(
                self,
                messages: list[Any],
                tools: list[dict[str, Any]] | None = None,
                *,
                max_tokens: int | None = None,
                temperature: float | None = None,
            ) -> Any:
                _ = tools, max_tokens, temperature
                user = next(
                    (
                        getattr(m, "content", "")
                        for m in messages
                        if getattr(m, "role", "") == "user"
                    ),
                    "",
                )
                system = next(
                    (
                        getattr(m, "content", "")
                        for m in messages
                        if getattr(m, "role", "") == "system"
                    ),
                    "",
                )
                from langchain.chains.question_answering import load_qa_chain
                from langchain_core.documents import Document

                docs = [Document(page_content=system)]
                chain = load_qa_chain(self._outer._llm, chain_type=self._outer.chain_type)
                content = str(chain.run(input_documents=docs, question=str(user)))

                @dataclass(slots=True)
                class _Resp:
                    content: str
                    model: str = "langchain-adapter"

                return _Resp(content=content)

        return _Provider(self)
