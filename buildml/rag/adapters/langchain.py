"""Use BuildML retrieval with a LangChain LLM you have already set up.

For teams with an existing LangChain investment: configured models, callbacks,
tracing: who want BuildML's retrieval and citation handling without rewriting
that half. The adapter takes BuildML hits, hands them to a LangChain QA chain,
and wraps the reply in a
:class:`~buildml.rag.results.GenerateResult` so the rest of BuildML sees a
familiar shape.

The seam is real and worth knowing about. LangChain's chains build their own
prompts, so BuildML's grounding instructions and ``[source:N]`` convention do not
apply. Citations are still attached, because they come from the retrieval side,
but nothing enforces that the answer references them: which is why faithfulness
scoring is left unset rather than reported as a passing score it did not earn.

Prefer :func:`~buildml.rag.generate.generate_grounded` for new work. Its prompt
is built for grounding and its results are verifiable end to end.

See Also
--------
buildml.rag.generate.generate_grounded : The native path.
buildml.rag.extras.require_langchain_community : The dependency gate.
"""

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
    """A LangChain model, wrapped so BuildML retrieval can drive it.

    Holds one configured LangChain LLM and exposes two ways to use it: pass hits
    directly via :meth:`generate_from_hits`, or obtain a
    :class:`~buildml.rag.generate.ChatProvider` shim via
    :meth:`as_chat_provider` and use it wherever BuildML expects a provider.

    Attributes
    ----------
    chain_type:
        The LangChain combine strategy, recorded on results for disclosure.

    Notes
    -----
    **Faithfulness is not scored on results from this adapter.** LangChain
    prompts do not request ``[source:N]`` markers, so the citation half of the
    heuristic would report zero coverage for an answer that may be perfectly
    grounded. Reporting nothing is more honest than reporting a misleading zero.

    **Requires ``buildml[rag-advanced]``**, checked at construction rather than
    at first use.

    Examples
    --------
    Retrieve with BuildML, answer with LangChain::

        adapter = LangChainGroundedAdapter(my_llm)
        hits = retrieve(index, "how do I cancel?", k=5).hits
        result = adapter.generate_from_hits("how do I cancel?", hits)

    See Also
    --------
    buildml.rag.generate.generate_grounded : The native, fully-verifiable path.
    """

    def __init__(
        self,
        llm: Any,
        *,
        chain_type: str = "stuff",
    ) -> None:
        """Wrap a LangChain LLM, checking the dependency up front.

        The check happens here rather than at first generation so a missing
        extra surfaces while you are wiring things up, not mid-request.

        Parameters
        ----------
        llm:
            Any configured LangChain LLM or chat model.
        chain_type:
            LangChain's combine strategy. ``'stuff'`` puts every passage in one
            prompt, which is what RAG normally wants; ``'map_reduce'`` and
            ``'refine'`` make multiple calls and suit contexts too large to fit.

        Raises
        ------
        MissingExtraError
            If ``buildml[rag-advanced]`` is not installed.
        """
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
        """Answer a question from BuildML hits, using the LangChain chain.

        Converts each hit into a LangChain ``Document`` carrying its chunk and
        document IDs, runs the QA chain, and wraps the reply in a BuildML result
        with citations attached.

        Parameters
        ----------
        query:
            The question.
        hits:
            Retrieved passages. Must be non-empty.
        config:
            Generation settings. Only ``k`` is honoured, bounding how many
            passages go into the recorded prompt context; templates,
            temperature, and token caps belong to the LangChain model.

        Returns
        -------
        GenerateResult
            The answer with citations and disclosures. ``faithfulness`` is
            ``None`` and ``usage`` is empty.

        Raises
        ------
        ValidationError
            If ``hits`` is empty, the LangChain QA imports fail, or the chain
            returns nothing.

        Notes
        -----
        **Most generation settings do not apply.** Temperature and token limits
        are configured on the LangChain model itself, not here.

        **Token usage is not reported**, because LangChain surfaces it through
        callbacks rather than the return value.
        """
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
        """Expose the LangChain model as something BuildML can call directly.

        Returns an object satisfying
        :class:`~buildml.rag.generate.ChatProvider`, so the LangChain model can
        be passed to :func:`~buildml.rag.generate.generate_grounded` and used
        anywhere a provider is expected.

        Returns
        -------
        ChatProvider
            A shim that forwards to the wrapped model.

        Notes
        -----
        **The shim passes BuildML's assembled context through as a single
        document.** The grounding prompt is built by BuildML and handed to
        LangChain whole, rather than LangChain building its own: so the
        ``[source:N]`` instructions do survive this path, unlike
        :meth:`generate_from_hits`.

        **Tools, token caps, and temperature are ignored.** Configure them on
        the LangChain model.

        See Also
        --------
        generate_from_hits : The more direct route when you have hits in hand.
        """

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
