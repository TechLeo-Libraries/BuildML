"""Report what this installation of the RAG stack can actually do.

Because RAG behaviour depends on optional dependencies, the same code retrieves
differently on a laptop with the full stack and a CI container without it. That
is a support burden unless the difference is inspectable, which is what this
module is for: one call that says which backends are present, which retrieval
modes are available, what the defaults resolve to here, and how to install what
is missing.

The matrix is honest in both directions. It reports absence plainly rather than
implying a degraded path is equivalent, and it states non-goals so nobody waits
for a hosted vector database that is not coming.

See Also
--------
buildml.rag.extras : The probes behind the availability flags.
buildml.rag.defaults : The defaults this reports.
"""

from __future__ import annotations

from typing import Any

from buildml.rag.defaults import default_retrieve_mode
from buildml.rag.extras import rag_advanced_available, rag_available


def rag_capability_matrix() -> dict[str, Any]:
    """Describe the RAG capabilities available in this environment.

    Probes the optional dependencies and assembles a plain dictionary covering
    embedding backends, retrieval modes, generation, install hints, and non-
    goals. The first thing to reach for when RAG behaves differently than
    expected, and useful to log alongside results so a run can be explained
    later.

    Returns
    -------
    dict
        Keys:

        ``backends``
            Per backend (``hashing``, ``semantic``, ``langchain``): whether it is
            available, which extra provides it, the implementing class, and a
            note on what it does.
        ``retrieve``
            Supported modes, the default resolved here, fusion methods, and
            whether reranking is possible.
        ``generate``
            Grounded generation facts, including that no LLM is bundled.
        ``default_embedder_when_installed``
            ``'semantic'`` or ``'hashing'`` — what ``"auto"`` picks here.
        ``install_hints``
            Copy-paste pip commands per extra.
        ``non_goals``
            What this stack deliberately does not try to be.
        ``rag_extra_present``, ``rag_advanced_present``
            Flat booleans for quick checks.

    Notes
    -----
    **Availability comes from real imports, not just spec lookups**, so an
    installed-but-broken torch reports as unavailable — which matches what the
    user would experience.

    **The probes make this slow on first call** in installs that have the
    extras, since importing torch is the expensive part.

    Examples
    --------
    Check before relying on reranking::

        matrix = rag_capability_matrix()
        if not matrix["retrieve"]["rerank"]:
            print(matrix["install_hints"]["rag"])
    """
    semantic = rag_available()
    advanced = rag_advanced_available()
    return {
        "backends": {
            "hashing": {
                "available": True,
                "extra": None,
                "embedder": "HashingEmbedder",
                "notes": "Core numpy/sklearn hashing embeddings — always available.",
            },
            "semantic": {
                "available": semantic,
                "extra": "rag",
                "embedder": "SentenceTransformerEmbedder",
                "notes": (
                    "sentence-transformers dense embeddings when buildml[rag] imports "
                    "cleanly (real import probe, not find_spec alone)."
                ),
            },
            "langchain": {
                "available": advanced,
                "extra": "rag-advanced",
                "embedder": "LangChain community retrieve hooks",
                "notes": "Optional LangChain retrieve adapters (buildml[rag-advanced]).",
            },
        },
        "retrieve": {
            "modes": ["dense", "hybrid", "bm25"],
            "default_mode": default_retrieve_mode(),
            "fusion": ["rrf", "weighted"],
            "rerank": semantic,
        },
        "generate": {
            "echo_grounded": True,
            "external_llm": "via buildml[ai] / provider hooks — not bundled here",
        },
        "default_embedder_when_installed": "semantic" if semantic else "hashing",
        "install_hints": {
            "rag": "pip install 'buildml[rag]'  # sentence-transformers + transformers",
            "rag-advanced": "pip install 'buildml[rag-advanced]'  # LangChain community",
        },
        "non_goals": [
            "Hosted vector DB product (Pinecone/Weaviate)",
            "Managed RAG SaaS orchestration",
            "Full LangChain agent frameworks",
        ],
        "rag_extra_present": semantic,
        "rag_advanced_present": advanced,
    }
