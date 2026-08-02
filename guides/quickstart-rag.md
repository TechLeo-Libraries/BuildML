# RAG quickstart

> **Install first (GitHub):** PyPI `buildml` is still legacy 1.x and does **not**
> install Session 2.x. Install 2.x from GitHub (or an editable checkout), then
> the RAG extra. See [installation](../docs/installation.rst).

Optional retrieval path on the same `Session` as classical ML and Torch: history,
explain catalog, and distinct artifact kinds.

**Recommended install (semantic defaults):**

```bash
pip install "buildml[rag]"   # sentence-transformers + transformers
# optional LangChain adapter:
pip install "buildml[rag-advanced]"
```

With `buildml[rag]` installed, defaults are **sentence-transformers embeddings**
and **hybrid BM25+dense retrieval**. Without the extra, BuildML falls back to
lexical hashing + dense-only retrieve (CI-safe, disclosed in results).

**Go deeper:** [RAG deep](rag-deep.md) ·
[Artifacts](artifacts-checkpoints-bundles.md) ·
[AI tools](ai-tools-operator-patterns.md) (RAG on the allowlist).

Classical `Session.fit` and Torch `*_torch` stay unchanged. RAG methods use the
`rag_*` prefix and store results in `session.rag_index_result` /
`session.rag_retrieve_result` / `session.rag_generate_result` /
`session.rag_eval_result`.

Distinct from tabular LTR (`fit_ranker` on labeled query–item feature rows) and
from recommenders (`fit_recommender` user–item CF) — see
[LTR quickstart](quickstart-ranking.md).

The path is **ingest → chunk → embed/index → retrieve → generate → evaluate →
bundle**. Grounded generate needs a chat provider (`ai_configure`, or pass
`provider=` such as `EchoGroundedProvider` for offline demos). The AI operator
(`buildml.ai`) can also call `rag_retrieve` / `rag_generate` as allowlisted tools.

```python
from buildml import Session

docs = [
    {
        "doc_id": "ml",
        "text": (
            "Supervised learning fits a model on labeled examples. "
            "Hold out a test partition for final estimates."
        ),
    },
    {
        "doc_id": "rag",
        "text": (
            "Retrieval indexes a corpus, retrieves relevant chunks, "
            "and optionally generates grounded answers later."
        ),
    },
    {
        "doc_id": "leak",
        "text": (
            "Evaluation contamination happens when labeled answers are "
            "indexed into the retrieval corpus."
        ),
        "metadata": {"topic": "hygiene"},
    },
]

session = Session()
session.rag_ingest_corpus(docs)
session.rag_chunk(size=160, overlap=32, strategy="recursive")
session.rag_embed_and_index()  # auto: ST when buildml[rag] installed

# Default retrieve is hybrid (BM25 + dense RRF) when rag extra is present
hybrid = session.rag_retrieve("corpus contamination indexed answers", k=3)
dense = session.rag_retrieve("corpus contamination indexed answers", k=3, mode="dense")
print(hybrid.mode, dense.hits[0].doc_id)

# Explicit lexical fallback (no model download):
# session.rag_embed_and_index(embedder="hashing")

from buildml.rag.generate import EchoGroundedProvider

answer = session.rag_generate(
    "What causes evaluation contamination?",
    provider=EchoGroundedProvider(),
    k=3,
)
print(answer.answer, [c.doc_id for c in answer.citations])

metrics = session.rag_evaluate(
    {
        "corpus contamination indexed answers": ["leak"],
        "supervised learning hold out test": ["ml"],
    },
    k=3,
)
print(metrics.recall_at_k, metrics.mrr, metrics.ndcg_at_k)

session.save_rag_bundle("artifacts/rag_bundle")
```

## Extras

| Extra | Purpose |
| --- | --- |
| `buildml[rag]` | HF sentence-transformers embeddings + cross-encoder rerank (recommended) |
| `buildml[rag-advanced]` | Optional LangChain QA adapter over BuildML retrieval hits |

## Explicit fallbacks

- `embedder="hashing"` — sklearn HashingVectorizer (lexical, deterministic CI)
- `mode="dense"` — cosine-only retrieve (no BM25 fusion)
- `rerank=False` — skip cross-encoder (default)

## Related

- [RAG deep guide](rag-deep.md)
- [Artifacts](artifacts-checkpoints-bundles.md)
- [AI operator safety](ai-operator-safety.md)
