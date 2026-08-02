# RAG deep guide

> **Install (GitHub 2.x + RAG):**
> ```bash
> pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"
> pip install "buildml[rag]"   # sentence-transformers for semantic backends
> ```
> Default hashing embedder works on core numpy/sklearn after the 2.x install.
> See [installation](../docs/installation.rst).

Retrieval-augmented generation on the same Session spine: history, explain,
and distinct artifact kinds. Short on-ramp: [quickstart-rag](quickstart-rag.md).

---

## Why RAG is a separate prefix

Tabular `ingest` builds a modeling `Dataset`. RAG corpus ingest builds an
**indexable document store** with optional `eval_only` roles so labeled answers
cannot contaminate retrieval. Methods use the `rag_*` prefix and results live in
`rag_*_result` properties.

Path: **ingest → chunk → embed/index → retrieve → (generate) → evaluate →
bundle**, with upsert/delete for incremental ops.

---

## Use case A — Dense, BM25, and hybrid retrieve

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
session.rag_chunk(size=160, overlap=32)
session.rag_embed_and_index()  # buildml.hashing_embed.v1 by default

dense = session.rag_retrieve("corpus contamination indexed answers", k=3, mode="dense")
bm25 = session.rag_retrieve("corpus contamination indexed answers", k=3, mode="bm25")
hybrid = session.rag_retrieve(
    "corpus contamination indexed answers",
    k=3,
    mode="hybrid",  # dense + BM25, RRF fusion by default
)
print(dense.hits[0].doc_id, bm25.hits[0].doc_id, hybrid.hits[0].doc_id)
```

---

## Use case B — Grounded generate with citations + faithfulness

```python
from buildml.rag.generate import EchoGroundedProvider, score_faithfulness

answer = session.rag_generate(
    "What causes evaluation contamination?",
    provider=EchoGroundedProvider(),  # offline demo
    k=3,
)
print(answer.answer)
print([c.doc_id for c in answer.citations])

# Cheap faithfulness hooks (Pass V): citation-marker coverage + lexical overlap.
# Attached automatically on GenerateResult when score_grounding is enabled (default).
print(answer.faithfulness)
if answer.faithfulness is not None:
    print(
        answer.faithfulness.citation_marker_coverage,
        answer.faithfulness.answer_context_token_overlap,
        answer.faithfulness.grounded,
    )

# Standalone helper over an answer + citations:
report = score_faithfulness(answer.answer, answer.citations)
print(report.to_dict())

# Production: configure a real chat provider (buildml[ai]) then:
# session.ai_configure(provider="openai")
# answer = session.rag_generate("...", k=3)  # uses configured provider
```

Grounded generate without a provider fails clearly. Citations are first-class;
do not treat echo providers as factual QA. Faithfulness is a **cheap heuristic**
(not NLI / LLM-as-judge) — high overlap does not prove factual correctness.

---

## Use case C — Evaluate with qrels

```python
metrics = session.rag_evaluate(
    {
        "corpus contamination indexed answers": ["leak"],
        "supervised learning hold out test": ["ml"],
    },
    k=3,
)
print(metrics.recall_at_k, metrics.mrr, metrics.ndcg_at_k, metrics.hit_rate_at_k)
```

---

## Use case D — eval_only hygiene (hard refuse)

```python
# Documents reserved for evaluation must not enter the index corpus.
eval_docs = [
    {
        "doc_id": "heldout_answer",
        "text": "SECRET labeled answer that must not be indexed.",
        # role handled via rag_ingest_corpus(..., role="eval_only") when supported
    }
]

# Pattern: ingest index docs with role="index" (default) and keep eval texts
# out of rag_embed_and_index. Indexing eval_only content raises LeakageError.
try:
    dirty = Session()
    dirty.rag_ingest_corpus(eval_docs, role="eval_only")
    dirty.rag_embed_and_index()
except Exception as exc:  # LeakageError when eval_only would contaminate
    print(type(exc).__name__, exc)
```

---

## Use case E — Semantic embedder, upsert, delete, bundle

```python
from buildml.rag.embed import SentenceTransformerEmbedder

# Requires buildml[rag] sentence-transformers pin
# session.rag_embed_and_index(
#     embedder=SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2"),
#     device="cpu",
# )

session.rag_upsert([{"doc_id": "new", "text": "Chunk update without full rebuild."}])
session.rag_delete(doc_ids=["new"])

bundle = session.save_rag_bundle("artifacts/rag_bundle")
restored = Session().load_rag_bundle(bundle)
again = restored.rag_retrieve("corpus contamination indexed answers", k=3)
assert again.hits[0].doc_id == dense.hits[0].doc_id
```

---

## Teaching surface

```python
before = session.explain("rag_retrieve", moment="before")
print(before.operation, before.prerequisites, before.risks)
```

AI operator tools can call `rag_retrieve` / `rag_generate` under confirmation
gates ([ai-tools](ai-tools-operator-patterns.md)).

---

## Artifacts

| Artifact | Contains | Does not |
| --- | --- | --- |
| `buildml.rag_bundle.v1` | embeddings, index, chunk config | Tabular dataset, Torch weights, API keys |
| Session checkpoint | tabular workflow | RAG index |

---

## Failure modes / limits

- Default embedder is **lexical hashing** — strong for CI, weaker for semantics.
- Semantic models need `buildml[ai]`/`[rag]` deps and download time.
- `eval_only` contamination → `LeakageError`.
- Generate quality depends entirely on the chat provider + retrieved context.
- Faithfulness hooks are lexical / citation-marker heuristics, not a judge model.
- Not a managed vector-DB cloud product.
- Not tabular learning-to-rank (`fit_ranker` on labeled query–item feature
  rows) and not recommender CF (`fit_recommender`). Shared metric names
  (nDCG/MRR) use different protocols — see [ranking-deep.md](ranking-deep.md).

---

## Related

- [RAG quickstart](quickstart-rag.md)
- [LTR / search ranking](ranking-deep.md) (tabular judgments; distinct path)
- [Artifacts](artifacts-checkpoints-bundles.md)
- [AI safety](ai-operator-safety.md)
- [Leakage](leakage-cv-recipes.md) (eval hygiene parallels)
