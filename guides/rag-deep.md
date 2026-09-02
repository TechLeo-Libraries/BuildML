# Retrieval-augmented generation

```bash
pip install buildml
# sentence-transformers + cross-encoder rerank: pip install "buildml[rag]"
# optional LangChain adapter: pip install "buildml[rag-advanced]"
```

You have documents, not a modelling table. You want to chunk them,
index them, retrieve, and optionally generate an answer with
citations. That is `session.rag.*`. It is not a hosted vector
database, not `session.nlp.fit_classifier` on a text column, not
tabular learning-to-rank (`session.ranking.fit`), and not recommender
CF (`session.recommender.fit`). Shared metric names (nDCG, MRR) use
different protocols. Do not mix those numbers.

Without an extra, `embedder="auto"` resolves to hashing (lexical,
deterministic). Retrieve defaults to `dense` because the "dense"
side is itself lexical; fusing it with BM25 would not add a second
signal. With `buildml[rag]` installed, auto picks sentence-transformers
and retrieve defaults to hybrid BM25+dense. Rerank stays off until
you turn it on. Generate needs a chat provider; nothing is bundled.

Documents ingested with `role="eval_only"` raise `LeakageError` at
`embed_and_index`. Labeled answers must not enter the index. You
choose k, mode, and whether to generate. The API refuses eval-only
contamination and generate without a provider.

Short on-ramp: [RAG quickstart](quickstart-rag.md). Proof:
[support-kb-rag](../proofs/support-kb-rag/).

## Ingest, chunk, index, retrieve

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
session.rag.ingest_corpus(docs)
session.rag.chunk(size=160, overlap=32, strategy="recursive")
session.rag.embed_and_index()  # hashing without extra; ST with buildml[rag]

hybrid = session.rag.retrieve("corpus contamination indexed answers", k=3)
dense = session.rag.retrieve(
    "corpus contamination indexed answers", k=3, mode="dense"
)
bm25 = session.rag.retrieve(
    "corpus contamination indexed answers", k=3, mode="bm25"
)
print(hybrid.mode, dense.hits[0].doc_id, bm25.hits[0].doc_id)
```

`ingest_corpus` builds a document store, not a modelling `Dataset`.
You can pass an in-memory sequence, a path, or `text_column` on an
already ingested frame. Default chunking is `size=512`,
`overlap=64`, `strategy="fixed"`. Retrieve `k` defaults to 5.
Fusion for hybrid is RRF (weighted is available). Check
`result.mode` rather than assuming: two machines can retrieve
differently when only one has the extra.

Hashing on purpose (CI / lexical-only):

```python
session.rag.embed_and_index(embedder="hashing")
```

## Generate with citations

Grounded generate without a provider fails clearly. Citations are
first-class. `EchoGroundedProvider` is an offline demo, not factual
QA.

```python
from buildml.rag.generate import EchoGroundedProvider, score_faithfulness

answer = session.rag.generate(
    "What causes evaluation contamination?",
    provider=EchoGroundedProvider(),
    k=3,
)
print(answer.answer)
print([c.doc_id for c in answer.citations])
print(answer.faithfulness)

report = score_faithfulness(answer.answer, answer.citations)
print(report.to_dict())
```

Faithfulness is a cheap heuristic: citation-marker coverage plus
lexical overlap. High overlap does not prove the answer is true. It
is not NLI and not LLM-as-judge. A production chat provider is
`session.ai.configure(...)` then `session.rag.generate` without an
echo provider. Pass `use_last_retrieve=True` to generate from the
last retrieve instead of retrieving again.

## Evaluate with qrels

```python
metrics = session.rag.evaluate(
    {
        "corpus contamination indexed answers": ["leak"],
        "supervised learning hold out test": ["ml"],
    },
    k=3,
)
print(metrics.recall_at_k, metrics.mrr, metrics.ndcg_at_k, metrics.hit_rate_at_k)
```

Default `relevance_mode` is `"document"`. These metrics score
retrieval against gold ids. They are not
`session.ranking.evaluate` on labeled query–item feature rows.

## eval_only is refused at index

```python
from buildml.core.errors import LeakageError

eval_docs = [
    {
        "doc_id": "heldout_answer",
        "text": "SECRET labeled answer that must not be indexed.",
    }
]

dirty = Session()
dirty.rag.ingest_corpus(eval_docs, role="eval_only")
try:
    dirty.rag.embed_and_index()
except LeakageError as exc:
    print(type(exc).__name__, exc)
```

Keep eval texts out of the index corpus. Ingest index docs with
`role="index"` (default).

## Upsert, delete, bundle

```python
session.rag.upsert([{"doc_id": "new", "text": "Chunk update without a full rebuild."}])
session.rag.delete(doc_ids=["new"])

bundle = session.rag.save_bundle("artifacts/rag_bundle")
restored = Session()
restored.rag.load_bundle(bundle)
again = restored.rag.retrieve("corpus contamination indexed answers", k=3)
```

`buildml.rag_bundle.v1` stores embeddings, index, and chunk config.
It does not store a tabular dataset, Torch weights, or API keys.
Session checkpoints do not embed the RAG index.

## Backends and extras

| Extra | What it provides |
| --- | --- |
| none (core) | HashingEmbedder, NumPy cosine store, in-process BM25 |
| `buildml[rag]` | sentence-transformers embeddings, hybrid default, cross-encoder rerank |
| `buildml[rag-advanced]` | optional LangChain retrieve/QA adapters |

| Embedder | When |
| --- | --- |
| `hashing` | always |
| `auto` | hashing without extra; semantic when `buildml[rag]` imports |
| sentence-transformer instance | you pass it |

Rerank needs the rag extra and is off by default (download plus a
forward pass per candidate):

```python
reranked = session.rag.retrieve(
    "corpus contamination indexed answers",
    k=3,
    mode="hybrid",
    rerank=True,
    config={"rerank_candidates": 12},
)
```

Semantic models need download time and compatible wheels. Generate
quality is entirely the provider plus retrieved context. This is
not a managed RAG SaaS.

## Benchmark

```bash
python benchmarks/rag/retrieval_quality.py
```

Compares hashing vs sentence-transformers vs hybrid+rerank on an
in-repo corpus with metric floors for CI.
