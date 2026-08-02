# BuildML RAG alpha quickstart (2.2.0a1)

Optional retrieval path on the same `Session` spine as classical ML and Torch DL:
history, explain catalog, and distinct artifact kinds. The default hashing
embedder uses core numpy/sklearn. Install the RAG extra for optional
sentence-transformer / cross-encoder backends and the declared install contract.

```bash
pip install "buildml[rag]"
```

Classical `Session.fit` and Torch `*_torch` stay unchanged. Retrieval methods use
the `rag_*` prefix and store results in `session.rag_index_result` /
`session.rag_retrieve_result` / `session.rag_eval_result`.

This alpha is **retrieve + evaluate + bundle**. There is no `rag_generate` and no
LLM operator (`buildml.ai`).

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
session.rag_embed_and_index()  # default: buildml.hashing_embed.v1

dense = session.rag_retrieve("corpus contamination indexed answers", k=3)
hybrid = session.rag_retrieve(
    "corpus contamination indexed answers",
    k=3,
    mode="hybrid",  # dense + BM25, RRF fusion by default
)
print(dense.hits[0].doc_id, hybrid.hits[0].doc_id)

metrics = session.rag_evaluate(
    {
        "corpus contamination indexed answers": ["leak"],
        "supervised learning hold out test": ["ml"],
    },
    k=3,
)
print(metrics.recall_at_k, metrics.mrr, metrics.ndcg_at_k, metrics.hit_rate_at_k)

session.rag_upsert([{"doc_id": "new", "text": "Chunk update without full rebuild."}])
session.rag_delete(doc_ids=["new"])

bundle = session.save_rag_bundle("artifacts/rag_bundle")
```

Reload into a fresh Session:

```python
restored = Session().load_rag_bundle(bundle)
again = restored.rag_retrieve("corpus contamination indexed answers", k=3)
assert again.hits[0].doc_id == dense.hits[0].doc_id
```

Optional semantic embedder (requires `buildml[rag]` sentence-transformers pin):

```python
from buildml.rag.embed import SentenceTransformerEmbedder

session.rag_embed_and_index(
    embedder=SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2"),
    device="cpu",
)
```

Explain catalog coverage:

```python
before = session.explain("rag_retrieve", moment="before")
print(before.operation, before.prerequisites)
```

## Artifacts

| Artifact | Schema | Contains | Does not contain |
| --- | --- | --- | --- |
| Session checkpoint | existing checkpoint formats | data, roles, splits, history | vector index, chunk embeddings |
| Torch trainer bundle | `buildml.torch_bundle.v1` | weights, optimizer, TrainConfig | RAG index |
| RAG bundle | `buildml.rag_bundle.v1` | chunk config, embedder id/dim, embeddings, chunk metadata | Session dataset rows, Torch weights |

Layout: `<path>/meta.json` + `<path>/chunks.jsonl` + `<path>/embeddings.npy`.

## Known limits (honest)

- **Hashing default is lexical, not semantic.** `buildml.hashing_embed.v1` is
  deterministic and CPU-only (no model download). It is not a substitute for
  sentence-transformer quality claims.
- **Local-first.** Default store is in-process NumPy cosine top-k. No hosted
  vector-DB product path in this alpha.
- **No generate / LLM operator.** Retrieve, evaluate, upsert/delete, and bundle
  save/load are in scope. Grounded generation and `buildml.ai` are later.
- **No Teaching Studio RAG cockpit.** Catalog, structured results, and
  walkthrough `rag_status` are the teaching surfaces.
- **Eval hygiene is caller-owned.** Index corpus and gold query/qrel sets must
  stay separate; documents marked `eval_only` refuse indexing (`LeakageError`).
- **CI merge gate.** RAG job runs on Python 3.11–3.12 CPU. GPU embed/rerank is
  optional when hardware and pins allow; not a PR blocker.

See [glossary.md](./glossary.md).
