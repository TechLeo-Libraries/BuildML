# support-kb-rag

You have a product support knowledge base and query-to-doc judgments. You
want grounded retrieval for agent assist, with offline retrieval metrics,
without indexing the labeled answers.

## Data

In-repo synthetic support articles plus query->doc judgments
(`load_support_kb_corpus`): license-clear.

## Leakage

The corpus indexes articles only: never labeled answer strings. Judgments
are used only in `session.rag.evaluate`. Embed/index is built before
evaluation; there is no test-time index mutation.

## How to run

```bash
python proofs/support-kb-rag/script.py
python proofs/support-kb-rag/baseline_industry.py
```

## What you'll get

`results/results.json` with recall@k, MRR, and nDCG@k on held-out judgments.
`results/comparison.json` is sklearn TF-IDF + cosine retrieval on the same
corpus and judgments. Judgments are never indexed. Embeddings use
sentence-transformers when available, else hashing. Generate uses
EchoGroundedProvider offline.

## Limitations

Tiny corpus; echo generate is not a production LLM.

Related: [RAG quickstart](../../guides/quickstart-rag.md),
[examples/rag_hashing_loop.py](../../examples/rag_hashing_loop.py).
