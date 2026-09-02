# policy-handbook-rag

You have a small employee policy handbook (leave, expenses, remote work,
security, conduct, travel) and relevance judgments. You want retrieval plus
grounded generation scaffolding, without indexing the answers.

## Data

Inline synthetic policy articles plus relevance judgments (same pattern as
`support-kb-rag`). Judgments are never indexed as answers.

## Leakage

The corpus contains policy articles only, not labeled answers. Judgments are
used solely in `session.rag.evaluate` (not indexed). Generate uses
EchoGroundedProvider so no live LLM is required. The TF-IDF twin uses the
same corpus and judgments.

## How to run

```bash
python proofs/policy-handbook-rag/script.py
python proofs/policy-handbook-rag/baseline_industry.py
```

## What you'll get

`results/results.json` with recall@k, MRR, and nDCG@k.
`results/comparison.json` is sklearn `TfidfVectorizer` + cosine on the same
corpus. Embeddings auto-use sentence-transformers when present, else hashing.

## Limitations

Tiny handbook; hashing embeddings are lexical, not semantic SOTA. Echo
generate is faithfulness scaffolding, not a production LLM.

Related: [RAG quickstart](../../guides/quickstart-rag.md),
[examples/rag_hashing_loop.py](../../examples/rag_hashing_loop.py).
