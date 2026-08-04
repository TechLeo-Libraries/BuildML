# policy-handbook-rag

## Business purpose

Retrieve answers from a small employee policy handbook (leave, expenses, remote work, security, conduct, travel) with grounded generation scaffolding.

## Data source

Inline synthetic policy articles + relevance judgments (same pattern as `support-kb-rag`). Judgments are never indexed as answers.

## Leakage controls

- Corpus contains policy articles only: not labeled answers
- Judgments used solely in `session.rag.evaluate` (not indexed)
- EchoGroundedProvider for offline generate (no live LLM required)
- Industry TF-IDF twin uses the same corpus and judgments

## BuildML API steps

1. `Session()` → `session.rag.ingest_corpus` → `session.rag.chunk`
2. `session.rag.embed_and_index` (auto when sentence-transformers present, else hashing)
3. `session.rag.retrieve` → `session.rag.generate` → `session.rag.evaluate`
4. `session.rag.save_bundle`

## Metrics

Primary retrieval: recall@k, MRR, nDCG@k (see `results/results.json`).

## Industry comparison (Tier C)

Industry twin: sklearn `TfidfVectorizer` + cosine twin via `baseline_industry.py` → `results/comparison.json`.

## Limitations

- Tiny handbook; hashing embeddings are lexical, not semantic SOTA
- Echo generate is faithfulness scaffolding, not a production LLM

## How to run

```bash
python proofs/policy-handbook-rag/script.py
python proofs/policy-handbook-rag/baseline_industry.py
```
