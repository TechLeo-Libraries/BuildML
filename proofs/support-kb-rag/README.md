# support-kb-rag

## Business purpose

Retrieve grounded answers from a product support knowledge base for agent
assist / self-serve help, with offline retrieval metrics and faithfulness
scaffolding.

## Data source

In-repo synthetic support articles + query→doc judgments
(`load_support_kb_corpus`): license-clear.

## Leakage controls

- Corpus indexes articles only: **never** labeled answer strings
- Judgments used only in `session.rag.evaluate`
- Embed/index built before evaluation; no test-time index mutation

## BuildML API steps

1. `session.rag.ingest_corpus` → `session.rag.chunk`
2. `session.rag.embed_and_index` (sentence-transformers when available; else hashing)
3. `session.rag.retrieve` / `session.rag.generate` (EchoGroundedProvider offline)
4. `session.rag.evaluate` → optional `session.rag.save_bundle`

## Metrics

recall@k, MRR, nDCG@k on held-out judgments.

## Industry comparison (Tier C)

Industry twin: `baseline_industry.py` runs sklearn TF-IDF + cosine retrieval on the same corpus/judgments (`results/comparison.json`). Judgments are never indexed.
## Limitations

Tiny corpus; echo generate is not a production LLM.
