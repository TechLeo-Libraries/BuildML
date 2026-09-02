# parchment-policy-copilot

This script composes `session.rag`, `session.ranking`, and `session.cbr` on
one synthetic policy handbook plus escalation cases. It is not a product BuildML ships.

The script indexes a handbook for hybrid RAG retrieve plus echo-grounded
generate, trains a query-group LTR ranker over synthetic policy judgments,
and fits case-based reasoning on prior escalation decisions.

## Data

Tiny synthetic handbook. Not a real policy corpus.

## Leakage

RAG corpus contains policy articles only: judgments never indexed. LTR
`group_split` on `query_id` before fit. CBR case memory built from train
only. Test retrieval / rank / CBR metrics after lock.

## What fails if leakage is ignored

Indexing labeled answers into the corpus turns RAG eval into a lookup.
Query leakage in LTR inflates nDCG on held-out policy questions. CBR memory
that includes test cases is not a fair retrieve-and-reuse bench.

## How to run

```bash
python proofs/parchment-policy-copilot/script.py
```

## What you'll get

`results/` summary and per-stage JSON.

## Upstream

`policy-handbook-rag`, `support-kb-rag`, `sponsored-ad-ltr`,
`search-relevance-ltr`, `warranty-cbr-memory`, `case-memory-claims`,
`pulse-support-copilot`.

## Limitations

Tiny handbook; Echo generate offline. CBR is not RAG.
