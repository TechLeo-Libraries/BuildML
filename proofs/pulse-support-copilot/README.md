# pulse-support-copilot

This script composes `session.rag`, `session.ranking`, `session.cbr`, and
`session.symbolic` on one synthetic support table. It is not a product BuildML ships.

The script retrieves from a knowledge-base corpus (answers never indexed),
re-ranks ticket-to-article candidates with a group-split LTR model, looks
up similar resolved cases via CBR, and induces symbolic decision-tree
guardrails on the same train split.

## Data

Synthetic KB and tickets. Not a live helpdesk.

## Leakage

RAG corpus is KB articles only; judgments never indexed as answers. LTR
`group_split` by `query_id` before ranker fit. CBR case memory is built
from train cases only. Symbolic rules are induced on the same train split
as CBR; test after lock.

## What fails if leakage is ignored

Indexing judgment answers into RAG inflates recall@k. Fitting the ranker on
test queries overstates NDCG. Putting test tickets into CBR memory makes
accuracy meaningless. Inducing guardrail rules on full data looks more
"safe" than they would in production.

## How to run

```bash
python proofs/pulse-support-copilot/script.py
```

## What you'll get

`results/` summary and per-stage JSON.

## Upstream

`support-kb-rag`, `search-relevance-ltr`, `case-memory-claims`,
`policy-rules-neuro-symbolic`.

## Limitations

Synthetic KB + tickets. Echo generate is offline scaffolding. Missing
extras are skipped with JSON disclosures (`MissingExtraError`).
