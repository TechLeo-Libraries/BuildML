# Parchment Policy Copilot

**Tier B** cross-domain product proof: RAG retrieval/generate + learning-to-rank
over policy queries + CBR case memory for escalations.

## Product narrative

Parchment is a policy assistant. A handbook corpus grounds answers; an LTR
stage re-ranks candidate articles; CBR recalls similar prior escalations.
The platform:

1. Indexes a policy handbook for hybrid RAG retrieve + echo-grounded generate
2. Trains a query-group LTR ranker over synthetic policy judgments
3. Fits case-based reasoning on prior escalation decisions

## Status

Run `script.py`. Outputs land under `results/` (summary and stage JSON)..

## How to run

```bash
python proofs\parchment-policy-copilot\script.py
```

## Leakage controls

- RAG corpus contains policy articles only: judgments never indexed
- LTR `group_split` on `query_id` before fit
- CBR case memory built from train only
- Test retrieval / rank / CBR metrics after lock

## What fails if leakage is ignored

- Indexing labeled answers into the corpus turns RAG eval into a lookup
- Query leakage in LTR inflates nDCG on held-out policy questions
- CBR memory that includes test cases is not a fair retrieve-and-reuse bench

## Upstream Tier A building blocks

`policy-handbook-rag`, `support-kb-rag`, `sponsored-ad-ltr`, `search-relevance-ltr`,
`warranty-cbr-memory`, `case-memory-claims`, `pulse-support-copilot`

## Limitations

Tiny handbook; Echo generate offline. CBR ≠ RAG.
