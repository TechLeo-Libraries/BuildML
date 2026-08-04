# Pulse Support Copilot

**Tier B** cross-domain product proof: RAG + ranking + CBR case memory +
symbolic guardrails for a synthetic support desk.

## Product narrative

Pulse helps agents answer tickets without contaminating retrieval metrics or
bypassing escalation policy:

1. Retrieves from a knowledge-base corpus (answers never indexed)
2. Re-ranks ticket→article candidates with a group-split LTR model
3. Looks up similar resolved cases via CBR for escalate-or-not
4. Induces symbolic decision-tree guardrails (PII / severity) on the same train split

## Status

Run `script.py`. Outputs land under `results/` (summary and stage JSON)..

## How to run

```bash
python proofs\pulse-support-copilot\script.py
```

## Leakage controls

- RAG corpus = KB articles only; judgments never indexed as answers
- LTR `group_split` by `query_id` before ranker fit
- CBR case memory built from train cases only
- Symbolic rules induced on the same train split as CBR; test after lock

## What fails if leakage is ignored

- Indexing judgment answers into RAG inflates recall@k
- Fitting the ranker on test queries overstates NDCG
- Putting test tickets into CBR memory makes accuracy meaningless
- Inducing guardrail rules on full data looks more “safe” than production

## Upstream Tier A building blocks

`support-kb-rag`, `search-relevance-ltr`, `case-memory-claims`,
`policy-rules-neuro-symbolic`

## Limitations

Synthetic KB + tickets: not a live helpdesk. Echo generate is offline
scaffolding. Missing extras are skipped with JSON disclosures (`MissingExtraError`).
