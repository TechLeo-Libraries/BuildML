# Zenith Support OS

**Tier B** cross-domain product proof: RAG retrieval + NLP ticket routing +
active-learning budget for a synthetic support operating system.

## Product narrative

Zenith helps agents answer tickets, route queues, and spend scarce labeling
budget honestly:

1. Retrieves from a knowledge-base corpus (answers never indexed)
2. Routes free-text tickets with TF-IDF + logistic (validation for selection)
3. Runs a margin-sampling active-learning loop on a train unlabeled pool only

## Status

Run `script.py`. Outputs land under `results/` (summary and stage JSON)..

## How to run

```bash
python proofs\zenith-support-os\script.py
```

## Leakage controls

- RAG corpus = KB articles only; judgments never indexed as answers
- NLP stratified split before TF-IDF fit; validation for selection
- Active-learning queries drawn from train unlabeled pool only
- Test evaluate after locks

## What fails if leakage is ignored

- Indexing judgment answers into RAG inflates recall@k
- Fitting the text vectorizer on test tickets invents queue accuracy
- Querying the test pool for labels makes active-learning curves meaningless

## Upstream Tier A building blocks

`support-kb-rag`, `policy-handbook-rag`, `ticket-routing-nlp`,
`active-labeling-budget`, `defect-active-budget`, `atlas-label-studio`

## Limitations

Synthetic KB + tickets: not a live helpdesk. Echo generate is offline
scaffolding. Active learning uses a simulated oracle.
