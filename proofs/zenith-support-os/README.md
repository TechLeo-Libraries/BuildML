# zenith-support-os

This script composes `session.rag`, `session.nlp`, and
`session.active_learning` on one synthetic support table. It is not a
product BuildML ships.

The script retrieves from a knowledge-base corpus (answers never indexed),
routes free-text tickets with TF-IDF + logistic (validation for selection),
and runs a margin-sampling active-learning loop on a train unlabeled pool
only.

## Data

Synthetic KB and tickets. Not a live helpdesk.

## Leakage

RAG corpus is KB articles only; judgments never indexed as answers. NLP
stratified split before TF-IDF fit; validation for selection.
Active-learning queries drawn from the train unlabeled pool only. Test
evaluate after locks.

## What fails if leakage is ignored

Indexing judgment answers into RAG inflates recall@k. Fitting the text
vectorizer on test tickets invents queue accuracy. Querying the test pool
for labels makes active-learning curves meaningless.

## How to run

```bash
python proofs/zenith-support-os/script.py
```

## What you'll get

`results/` summary and per-stage JSON.

## Upstream

`support-kb-rag`, `policy-handbook-rag`, `ticket-routing-nlp`,
`active-labeling-budget`, `defect-active-budget`, `atlas-label-studio`.

## Limitations

Synthetic KB + tickets. Echo generate is offline scaffolding. Active
learning uses a simulated oracle.
