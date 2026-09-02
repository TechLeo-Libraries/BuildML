# helix-knowledge-mesh

This script composes `session.kg`, `session.rag`, and `session.symbolic` on
one synthetic systems/teams/policies table plus a handbook corpus. It is
not a product BuildML ships.

The script fits TransE link prediction on a disjoint triple split, indexes
a policy handbook for hybrid RAG retrieve plus echo-grounded generate, and
induces decision-tree guardrails for block/allow on answer risk features.

## Data

Synthetic mesh and handbook.

## Leakage

KG triple split before TransE fit. RAG corpus contains policy articles
only: judgments never indexed. Symbolic guardrails fit on train; test after
lock.

## What fails if leakage is ignored

Training TransE on all triples makes link metrics meaningless. Indexing
labeled answers into the corpus turns RAG eval into a lookup. Inducing
guardrail rules on the full table overstates compliance.

## How to run

```bash
python proofs/helix-knowledge-mesh/script.py
```

## What you'll get

`results/` summary and per-stage JSON.

## Upstream

`logistics-kg-linkpred`, `kg-biomed-linkpred`, `policy-handbook-rag`,
`support-kb-rag`, `compliance-neuro-symbolic`,
`policy-rules-neuro-symbolic`.

## Limitations

Synthetic mesh / handbook. Missing extras skip with JSON disclosures.
