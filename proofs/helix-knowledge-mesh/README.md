# Helix Knowledge Mesh

**Tier B** cross-domain product proof: knowledge-graph link prediction + RAG
retrieval/generate + symbolic answer guardrails.

## Product narrative

Helix is an enterprise knowledge mesh: systems, teams, and policies form a KG;
a handbook corpus grounds answers; symbolic rules block high-risk / ungrounded
responses. The platform:

1. Fits TransE link prediction on a disjoint triple split
2. Indexes a policy handbook for hybrid RAG retrieve + echo-grounded generate
3. Induces decision-tree guardrails for block/allow on answer risk features

## Status

Run `script.py`. Outputs land under `results/` (summary and stage JSON)..

## How to run

```bash
python proofs\helix-knowledge-mesh\script.py
```

## Leakage controls

- KG triple split before TransE fit
- RAG corpus contains policy articles only: judgments never indexed
- Symbolic guardrails fit on train; test after lock

## What fails if leakage is ignored

- Training TransE on all triples makes link metrics meaningless
- Indexing labeled answers into the corpus turns RAG eval into a lookup
- Inducing guardrail rules on the full table overstates compliance

## Upstream Tier A building blocks

`logistics-kg-linkpred`, `kg-biomed-linkpred`, `policy-handbook-rag`,
`support-kb-rag`, `compliance-neuro-symbolic`, `policy-rules-neuro-symbolic`

## Limitations

Synthetic mesh / handbook. Missing extras skip with JSON disclosures.
