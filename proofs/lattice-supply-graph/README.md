# Lattice Supply Graph

**Tier B** cross-domain product proof: classical graph node features +
knowledge-graph link prediction + classical supervised late-risk scoring.

## Product narrative

Lattice models a synthetic supplier network. Community graphs feed classical
node features; a logistics KG predicts missing links; a logistic model scores
late-delivery risk. The platform:

1. Fits classical inductive graph features on a stratified node split
2. Fits TransE link prediction on warehouse–route–hub triples
3. Trains classical late-risk scoring on the same node split

## Status

`completed`: run `script.py`; see `results/summary.json` and stage JSONs.

## How to run

```bash
.\.venv\Scripts\python.exe proofs\lattice-supply-graph\script.py
```

## Leakage controls

- Stratified node split before graph / supervised fit
- Classical graph features from train graph view
- KG triple split before TransE
- Test evaluate after each stage locks

## What fails if leakage is ignored

- Graph features conditioned on test labels overstate community risk
- Training TransE on all triples makes link metrics meaningless
- Supervised late-risk trained with test rows overstates TMS readiness

## Upstream Tier A building blocks

`peer-lending-graph`, `graph-fraud-rings`, `logistics-kg-linkpred`,
`kg-biomed-linkpred`, `loan-approval-classical`

## Limitations

Synthetic supplier communities. Classical graph path is primary.
