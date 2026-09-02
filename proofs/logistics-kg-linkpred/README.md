# logistics-kg-linkpred

You have warehouse-route-hub-carrier triples. You want missing-link
prediction with TransE for network completion, on a disjoint triple split.

## Data

Inline synthetic logistics triples (warehouse-route-hub-carrier motifs). Not
a real TMS extract.

## Leakage

Triple split before fit. Train-only TransE. Test link metrics after lock.
The PMI twin uses the same triple split.

## How to run

```bash
python proofs/logistics-kg-linkpred/script.py
python proofs/logistics-kg-linkpred/baseline_industry.py
```

## What you'll get

`results/results.json` with hits@k, mean rank, and MRR.
`results/comparison.json` is a train co-occurrence PMI filtered-ranking twin
on the same split. Bundle save/load re-evaluates the holdout.

## Limitations

Synthetic logistics motifs; not a licensed TMS / network extract; single
seed.

Related: [Knowledge-graph quickstart](../../guides/quickstart-kg.md),
[examples/kg_transe_loop.py](../../examples/kg_transe_loop.py).
