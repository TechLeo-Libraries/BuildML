# logistics-kg-linkpred

## Business purpose

Predict missing logistics links (warehouse → route → hub / carrier) with TransE embeddings for network completion and routing discovery.

## Data source

Inline synthetic logistics triples (warehouse–route–hub–carrier motifs). **Not** a real TMS extract.

## Leakage controls

- Triple split before fit
- Train-only TransE
- Test link metrics after lock
- Industry PMI twin uses the same triple split

## BuildML API steps

1. `Session.ingest` → `set_roles` → `split`
2. `fit_kg(method="transe")`
3. `predict_links` → `evaluate_kg(test)`
4. `save_kg_bundle`

## Metrics

Primary holdout: hits@k, mean rank, MRR (see `results/results.json`).

## Industry comparison (Tier C)

Filled — train co-occurrence PMI filtered-ranking twin via `baseline_industry.py` → `results/comparison.json`.

## Limitations

- Synthetic logistics motifs; not a licensed TMS / network extract
- Single seed

## How to run

```bash
python proofs/logistics-kg-linkpred/script.py
python proofs/logistics-kg-linkpred/baseline_industry.py
```
