# Beacon Label Factory

**Tier B** cross-domain product proof: SSL pretext + semi-supervised
propagation + active-learning budget loop for scarce inspection labels.

## Product narrative

Beacon is a labeling factory for manufacturing inspection features. Most train
labels are masked; holdouts keep full labels for evaluation. The platform:

1. Fits masked-tabular SSL pretext (+ optional probe head) on train features
2. Runs label propagation on the scarce labeled pool
3. Runs a margin-sampling active-learning budget loop querying **train** only
4. Uses a simulated oracle (ground-truth for queried train indices)

## Status

Run `script.py`. Outputs land under `results/` (summary and stage JSON)..

## How to run

```bash
python proofs\beacon-label-factory\script.py
```

## Leakage controls

- Stratified split before masking / pretext / AL
- Label masking applied to train indices only
- Holdouts retain full labels solely for evaluation
- AL queries drawn from the train unlabeled pool only

## What fails if leakage is ignored

- Masking validation/test then recovering labels via the graph overstates SSL gains
- Allowing AL to query the test pool turns the budget curve into a cheat sheet
- Fitting SSL pretext on the full table leaks holdout geometry into embeddings

## Upstream Tier A building blocks

`radiology-semi-labels`, `semi-label-efficiency`, `active-labeling-budget`,
`defect-active-budget`, `tabular-ssl-probe`, `ssl-representation-probe`,
`atlas-label-studio`

## Limitations

Simulated oracle; tabular inspection proxies. Missing extras skip with JSON
disclosures.
