# loan-fairness-observational

## Business purpose

Observational group-fairness gaps (demographic parity / disparate impact /
equalized odds) on a synthetic consumer-credit holdout, with `region` as a
caller-declared sensitive attribute.

## Data source

In-repo synthetic credit table (`load_credit_approval_synthetic`): license-clear,
deterministic. **Not** a real FCRA / bureau extract. Sensitive groups are
caller-declared (`region`); BuildML does not infer protected class.

## Leakage controls

- Stratified train / validation / test before fit
- Sensitive column `role=ignore` (not a predictor)
- Classifier fitted on train only
- Fairness metrics on holdout test predictions only

## BuildML API steps

1. Load synthetic credit table (+ provenance meta)
2. `Session.ingest` → roles → stratified `split` → `impute` → `scale` → `fit`
3. `session.fairness.evaluate(sensitive_column="region", partition="test")`

## Metrics

Observational fairness gaps from `FairnessReport` (selection rates, demographic
parity difference, disparate impact ratio, equalized odds ΔTPR / ΔFPR). See
`results/results.json`.

## How to run

```bash
python proofs/loan-fairness-observational/script.py
```

## Limitations

Observational only; not a legal audit. Synthetic table; no bias mitigation /
reweighing applied in this proof.
