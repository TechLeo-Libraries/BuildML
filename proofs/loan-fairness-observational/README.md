# loan-fairness-observational

You have a synthetic consumer-credit holdout and a caller-declared sensitive
attribute (`region`). You want observational group-fairness gaps
(demographic parity / disparate impact / equalized odds) on test
predictions, not a legal audit.

## Data

In-repo synthetic credit table (`load_credit_approval_synthetic`):
license-clear, deterministic. Not a real FCRA / bureau extract. Sensitive
groups are caller-declared (`region`); BuildML does not infer protected
class.

## Leakage

Stratified train / validation / test before fit. The sensitive column has
`role=ignore` (not a predictor). The classifier fits on train only.
Fairness metrics are computed on holdout test predictions only.

## How to run

```bash
python proofs/loan-fairness-observational/script.py
```

## What you'll get

`results/results.json` with observational fairness gaps from
`FairnessReport` (selection rates, demographic parity difference,
disparate impact ratio, equalized odds Delta TPR / Delta FPR). There is no
sklearn metric twin for this proof.

## Limitations

Observational only; not a legal audit. Synthetic table; no bias mitigation /
reweighing in this proof.

Related: [Fairness quickstart](../../guides/quickstart-fairness.md).
