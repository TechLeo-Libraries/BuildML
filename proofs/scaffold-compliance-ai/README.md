# scaffold-compliance-ai

This script composes `session.symbolic`, classical `session.fit`, and
`session.decision` on one synthetic wire-review table. It is not a product BuildML ships.

Rule-ish labels come from amount x jurisdiction and young-account x PEP
patterns. The script induces symbolic decision-tree guardrails on a
stratified train split, optionally fits a neuro-symbolic NAM when torch
paths are enabled, and selects review threshold / knapsack on validation.

## Data

Synthetic wires. Not a legal case file.

## Leakage

Stratified split before symbolic / neuro-symbolic / decisions. Symbolic +
NAM fit on train only. Review capacity / threshold tuned on validation
only. Test evaluate after each stage locks.

## What fails if leakage is ignored

Inducing rules on the full book looks more "compliant" than they would in
production. Tuning escalation thresholds on test understates review cost.
Fitting NAM with test rows invents holdout fidelity.

## How to run

```bash
python proofs/scaffold-compliance-ai/script.py
```

## What you'll get

`results/` summary and per-stage JSON.

## Upstream

`compliance-neuro-symbolic`, `policy-rules-neuro-symbolic`,
`cost-sensitive-collections`, `loan-approval-classical`.

## Limitations

Not legal advice; rule fidelity is not compliance certification.
Neuro-symbolic NAM skipped when torch paths are disabled.
