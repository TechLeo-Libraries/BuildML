# Scaffold Compliance AI

**Tier B** cross-domain product proof: symbolic KYC/AML rules + optional
neuro-symbolic NAM + validation-tuned escalation capacity.

## Product narrative

Scaffold screens synthetic wires for escalation. Rule-ish labels come from
amount × jurisdiction and young-account × PEP patterns. The desk:

1. Induces symbolic decision-tree guardrails on a stratified train split
2. Optionally fits a neuro-symbolic NAM when torch paths are enabled
3. Scores with classical logistic and selects review threshold / knapsack on validation

## Status

`completed` / `partial`: run `script.py`; see `results/summary.json` and stage JSONs.

## How to run

```bash
.\.venv\Scripts\python.exe proofs\scaffold-compliance-ai\script.py
```

## Leakage controls

- Stratified split before symbolic / neuro-symbolic / decisions
- Symbolic + NAM fit on train only
- Review capacity / threshold tuned on validation only
- Test evaluate after each stage locks

## What fails if leakage is ignored

- Inducing rules on the full book looks more “compliant” than production
- Tuning escalation thresholds on test understates review cost
- Fitting NAM with test rows invents holdout fidelity

## Upstream Tier A building blocks

`compliance-neuro-symbolic`, `policy-rules-neuro-symbolic`,
`cost-sensitive-collections`, `loan-approval-classical`

## Limitations

Not legal advice; rule fidelity ≠ compliance certification. Neuro-symbolic NAM
skipped when torch paths are disabled.
