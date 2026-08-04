# Vector Control Deck

**Tier B** cross-domain product proof: imitation learning (+ optional gym RL)
+ decision/optimize allocation + classical supervised action baseline.

## Product narrative

Vector is a control deck for synthetic cartpole-ish demos. Expert actions are
cloned, optionally probed in Gymnasium, scored with a classical baseline, and
allocated under a capacity budget. The platform:

1. Fits behavioral cloning on train expert rows
2. Optionally runs a short Gymnasium REINFORCE probe (skips if missing)
3. Trains a classical logistic action baseline
4. Selects threshold / knapsack intervention policies on validation

## Status

Run `script.py`. Outputs land under `results/` (summary and stage JSON)..
Core stages keep the product completed when gym RL skips.

## How to run

```bash
python proofs\vector-control-deck\script.py
```

## Leakage controls

- BC / classical fit on train expert rows only
- Optional gym RL is a separate env probe
- Decision policies selected on validation only
- Test imitation / supervised metrics after lock

## What fails if leakage is ignored

- BC trained on test trajectories overstates policy cloning skill
- Capacity policies tuned on test understate intervention cost
- Reporting gym returns without disclosing env/eval separation misleads

## Upstream Tier A building blocks

`imitation-cartpole-control`, `tabular-q-frozenlake`, `campaign-budget-optimize`,
`loan-approval-classical`

## Limitations

Synthetic demos. Gymnasium RL optional; skips disclosed in JSON.
