# vector-control-deck

This script composes `session.rl` (imitation, optional gym), classical
`session.fit`, and `session.decision` on one synthetic expert-action table.
It is not a product BuildML ships.

The script fits behavioral cloning on train expert rows, optionally runs a
short Gymnasium REINFORCE probe (skips if missing), trains a classical
logistic action baseline, and selects threshold / knapsack intervention
policies on validation. Classical and decision stages still run when gym
RL skips.

## Data

Synthetic cartpole-style demos.

## Leakage

BC / classical fit on train expert rows only. Optional gym RL is a
separate env probe. Decision policies selected on validation only. Test
imitation / supervised metrics after lock.

## What fails if leakage is ignored

BC trained on test trajectories overstates policy cloning skill. Capacity
policies tuned on test understate intervention cost. Reporting gym returns
without disclosing env/eval separation misleads.

## How to run

```bash
python proofs/vector-control-deck/script.py
```

## What you'll get

`results/` summary and per-stage JSON. Gymnasium skip is disclosed in JSON.

## Upstream

`imitation-cartpole-control`, `tabular-q-frozenlake`,
`campaign-budget-optimize`, `loan-approval-classical`.

## Limitations

Synthetic demos. Gymnasium RL optional; skips disclosed in JSON.
