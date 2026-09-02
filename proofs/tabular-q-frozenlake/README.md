# tabular-q-frozenlake

You want tabular Q-learning (`mode='tabular_q'`, `algorithm='q_learning'`)
on FrozenLake: fit, evaluate, act, and save a bundle. This is a teaching-
scale discrete env, not robotics.

## Data

Gymnasium FrozenLake-v1 when `buildml[rl]` is installed. The script skips
when the extra is absent.

## Leakage

The agent learns from training rollouts in the env. Evaluation episodes are
separate from the fit loop. A random-policy twin is not trained on eval
returns.

## How to run

```bash
python proofs/tabular-q-frozenlake/script.py
python proofs/tabular-q-frozenlake/baseline_industry.py
```

## What you'll get

`results/results.json` with mean return and related tabular-Q probe fields.
`results/comparison.json` is a uniform random FrozenLake policy (50
episodes) when gymnasium is available.

## Limitations

Teaching-scale discrete env only (FrozenLake). Not MuJoCo, robotics, or
batch offline RL. Deep value methods live under `mode='gym_sb3'`.

Related: [Imitation / RL quickstart](../../guides/quickstart-imitation-rl.md).
