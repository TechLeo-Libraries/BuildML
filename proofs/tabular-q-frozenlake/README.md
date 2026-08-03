# Tabular Q-learning proof (FrozenLake)

Tier **A** proof that foundational tabular TD control (`mode='tabular_q'`,
`algorithm='q_learning'`) runs end-to-end: fit, evaluate, act, bundle save.

## Run

```bash
python proofs/tabular-q-frozenlake/script.py
```

Requires `buildml[rl]` (Gymnasium). Skips gracefully when the extra is absent.

## Honesty boundary

- Teaching-scale discrete env only (FrozenLake).
- Not MuJoCo, robotics, or batch offline RL.
- Deep value methods live under `mode='gym_sb3'`.
