# torch-tabular-underwrite

You have mortgage tabular features and a default label. You want a short
Torch MLP on an honest split, with a skip when Torch is unavailable.

## Data

In-repo synthetic mortgage table (`load_mortgage_default_synthetic`):
license-clear, deterministic. Not a real servicing extract.

## Leakage

Stratified train / validation / test before impute, encode, or loaders.
Torch normalize statistics come from the train loader only. Test
`session.dl.evaluate` after lock. The sklearn `MLPClassifier` twin uses the
same `SplitPlan`.

## How to run

```bash
python proofs/torch-tabular-underwrite/script.py
python proofs/torch-tabular-underwrite/baseline_industry.py
```

## What you'll get

`results/results.json` with holdout accuracy / F1 / ROC-AUC (or Torch report
metrics) on test. If `TORCH_STATUS` says `skip_torch_paths`, the script
writes `skipped_missing_extra`. `results/comparison.json` is a sklearn
`MLPClassifier` twin on the same split.

## Limitations

3-epoch CPU MLP smoke; not an underwriting network you would ship. Honest
skip when Torch is missing or unhealthy.

Related: [Torch quickstart](../../guides/quickstart-torch.md).
