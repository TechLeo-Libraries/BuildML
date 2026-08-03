# torch-tabular-underwrite

## Business purpose

Train a small Torch MLP on mortgage tabular features for default risk underwriting, with an honest skip when Torch is unavailable.

## Data source

In-repo synthetic mortgage table (`load_mortgage_default_synthetic`): license-clear, deterministic. **Not** a real servicing extract.

## Leakage controls

- Stratified train / validation / test before impute / encode / loaders
- Torch normalize statistics from train loader only
- Test `evaluate_torch` after lock
- Industry MLPClassifier twin uses the same SplitPlan

## BuildML API steps

1. Probe `TORCH_STATUS`; if `skip_torch_paths` → `skipped_missing_extra`
2. `Session.ingest` → `set_roles` → `split` → `impute` → `encode`
3. `make_torch_loaders` → `fit_torch(epochs=3)` → `evaluate_torch`
4. `save_torch_bundle` when available

## Metrics

Primary holdout: accuracy / F1 / ROC-AUC (or Torch report metrics) on test.

## Industry comparison (Tier C)

Filled: sklearn `MLPClassifier` twin via `baseline_industry.py` → `results/comparison.json`.

## Limitations

- 3-epoch CPU MLP smoke; not a production underwriting network
- Honest skip when Torch is missing or unhealthy

## How to run

```bash
python proofs/torch-tabular-underwrite/script.py
python proofs/torch-tabular-underwrite/baseline_industry.py
```
