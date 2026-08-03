# torch-text-intent

## Business purpose

Route free-text support tickets to a queue using BuildML's Torch text path (`make_text_torch_loaders` + `fit_torch`), with an honest skip when Torch is unavailable.

## Data source

In-repo synthetic support tickets (`load_support_tickets_synthetic`): license-clear, deterministic. **Not** a real ticketing corpus.

## Leakage controls

- Stratified train / validation / test before text loaders
- Vocabulary / normalize from train only
- Test evaluated after lock
- Industry Tfidf+LR twin uses the same SplitPlan

## BuildML API steps

1. Probe `TORCH_STATUS`; if `skip_torch_paths` → `skipped_missing_extra`
2. `Session.ingest` → `set_roles` → `split`
3. `make_text_torch_loaders` → `fit_torch(epochs=3)` → `evaluate_torch`
4. Fallback: `fit_text_classifier` if Torch text APIs raise after Torch is available

## Metrics

Primary holdout: accuracy / F1 (weighted) on test (see `results/results.json`).

## Industry comparison (Tier C)

Filled: sklearn `TfidfVectorizer` + `LogisticRegression` twin via `baseline_industry.py` → `results/comparison.json`.

## Limitations

- Synthetic tickets; short Torch training budget
- Honest skip when Torch is missing

## How to run

```bash
python proofs/torch-text-intent/script.py
python proofs/torch-text-intent/baseline_industry.py
```
