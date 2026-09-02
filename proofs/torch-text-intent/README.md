# torch-text-intent

You have free-text support tickets and a queue label. You want BuildML's
Torch text path (`session.dl.make_text_loaders` + `session.dl.fit`), with a
skip when Torch is unavailable.

## Data

In-repo synthetic support tickets (`load_support_tickets_synthetic`):
license-clear, deterministic. Not a real ticketing corpus.

## Leakage

Stratified train / validation / test before text loaders. Vocabulary and
normalize stats come from train only. Test evaluated after lock. The
Tfidf+LR twin uses the same `SplitPlan`.

## How to run

```bash
python proofs/torch-text-intent/script.py
python proofs/torch-text-intent/baseline_industry.py
```

## What you'll get

`results/results.json` with holdout accuracy / F1 (weighted) on test. If
Torch is missing, the script writes `skipped_missing_extra`. If Torch text
APIs raise after Torch is available, the script falls back to
`session.nlp.fit_classifier`. `results/comparison.json` is sklearn
`TfidfVectorizer` + `LogisticRegression` on the same split.

## Limitations

Synthetic tickets; short Torch training budget. Honest skip when Torch is
missing.

Related: [Torch quickstart](../../guides/quickstart-torch.md).
