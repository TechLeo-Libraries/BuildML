# rivulet-stream-risk

This script composes `session.online`, `session.anomaly`, classical
`session.fit`, and `session.decision` on one synthetic payment-rail table.
It is not a product BuildML ships.

Rare attacks arrive in a continuous feed. The script streams train-cursor
`partial_fit` updates (test never enters the stream), runs unsupervised
anomaly with validation-only threshold tuning, and selects cost-sensitive
threshold / knapsack policies on validation.

## Data

Synthetic payment rail. Not a card-network extract.

## Leakage

Stratified split before online / anomaly / supervised / decisions. Online
`partial_fit` consumes the train cursor only. Anomaly threshold + decision
policies tuned on validation only. Test evaluated once per stage after that
stage locks.

## What fails if leakage is ignored

Streaming updates that include test rows make online metrics meaningless.
Tuning thresholds on test inflates F1 and understates review cost. Fitting
the supervised scorer on the full table invents holdout ROC.

## How to run

```bash
python proofs/rivulet-stream-risk/script.py
```

## What you'll get

`results/` summary and per-stage JSON.

## Upstream

`stream-fraud-online`, `clickstream-online`, `payment-rail-anomaly`,
`network-intrusion-anomaly`, `cost-sensitive-collections`.

## Limitations

Synthetic payment rail. Missing extras are skipped with JSON disclosures
(`MissingExtraError`).
