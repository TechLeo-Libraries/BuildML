# Rivulet Stream Risk

**Tier B** cross-domain product proof: online stream scoring + unsupervised
anomaly + validation-tuned decision thresholds for a synthetic payment rail.

## Product narrative

Rivulet is a stream-risk desk for ACH / card-style authorizations. Rare attacks
arrive in a continuous feed. The platform:

1. Streams train-cursor `partial_fit` updates (test never enters the stream)
2. Runs unsupervised anomaly detection with **validation-only** threshold tuning
3. Trains a supervised logistic scorer and selects cost-sensitive threshold /
   knapsack policies on validation

## Status

Run `script.py`. Outputs land under `results/` (summary and stage JSON)..

## How to run

```bash
python proofs\rivulet-stream-risk\script.py
```

## Leakage controls

- Stratified split before online / anomaly / supervised / decisions
- Online `partial_fit` consumes train cursor only
- Anomaly threshold + decision policies tuned on validation only
- Test evaluated once per stage after that stage locks

## What fails if leakage is ignored

- Streaming updates that include test rows make online metrics meaningless
- Tuning thresholds on test inflates F1 and understates review cost
- Fitting the supervised scorer on the full table invents holdout ROC

## Upstream Tier A building blocks

`stream-fraud-online`, `clickstream-online`, `payment-rail-anomaly`,
`network-intrusion-anomaly`, `cost-sensitive-collections`

## Limitations

Synthetic payment rail: not a card-network extract. Missing extras are skipped
with JSON disclosures (`MissingExtraError`).
