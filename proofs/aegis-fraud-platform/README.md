# aegis-fraud-platform

This script composes `session.graph`, `session.anomaly`, classical
`session.fit`, `session.online`, `session.decision`, and optional
`session.symbolic` on one synthetic payments table. It is not a product BuildML ships.

Accounts form community graphs; rare fraud is denser in one community. The
script scores rings, flags anomalies, updates an online classifier from a
train cursor, and selects a review threshold on validation.

## Data

Synthetic payments portfolio. Not a real card network.

## Leakage

Stratified node split before any graph, anomaly, or supervised fit. Anomaly
threshold and decision policies tuned on validation only. Online
`partial_fit` consumes the train cursor only. Test is evaluated once per
stage after that stage locks.

Default `scale` skips `ignore`/`id` so `review_cost` stays non-negative.

## What fails if leakage is ignored

Tuning thresholds on test inflates F1 and understates review cost. Graph
features conditioned on test labels overstate ring detection. Streaming
updates that include test rows make online metrics meaningless. Symbolic
rules induced on the full table look more "compliant" than they would in
production.

## How to run

```bash
python proofs/aegis-fraud-platform/script.py
```

## What you'll get

`results/` summary and per-stage JSON.

## Upstream

`graph-fraud-rings`, `network-intrusion-anomaly`, `loan-approval-classical`,
`stream-fraud-online`, `cost-sensitive-collections`,
`policy-rules-neuro-symbolic`.

## Limitations

Synthetic portfolio. Classical graph path is primary. Missing extras are
skipped with JSON disclosures (`MissingExtraError`).
