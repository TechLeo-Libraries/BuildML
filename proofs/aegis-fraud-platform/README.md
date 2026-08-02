# Aegis Fraud Platform

**Tier B** cross-domain product proof — graph rings + anomaly + supervised scoring
+ online stream updates + validation-tuned decision thresholds + optional
symbolic guardrails.

## Product narrative

Aegis is a fraud review desk for a synthetic payments portfolio. Accounts form
community graphs; rare fraud is denser in one community. The platform:

1. Fits classical graph node features on a stratified node split
2. Runs unsupervised anomaly detection with **validation-only** threshold tuning
3. Trains a supervised logistic scorer for calibrated review scores
4. Streams train-cursor `partial_fit` updates (test never enters the stream)
5. Selects cost-sensitive threshold / knapsack on validation
   (default `scale` skips `ignore`/`id` so `review_cost` stays non-negative)
6. Optionally induces symbolic decision-tree guardrails for explainable denies

## Status

`completed` — run `script.py`; see `results/summary.json` and stage JSONs.

## How to run

```bash
.\.venv\Scripts\python.exe proofs\aegis-fraud-platform\script.py
```

## Leakage controls

- Stratified node split before any graph / anomaly / supervised fit
- Anomaly threshold + decision policies tuned on validation only
- Online `partial_fit` consumes train cursor only
- Test evaluated once per stage after that stage locks

## What fails if leakage is ignored

- Tuning thresholds on test inflates F1 and understates review cost
- Graph features conditioned on test labels overstate ring detection
- Streaming updates that include test rows make online metrics meaningless
- Symbolic rules induced on the full table look more “compliant” than production

## Upstream Tier A building blocks

`graph-fraud-rings`, `network-intrusion-anomaly`, `loan-approval-classical`,
`stream-fraud-online`, `cost-sensitive-collections`, `policy-rules-neuro-symbolic`

## Limitations

Synthetic portfolio — not a real card network. Classical graph path is primary.
Missing extras are skipped with JSON disclosures (`MissingExtraError`).
