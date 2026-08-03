# Prism Shape Monitor

**Tier B** cross-domain product proof: TDA shape descriptors + unsupervised
anomaly + classical supervised pass/fail for process monitoring.

## Product narrative

Prism monitors manufacturing process clouds. In-spec vs drifted sensor
geometry is scored three ways:

1. Optional TDA persistence-image head (skips if `ripser`/`persim` missing)
2. Unsupervised anomaly with **validation-only** threshold tuning
3. Classical logistic pass/fail scorer on the same honest split

## Status

`completed`: run `script.py`; see `results/summary.json` and stage JSONs.
Core stages (anomaly + supervised) keep the product completed when TDA skips.

## How to run

```bash
.\.venv\Scripts\python.exe proofs\prism-shape-monitor\script.py
```

## Leakage controls

- Stratified split before TDA / anomaly / supervised fit
- Scale + TDA fit on train only
- Anomaly threshold tuned on validation only
- Test used once per stage after lock

## What fails if leakage is ignored

- Fitting persistence images on the full cloud leaks holdout geometry
- Tuning anomaly thresholds on test inflates F1 for drift alerts
- Supervised pass/fail trained with test rows overstates SPC readiness

## Upstream Tier A building blocks

`process-tda-shape`, `credit-tda-shape`, `iot-sensor-anomaly`,
`network-intrusion-anomaly`, `loan-approval-classical`

## Limitations

Synthetic process clouds. TDA extras optional; skips disclosed in JSON.
