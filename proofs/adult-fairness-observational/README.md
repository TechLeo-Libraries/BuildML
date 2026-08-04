# adult-fairness-observational

## Business purpose

Observational group-fairness gaps (demographic parity / disparate impact /
equalized odds) on a **real public** credit/income table when available.

## Data source

**REAL_PUBLIC_DATASET** loader preference (see `load_fairness_public_dataset`):

1. OpenML Adult (`data_id=1590`, sensitive=`sex`) — network/cache
2. OpenML German Credit `credit-g` (sensitive stand-in from `personal_status`)
3. Offline CI fallback: sklearn breast cancer + **disclosed constructed**
   `radius_intensity_proxy` (median-split of `mean_radius`) — **not** a
   protected demographic class; see `data.proxy_disclosure` in results JSON

Adult rows are capped at 2500 for CI runtime when the full table loads.

## Leakage controls

- Stratified split before fit
- Sensitive column `role=ignore` (not a predictor)
- Fairness metrics on holdout test predictions only

## BuildML API steps

1. Load public fairness table (+ provenance meta)
2. `Session.ingest` → roles → stratified `split` → `impute` → `scale` → `fit`
3. `session.evaluate(test)` (perfect-score refusal gate)
4. `session.fairness.evaluate(sensitive_column=..., partition="test")`

## Metrics

Holdout classification metrics plus observational fairness gaps. Refuses
perfect accuracy/F1/ROC-AUC ≥ 1.0.

## How to run

```bash
python proofs/adult-fairness-observational/script.py
```

## Limitations

Observational only; not a legal audit. Offline CI may exercise the disclosed
proxy path when OpenML is unavailable.
