# adult-fairness-observational

You have a public income / credit-style table and a sensitive column. You
want observational group-fairness gaps (demographic parity / disparate
impact / equalized odds) on holdout predictions, not a legal audit.

## Data

**REAL_PUBLIC_DATASET** loader preference (see `load_fairness_public_dataset`):

1. OpenML Adult (`data_id=1590`, sensitive=`sex`) -- network/cache
2. OpenML German Credit `credit-g` (sensitive stand-in from `personal_status`)
3. Offline CI fallback: sklearn breast cancer plus a disclosed constructed
   `radius_intensity_proxy` (median-split of `mean_radius`). That proxy is
   not a protected demographic class; see `data.proxy_disclosure` in results
   JSON.

Adult rows are capped at 2500 for CI runtime when the full table loads.
Provenance fields are written under `results/results.json` -> `data`.

## Leakage

Stratified split before fit. The sensitive column has `role=ignore` (not a
predictor). Fairness metrics are computed on holdout test predictions only.
The script refuses perfect accuracy/F1/ROC-AUC >= 1.0.

## How to run

```bash
python proofs/adult-fairness-observational/script.py
python proofs/adult-fairness-observational/baseline_industry.py
```

## What you'll get

`results/results.json` with holdout classification metrics plus
observational fairness gaps. `results/comparison.json` is a sklearn
`Pipeline` (`SimpleImputer` + `StandardScaler` + `LogisticRegression`) plus
holdout group selection rates (demographic parity / disparate impact) on
the same `SplitPlan`. Observational only.

## Limitations

Observational only; not a legal audit. Offline CI may exercise the disclosed
proxy path when OpenML is unavailable.

Related: [Fairness quickstart](../../guides/quickstart-fairness.md).
