# Terrace Retail Mesh

**Tier B** cross-domain product proof: multitask SKU heads + chronological
demand forecast + collaborative recommenders for synthetic retail.

## Product narrative

Terrace joins merchandising, demand, and personalization surfaces:

1. Fits multi-output buy / high-margin heads on SKU features
2. Forecasts store sales with lag-ridge under an honest `time_split`
3. Recommends catalog items with ALS / item-kNN on a held-out interaction split

## Status

Run `script.py`. Outputs land under `results/` (summary and stage JSON)..

## How to run

```bash
python proofs\terrace-retail-mesh\script.py
```

## Leakage controls

- Multitask split before multi-output fit
- Forecast uses `time_split`; lag features from past only
- Recommender split before ALS / item_knn fit
- Test evaluate after locks

## What fails if leakage is ignored

- Fitting multitask heads on the full SKU table invents holdout F1
- Using future sales in lag features invents forecast MAE
- Fitting recommenders on test interactions invents recall@k

## Upstream Tier A building blocks

`sku-multitask-retail`, `multi-target-underwriting`, `store-sales-forecast`,
`energy-load-forecast`, `catalog-recs-implicit`, `movie-recs-collaborative`

## Limitations

Three synthetic retail surfaces stitched into one product narrative: not a
production merchandising stack.
