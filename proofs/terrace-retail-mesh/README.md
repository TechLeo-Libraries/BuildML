# terrace-retail-mesh

This script composes `session.multitask`, `session.forecast`, and
`session.recommender` on synthetic merchandising, demand, and interaction
tables. It is not a product BuildML ships.

The script fits multi-output buy / high-margin heads on SKU features,
forecasts store sales with lag-ridge under an honest `time_split`, and
recommends catalog items with ALS / item-kNN on a held-out interaction
split.

## Data

Three synthetic retail surfaces stitched for harness coverage. Not a
production merchandising stack.

## Leakage

Multitask split before multi-output fit. Forecast uses `time_split`; lag
features from past only. Recommender split before ALS / item_knn fit. Test
evaluate after locks.

## What fails if leakage is ignored

Fitting multitask heads on the full SKU table invents holdout F1. Using
future sales in lag features invents forecast MAE. Fitting recommenders on
test interactions invents recall@k.

## How to run

```bash
python proofs/terrace-retail-mesh/script.py
```

## What you'll get

`results/` summary and per-stage JSON.

## Upstream

`sku-multitask-retail`, `multi-target-underwriting`, `store-sales-forecast`,
`energy-load-forecast`, `catalog-recs-implicit`,
`movie-recs-collaborative`.

## Limitations

Three synthetic retail surfaces stitched together. Not a production
merchandising stack.
