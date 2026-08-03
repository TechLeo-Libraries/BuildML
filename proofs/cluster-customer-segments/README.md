# cluster-customer-segments

## Business purpose

Segment customers from RFM-like features for marketing treatment design.
Clustering is unsupervised; synthetic latent segments are used only for
external validation metrics.

## Data source

Synthetic RFM table (`load_customer_segments_synthetic`): license-clear.

## Leakage controls

- Random train / validation / test before scale / PCA / cluster fit
- Scale + PCA + `fit_clusters` on **train** only
- `true_segment` role = `ignore` (never a fit target)
- Test `evaluate_clusters` after the model is locked

## BuildML API steps

1. `ingest` → roles → `split` → `scale` → `reduce_dimensions(pca)`
2. `fit_clusters(method="kmeans")`
3. `evaluate_clusters` on validation/test with external labels
4. Optional HDBSCAN probe when installed
5. `save_unsupervised_bundle`

## Metrics

Internal cluster quality + external agreement (ARI/NMI-style: see JSON).

## Industry comparison (Tier C)

Filled: `baseline_industry.py` runs sklearn StandardScaler→PCA→KMeans with silhouette / ARI / NMI on the same split (`results/comparison.json`).
## Limitations

Ground-truth segments are synthetic; real CRM clusters are unlabeled.
