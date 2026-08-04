# cluster-customer-segments

## Business purpose

Segment customers from RFM-like features for marketing treatment design.
Clustering is unsupervised; synthetic latent segments are used only for
external validation metrics.

## Data source

Synthetic RFM table (`load_customer_segments_synthetic`): license-clear.

## Leakage controls

- Random train / validation / test before scale / PCA / cluster fit
- Scale + PCA + `session.unsupervised.fit` on **train** only
- `true_segment` role = `ignore` (never a fit target)
- Test `session.unsupervised.evaluate` after the model is locked

## BuildML API steps

1. `ingest` → roles → `split` → `scale` → `reduce_dimensions(pca)`
2. `session.unsupervised.fit(method="kmeans")`
3. `session.unsupervised.evaluate` on validation/test with external labels
4. Optional HDBSCAN probe when installed
5. `session.unsupervised.save_bundle`

## Metrics

Internal cluster quality + external agreement (ARI/NMI-style: see JSON).

## Industry comparison (Tier C)

Industry twin: `baseline_industry.py` runs sklearn StandardScaler→PCA→KMeans with silhouette / ARI / NMI on the same split (`results/comparison.json`).
## Limitations

Ground-truth segments are synthetic; real CRM clusters are unlabeled.
