# cluster-customer-segments

You have RFM-like customer features and want segments for treatment design.
Clustering is unsupervised. Synthetic latent segments exist only so you can
read external agreement; they are not a fit target.

## Data

Synthetic RFM table (`load_customer_segments_synthetic`): license-clear.

## Leakage

Random train / validation / test before scale, PCA, or cluster fit. Scale,
PCA, and `session.unsupervised.fit` run on train only. `true_segment` has
role `ignore` (never a fit target). Test `session.unsupervised.evaluate`
runs after the model is locked.

## How to run

```bash
python proofs/cluster-customer-segments/script.py
python proofs/cluster-customer-segments/baseline_industry.py
```

## What you'll get

`results/results.json` with internal cluster quality plus external agreement
(ARI / NMI-style). `results/comparison.json` is sklearn StandardScaler ->
PCA -> KMeans with silhouette / ARI / NMI on the same split. An optional
HDBSCAN probe runs when that extra is installed.

## Limitations

Ground-truth segments are synthetic. Real CRM clusters are unlabeled.

Related: [Unsupervised quickstart](../../guides/quickstart-unsupervised.md),
[examples/unsupervised_cluster_loop.py](../../examples/unsupervised_cluster_loop.py).
