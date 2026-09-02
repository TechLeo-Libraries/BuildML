# Unsupervised learning (deep)

```bash
pip install buildml
# HDBSCAN / UMAP: pip install "buildml[unsupervised]"
# DEC / IDEC: pip install "buildml[torch]"
```

Cluster on the same Session as classical work. No target is required. Fit
is train-only. `assign` and `evaluate` need a plan first. Default is
sklearn KMeans (`n_clusters=8`). PCA stays on `session.reduce_dimensions`;
you can cluster those train-fitted components.

This is not the EDA IsolationForest screen. That stays descriptive. It is
not `session.anomaly` and not a ground-truth taxonomy.

Short on-ramp: [unsupervised quickstart](quickstart-unsupervised.md).

## A first loop

```python
import numpy as np
import pandas as pd
from buildml import Session

rng = np.random.default_rng(1)
a = rng.normal([0, 0], 0.35, size=(60, 2))
b = rng.normal([2.5, 2.5], 0.35, size=(60, 2))
frame = pd.DataFrame(np.vstack([a, b]), columns=["f1", "f2"])
frame["group_id"] = [0] * 60 + [1] * 60

session = (
    Session.ingest(frame)
    .set_roles({"f1": "feature", "f2": "feature", "group_id": "ignore"})
    .split(test_size=0.2, validation_size=0.2, random_state=0)
    .scale(method="standard")
)

fit = session.unsupervised.fit(method="kmeans", n_clusters=2, random_state=0)
val = session.unsupervised.evaluate(partition="validation")
test = session.unsupervised.evaluate(
    partition="test",
    external_label_column="group_id",
)
print(fit.to_dict())
print(val.metrics, test.external_metrics)
```

`session.unsupervised.fit` calls `assert_can_fit("train")`. Assign and
evaluate reuse the frozen plan. Do not `fit_predict` on concatenated
partitions outside Session and then claim holdout validity.

## Methods

| Method | Backend | Holdout assign | Notes |
| --- | --- | --- | --- |
| `kmeans` | sklearn | Native `predict` | `auto_k` elbow on train |
| `agglomerative` | sklearn | Nearest train centroid | Disclosed approximation |
| `dbscan` | sklearn | Nearest core within `eps` | Density-driven k; else `-1` |
| `gmm` | sklearn | Native `predict` | BIC model selection |
| `hdbscan` | hdbscan | `approximate_predict` / nearest core | Needs `[unsupervised]` |
| `spectral` | sklearn | Nearest centroid | Transductive on train |
| `optics` | sklearn | Nearest centroid | Transductive |
| `mean_shift` | sklearn | Nearest centroid | Bandwidth-driven k |
| `dec` / `idec` | Torch | Native encoder assign | Needs `[torch]` |

```python
session.unsupervised.fit(method="agglomerative", n_clusters=2, linkage="ward")
print(session.unsupervised.plan.assign_strategy)  # nearest_centroid

session.unsupervised.fit(method="dbscan", eps=0.8, min_samples=5, n_clusters=None)
print(session.unsupervised.plan.n_clusters, session.unsupervised.fit_result.warnings)
```

## PCA stays on `reduce_dimensions`

Do not fork a second PCA. Cluster the train-fitted components:

```python
session = (
    Session.ingest(frame)
    .set_roles({"f1": "feature", "f2": "feature", "group_id": "ignore"})
    .split(test_size=0.25, random_state=0)
    .scale(method="standard")
    .reduce_dimensions(method="pca", n_components=2, prefix="pc")
)
session.unsupervised.fit(method="kmeans", n_clusters=2, prefer_reduce_components=True)
assert session.unsupervised.fit_result.used_reduce_components
```

Set `prefer_reduce_components=False` or pass `columns=` to cluster raw
scaled features. Fold-local PCA inside CV remains
`PreprocessRecipe(reduce="pca")` on the **supervised** path. Clustering
itself is a Session-global plan.

| Reduce method | Extra | Holdout transform |
| --- | --- | --- |
| `pca` | core | Native |
| `umap` | `[unsupervised]` | Native `transform` |
| `tsne` | core | Nearest-neighbor train embed transfer (disclosed) |

## Assign

```python
holdout = session.unsupervised.assign(partition="test")
print(holdout.labels[:10], holdout.n_noise)

session.unsupervised.assign(partition="all", attach=True)
assert "cluster_id" in session.dataset.columns
```

`attach=True` requires `partition="all"` so row alignment cannot drift.

## Evaluate

Internal metrics describe cohesion under the feature geometry:
silhouette, Calinski-Harabasz, Davies-Bouldin, and `noise_rate` when
DBSCAN produces `-1`. Optional bootstrap stability
(`compute_stability=True`) and an elbow curve (`compute_elbow=True`)
stay on train.

`external_label_column` adds ARI / NMI **after** fit. Those labels never
train the clusterer. Agreement is not a causal structure and not ROI.

Default `evaluate(partition="validation")` falls back to `test` when no
validation partition was carved.

## Bundle

`buildml.unsupervised_bundle.v2` (v1 loadable) stores the `ClusterPlan`.
It does not store the dataset, the split, or a classical estimator.
`checkpoint_load` will not restore `session.unsupervised.plan`.

```python
path = session.unsupervised.save_bundle("artifacts/clusters")
other = Session.ingest(...).set_roles(...).split(...).scale(...)
other.unsupervised.load_bundle(path)
other.unsupervised.assign(partition="test")
```

## What usually goes wrong

- Fit without a split: `LeakageError`.
- Nulls in features: impute first; scale before distance methods.
- `n_clusters` larger than train: `ValidationError`.
- DBSCAN with too-small `eps`: all noise; read the warnings.
- Publishing silhouette as accuracy.

[Preprocess](preprocess-depth.md) · [Artifacts](artifacts-checkpoints-bundles.md)
