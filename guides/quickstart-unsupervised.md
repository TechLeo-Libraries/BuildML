# Unsupervised quickstart

```bash
pip install buildml
# HDBSCAN / UMAP: pip install "buildml[unsupervised]"
# DEC / IDEC: pip install "buildml[torch]"
```

Cluster on the same Session. No target is required. Fit is train-only;
`assign` and `evaluate` need a plan first. Default is sklearn KMeans
(n_clusters=8). PCA stays on `session.reduce_dimensions`; you can cluster
those train-fitted components.

This is not the EDA IsolationForest screen. That stays descriptive.

[Unsupervised deep](unsupervised-deep.md) ·
[cluster-customer-segments](../proofs/cluster-customer-segments/)

```python
import numpy as np
import pandas as pd

from buildml import Session

rng = np.random.default_rng(0)
frame = pd.DataFrame(
    {
        "x": np.concatenate([rng.normal(0, 0.4, 40), rng.normal(3, 0.4, 40)]),
        "y": np.concatenate([rng.normal(0, 0.4, 40), rng.normal(3, 0.4, 40)]),
        "segment": [0] * 40 + [1] * 40,  # optional reference labels for ARI/NMI only
    }
)

session = (
    Session.ingest(frame)
    .set_roles({"x": "feature", "y": "feature", "segment": "ignore"})
    .split(test_size=0.25, random_state=0)
    .scale(method="standard")
)

fit = session.unsupervised.fit(method="kmeans", n_clusters=2)
print(fit.cluster_sizes, fit.assign_strategy)

labels = session.unsupervised.assign(partition="test")
print(labels.n_rows, set(labels.labels))

metrics = session.unsupervised.evaluate(
    partition="test",
    external_label_column="segment",  # optional agreement check: not used in fit
)
print(metrics.metrics, metrics.external_metrics)

bundle = session.unsupervised.save_bundle("artifacts/unsupervised_bundle")
# Bundle stores the ClusterPlan only: reload features/splits via checkpoint or re-ingest.
fresh = Session.ingest(session.to_pandas()).set_roles(
    {"x": "feature", "y": "feature", "segment": "ignore"}
)
fresh.split(test_size=0.25, random_state=0).scale(method="standard")
fresh.unsupervised.load_bundle(bundle)
again = fresh.unsupervised.assign(partition="test")
print(again.labels[:5])
```

PCA then cluster (same ReducePlan: no forked PCA):

```python
session = (
    Session.ingest(frame)
    .set_roles({"x": "feature", "y": "feature", "segment": "ignore"})
    .split(test_size=0.25, random_state=0)
    .scale(method="standard")
    .reduce_dimensions(method="pca", n_components=2, prefix="pc")
)
session.unsupervised.fit(method="kmeans", n_clusters=2)  # prefers pc_* columns
assert session.unsupervised.fit_result.used_reduce_components
```

Explain catalog coverage:

```python
print(session.explain("fit_clusters", moment="before").operation)
print(session.explain("evaluate_clusters", moment="before").concept_links)
```

## Honesty limits

- Internal metrics (silhouette, Calinski–Harabasz, Davies–Bouldin) measure
  **geometry**, not ground-truth taxonomy or business value.
- Agglomerative holdout assign is nearest-centroid; DBSCAN holdout assign is
  nearest-core within `eps` (else noise `-1`): both are disclosed approximations.
- Unsupervised bundles are complementary to Session checkpoints (data/splits/
  classical plans) and to Torch/RAG bundles: not interchangeable.
- Dedicated anomaly/fraud scoring is ``session.anomaly.fit`` (separate Session path);
  do not treat clustering or EDA IsolationForest as that product.
