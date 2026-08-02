# Unsupervised learning (deep)

> **Install (GitHub 2.x):**
> `pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"`
> Clustering is core (sklearn). See [installation](../docs/installation.rst).

This guide covers the Session unsupervised path: train-fit clustering, holdout
assign, geometric evaluation, PCA integration via `reduce_dimensions`, and
`buildml.unsupervised_bundle.v1`. It matches the depth bar of classical / Torch /
RAG guides: leakage discipline, disclosures, artifact boundaries, and failure
modes.

**Related:** [Quickstart](quickstart-unsupervised.md) ·
[Preprocess depth](preprocess-depth.md) ·
[Artifacts](artifacts-checkpoints-bundles.md) ·
[Leakage](leakage-cv-recipes.md)

---

## What this path is (and is not)

| Is | Is not |
| --- | --- |
| Train-fitted `ClusterPlan` with holdout assign | Supervised `Session.fit` |
| Optional use of train-fitted PCA components | A second private PCA implementation |
| Internal validity + optional external ARI/NMI | Ground-truth taxonomy certification |
| Distinct unsupervised bundle | Session checkpoint / Torch / RAG bundle |
| Production-shaped Session API | EDA IsolationForest / correlation-cluster screens |

Causal claims stay out of this path (and out of EDA). A later causal API will
require explicit estimand/assumption objects — clustering labels are not causal
effects.

---

## Core loop

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

fit = session.fit_clusters(method="kmeans", n_clusters=2, random_state=0)
val = session.evaluate_clusters(partition="validation")
test = session.evaluate_clusters(
    partition="test",
    external_label_column="group_id",
)
print(fit.to_dict())
print(val.metrics, test.external_metrics)
session.explain("fit_clusters", moment="after")
```

**Leakage contract:** `fit_clusters` calls `assert_can_fit("train")`. Assign and
evaluate reuse the frozen plan. Do not `fit_predict` on concatenated partitions
outside Session and then claim holdout validity.

---

## Methods and assign strategies

| Method | Fit | Holdout assign | Notes |
| --- | --- | --- | --- |
| `kmeans` | sklearn `KMeans` on train | Native `predict` | Primary full API |
| `agglomerative` | `AgglomerativeClustering` | Nearest train centroid | Disclosed approximation |
| `dbscan` | `DBSCAN` | Nearest train core within `eps`, else `-1` | `n_clusters` is observed |

```python
session.fit_clusters(method="agglomerative", n_clusters=2, linkage="ward")
print(session.cluster_plan.assign_strategy)  # nearest_centroid

session.fit_clusters(method="dbscan", eps=0.8, min_samples=5, n_clusters=None)
print(session.cluster_plan.n_clusters, session.cluster_fit_result.warnings)
```

---

## PCA integration (do not fork)

`Session.reduce_dimensions(method="pca")` remains the dimensionality-reduction
plan. Clustering optionally consumes those components:

```python
session = (
    Session.ingest(frame)
    .set_roles({"f1": "feature", "f2": "feature", "group_id": "ignore"})
    .split(test_size=0.25, random_state=0)
    .scale(method="standard")
    .reduce_dimensions(method="pca", n_components=2, prefix="pc")
)
session.fit_clusters(method="kmeans", n_clusters=2, prefer_reduce_components=True)
assert session.cluster_fit_result.used_reduce_components
# Explained variance is still unsupervised — not cluster quality:
print(session.reduce_plan.to_dict()["total_explained_variance"])
```

Set `prefer_reduce_components=False` or pass explicit `columns=` to cluster raw
scaled features instead. Fold-local PCA inside CV remains
`PreprocessRecipe(reduce="pca")` for **supervised** selection — unsupervised
clustering is a Session-global plan path today (honest limit).

---

## Assign and attach

```python
holdout = session.assign_clusters(partition="test")
print(holdout.labels[:10], holdout.n_noise)

# Attach labels to the full frame (aligned write):
session.assign_clusters(partition="all", attach=True)
assert "cluster_id" in session.dataset.columns
```

`attach=True` requires `partition="all"` so row alignment cannot silently drift.

---

## Evaluation honesty

Internal metrics describe cohesion/separation under the feature geometry:

- `silhouette` (optionally subsampled via `sample_size`)
- `calinski_harabasz`
- `davies_bouldin`
- `noise_rate` when DBSCAN produces `-1`

Optional `external_label_column` adds ARI / NMI **after** fit. Those labels are
never used to train the clusterer. Agreement ≠ causal structure ≠ business ROI.

Default `evaluate_clusters(partition="validation")` falls back to `test` when
no validation partition was carved.

---

## Bundles vs checkpoints

| Artifact | Contains | Does not |
| --- | --- | --- |
| `save_unsupervised_bundle` | `ClusterPlan`, meta, disclosures | Dataset, splits, classical estimator |
| `checkpoint_save` | data, roles, splits, history, classical preprocess plans | ClusterPlan / Torch / RAG |
| `reduce_dimensions` plan | Inside classical `plans.joblib` when checkpointed | Cluster labels |

```python
path = session.save_unsupervised_bundle("artifacts/clusters")
# Later: re-attach features, then:
other = Session.ingest(...).set_roles(...).split(...).scale(...)
other.load_unsupervised_bundle(path)
other.assign_clusters(partition="test")
```

Schema: `buildml.unsupervised_bundle.v1`. See
`buildml.unsupervised.checkpoint.CHECKPOINT_BOUNDARY`.

---

## Failure modes

- Fitting without a split → `LeakageError` / fit refusal.
- Nulls in features → impute first; scale before distance methods.
- `n_clusters` > `n_train` → validation error.
- DBSCAN with too-small `eps` → all noise; read warnings.
- Expecting `checkpoint_load` to restore `cluster_plan` → it will not.
- Publishing silhouette as “accuracy” → teaching anti-pattern (catalog + concepts).

---

## Phase coverage note

Phase 1 depth-first order (**complete**): unsupervised (this guide) → ensembles →
AutoML → forecasting → anomaly/fraud (see [Anomaly deep](anomaly-deep.md)).
Later paradigms (semi/self-supervised, federated,
causal path, GNNs, RL, …) wait until Phase 1 items hit this bar. Explicit
non-goals (TTS, swarm zoo, digital twins, AV/robotics product stacks, full COCO
detection suite, …) stay undocumented as product surfaces.

---

## Teaching surface

```python
from buildml.explain.catalog import OPERATION_CATALOG

for name in (
    "fit_clusters",
    "assign_clusters",
    "evaluate_clusters",
    "save_unsupervised_bundle",
    "load_unsupervised_bundle",
):
    assert name in OPERATION_CATALOG

session.explain("evaluate_clusters", moment="before")
```

Concepts: `unsupervised-train-fit-holdout-assign`, `cluster-validity-not-truth`,
`pca-cluster-integration`, `unsupervised-bundle-boundary`.
