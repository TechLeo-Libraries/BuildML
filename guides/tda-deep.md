# Topological data analysis deep

```bash
pip install "buildml[tda]"
# giotto-tda Betti curves / Mapper summary: pip install "buildml[tda-industry]"
```

You want local shape: for each train row, take its k nearest train
neighbors, compute Vietoris-Rips persistence, vectorize the diagrams,
optionally put a sklearn head on those vectors. Holdout rows reuse the
frozen neighbor index and the frozen vectorizer. They do not refit
homology.

This extra is required. There is no core-only TDA fallback.
`session.tda.fit()` uses `vectorization="persistence_image"`. With
`backend=None` that image is built by **giotto** when
`buildml[tda-industry]` imported, otherwise by the **native** ripser +
persim stack. `silhouette` always routes native. `betti_curve` and
`persistence_landscape` always route giotto. You can pin `backend=`
yourself; an impossible pairing raises.

Default head is `logistic_regression`. Default `knn` is 16, homology
dims `(0, 1)`, `n_bins=20`, `standardize=True` on train,
`max_points_guard=4000`. `subsample_strategy="error"` refuses when train
is larger than that guard; pass `"random"` or `"stratified"` if you mean
to subsample. `mapper=True` is a KeplerMapper **train summary** on
giotto, not a Mapper research product.

Short on-ramp: [TDA quickstart](quickstart-tda.md). Proof:
[credit-tda-shape](../proofs/credit-tda-shape/).

## Fit, transform, predict, evaluate

Need a split and at least two numeric feature columns. Fit is train
only. `transform` and `predict` default to test. `evaluate` defaults to
validation. `head="none"` skips the supervised head: you can still
`transform`, but `predict` / `evaluate` refuse.

```python
import numpy as np
import pandas as pd

from buildml import Session

rng = np.random.default_rng(0)
a = rng.normal(size=(120, 4))
b = rng.normal(size=(120, 4)) * 1.8 + np.array([3, 0, 0, 0])
x = np.vstack([a, b])
y = np.array([0] * 120 + [1] * 120)
frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(4)])
frame["y"] = y

session = (
    Session.ingest(frame)
    .set_roles({**{f"f{i}": "feature" for i in range(4)}, "y": "target"})
    .split(test_size=0.2, validation_size=0.2, random_state=0, stratify=True)
    .scale(method="standard")
)

fit = session.tda.fit(
    vectorization="persistence_image",
    knn=12,
    n_bins=12,
    head="logistic_regression",
)
print(fit.feature_dim, fit.train_score)

feats = session.tda.transform(partition="test")
print(feats.features.shape)

ev = session.tda.evaluate(partition="validation")
print(ev.metrics)

session.tda.save_bundle("artifacts/tda_bundle")
```

What that fit actually did, in order:

1. Resolve numeric features (and optional `reduce_dimensions` components
   when `prefer_reduce_components=True`, the default).
2. Optional train mean/scale.
3. Optional subsample against `max_points_guard`.
4. Fit `NearestNeighbors` on **train** points.
5. For each train row, local cloud = `knn` train neighbors, then
   Vietoris-Rips diagrams.
6. Fit vectorizer ranges and grids from **train diagrams only**.
7. Optional sklearn head on the train topological vectors.
8. Holdout: same NN index, frozen vectorizer, frozen head.

## Backends and vectorizations

| Backend | Extra | Engine | Vectorizations |
| --- | --- | --- | --- |
| `native` | `tda` (required) | ripser Vietoris-Rips | `persistence_image`, `landscape`, `silhouette` |
| `giotto` | `tda-industry` (includes `tda`) | `gtda.homology.VietorisRipsPersistence` | `persistence_image`, `landscape`, `betti_curve`, `persistence_landscape` |

giotto-tda is marked off Python 3.13 (no reliable wheels). Native still
runs.

| `vectorization` | Where it lives | Notes |
| --- | --- | --- |
| `persistence_image` (default) | persim or gtda | Birth × persistence raster |
| `landscape` | in-tree or gtda | Layered tents on a train-fitted t-grid (`n_layers=3`) |
| `silhouette` | in-tree, native only | Weighted average of tents |
| `betti_curve` | gtda only | Industry |
| `persistence_landscape` | gtda only | Industry |

Heads: `logistic_regression`, `random_forest`, `ridge`,
`hist_gradient_boosting`, or `none`. Task is inferred when `task=None`.

`evaluate(..., compare_diagram_distances=True)` can add Wasserstein or
bottleneck distances (`diagram_distance_metric`, default
`"wasserstein"`, `diagram_distance_dim=1`). That compares diagrams. It
does not replace holdout classification or regression metrics.

## Bundles

`session.tda.save_bundle` writes `buildml.tda_bundle.v2` (`meta.json` +
`tda_plan.joblib`). v1 bundles still load. Session checkpoints do not
embed `TdaPlan`. `trusted=True` only for a file you made.

Paste: [`examples/tda_loop.py`](../examples/tda_loop.py).
Benchmark: `python benchmarks/tda/persistence_pipeline.py`.

## When it refuses

| What you see | What happened |
| --- | --- |
| `MissingExtraError` for `tda` | ripser/persim not installed; this path has no core fallback |
| `MissingExtraError` for `tda-industry` | You asked for giotto (or a giotto-only vectorization) without that extra |
| Vectorization not valid for backend | `silhouette` on giotto, or `betti_curve` on native, for example |
| No split | `fit` before `split` |
| Fewer than two numeric features | Nothing to build a point cloud from |
| Train larger than `max_points_guard` | Default `subsample_strategy="error"`; choose random/stratified or raise the guard |
| Predict / evaluate with `head="none"` | No supervised head was fitted |
| `mapper=True` on native | Mapper summary is giotto-only |

[TDA quickstart](quickstart-tda.md) ·
[credit-tda-shape](../proofs/credit-tda-shape/) ·
[Artifacts](artifacts-checkpoints-bundles.md)
