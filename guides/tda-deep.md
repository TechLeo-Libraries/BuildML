# TDA deep guide

Session-shaped persistent homology for tabular workflows: native and industry backends.

## Backend catalog

| Backend | Extra | PH engine | Vectorizations |
|---------|-------|-----------|----------------|
| **native** (default when only `tda` installed) | `buildml[tda]` | ripser VR | persistence_image, landscape, silhouette |
| **giotto** (default when industry installed) | `buildml[tda-industry]` | gtda VietorisRipsPersistence | betti_curve, persistence_image, persistence_landscape, landscape |

```python
Session.tda_capability_matrix()
```

## Why ripser + persim (native) vs giotto-tda (industry)

| Option | Trade-off |
|--------|-----------|
| **ripser + persim** (`buildml[tda]`) | Light, standard VR backend + persistence images; landscapes/silhouettes in-tree |
| **giotto-tda** (`buildml[tda-industry]`) | Sklearn-style PH pipelines, Betti curves, gtda vectorizers, optional KeplerMapper train summary |
| Multiple half-wired stacks | Rejected: two honest backends behind one Session API |

`import buildml` never imports ripser/persim/gtda. Missing installs raise
`MissingExtraError("tda", ...)` or `MissingExtraError("tda-industry", ...)`.

## Pipeline

1. Resolve ≥2 numeric feature columns (optional `reduce_dimensions` components).
2. Optional train mean/scale standardization.
3. Optional train subsample when above `max_points_guard` (`subsample_strategy`: error | random | stratified).
4. Fit `NearestNeighbors` on **train** points.
5. For each train row: local cloud = `knn` train neighbors → Vietoris–Rips diagrams.
6. Fit vectorizer ranges/grids from **train diagrams only**.
7. Optional sklearn head on train topological vectors.
8. Holdout: same NN index + frozen vectorizer (+ head); never refit.

## APIs

| Method | Role |
|--------|------|
| `fit_tda` | Train PH + vectorizer ± head (`backend=`, `mapper=` on giotto) |
| `transform_tda` | Topological feature matrix |
| `predict_tda` | Head predictions |
| `evaluate_tda` | Holdout metrics; optional Wasserstein/bottleneck diagram distances |
| `tda_capability_matrix` | Honest backend / vectorization matrix |
| `save_tda_bundle` / `load_tda_bundle` | `buildml.tda_bundle.v2` (v1 loadable) |

## Bundle boundary

`meta.json` + `tda_plan.joblib`. Session checkpoints do **not** embed
`TdaPlan`. Reload workflow via `checkpoint_load`, then `load_tda_bundle`.

## What this is not

- Full Mapper research / interactive visualization suite (train summary only on giotto)
- Every TDA paper (multiparameter, zigzag, sheaves, …)
- Domain-specific credit-risk product surface

## Benchmark

`benchmarks/tda/persistence_pipeline.py` compares native vs giotto vectorizations.

## Teaching surfaces

Concepts: `tda-persistent-homology`, `tda-vectorization`, `tda-supervised-head`,
`tda-bundle-boundary`, `tda-extra-boundary`, `tda-giotto-backend`. Overlays +
AI allowlist + walkthrough `tda_status` are wired.

