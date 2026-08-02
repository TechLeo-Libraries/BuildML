# TDA deep guide

Session-shaped persistent homology for tabular workflows.

## Why ripser + persim

| Option | Trade-off |
|--------|-----------|
| **ripser + persim** (chosen) | Light, standard VR backend + persistence images; landscapes/silhouettes in-tree |
| giotto-tda | Richer sklearn pipelines, heavier transitive deps |
| Multiple half-wired stacks | Rejected — one coherent path |

`import buildml` never imports ripser/persim. Missing installs raise
`MissingExtraError("tda", ...)`.

## Pipeline

1. Resolve ≥2 numeric feature columns (optional `reduce_dimensions` components).
2. Optional train mean/scale standardization.
3. Fit `NearestNeighbors` on **train** points.
4. For each train row: local cloud = `knn` train neighbors → Vietoris–Rips
   diagrams via ripser (`maxdim`, optional `thresh`).
5. Fit vectorizer ranges/grids from **train diagrams only**.
6. Optional sklearn head on train topological vectors.
7. Holdout: same NN index + frozen vectorizer (+ head); never refit.

## APIs

| Method | Role |
|--------|------|
| `fit_tda` | Train PH + vectorizer ± head |
| `transform_tda` | Topological feature matrix |
| `predict_tda` | Head predictions |
| `evaluate_tda` | Holdout metrics |
| `save_tda_bundle` / `load_tda_bundle` | `buildml.tda_bundle.v1` |

## Bundle boundary

`meta.json` + `tda_plan.joblib`. Session checkpoints do **not** embed
`TdaPlan`. Reload workflow via `checkpoint_load`, then `load_tda_bundle`.

## What this is not

- Full Mapper research / visualization suite (deep PH path is the product bar)
- Every TDA paper (multiparameter, zigzag, sheaves, …)
- Domain-specific credit-risk product surface
- A reason to pull giotto-tda into core

## Teaching surfaces

Concepts: `tda-persistent-homology`, `tda-vectorization`, `tda-supervised-head`,
`tda-bundle-boundary`, `tda-extra-boundary`. Overlays + AI allowlist +
walkthrough `tda_status` are wired.

## Tracker

Phase 2 TDA → **PASS**. Phase 3 starts at **recommendation systems**
(depth-first). After recommenders PASS: LTR, knowledge graphs,
optimisation/decision helpers, synthetic-data. NLP/CV deepenings if still
partial.
