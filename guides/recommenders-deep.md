# Recommendation systems (deep)

Session-shaped collaborative filtering and content-based recommenders with
leakage discipline, ranking metrics, industry backends, and a dedicated bundle
boundary.

Honesty: this is **not** a Netflix-scale recsys platform (no streaming feature
store, no multi-stage industrial cascade). With ``buildml[recommenders-industry]``
it uses real industry libraries (``implicit`` ALS/BPR, LightFM hybrid) as
defaults for implicit feedback — not a from-scratch reimplementation.

---

## What it is / is not

| Is | Is not |
|----|--------|
| User/item/interaction tables → top-K | RAG document retrieve/generate |
| Train-only CF / SVD / NMF / content | Diagnostic EDA `Recommendation` Findings |
| implicit ALS/BPR + LightFM (optional extra) | surprise / full recsys platform |
| Precision@K, Recall@K, nDCG@K, MAP@K | Classical accuracy on engineered rows |
| `buildml.recommender_bundle.v1` | Session checkpoint payload |

EDA **Recommendation** objects (`buildml.explain.schemas.Recommendation`) are
teaching advice linked to Findings — they never rank items. RAG
(`rag_retrieve` / `rag_generate`) ranks **documents**, not catalog items from
an interaction matrix.

---

## Backend catalog

Inspect honest capabilities:

```python
from buildml.recommenders import recommender_capability_matrix

print(recommender_capability_matrix())
```

| Backend | Extra | Methods | Default when |
|---------|-------|---------|--------------|
| `sklearn` | (core) | item_knn, user_knn, svd, nmf, content | explicit feedback |
| `implicit` | recommenders-industry | als, bpr | implicit feedback when installed |
| `lightfm` | recommenders-industry | lightfm | hybrid with side features |

Install industry backends:

```text
pip install 'buildml[recommenders-industry]'
```

Routing: pass ``method=`` and/or ``backend=``; omit ``method`` to get
feedback-aware defaults (ALS for implicit when ``implicit`` is installed).

---

## Data model

Interactions are rows with:

1. **User id** — `user_column=` (required kwargs; not a dedicated `ColumnRole`)
2. **Item id** — `item_column=`
3. **Rating / signal** — Session `target` or `rating_column=` for explicit;
   `feedback='implicit'` for presence-only positives

Suggested roles: mark user/item as `id` or `ignore` so classical `fit()` does
not treat them as features. Optional numeric **item features** support
`method='content'` and LightFM hybrid (`item_feature_columns=` /
`user_feature_columns=`).

---

## Algorithms

### Core (sklearn / numpy)

- **item_knn** — cosine item–user similarity
- **user_knn** — cosine user–user similarity
- **svd** / **nmf** — matrix factorization
- **content** — rating-weighted item feature profiles

### Industry (`recommenders-industry`)

- **als** / **bpr** — ``implicit`` library on sparse implicit-feedback matrices
- **lightfm** — hybrid WARP with optional user/item side features

All methods restrict candidates to the **train item catalog** (known-item
protocol). Holdout-only items are never collaborative candidates.

---

## Cold start

| Case | Behavior |
|------|----------|
| User absent from train | `cold_start='popularity'` → train popularity list; `'skip'` → empty |
| Item absent from train | Excluded from candidates and from eval relevant sets (warned) |
| Warm user, empty scores | Disclosed popularity fallback |

---

## Evaluation protocol

For each **warm** holdout user with ≥1 known (train-catalog) holdout item:

1. Relevant set = holdout items ∩ train catalog
2. Recommend top-K among train items, excluding the user's **train** history
3. Score Precision@K, Recall@K, nDCG@K, MAP@K
4. Macro-average over scored users; count cold-start users separately

Never train on test interactions (`assert_can_fit` / `assert_fit_partition`).

---

## Bundle boundary

`save_recommender_bundle` / `load_recommender_bundle` write
`buildml.recommender_bundle.v1` (`meta.json` + `recommender_plan.joblib`).

Session checkpoints do **not** embed `RecommenderPlan`. Reload workflow via
`checkpoint_load`, then `load_recommender_bundle`.

---

## API surface

| Session method | Role |
|----------------|------|
| `fit_recommender` | Train-only fit (backend/method routing) |
| `recommend` | Top-K lists |
| `evaluate_recommender` | Holdout ranking metrics |
| `save_recommender_bundle` / `load_recommender_bundle` | Persist / restore |

Walkthrough exposes `recommender_status`; AI allowlist includes the five ops.

---

## Worked examples

Implicit feedback with industry default (ALS when installed):

```python
session.fit_recommender(
    user_column="user_id",
    item_column="item_id",
    feedback="implicit",
    n_factors=32,
)
session.evaluate_recommender(k=10)
```

Explicit core + industry hybrid:

```python
session.fit_recommender(
    method="lightfm",
    user_column="user_id",
    item_column="item_id",
    item_feature_columns=["f1", "f2"],
    user_feature_columns=["age"],
)
```

Core method swap:

```python
for method in ("item_knn", "user_knn", "svd", "nmf"):
    session.fit_recommender(
        method=method,
        user_column="user_id",
        item_column="item_id",
        n_neighbors=25,
        n_factors=16,
        random_state=0,
    )
    print(method, session.evaluate_recommender(k=10).metrics)
```

Benchmark: ``python benchmarks/recommenders/ranking_quality.py``

---

## Tracker

Phase 3 application systems — **recommenders** (this guide) — PASS (R5.3 industry depth).
Search/LTR — PASS. Knowledge graphs — PASS.
**Next (R5.4):** Causal inference (dowhy/econml).
