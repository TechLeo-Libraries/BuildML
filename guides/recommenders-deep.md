# Recommendation systems (deep)

Session-shaped collaborative filtering and content-based recommenders with
leakage discipline, ranking metrics, and a dedicated bundle boundary.

Honesty: this is **not** a Netflix-scale recsys platform (no streaming feature
store, no multi-stage industrial cascade, no surprise/implicit required). It is
a complete Session path: fit → recommend → evaluate → bundle.

---

## What it is / is not

| Is | Is not |
|----|--------|
| User/item/interaction tables → top-K | RAG document retrieve/generate |
| Train-only CF / SVD / NMF / content | Diagnostic EDA `Recommendation` Findings |
| Precision@K, Recall@K, nDCG@K, MAP@K | Classical accuracy on engineered rows |
| `buildml.recommender_bundle.v1` | Session checkpoint payload |

EDA **Recommendation** objects (`buildml.explain.schemas.Recommendation`) are
teaching advice linked to Findings — they never rank items. RAG
(`rag_retrieve` / `rag_generate`) ranks **documents**, not catalog items from
an interaction matrix.

---

## Data model

Interactions are rows with:

1. **User id** — `user_column=` (required kwargs; not a dedicated `ColumnRole`)
2. **Item id** — `item_column=`
3. **Rating / signal** — Session `target` or `rating_column=` for explicit;
   `feedback='implicit'` for presence-only positives

Suggested roles: mark user/item as `id` or `ignore` so classical `fit()` does
not treat them as features. Optional numeric **item features** support
`method='content'`.

---

## Algorithms

### Neighborhood CF

- **item_knn** — cosine similarity on item–user vectors; score candidates from
  items the user already consumed (train history).
- **user_knn** — cosine similarity on user–item vectors; aggregate neighbor
  ratings.

### Matrix factorization

- **svd** — `TruncatedSVD` on mean-centered train matrix; score ≈ global mean +
  user factors · item factors.
- **nmf** — non-negative factorization for non-negative interactions.

### Content

- **content** — standardize numeric `item_feature_columns` on train item rows;
  user profile = rating-weighted mean of consumed item vectors; cosine score.

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
| `fit_recommender` | Train-only fit |
| `recommend` | Top-K lists |
| `evaluate_recommender` | Holdout ranking metrics |
| `save_recommender_bundle` / `load_recommender_bundle` | Persist / restore |

Walkthrough exposes `recommender_status`; AI allowlist includes the five ops.

---

## Worked method swap

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

Content path:

```python
session.fit_recommender(
    method="content",
    user_column="user_id",
    item_column="item_id",
    item_feature_columns=["f1", "f2"],
)
```

---

## Tracker

Phase 3 application systems — **recommenders** (this guide) — PASS.
Search/LTR — PASS. Knowledge graphs — next; then optimisation helpers.  
**Next:** Search / learning-to-rank (LTR) — see [ranking-deep.md](ranking-deep.md).
Distinct from LTR: recommenders use user–item interactions; LTR uses
labeled query–item feature rows (`fit_ranker`).
