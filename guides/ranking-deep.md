# Learning-to-rank (tabular search ranking) — deep guide

> Core Session path. No LightGBM/XGBoost required.
> Quickstart: [quickstart-ranking.md](quickstart-ranking.md).

## What this is (and is not)

**Is:** a Session-shaped tabular LTR loop —

1. Ingest judgment rows `(query_id, item_id, features…, relevance)`
2. Prefer `group_split` on the query id
3. `fit_ranker` on **train only**
4. `rank` to order candidates per query
5. `evaluate_ranker` with graded nDCG@K, MAP@K, MRR@K
6. `save_ranker_bundle` / `load_ranker_bundle`

**Is not:**

- A search-engine product (no crawler, inverted index, or serving stack)
- RAG (`rag_retrieve` / chunk embeddings / generate) — see [rag-deep.md](rag-deep.md)
- Recommenders (`fit_recommender` user–item CF) — see [recommenders-deep.md](recommenders-deep.md)
- Hyperparameter `evolutionary_search` / classical model search

Metric names may overlap (nDCG, MRR) across RAG / recommenders / LTR; the
**protocol** differs. Do not mix `rag_evaluate`, `evaluate_recommender`, and
`evaluate_ranker` numbers.

---

## Data model

| Column | Role suggestion | Meaning |
|--------|-----------------|---------|
| `query_column` | `group` (preferred) or `id` | Query / request id |
| `item_column` | `id` or `ignore` | Item or document id |
| `relevance_column` | `target` | Graded or binary relevance |
| feature columns | `feature` | Numeric query–item features |

Each **row** is one labeled judgment. Multiple rows share a `query_id`.

---

## Leakage discipline

- `fit_ranker` calls `assert_can_fit("train")` — holdout rows never update weights.
- Prefer `Session.group_split(group_column=query_column)` so **no query id**
  appears in more than one partition (test labels cannot leak into train).
- Random row `split` is allowed but **disclosed with warnings** when query ids
  overlap partitions — ranking structure can still leak even if fit ignores
  holdout rows.
- At eval time, relevance labels are used only to **score** frozen rankings,
  never to refit.

---

## Algorithms

### Pointwise (`method='pointwise'`)

- `pointwise_estimator='ridge'` (default): `sklearn.linear_model.Ridge`
- `pointwise_estimator='hgb'`: `HistGradientBoostingRegressor`
- Features are standardized on **train** means/scales only.
- Inference score = predicted relevance; sort descending within query.

### Pairwise RankSVM-lite (`method='pairwise'`)

- Sample within-query pairs with distinct grades (budget
  `max_pairs_per_query`).
- Build difference features `x_i − x_j` and train `LinearSVC`.
- Inference score = `w·x` (linear scoring function).
- Honesty: RankSVM-**lite**, not LambdaMART / LightGBM ranker. Optional
  boosted listwise extras are out of scope until a complete path is justified.

---

## Metrics

Macro-averaged over holdout queries that have ≥1 relevant item
(`relevance > relevance_threshold`):

| Metric | Definition in BuildML LTR |
|--------|---------------------------|
| `ndcg_at_k` | Graded nDCG with gain `2^rel − 1` |
| `map_at_k` | Mean average precision (binaryized grades) |
| `mrr_at_k` | Mean reciprocal rank of first relevant |

---

## Bundles

Schema `buildml.ranker_bundle.v1`:

- `meta.json` — format, plan summary, optional fit/eval/rank summaries
- `ranker_plan.joblib` — `RankerPlan` + estimator + standardization

Session checkpoints do **not** embed `RankerPlan`. See
[artifacts-checkpoints-bundles.md](artifacts-checkpoints-bundles.md).

---

## Worked comparison sketch

```python
# After group_split + roles...
session.fit_ranker(
    method="pointwise",
    query_column="query_id",
    item_column="item_id",
)
pw = session.evaluate_ranker(k=5).metrics

session.fit_ranker(
    method="pairwise",
    query_column="query_id",
    item_column="item_id",
    max_pairs_per_query=60,
)
pr = session.evaluate_ranker(k=5).metrics
print(pw, pr)
```

---

## Tracker

Phase 3 application systems — depth-first:

1. Recommenders — **PASS**
2. Search / LTR — **this module**
3. Knowledge graphs — next after LTR PASS
4. Then optimisation / decision helpers, synthetic-data systems
