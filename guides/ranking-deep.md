# Learning-to-rank (tabular search ranking) — deep guide

> Core Session path with sklearn fallback; industry GBDT rankers via
> `buildml[ranking-industry]`; torch listwise-lite via `buildml[torch]`.
> Quickstart: [quickstart-ranking.md](quickstart-ranking.md).

## What this is (and is not)

**Is:** a Session-shaped tabular LTR loop —

1. Ingest judgment rows `(query_id, item_id, features…, relevance)`
2. Prefer `group_split` on the query id
3. `fit_ranker` on **train only** (industry backend default when installed)
4. `rank` to order candidates per query
5. `evaluate_ranker` with graded nDCG@K, MAP@K, MRR@K
6. `save_ranker_bundle` / `load_ranker_bundle`

**Is not:**

- A search-engine product (no crawler, inverted index, or serving stack)
- RAG (`rag_retrieve` / chunk embeddings / `rag_evaluate`) — see [rag-deep.md](rag-deep.md)
- Recommenders (`fit_recommender` user–item CF) — see [recommenders-deep.md](recommenders-deep.md)
- Hyperparameter `evolutionary_search` / classical model search

Metric names may overlap (nDCG, MRR) across RAG / recommenders / LTR; the
**protocol** differs. Do not mix `rag_evaluate`, `evaluate_recommender`, and
`evaluate_ranker` numbers.

Inspect installed backends:

```python
from buildml import Session
Session.ranking_capability_matrix()
```

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

## Backends and algorithms

### Sklearn fallback (`backend='sklearn'`)

Always available — no extra required.

| `method` | Estimator |
|----------|-----------|
| `pointwise` | Ridge (default) or HistGradientBoostingRegressor (`pointwise_estimator='hgb'`) |
| `pairwise` | RankSVM-lite: LinearSVC on within-query feature differences |

Features are standardized on **train** means/scales only.

### Industry GBDT rankers (`backend='industry'`, `buildml[ranking-industry]`)

**Default backend when LightGBM/XGBoost/CatBoost are installed.**

| `method` | Library | Objective |
|----------|---------|-----------|
| `lambdarank_lgbm` | LightGBM | LambdaRank (ndcg metric) |
| `rank_ndcg_xgb` | XGBoost | `rank:ndcg` |
| `yetirank_catboost` | CatBoost | YetiRank |

Query groups are sorted contiguously for listwise training; inference scores
each row independently then sorts within query.

### Torch listwise-lite (`backend='torch'`, `buildml[torch]`)

| `method` | Idea |
|----------|------|
| `listwise_lite` | Small MLP + per-query softmax cross-entropy on normalized relevance grades (ListNet-style lite) |

---

## Metrics

Macro-averaged over holdout queries that have ≥1 relevant item
(`relevance > relevance_threshold`):

| Metric | Definition in BuildML LTR |
|--------|---------------------------|
| `ndcg_at_k` | Graded nDCG with gain `2^rel − 1` |
| `map_at_k` | Mean average precision (binaryized grades) |
| `mrr_at_k` | Mean reciprocal rank of first relevant |

These are **judgment-table** metrics. RAG chunk nDCG and recommender known-item
nDCG use different candidate sets and protocols.

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
    backend="sklearn",
    method="pointwise",
    query_column="query_id",
    item_column="item_id",
)
pw = session.evaluate_ranker(k=5).metrics

# Industry default when buildml[ranking-industry] installed:
session.fit_ranker(
    backend="industry",
    method="lambdarank_lgbm",
    query_column="query_id",
    item_column="item_id",
)
gbdt = session.evaluate_ranker(k=5).metrics
print("pointwise", pw, "lambdarank", gbdt)
```

Benchmark: `python benchmarks/ranking/ndcg_lift.py` compares industry default
vs sklearn pointwise on synthetic judgments.

---

## Tracker

Phase 3 application systems — depth-first:

1. Recommenders — **PASS**
2. Search / LTR — **PASS** (R6.8 industry depth)
3. Knowledge graphs — **PASS**
4. Optimisation / decision helpers — next (R6.9)
