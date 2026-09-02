# Learning-to-rank deep

```bash
pip install buildml
# LightGBM / XGBoost / CatBoost rankers: pip install "buildml[ranking-industry]"
# ListNet-style MLP: pip install "buildml[torch]"
```

You have labeled query-item rows (a judgment table) and you want a
ranker that orders candidates for a query. `query_column` and
`item_column` are required. They are not inferred from roles.
`relevance_column` defaults to the Session target when you have one.

Prefer `group_split` on the query so a query id does not appear in more
than one partition. Random row `split` is allowed; overlapping query ids
are disclosed with warnings because ranking structure can still leak
even if fit ignores holdout rows.

`session.ranking.fit(query_column=..., item_column=...)` with
`backend=None` and `method=None` picks **LightGBM LambdaRank** when
`buildml[ranking-industry]` imported (then XGBoost `rank:ndcg`, then
CatBoost YetiRank). On a core install it is sklearn **pointwise** Ridge.
That is one of the surfaces where omitting both knobs can select
industry.

This is tabular LTR. It is not a search engine, not
`session.rag.retrieve`, and not `session.recommender`.

Short on-ramp: [ranking quickstart](quickstart-ranking.md). Proof:
[search-relevance-ltr](../proofs/search-relevance-ltr/).

## Fit, rank, evaluate

Each row is one labeled judgment. Several rows share a query id. Mark
the query `group` (preferred for `group_split`) and the item `id` or
`ignore`. Features are numeric query-item columns. Relevance must be
numeric (graded or binary).

Fit is train only. Features are standardized on train means and scales.
`rank` defaults to the **test** partition when you pass neither
`partition` nor `query_ids` (`k=10`). `evaluate` defaults to test as
well. Relevance labels at eval time score frozen rankings. They do not
refit.

```python
import numpy as np
import pandas as pd

from buildml import Session

rng = np.random.default_rng(0)
rows = []
for q in range(40):
    for item in range(8):
        f1 = float(rng.normal(q % 5, 1.0))
        f2 = float(rng.normal(item, 1.0))
        rel = float(max(0, int(3 - abs(f1 - (q % 5)) + (item % 3 == 0))))
        rows.append(
            {
                "query_id": f"q{q}",
                "item_id": f"i{item}",
                "f1": f1,
                "f2": f2,
                "bm25": float(rng.random()),
                "relevance": rel,
            }
        )
frame = pd.DataFrame(rows)

session = (
    Session.ingest(frame)
    .set_roles(
        {
            "query_id": "group",
            "item_id": "id",
            "relevance": "target",
            "f1": "feature",
            "f2": "feature",
            "bm25": "feature",
        }
    )
    .group_split(test_size=0.25, validation_size=0.15, random_state=0)
)

fit = session.ranking.fit(
    backend="sklearn",
    method="pointwise",
    query_column="query_id",
    item_column="item_id",
    pointwise_estimator="ridge",
)
print(fit.backend, fit.method)

ranked = session.ranking.rank(partition="test", k=5)
print(ranked.n_queries)

ev = session.ranking.evaluate(partition="test", k=5)
print(ev.metrics)

session.ranking.save_bundle("artifacts/ranker_bundle")
```

Or omit `backend` and `method` to take the industry default when that
extra imported:

```python
session.ranking.fit(query_column="query_id", item_column="item_id")
```

Need at least four train rows. `query_column` and `item_column` must
differ and must exist on the frame.

## Backends

| Backend | Extra | Methods |
| --- | --- | --- |
| `sklearn` | core | `pointwise`, `pairwise` |
| `industry` | `ranking-industry` | `lambdarank_lgbm`, `rank_ndcg_xgb`, `yetirank_catboost` |
| `torch` | `torch` | `listwise_lite` |

Aliases `lambdarank`, `rank_ndcg`, and `yetirank` resolve to the
canonical names above.

### Sklearn

Always available. `pointwise` is Ridge (`pointwise_estimator="ridge"`)
or `HistGradientBoostingRegressor` (`"hgb"`). `pairwise` is RankSVM-lite:
LinearSVC on within-query feature differences, at most
`max_pairs_per_query=80` oriented pairs per train query, `C=1.0`.

### Industry GBDT

Query groups are sorted contiguously for listwise training. Inference
scores each row independently, then sorts within query.
`n_estimators=120`, `learning_rate=0.08`. Default industry method when
installed is LightGBM, then XGB, then CatBoost, matching what actually
imported.

```python
session.ranking.fit(
    backend="industry",
    method="lambdarank_lgbm",
    query_column="query_id",
    item_column="item_id",
)
print(session.ranking.evaluate(k=5).metrics)
```

### Torch listwise-lite

Small MLP plus per-query softmax cross-entropy on normalized relevance
grades (ListNet-style). `hidden_dim=64`, `epochs=40`, `device="cpu"`.

## Metrics

Macro-averaged over holdout queries that have at least one relevant item
(`relevance > relevance_threshold`, default 0.0):

| Metric | Meaning here |
| --- | --- |
| `ndcg_at_k` | Graded nDCG with gain `2^rel - 1` |
| `map_at_k` | Mean average precision on binaryized grades |
| `mrr_at_k` | Mean reciprocal rank of the first relevant item |

Those are judgment-table metrics. Recommender known-item nDCG and RAG
chunk nDCG use different candidate sets. Do not mix
`session.ranking.evaluate`, `session.recommender.evaluate`, and
`session.rag.evaluate` numbers.

If you pass `backend=` at `rank` / `evaluate` time, it must match the
frozen plan. A mismatch raises.

## Bundles

`session.ranking.save_bundle` writes `buildml.ranker_bundle.v1`:
`meta.json` plus `ranker_plan.joblib` (estimator and train
standardization). A Session checkpoint does not embed `RankerPlan`.
`trusted=True` only for a file you made.

Runnable mirror:
[`examples/ranking_pointwise_loop.py`](../examples/ranking_pointwise_loop.py).
Benchmark: `python benchmarks/ranking/ndcg_lift.py`.

## When it refuses

| What you see | What happened |
| --- | --- |
| `query_column` and `item_column` required | Ids are not inferred from roles |
| Relevance required | No `relevance_column` and no Session target |
| Relevance not numeric | Graded or binary labels must be numeric |
| No split | `fit` before `split` |
| Fewer than 4 train rows | Not enough judgments to fit |
| `MissingExtraError` for `ranking-industry` | You asked for a GBDT ranker without that extra |
| `MissingExtraError` for `torch` | You asked for `listwise_lite` without Torch |
| Method not valid for backend | Pairing the catalog does not advertise |
| Backend does not match frozen plan | `rank` / `evaluate` `backend=` disagrees with fit |
| NaN/Inf features at score | Clean inputs before `rank` |

[Ranking quickstart](quickstart-ranking.md) ·
[search-relevance-ltr](../proofs/search-relevance-ltr/) ·
[Artifacts](artifacts-checkpoints-bundles.md)
