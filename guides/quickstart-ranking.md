# Quickstart: Search / learning-to-rank (LTR)

**Proof:** [search-relevance-ltr](../proofs/search-relevance-ltr/) (+ Tier C Ridge pointwise twin).

> **Install (GitHub 2.x):**
> `pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"`
> Core sklearn path: no extra required. For GBDT rankers:
> `pip install "buildml[ranking-industry]"`.
> See [installation](../docs/installation.rst).

Session tabular learning-to-rank on query–item (or query–document) feature
rows with relevance labels. Train-only fit, query-group split disclosure,
and per-query ranking metrics (nDCG@K, MAP@K, MRR@K).

**Not** a search-engine product. **Not** RAG retrieve/generate (chunk index +
embedding nDCG). **Not** recommendation systems (`fit_recommender` user–item CF).

Runnable mirror: [`examples/ranking_pointwise_loop.py`](../examples/ranking_pointwise_loop.py).
Deep guide: [ranking-deep.md](ranking-deep.md).

---

## Fit → rank → evaluate → bundle

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

# Sklearn fallback (always works):
fit = session.fit_ranker(
    backend="sklearn",
    method="pointwise",
    query_column="query_id",
    item_column="item_id",
    pointwise_estimator="ridge",
)
print(fit.to_dict())

# Or omit backend/method to use industry default when installed:
# fit = session.fit_ranker(query_column="query_id", item_column="item_id")

ranked = session.rank(partition="test", k=5)
print(ranked.to_dict())

ev = session.evaluate_ranker(partition="test", k=5)
print(ev.metrics)

session.save_ranker_bundle("artifacts/ranker_demo_bundle")
```

---

## Backends

| `backend` | Extra | Methods |
|-----------|-------|---------|
| `sklearn` | core | `pointwise`, `pairwise` |
| `industry` | `ranking-industry` | `lambdarank_lgbm`, `rank_ndcg_xgb`, `yetirank_catboost` |
| `torch` | `torch` | `listwise_lite` |

When extras are installed, `fit_ranker()` defaults to the industry backend.
Use `Session.ranking_capability_matrix()` to inspect what is available.

Prefer `group_split` with `query_id` as `role='group'` so holdout queries
(and their labels) never appear in train.

---

## Distinct from

| Path | What it ranks |
|------|----------------|
| **LTR (this)** | Labeled query–item feature rows |
| [Recommenders](quickstart-recommenders.md) | User–item interactions (CF / content) |
| [RAG](quickstart-rag.md) | Document chunks via embeddings / hybrid retrieve |

Same metric names (nDCG, MRR) can appear in all three: **do not compare**
`evaluate_ranker`, `evaluate_recommender`, and `rag_evaluate` numbers directly.

---

## Next

- Deep dive: [ranking-deep.md](ranking-deep.md)
- Benchmark: `benchmarks/ranking/ndcg_lift.py`
- After LTR PASS: optimisation helpers (R6.9)
