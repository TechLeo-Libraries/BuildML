# Quickstart: Recommendation systems

**Proof:** [movie-recs-collaborative](../proofs/movie-recs-collaborative/) (+ Tier C item-cosine twin).

> **Install (GitHub 2.x):**
> `pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"`
> Core path (numpy/sklearn CF + content): no extra required.
> Industry ALS/BPR + LightFM: `pip install 'buildml[recommenders-industry]'`.
> See [installation](../docs/installation.rst).

Session collaborative filtering and optional content-based scoring on
user/item/interaction tables. Train-only fit, known-item protocol, cold-start
disclosure, and ranking metrics (Precision@K, Recall@K, nDCG@K, MAP@K).

**Not** a Netflix-scale recsys platform. **Not** RAG retrieve/generate.
**Not** diagnostic EDA `Recommendation` Finding objects (teaching advice).

Runnable mirror: [`examples/recommender_item_knn_loop.py`](../examples/recommender_item_knn_loop.py).
Deep guide: [recommenders-deep.md](recommenders-deep.md).

---

## Fit → recommend → evaluate → bundle

```python
import numpy as np
import pandas as pd
from buildml import Session

rng = np.random.default_rng(0)
rows = []
for user in range(40):
    liked = rng.choice(30, size=8, replace=False)
    for item in liked:
        rows.append(
            {
                "user_id": f"u{user}",
                "item_id": f"i{item}",
                "rating": float(rng.integers(3, 6)),
                "f1": float(item % 5),
                "f2": float(item // 5),
            }
        )
frame = pd.DataFrame(rows)

session = (
    Session.ingest(frame)
    .set_roles(
        {
            "user_id": "id",
            "item_id": "id",
            "rating": "target",
            "f1": "feature",
            "f2": "feature",
        }
    )
    .split(test_size=0.2, validation_size=0.15, random_state=0)
)

fit = session.fit_recommender(
    method="item_knn",
    user_column="user_id",
    item_column="item_id",
    n_neighbors=20,
)
print(fit.to_dict())

recs = session.recommend(partition="test", k=5)
print(recs.to_dict())

ev = session.evaluate_recommender(partition="test", k=5)
print(ev.metrics)

session.save_recommender_bundle("artifacts/recommender_demo_bundle")
```

---

## Methods

| `method` | Idea |
|----------|------|
| `item_knn` | Item–item cosine CF (default) |
| `user_knn` | User–user cosine CF |
| `svd` | TruncatedSVD latent factors |
| `nmf` | Non-negative matrix factorization |
| `content` | Rating-weighted item-feature profiles (`item_feature_columns=`) |

---

## Column conventions

| Column | How to pass | Suggested role |
|--------|-------------|----------------|
| User id | `user_column=` (required) | `id` or `ignore` |
| Item id | `item_column=` (required) | `id` or `ignore` |
| Rating | `rating_column=` or Session `target` | `target` |
| Implicit | `feedback='implicit'` (presence = positive) |: |

---

## Leakage / honesty

| Rule | Behavior |
|------|----------|
| Fit | Train interactions only |
| Candidates | Train item catalog (known-item protocol) |
| Cold users | `cold_start='popularity'` or `'skip'` (disclosed) |
| Eval | Holdout ranking metrics; no refit |
| Scope | Session CF/content: not Netflix-scale; ≠ RAG; ≠ EDA Findings |

---

## Next

- Deep guide: [recommenders-deep.md](recommenders-deep.md)
- Distinct tabular ranking path: [Search / LTR](quickstart-ranking.md)
  (`fit_ranker`: query–item feature rows; not CF)
