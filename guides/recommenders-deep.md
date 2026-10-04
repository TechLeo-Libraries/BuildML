# Recommenders deep

```bash
pip install buildml
# implicit ALS / BPR: pip install "buildml[recommenders-industry]"
# LightFM hybrid: pip install "buildml[recommenders-lightfm]"
```

You have user-item interactions and you want top-K items from the
**train catalog**. `user_column` and `item_column` are required kwargs.
They are not inferred from roles. Mark them `id` or `ignore` so a later
classical `fit()` does not treat ids as features.

`session.recommender.fit()` with the mixin defaults (`feedback="explicit"`,
`method=None`) is item kNN on sklearn. That stays sklearn even when
`implicit` is installed, because this adapter routes explicit feedback to the sklearn methods. `feedback="implicit"` with `method=None` picks ALS when
`buildml[recommenders-industry]` imported, otherwise sklearn NMF.
LightFM is `method="lightfm"` (extra `recommenders-lightfm`, not the
implicit extra). This extra excludes native Windows and Python 3.12 or newer.
LightFM 1.17's published C extension failed to build on Python 3.12 in our
installation checks. The example below was verified on Ubuntu with Python 3.11.

For that environment, install LightFM in a dedicated virtual environment before
installing the BuildML extra. LightFM 1.17 requires the legacy build path shown
here; a C compiler and Python development headers must already be installed:

```bash
python3.11 -m venv .venv-lightfm
. .venv-lightfm/bin/activate
python -m pip install "pip<25" setuptools wheel "numpy<2" scipy
python -m pip install --no-use-pep517 --no-build-isolation lightfm==1.17
python -m pip install "buildml[recommenders-industry,recommenders-lightfm]"
```

This is collaborative filtering and content profiles on a Session split.
RAG retrieval, supervised learning-to-rank, and EDA recommendations have
separate APIs and purposes.

Quickstart: [recommenders quickstart](quickstart-recommenders.md).
Proof: [movie-recs-collaborative](https://github.com/TechLeo-Libraries/BuildML/tree/main/proofs/movie-recs-collaborative).

## Fit, recommend, evaluate

Fit is train-only. For explicit feedback, `rating_column` defaults to
the Session target. For `feedback="implicit"`, ratings are ignored and
presence is the signal. `recommend` needs either `partition=` or
`user_ids=`, not both and not neither. `evaluate` defaults to test with
`k=10`. `exclude_train_items=True` (the default on `recommend`) hides
items the user already had in train.

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
                "age": float(20 + user),
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
            "age": "feature",
            "f1": "feature",
            "f2": "feature",
        }
    )
    .split(test_size=0.2, validation_size=0.15, random_state=0)
)

fit = session.recommender.fit(
    method="item_knn",
    user_column="user_id",
    item_column="item_id",
    n_neighbors=20,
)
print(fit.method, fit.backend)

recs = session.recommender.recommend(partition="test", k=5)
print(recs.to_dict())

ev = session.recommender.evaluate(partition="test", k=5)
print(ev.metrics)

session.recommender.save_bundle("artifacts/recommender_bundle")
```

Candidates are always the train item catalog. A holdout-only item is
never a collaborative candidate.

## Methods and backends

| Method | Backend | Extra | Typical feedback |
| --- | --- | --- | --- |
| `item_knn` (explicit default) | sklearn | core | cosine item-user CF, `n_neighbors=40` |
| `user_knn` | sklearn | core | cosine user-user CF |
| `svd` / `nmf` | sklearn | core | matrix factorization, `n_factors=32` |
| `content` | sklearn | core | rating-weighted item feature profiles (`item_feature_columns=`) |
| `als` / `bpr` | implicit | `recommenders-industry` | implicit only |
| `lightfm` | lightfm | `recommenders-lightfm` | hybrid WARP/BPR, optional `user_feature_columns` / `item_feature_columns` |

`backend="implicit"` with `feedback="explicit"` is refused. Use sklearn
`svd` / `nmf` / `item_knn` for ratings, or LightFM for hybrid.

```python
# Requires: pip install "buildml[recommenders-industry,recommenders-lightfm]".
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
                "age": float(20 + user),
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
            "age": "feature",
            "f1": "feature",
            "f2": "feature",
        }
    )
    .split(test_size=0.2, validation_size=0.15, random_state=0)
)

fit = session.recommender.fit(
    method="item_knn",
    user_column="user_id",
    item_column="item_id",
    n_neighbors=20,
)
print(fit.method, fit.backend)

recs = session.recommender.recommend(partition="test", k=5)
print(recs.to_dict())

ev = session.recommender.evaluate(partition="test", k=5)
print(ev.metrics)

session.recommender.save_bundle("artifacts/recommender_bundle")

# Implicit industry default when implicit is installed:
session.recommender.fit(
    user_column="user_id",
    item_column="item_id",
    feedback="implicit",
    n_factors=32,
)

# LightFM hybrid when that extra imported:
session.recommender.fit(
    method="lightfm",
    user_column="user_id",
    item_column="item_id",
    item_feature_columns=["f1", "f2"],
    user_feature_columns=["age"],
)
```

`n_iterations=15` (sklearn-style loops) and `lightfm_epochs=10` are the
library defaults if you do not pass them.

## Cold start

You pick the policy with `cold_start=` (`"popularity"` default, or
`"skip"`).

| Case | Behavior |
| --- | --- |
| User absent from train | popularity list from train, or empty lists if `skip` |
| Item absent from train | Dropped from candidates and from eval relevant sets, with a warning |
| Warm user, empty scores | Disclosed popularity fallback |

## Evaluation protocol

For each **warm** holdout user with at least one holdout item that
exists in the train catalog:

1. Relevant set = holdout items ∩ train catalog.
2. Recommend top-K among train items, excluding that user's train
   history.
3. Precision@K, Recall@K, nDCG@K, MAP@K.
4. Macro-average over scored users. Cold-start users are counted
   separately, not scored as if they were warm.

Those nDCG numbers are known-item recommender metrics. Do not compare
them to `session.ranking.evaluate` (judgment tables) or
`session.rag.evaluate` (chunks). Same names, different protocols.

## Bundles

`session.recommender.save_bundle` writes `buildml.recommender_bundle.v1`
(train catalog, matrix, similarities or factors). A Session checkpoint
does not embed `RecommenderPlan`. `trusted=True` only for a file you
made.

Runnable example:
[`examples/recommender_item_knn_loop.py`](../examples/recommender_item_knn_loop.py).
Benchmark: `python benchmarks/recommenders/ranking_quality.py`.

## Validation errors and prerequisites

| What you see | What happened |
| --- | --- |
| `user_column` and `item_column` required | Ids are not inferred from roles |
| No split | `fit` before `split` |
| Explicit needs a rating | No `rating_column` and no Session target |
| `backend='implicit'` requires `feedback='implicit'` | ALS/BPR on explicit ratings |
| `MissingExtraError` for `recommenders-industry` | You asked for ALS/BPR without `implicit` |
| `MissingExtraError` for `recommenders-lightfm` | You asked for LightFM without that extra |
| `recommend()` needs `user_ids` or `partition` | Neither (or both) were passed |
| Method not valid for backend | Pairing the catalog does not advertise |

[Recommenders quickstart](quickstart-recommenders.md) ·
[movie-recs-collaborative](https://github.com/TechLeo-Libraries/BuildML/tree/main/proofs/movie-recs-collaborative) ·
[Artifacts](artifacts-checkpoints-bundles.md)
