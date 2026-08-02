# Quickstart: Case-based reasoning (CBR)

Session path for **tabular case memory**: build cases from train
(features + solution), retrieve k nearest neighbors, reuse/adapt solutions,
explain which cases influenced the answer, and persist via
`buildml.cbr_bundle.v1`.

Honesty: **not** RAG (document retrieval for generation), **not** a vector DB
product, **not** a full cognitive CBR research suite. Core stays light
(numpy / pandas / sklearn distances).

**Go deeper:** [CBR deep](cbr-deep.md) ·
[Artifacts](artifacts-checkpoints-bundles.md)

```python
import numpy as np
import pandas as pd
from buildml import Session

rng = np.random.default_rng(0)
x = rng.normal(size=(220, 2))
y = (x[:, 0] + 0.3 * x[:, 1] > 0).astype(int)
frame = pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "y": y})

session = (
    Session.ingest(frame)
    .set_roles({"a": "feature", "b": "feature", "y": "target"})
    .split(test_size=0.2, validation_size=0.2, random_state=0, stratify=True)
    .scale(method="standard")
)

fit = session.fit_cbr(
    task="classification",
    metric="euclidean",
    reuse="distance_weighted",
    k=5,
)
print(fit.n_cases, fit.metric, fit.reuse)

neighbors = session.retrieve_cases(partition="test", k=3)
print(neighbors.traces[0].neighbor_case_ids, neighbors.traces[0].distances)

pred = session.predict_cbr(partition="test", return_traces=True)
print(pred.traces[0].neighbor_solutions, pred.traces[0].prediction)

ev = session.evaluate_cbr(partition="validation")
print(ev.metrics, ev.mean_neighbor_distance)

session.save_cbr_bundle("artifacts/cbr_bundle")
```

| In scope | Out of scope |
| --- | --- |
| Train-only case memory | Building memory from Session test |
| euclidean / manhattan / cosine / mixed | Vector DB / ANN products |
| Majority / distance-weighted / local Ridge | Full revise cognitive suite |
| CaseTrace explanations | RAG `rag_generate` / citations |
| `buildml.cbr_bundle.v1` | Session checkpoint embedding the plan |

Next Phase 2 item after this: **Imitation learning + Reinforcement learning**.
