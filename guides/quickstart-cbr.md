# Case-based reasoning quickstart

```bash
pip install buildml
```

Train-only case memory, then kNN retrieve / reuse. Default is k=5 and
euclidean. `backend=None` picks the industry ANN when hnswlib or faiss
imports (`buildml[cbr-industry]` or `buildml[cbr-faiss]`), otherwise
exact sklearn kNN.
`retain` refuses validation and test rows. This is not RAG.

[CBR deep](cbr-deep.md) ·
[case-memory-claims](../proofs/case-memory-claims/)

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

# backend=None → industry ANN when hnswlib or faiss imports, else sklearn
print(session.cbr.capability_matrix()["default_backend_when_installed"])

fit = session.cbr.fit(
    task="classification",
    metric="euclidean",
    reuse="distance_weighted",
    k=5,
)
print(fit.backend, fit.n_cases, fit.metric, fit.reuse)

neighbors = session.cbr.retrieve(partition="test", k=3)
print(neighbors.traces[0].neighbor_case_ids, neighbors.traces[0].distances)

pred = session.cbr.predict(partition="test", return_traces=True)
print(pred.traces[0].neighbor_solutions, pred.traces[0].prediction)

ev = session.cbr.evaluate(partition="validation")
print(ev.metrics, ev.mean_neighbor_distance)

session.cbr.save_bundle("artifacts/cbr_bundle")
```

| In scope | Out of scope |
| --- | --- |
| Train-only case memory | Building memory from Session test |
| sklearn exact kNN + industry ANN | Vector DB / Pinecone products |
| Text embedding cases (`backend='embedding'`) | RAG `session.rag.generate` / citations |
| Majority / distance-weighted / local Ridge | Full revise cognitive suite |
| CaseTrace explanations (all backends) | Session checkpoint embedding the plan |
| `buildml.cbr_bundle.v1` | Calling CBR “tabular RAG” |

Optional extras: `buildml[cbr-industry]` (hnswlib ANN),
`buildml[cbr-faiss]` (faiss-cpu peer), `buildml[rag|ssl]`
(text embeddings), `buildml[torch]` (learned metric encoder). Included in
`buildml[production]`.

Related next: learning to rank (LTR).
