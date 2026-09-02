# Case-based reasoning deep

```bash
pip install buildml
# hnswlib ANN: pip install "buildml[cbr-industry]"
# text case embeddings: pip install "buildml[rag]"   # or buildml[ssl]
# learned metric encoder: pip install "buildml[torch]"
```

You want to answer a new row by retrieving similar **train** cases and
reusing their labels or numbers, with a trace of which cases mattered.
The training table is the memory. There is no compressed model in the
usual sense.

`session.cbr.fit()` with `backend=None` picks the **industry ANN**
(hnswlib, else faiss) when `buildml[cbr-industry]` is installed, otherwise
exact sklearn kNN. Default metric is `euclidean`, default `k` is 5,
default reuse is `distance_weighted`, default adapt is `none`, and
`standardize=True` fits mean/scale on train only. Torch is never probed
while inferring a backend: you have to ask for `backend="torch"`.

`session.cbr.retain` will not take validation or test rows. It also
requires a non-empty `source_disclosure`. This is not RAG: CBR reuses a
solution from similar cases, it does not retrieve passages for a
generator.

Short on-ramp: [CBR quickstart](quickstart-cbr.md). Proof:
[case-memory-claims](../proofs/case-memory-claims/).

## Fit, retrieve, predict, evaluate

Fit needs a split. The case base is train only. `retrieve` and `predict`
default to test. `evaluate` defaults to validation (accuracy / F1 or
RMSE / R², plus mean neighbor distance). Task is inferred from the
target: numeric with more than 20 unique values is treated as
regression, otherwise classification. Say `task=` yourself when an
integer label would look like a quantity.

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

fit = session.cbr.fit(
    task="classification",
    metric="euclidean",
    reuse="distance_weighted",
    k=5,
)
print(fit.backend, fit.n_cases, fit.metric)

neighbors = session.cbr.retrieve(partition="test", k=3)
print(neighbors.traces[0].neighbor_case_ids, neighbors.traces[0].distances)

pred = session.cbr.predict(partition="test", return_traces=True)
print(pred.traces[0].neighbor_solutions, pred.traces[0].prediction)

ev = session.cbr.evaluate(partition="validation")
print(ev.metrics, ev.mean_neighbor_distance)

session.cbr.save_bundle("artifacts/cbr_bundle")
```

Do not quote in-sample `train_score` as holdout performance. A train row
is usually its own nearest neighbor.

## Backends and metrics

| Backend | Extra | Retrieval | Metrics it will honor |
| --- | --- | --- | --- |
| `sklearn` | core | Exact kNN | `euclidean`, `manhattan`, `cosine`, `mixed` |
| `industry` | `cbr-industry` | hnswlib (preferred) or faiss ANN | `euclidean`, `cosine` |
| `embedding` | `rag` or `ssl` | sentence-transformer case vectors, then ANN if industry is also installed, else exact cosine kNN | `cosine`, `euclidean` |
| `torch` | `torch` | Supervised metric MLP on train, then kNN in that space | `euclidean`, `cosine` |

`backend=None` with no text columns and a metric the ANN can compute
selects industry when it imported, else sklearn. `manhattan` and `mixed`
force sklearn: approximate indexes do not implement them, and silently
swapping the metric would change what "similar" means. Text columns
force `embedding`. Naming an unavailable backend raises
`MissingExtraError`; inference falls back.

An impossible metric/backend pair raises `ValidationError`. It will not
substitute cosine because you asked for Manhattan on HNSW.

| `metric` | Meaning |
| --- | --- |
| `euclidean` | L2 on (optionally z-scored) numeric or embedding features |
| `manhattan` | L1; sklearn only |
| `cosine` | `1 - cosine_similarity` |
| `mixed` | Gower-style: range-normalized numeric absolute difference plus categorical mismatch; sklearn only |

Categorical columns for `mixed` are the list you pass as
`categorical_columns=`. Train-fit transforms (mean/scale, ranges,
vocabularies, encoders, ANN indexes) freeze at `fit` and are reused at
score and retain. They are never refit on holdout or retained rows.

```python
# Text / hybrid cases when sentence-transformers are installed:
session.cbr.fit(
    backend="embedding",
    text_columns=["description"],
    text_model_name="sentence-transformers/all-MiniLM-L6-v2",
    metric="cosine",
    k=7,
)
```

`embedding` without `text_columns` is refused.

## Reuse and adapt

You pick how neighbors become an answer.

| `reuse` | Task | Behavior |
| --- | --- | --- |
| `majority` | classification only | Unweighted vote |
| `distance_weighted` (default) | both | Weights `1/(d+ε)` with `distance_eps=1e-8` |
| `local_mean` | regression only | Unweighted mean of neighbor solutions |
| `local_ridge` | regression only | Tiny Ridge on the k neighbors' features |

`adapt="offset"` is a fixed half-and-half blend toward the neighbor mean
(regression). `adapt="none"` leaves the reuse result as-is. Wrong
reuse-for-task pairings raise instead of quietly averaging a class
label.

Traces (`CaseTrace`) carry neighbor ids, row indices, distances,
weights, neighbor solutions, and the prediction, on every backend.

## Retain

New labeled cases can enter memory after fit. Validation and test
indices are refused, not warned about: retaining a holdout row makes it
its own nearest neighbor the next time you score that partition.
`source_disclosure` is required so the origin of those cases is on the
plan.

```python
new_cases = pd.DataFrame({"a": [0.1, -0.4], "b": [0.2, 0.3], "y": [1, 0]})
session.cbr.retain(
    labeled_frame=new_cases,
    source_disclosure="Human review of production traffic, Q3.",
)
```

Pass either `labeled_frame` or `row_indices`, not both. Null solutions
are refused. `allow_overlap_with_train=True` (the default) permits
overlap with existing train ids; turn it off if duplicates should fail.

## CBR is not RAG

CBR memory is train tabular cases with a solution. RAG memory is a text
corpus for grounding generation. Sharing nearest-neighbor search or
sentence-transformers does not make this a submodule of
`session.rag`. The bundles are different:
`buildml.cbr_bundle.v1` vs `buildml.rag_bundle.v1`. Do not call CBR
"tabular RAG", and do not route cases through `session.rag.retrieve`.

## Bundles

`session.cbr.save_bundle` stores the case memory, metric, reuse, and
frozen transforms. A Session checkpoint does not embed `CbrPlan`. Reload
the table, then `session.cbr.load_bundle(..., trusted=True)` for a file
you made.

Runnable mirror: [`examples/cbr_knn_loop.py`](../examples/cbr_knn_loop.py).
Benchmark: `python benchmarks/cbr/retrieval_accuracy.py`.

## When it refuses

| What you see | What happened |
| --- | --- |
| No split | `fit` before `split` |
| `MissingExtraError` for `cbr-industry` | You named `backend="industry"` without the extra |
| `MissingExtraError` for `rag or ssl` | You named `embedding` without sentence-transformers |
| Metric not valid for backend | Manhattan/mixed on ANN, or similar mismatch |
| `embedding` requires `text_columns` | Backend named without text |
| retain refused holdout index | You tried to absorb validation or test labels |
| empty `source_disclosure` | Retain without saying where the cases came from |
| reuse vs task mismatch | `majority` on regression, or `local_mean` on classification |

[CBR quickstart](quickstart-cbr.md) ·
[case-memory-claims](../proofs/case-memory-claims/) ·
[Artifacts](artifacts-checkpoints-bundles.md)
