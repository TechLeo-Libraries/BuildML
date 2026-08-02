# Case-based reasoning deep guide

## What shipped

BuildML’s CBR path is a **Session-native tabular case memory** with retrieve →
reuse/adapt → optional retain, explanation traces, and a dedicated bundle.

| Surface | Role |
| --- | --- |
| `fit_cbr` | Build case base from Session **train** |
| `retrieve_cases` | kNN neighbors (no reuse) |
| `predict_cbr` | Retrieve + reuse (+ `CaseTrace`) |
| `evaluate_cbr` | Holdout accuracy/RMSE (+ mean neighbor distance) |
| `retain_cbr` | Lite retain with disclosure; refuse holdout indices |
| `save_cbr_bundle` / `load_cbr_bundle` | `buildml.cbr_bundle.v1` |
| `cbr_capability_matrix()` | Honest backend / extra matrix |

## Backends (R6.7 industry depth)

| `backend` | Extra | Retrieval |
| --- | --- | --- |
| `sklearn` (fallback) | core | Exact kNN — euclidean / manhattan / cosine / mixed |
| `industry` (default when installed) | `buildml[cbr-industry]` | hnswlib (preferred) or faiss ANN on numeric features |
| `embedding` | `buildml[rag]` or `buildml[ssl]` | sentence-transformer case embeddings (+ optional numeric concat) |
| `torch` | `buildml[torch]` | Learned metric MLP encoder + kNN |

Pass `backend=` on `fit_cbr`, `retrieve_cases`, and `predict_cbr`. Case
influence traces (`CaseTrace`) are preserved for all backends.

```python
matrix = Session.cbr_capability_matrix()
print(matrix["default_backend_when_installed"])

session.fit_cbr(backend="industry", metric="euclidean", k=5)
# or backend=None → honest default when cbr-industry is installed
```

Text/hybrid cases:

```python
session.fit_cbr(
    backend="embedding",
    text_columns=["description"],
    text_model_name="sentence-transformers/all-MiniLM-L6-v2",
    metric="cosine",
    k=7,
)
```

## Distance metrics (documented)

| `metric` | Definition |
| --- | --- |
| `euclidean` | L2 on (optionally z-scored) numeric or embedding features |
| `manhattan` | L1 on (optionally z-scored) numeric features (**sklearn only**) |
| `cosine` | `1 - cosine_similarity` on numeric or embedding features |
| `mixed` | Gower-style: range-normalized numeric \|Δ\| + categorical mismatch (**sklearn only**) |

Categorical columns are **explicit** via `categorical_columns=` (used by
`mixed` on the sklearn backend). Train-fit transforms (mean/scale, ranges, cat
vocabularies, encoders, ANN indexes) are frozen at `fit_cbr` and reused at
score/retain time.

## Reuse / adapt

| `reuse` | Task | Behavior |
| --- | --- | --- |
| `majority` | classification | Unweighted majority vote |
| `distance_weighted` | both | Weights `1/(d+ε)` for vote or average |
| `local_mean` | regression | Unweighted mean of neighbor solutions |
| `local_ridge` | regression | Tiny Ridge on the k neighbors |

`adapt='offset'` is a lite blend toward the neighbor mean (regression).

## CBR ≠ RAG

| | **CBR** | **RAG** |
| --- | --- | --- |
| Memory | Train tabular cases (features + solution) | Text corpus / chunks |
| Goal | Reuse/adapt a label or numeric outcome | Ground generation / citations |
| Extras | Core + optional `cbr-industry` / `rag` for embeddings | `buildml[rag]` |
| Bundle | `buildml.cbr_bundle.v1` | `buildml.rag_bundle.v1` |
| Traces | `CaseTrace` — which cases influenced the prediction | Chunk citations for generation |

Sharing “nearest neighbors” or sentence-transformers does **not** make CBR a
RAG submodule. Do not call CBR “tabular RAG.” CBR embeds **cases with
solutions** for supervised-style reuse; RAG retrieves **documents** for
grounding LLM output.

## Leakage discipline

- Require `SplitPlan` before `fit_cbr`.
- Case memory at fit: **train only**.
- Holdout partitions: retrieve / predict / evaluate only.
- `retain_cbr` hard-refuses validation/test **label** indices and requires
  `source_disclosure`.
- Distance transforms and ANN indexes are never refit on holdout or retained rows.
- Bundles store the plan; Session checkpoints do **not**.

## Anti-patterns

- Building the case base from the full frame before `split`.
- Retaining Session test rows “to improve accuracy.”
- Treating in-sample `train_score` as holdout performance (self is usually
  nearest).
- Routing CBR through `rag_retrieve` / `rag_generate`.
- Expecting `checkpoint_load` to restore `CbrPlan`.
- Calling embedding backend “RAG” because it uses sentence-transformers.

## Bundle boundary

See `buildml.cbr.checkpoint.CHECKPOINT_BOUNDARY`. Reload workflow via
`checkpoint_load`; reload the learner via `load_cbr_bundle`.

## Benchmark

`benchmarks/cbr/retrieval_accuracy.py` — k vs holdout accuracy and retrieve
latency for sklearn / industry / torch backends (skips missing extras).
