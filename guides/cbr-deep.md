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

## Distance metrics (documented)

| `metric` | Definition |
| --- | --- |
| `euclidean` | L2 on (optionally z-scored) numeric features |
| `manhattan` | L1 on (optionally z-scored) numeric features |
| `cosine` | `1 - cosine_similarity` on numeric features |
| `mixed` | Gower-style: range-normalized numeric \|Δ\| + categorical mismatch, weighted by feature counts |

Categorical columns are **explicit** via `categorical_columns=` (used by
`mixed`). Train-fit transforms (mean/scale, ranges, cat vocabularies) are
frozen at `fit_cbr` and reused at score/retain time.

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
| Extras | Core (numpy/sklearn) | Often `buildml[rag]` |
| Bundle | `buildml.cbr_bundle.v1` | `buildml.rag_bundle.v1` |

Sharing “nearest neighbors” does **not** make CBR a RAG submodule. Do not call
CBR “tabular RAG.”

## Leakage discipline

- Require `SplitPlan` before `fit_cbr`.
- Case memory at fit: **train only**.
- Holdout partitions: retrieve / predict / evaluate only.
- `retain_cbr` hard-refuses validation/test **label** indices and requires
  `source_disclosure`.
- Distance transforms are never refit on holdout or retained rows.
- Bundles store the plan; Session checkpoints do **not**.

## Anti-patterns

- Building the case base from the full frame before `split`.
- Retaining Session test rows “to improve accuracy.”
- Treating in-sample `train_score` as holdout performance (self is usually
  nearest).
- Routing CBR through `rag_retrieve` / `rag_generate`.
- Expecting `checkpoint_load` to restore `CbrPlan`.

## Bundle boundary

See `buildml.cbr.checkpoint.CHECKPOINT_BOUNDARY`. Reload workflow via
`checkpoint_load`; reload the learner via `load_cbr_bundle`.
