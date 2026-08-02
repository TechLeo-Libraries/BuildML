# Online / continual learning (deep)

> **Install (GitHub 2.x):**
> `pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"`
> Core sklearn path — no optional extra. Industry + torch backends below.

## What this is

Session-facing incremental learning with honest backend selection:

1. `fit_online` — warm-start on an initial **train** chunk
2. `partial_fit_online` — update on subsequent train chunks (or role-aligned frames)
3. `evaluate_online` / `predict_online` — holdout inference (never for updates)
4. `save_online_bundle` / `load_online_bundle` — `buildml.online_bundle.v1`

Inspect backends with `buildml.online.online_capability_matrix()`.

| Backend | Extra | Estimators (examples) | Drift hooks |
| --- | --- | --- | --- |
| `sklearn` | core | `sgd_classifier`, `passive_aggressive_*`, `perceptron`, `*_nb` | mean-shift disclosure |
| `industry` | `buildml[online-industry]` | `river_logistic`, `river_hoeffding`, `river_pa`, … | ADWIN, Page-Hinkley, mean-shift |
| `torch` | `buildml[torch]` | `replay_mlp`, `ewc_mlp` | mean-shift disclosure |

When extras are installed, `fit_online()` defaults to the industry backend (`river_logistic`) if River is present, else torch (`replay_mlp`), else sklearn (`sgd_classifier`).

| In scope | Out of scope (next / never-as-product) |
| --- | --- |
| Sklearn `partial_fit` + River streaming + lite torch replay/EWC | Distributed streaming platforms |
| Train-cursor chunk carving + external frames | Full lifelong-learning research suites |
| Class vocabulary contract on first fit | Silent full refits (opt-in + disclosed only) |
| Drift disclosure on updates/evaluate | Using holdout rows for `partial_fit` |

## Install

```bash
pip install "buildml[online-industry,torch]"
```

## Leakage discipline

- Updates use **train** rows (or user frames with matching feature/target columns).
- Validation/test indices are refused.
- Evaluation never feeds `partial_fit`.
- Classifiers: `classes=` on first fit — explicit or discovered from the **full train target column** (labels only).

## Chunk / stream ingestion

| Source | API | Cursor |
| --- | --- | --- |
| Session train partition | `partial_fit_online(n_rows=…)` | Advances |
| Explicit train indices | `partial_fit_online(indices=…)` | Advances past max index |
| External aligned frame | `partial_fit_online(frame=…)` | Unchanged |

## Drift-aware evaluate

- `drift_detector='mean_shift'` — compare chunk/holdout feature means vs init (all backends).
- `drift_detector='adwin'` / `'page_hinkley'` — River error-stream detectors on updates and `evaluate_online(drift_check=True)` (industry backend only).
- Results expose `drift_detected` and `drift_notes` on `OnlineEvalResult` and update results.

## Benchmark

```bash
python benchmarks/online/stream_accuracy.py
```

Compares accuracy over train chunks vs a full-batch SGD refit baseline on synthetic tabular data.

## Bundle boundary

`buildml.online_bundle.v1` stores `OnlinePlan` (backend, estimator, cursor, seen indices, update history, classes). Session checkpoints do **not** embed it. See [Artifacts](artifacts-checkpoints-bundles.md).

## Teaching surfaces

- Concepts: `online-partial-fit`, `online-class-discovery`, `online-drift-disclose`, `online-bundle-boundary`
- Overlays for all Session ops; AI allowlist: `fit_online`, `partial_fit_online`, `evaluate_online`, save/load bundle
- Walkthrough / audit include online status with backend disclosure

## Phase tracker

Phase 2 items 1–4 (semi / self / active / **online industry depth**) are done. **Next:** multi-task learning (R6.4).
