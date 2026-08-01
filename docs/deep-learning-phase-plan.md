# Deep learning phase plan

Next-phase plan after classical alpha `2.0.0a1`.  
Grounded in: [reconstruction-roadmap.md](./reconstruction-roadmap.md) ·
[classical-ml-capability-map.md](./classical-ml-capability-map.md) ·
[classical-alpha-gate.md](./classical-alpha-gate.md) ·
[ingest-engine-checkpoint-design.md](./ingest-engine-checkpoint-design.md) ·
[quality-bar.md](./quality-bar.md) · [editorial-standards.md](./editorial-standards.md)

**Status:** M0 locked · M1 in progress (thin tabular Torch slice).  
**Sequencing (locked):** Classical ML → Deep Learning → RAG / modern methods → LLM operator last.  
**North star:** flexibility · depth · functionality.  
**M0 lock artifact:** [dl-m0-lock.md](./dl-m0-lock.md).

---

## 1. Goals and non-goals

### Goals (DL phase / v3 domain)

1. Attach a **deep-learning domain** (`buildml.dl`) that uses the same Session language
   (roles, partitions, history, explain, checkpoints) without turning `Session` into a
   second god-object.
2. Ship a **leakage-safe** supervised DL path: split first; train-time transforms and
   normalization fit on train (or declared train folds) only; eval partitions receive
   frozen transforms.
3. Cover a **complete first workflow turn** for tabular supervised nets (and a clear
   extension point for tensor / image / sequence inputs later): dataset → loader →
   train loop → metrics → checkpoint → explain hooks.
4. Keep **core install lean**: `import buildml` never requires Torch (or any DL stack).
5. Meet the same quality bar as classical: typed results, catalog coverage, tests,
   docstring standard, honest scale notes — not thin framework wrappers.
6. Reuse Teaching Studio / explain **principles** (evidence, limitations, progressive
   disclosure) with DL-appropriate surfaces; do not force classical EDA boards onto
   neural training blindly.

### Non-goals (this phase)

| Non-goal | Rationale |
| --- | --- |
| Replacing classical `Session.fit` / sklearn path | Classical remains the default supervised path |
| Full AutoML / architecture search product | Later; optional wrappers only after a solid trainer spine |
| Distributed / multi-node training as day-one identity | Document limits; single-process + optional device first |
| Production serving stack (TorchServe, ONNX runtime productization) | Export hooks later; library-first |
| RAG / embeddings index / retrieval | Separate `buildml.rag` after DL spine |
| LLM natural-language operator | Separate `buildml.ai` after method surface is solid |
| Forcing Polars/DuckDB into Torch DataLoaders on day one | Reuse Dataset partitions; materialize batches honestly |
| Claiming GPU CI on every PR | Optional/manual or scheduled job; CPU smoke is the merge gate |
| Porting classical preprocess recipes 1:1 into nets | Some steps apply; many do not — document the boundary |

### Relationship to classical Session

| Concern | Classical (`buildml.model`) | DL (`buildml.dl`) |
| --- | --- | --- |
| Estimator contract | sklearn-compatible | `nn.Module` (+ config) |
| Fit API | `Session.fit(estimator)` | `Session.fit_torch(...)` or thin `Session.dl_*` delegates |
| Design matrix | In-memory `X, y` | Batched tensors / DataLoaders |
| Artifacts | pipeline / model bundle | trainer checkpoint + optional export |
| CV / search | fold-local `PreprocessRecipe` | fold-aware train loops (later depth); no fake sklearn CV |
| Shared spine | Dataset, roles, splits, checkpoint, history, explain, core errors | same |

Classical APIs stay authoritative for tabular sklearn workflows. DL methods are additive
and may refuse when Session state is sklearn-only (e.g. fitted `FitResult` without a
DL trainer) with clear errors — never silent cross-wiring.

---

## 2. Architecture (no god-object)

### Package boundaries

```text
buildml/
  session/          # thin delegates only — no train-loop bodies
  data/             # Dataset, roles, splits (shared)
  checkpoint/       # workflow data resume (shared); DL weights elsewhere
  explain/          # catalog + concepts (shared schemas; DL ops register here)
  core/             # types, results, MissingExtraError, LeakageError
  model/            # classical sklearn only
  preprocess/       # classical train-fitted plans (shared where applicable)
  dl/               # NEW — all Torch-facing implementation
    __init__.py     # lazy public exports; no eager torch import at package root
    extras.py       # require_torch() → MissingExtraError("torch", ...)
    types.py        # TaskSpec, DeviceSpec, TrainConfig dataclasses
    dataset.py      # partition → TensorDataset / IterableDataset adapters
    loaders.py      # DataLoader factory (batch, shuffle train-only, workers)
    transforms.py   # train-fit normalize / encode bridges; frozen apply
    train.py        # train_loop, early stopping, grad clip hooks
    metrics.py      # epoch metrics + evaluate on partition
    checkpoint.py   # weight + optimizer + TrainConfig bundle (≠ Session checkpoint)
    results.py      # TrainResult, DLEvaluateResult, LoaderReport
    explain_hooks.py  # history summaries + catalog-facing result reading
```

Optional later (not v1 DL): `buildml.dl.vision`, `buildml.dl.sequence` as submodules
under the same extra — still not new Session mega-methods.

### Session attachment rules

1. **Delegate or do not exist.** `Session` methods call into `buildml.dl.*` and record
   history; they do not contain optimizer/step code.
2. **Lazy import.** `import buildml` and `from buildml import Session` must succeed
   without Torch. Torch is imported inside `buildml.dl.extras.require_torch()` when a
   DL entrypoint runs.
3. **Separate result slots.** Prefer `session.dl_train_result` (name TBD at impl) rather
   than overloading `session.fit_result`. Classical `FitResult` stays sklearn-shaped.
4. **Separate artifact kinds.**
   - Session **checkpoint** = data + roles + splits + history (+ optional classical plans).
   - DL **trainer bundle** = weights, optimizer state, config, metric history, feature /
     label contracts. Do not embed estimator weights in Session checkpoints (mirrors
     classical: checkpoint ≠ model bundle).
5. **One implementation path.** No parallel “handy” train loop in Session and another in
   `buildml.dl`.

### Integration sketch

```text
Session.ingest → set_roles → split
  → [optional classical preprocess on train]
  → Session.make_torch_loaders(...)     # → buildml.dl.loaders
  → Session.fit_torch(module, config)   # → buildml.dl.train
  → Session.evaluate_torch(partition=...)
  → Session.save_torch_bundle(...) / load_torch_bundle(...)
  → session.explain("fit_torch") / workflow() sees DL ops when registered
```

Naming is indicative; final public names land in the design milestone with catalog
entries. Prefer a small, consistent `*_torch` or `dl_*` prefix so classical
`fit` / `evaluate` remain unambiguous.

### Shared spine reuse

| Shared | Reuse how |
| --- | --- |
| `Dataset` + `SplitPlan` | Source of partition membership; forbid train-shuffle of test |
| `assert_fit_partition` / leakage guards | Call before any train-fit transform or training |
| `MissingExtraError` | Extra name `torch` (and meta `dl` if aliased) |
| Operation history | Record DL ops with typed `result_summary` dicts |
| Explain catalog schemas | New `OperationSpec` rows for DL ops |
| Checkpoint save/load | Resume data workflow; user reattaches then reloads DL bundle |

---

## 3. Capability map — v1 DL

Status tags: **M0** design · **M1** thin vertical slice · **M2** depth · **M3** docs/alpha · **L** later · **X** non-goal for DL v1.

### 3.1 Datasets and loaders

| Capability | Tag | Notes |
| --- | --- | --- |
| Partition → TensorDataset from feature/target roles | M1 | Numeric tabular first; dtype/contract checks |
| Train / val / test DataLoaders | M1 | Shuffle **train only**; seeded generators |
| Batch size / num_workers / pin_memory config | M1 | Sensible CPU defaults; document CUDA knobs |
| Class / sample weights into sampler or loss | M2 | Align with classical imbalance intent; train-only |
| Custom collate / user Dataset injection | M2 | Validated against roles/schema |
| Image / sequence / multimodal loaders | L | After tabular spine |
| Streaming / IterableDataset for huge tables | L | Honest materialization gates first |
| Automatic Polars→Torch zero-copy | X/L | Not required for v1; Pandas/NumPy bridge OK |

### 3.2 Train loop

| Capability | Tag | Notes |
| --- | --- | --- |
| Caller-supplied `nn.Module` + loss + optimizer factory | M1 | BuildML does not invent mystery architectures |
| Epoch loop with train + optional val metrics | M1 | Typed `TrainResult` |
| Device selection (`cpu` / `cuda` / `mps`) with clear fallback | M1 | No silent wrong-device claims |
| Early stopping on val metric | M2 | Patience / mode / restore-best |
| Gradient clipping, LR schedulers | M2 | Config-driven, documented |
| Mixed precision | L | After stable CPU/CUDA path |
| DistributedDataParallel | X | Explicit non-goal for DL alpha |
| Built-in model zoo as product identity | X | Optional examples in docs only |

### 3.3 Metrics and evaluation

| Capability | Tag | Notes |
| --- | --- | --- |
| Classification / regression metric suites on a partition | M1 | Reuse classical metric names where comparable |
| Per-epoch history tables | M1 | For plots and Teaching Studio later |
| Confusion / residual-style diagnostics | M2 | Structured, not plot-only |
| Calibration / threshold tools for net probabilities | L | After predict_proba-equivalent path is solid |
| Compare Torch run vs sklearn `FitResult` | L | Nice teaching feature; not alpha-blocking |

### 3.4 Checkpointing and persistence

| Capability | Tag | Notes |
| --- | --- | --- |
| Save/load trainer bundle (weights, opt, config, history) | M1 | Distinct schema id, e.g. `buildml.torch_bundle.v1` |
| Resume interrupted training | M2 | Same bundle; validate config hash / schema |
| Session checkpoint mid-loop (data only) | M1 | Existing API; document DL resume recipe |
| Export TorchScript / ONNX | L | Escape hatch after bundle round-trip |
| Pickle arbitrary Python into Session checkpoint | X | Keep checkpoint trust model |

### 3.5 Explain / teaching hooks

| Capability | Tag | Notes |
| --- | --- | --- |
| Catalog entries for loader / fit_torch / evaluate_torch / save bundle | M1 | Prerequisites, leakage, result reading |
| Concept notes: overfitting, early stop, batch leakage, device | M1 | Link from operations |
| History + walkthrough awareness of DL ops | M1–M2 | Same resolver; DL-specific status fields |
| Loss/metric curves in offline HTML or Studio panel | M2 | New board — not classical EDA domains |
| Concept Academy notes for backprop / generalization | M2 | Editorial standards apply |
| SHAP / saliency productization | L | Careful deps; classical map already marks SHAP later |

### 3.6 Leakage and split discipline

| Capability | Tag | Notes |
| --- | --- | --- |
| Refuse training without split (guided mode) | M1 | Same philosophy as classical |
| Train-only fit for normalize / target encode used before tensors | M1 | Frozen stats on val/test |
| No test rows in DataLoader shuffle or oversampling | M1 | Tests must prove it |
| Validation-only early stopping; test once at end | M1 | Document anti-pattern of test-tuned early stop |
| Fold-local DL CV | M2/L | Depth item; do not fake it in M1 |
| Group / time split compatibility | M2 | Reuse `group_split` / `time_split`; loaders honor membership |
| Injected external partitions | M1 | `inject_split` remains valid source of membership |

---

## 4. Explain catalog and Teaching Studio

### Reuse

- `OperationSpec` / prerequisites / leakage / anti-patterns / result_reading
- `CONCEPT_NOTES` registry and Concept Academy shape
- History records + `workflow()` / `walkthrough()` / `dry_run` resolution
- Editorial standards: observation → finding → recommendation; no overclaim
- Local-only HTML defaults; no network dependency for reports

### Do not force classical EDA UI onto nets

| Classical surface | DL v1 approach |
| --- | --- |
| EDA domain boards (quality, VIF, drift, …) | Still valid **on the tabular Dataset** before training; not a substitute for training diagnostics |
| Eval plot boards (ROC, residuals) | Add **training curve / metric history** boards; reuse metric plots only where predictions exist |
| Teaching Studio “preprocess scope” | Add DL analogs: device, early-stop partition, train-only transform scope |
| `session.eda_app()` as primary DL cockpit | Optional later; M1 ships catalog + structured results + optional matplotlib/plotly curves behind extras |

### Disclosure principles for DL copy

- State **partition** used for early stopping vs final test metrics.
- State **device** and whether CUDA was requested but unavailable.
- State that batch metrics are noisy; prefer epoch aggregates for claims.
- Never imply Session checkpoint contains weights.
- Never imply catalog “available” means architecture is appropriate.

---

## 5. Packaging, Python, CI

### Extras

| Extra | Contents | Notes |
| --- | --- | --- |
| `torch` | `torch` (bounded lower version TBD at impl against 3.10–3.13 wheels) | Primary install hint |
| `dl` | Alias meta-extra → `buildml[torch]` (+ later light helpers if any) | Matches roadmap naming |
| `all` (future) | classical + dl (+ rag/ai when they exist) | Do not fold Torch into `all-classical` |

Rules (unchanged product law):

- `import buildml` on core alone.
- Missing Torch → `MissingExtraError("torch", feature=...)` with
  `pip install 'buildml[torch]'` (or `[dl]` if that is the documented alias).
- No eager `import torch` in `buildml/__init__.py` or `session.py` module top-level.

### Python support

| Item | Direction |
| --- | --- |
| Classical alpha | 3.10–3.13 (locked) |
| DL phase | Keep 3.10–3.13 **if** Torch wheels support the matrix; otherwise document a **DL CI subset** (e.g. 3.11–3.12) without dropping classical support |
| Decision point | Design milestone M0: pin Torch version range after checking current wheels |

Roadmap already says: revisit Python when DL domain lands — that revisit is an evidence
pass, not a silent drop of 3.10/3.13 for core.

### CI shape

| Job | Role |
| --- | --- |
| Existing `test` | Core + classical; **must stay green without Torch** |
| New `torch` (or `dl`) | `pip install -e ".[dev,torch]"`; DL unit + integration smoke; Python subset matrix |
| Optional `torch-cuda` | Manual / scheduled; not a PR blocker |
| Import smoke | Assert `import buildml` still works in an env **without** Torch |

Mirror the `engines` / `optuna` pattern: skip-friendly dedicated job, not weight on every
core matrix cell.

---

## 6. Phased milestones and exit criteria

### M0 — Design lock (docs + spikes)

**Deliverables**

- This plan approved (or amended with decision log entries).
- Short design addendum (can live in this file §Decision log): public method names,
  bundle schema id, Torch version pin, Python matrix for DL CI.
- Spike notes: partition→DataLoader latency on a fixture; device fallback behavior.

**Exit**

- [x] Public API sketch agreed (method names + result types)
- [x] Bundle vs Session checkpoint boundary written and accepted
- [x] Torch pin + CI Python subset chosen
- [x] No production code required beyond optional spikes in a branch

### M1 — Thin vertical slice

**Deliverables**

- Package `buildml.dl` with lazy Torch import
- Tabular partition → loaders → train loop → evaluate → trainer bundle save/load
- Session delegates + history recording
- Catalog entries for the slice operations
- Integration smoke test (CPU) gated on `buildml[torch]`
- CI `torch` job

**Canonical smoke**

```text
Session.ingest → set_roles → split
  → make_torch_loaders
  → fit_torch (small MLP, few epochs, CPU)
  → evaluate_torch(partition="test")
  → save_torch_bundle / load_torch_bundle
  → explain("fit_torch")  # catalog hit
```

**Exit**

- [x] Core CI unchanged (no Torch required)
- [x] DL smoke green on CI CPU job (`torch` job; 3.11–3.12)
- [x] Leakage tests: test partition never shuffled into train loader; normalize fit train-only
- [x] `MissingExtraError` path tested without Torch installed
- [x] Typed results + docstrings for public delegates

### M2 — Depth

**Deliverables**

- Early stopping, schedulers, grad clip, richer metrics/diagnostics
- Group/time split honored by loaders
- Resume training from trainer bundle
- Training-curve report (viz or dashboard extra — structured data in core DL results)
- Walkthrough / workflow status for DL ops
- Fold-aware or repeated holdout discipline docs (implement what is tested)

**Exit**

- [ ] Capability map M2 rows green or explicitly deferred with reason
- [ ] Teaching disclosures for early-stop partition + device
- [ ] Quality bar: not “accuracy-only”; structured diagnostics present

### M3 — Docs and DL alpha gate

**Deliverables**

- Quickstart for DL slice; glossary terms; known limits list
- DL alpha gate doc (sibling to classical-alpha-gate) with must IDs
- README extras list includes `torch` / `dl`
- Version bump policy: e.g. `2.1.0a1` or `3.0.0a1` — **decide in M0**; do not silently
  reuse classical alpha numbering without a changelog story

**Exit**

- [ ] DL alpha gate musts green
- [ ] Editorial lint clean for new user-facing strings
- [ ] Changelog + known limits do not claim RAG/LLM

---

## 7. Risks and classical-only surfaces

### Risks

| Risk | Mitigation |
| --- | --- |
| Session grows another mega-facade | Hard rule: no train-loop code in `session.py`; review diffs for body length |
| Torch in core import graph | Lazy require + CI job that installs core-only and imports buildml |
| Leakage via global normalize / augment | Train-fit transforms module + tests; catalog leakage fields |
| Users equate val early-stopping with test performance | Disclosures + evaluate_torch default partition = test only after stop on val |
| Bundle / checkpoint confusion | Distinct schema ids; docs + errors if wrong loader used |
| Python / Torch wheel matrix breaks 3.13 | DL CI subset; classical keeps full matrix |
| GPU flakiness in CI | CPU merge gate only |
| Scope creep into RAG/LLM | Explicit sequencing (§8); reject feature PRs that invert order without decision log |

### Stays classical-only (for now)

- `Session.fit` / `cv_score` / `grid_search` / `nested_cv_score` / Optuna recipe search
- `PreprocessRecipe` fold-local sklearn pipeline
- Pipeline bundle `buildml.pipeline_bundle.v2` + `predict_from_pipeline`
- Classical EDA Teaching Studio domain boards as the primary exploration product
- Imbalanced-learn resample strategies (DL may add weighted sampling later — separate)
- `all-classical` extra contents

Classical may still **prepare tabular features** that DL loaders consume; that is
composition, not ownership transfer.

---

## 8. Later sequencing: RAG then LLM

Locked vision (reconstruction roadmap): **Classical → DL → RAG → LLM operator**.

| Horizon | Package | Depends on | Starts when |
| --- | --- | --- | --- |
| Done / alpha | Classical Session spine | — | `2.0.0a1` gate |
| This plan | `buildml.dl` + extra `torch`/`dl` | Shared data/checkpoint/explain | After M0 approval |
| Next domain | `buildml.rag` + extra `rag` | Stable Dataset/checkpoint; embeddings likely need Torch or sibling stack | After DL M1 smoke (prefer after M2 so trainer/embed patterns exist) |
| Last | `buildml.ai` + extra `ai` | Broad, stable method catalog for tool-calling | After DL (+ RAG if operator should drive retrieval) method surface is documented |

**RAG must not start as a fork of Session.** It attaches like DL: domain package,
extras, catalog ops, separate index artifacts vs Session checkpoints.

**LLM operator constraints (preview, not this phase)**

- Maps natural language → **allowed existing methods** (dry-run / execute)
- Optional; core and DL must work with no API keys
- No LLM dependency in `import buildml`

---

## 9. Recommended immediate first implementation slice

After plan approval, implement **only** this vertical slice before any model zoo,
dashboard redesign, or RAG spikes:

1. Add `buildml/dl/` with `require_torch()` and empty typed result stubs.
2. `partition_frame_to_tensors` + `make_loaders(session_or_dataset, split_plan, ...)`.
3. `train_supervised_module(...)` CPU loop: train epochs, optional val metric each epoch.
4. `evaluate_module(...)` on a named partition.
5. `save_torch_bundle` / `load_torch_bundle` round-trip.
6. Thin `Session` delegates + history keys.
7. Catalog ops + two concept notes (batch leakage; early stopping partition).
8. Tests: leakage shuffle, missing-extra, CPU smoke; CI job `torch`.

**Explicitly defer in that first PR series:** AMP, DDP, vision datasets, Optuna-over-Torch,
EDA Studio redesign, ONNX export, RAG embeddings.

---

## Decision log (locked in M0)

| Topic | Status | Notes |
| --- | --- | --- |
| Public method prefix (`fit_torch` vs `dl_fit`) | Locked | `*_torch` Session methods; result slot `dl_train_result` |
| Version line for DL alpha | Locked | `2.1.0a1` (classical remains `2.0.0a1`) — apply at M3 |
| Extra name canonical (`torch` vs `dl` vs both) | Locked | `torch` = deps; `dl` = alias meta-extra |
| Minimum Torch version | Locked | `torch>=2.2` in `pyproject.toml` |
| DL CI Python versions | Locked | 3.11 + 3.12 (`torch` CI job) |
| Whether classical preprocess plans auto-apply before loaders | Locked | No auto-apply; call classical prep explicitly first |
| Trainer bundle serializer | Locked | `torch.save` (`trainer.pt`) + JSON `meta.json`; schema `buildml.torch_bundle.v1` |
| Bundle vs Session checkpoint | Locked | See [dl-m0-lock.md](./dl-m0-lock.md) |

---

## References

- Architecture anti-pattern to avoid: mutable facade reimplementation
  ([architecture-review.md](./architecture-review.md))
- Domain attachment rule: [reconstruction-roadmap.md](./reconstruction-roadmap.md) §C–D, §H
- Classical completeness target (DL marked X): [classical-ml-capability-map.md](./classical-ml-capability-map.md) §10
- Materialization sketch: [ingest-engine-checkpoint-design.md](./ingest-engine-checkpoint-design.md) §8
- Classical alpha out-of-scope confirmation: [classical-alpha-gate.md](./classical-alpha-gate.md) known limit 6
