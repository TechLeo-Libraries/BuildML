# BuildML reconstruction roadmap

Decision-oriented planning map for a full system rewrite · grounded in the 1 Aug 2026 architecture audit · updated with owner planning decisions (Aug 2026).

> Related planning docs:  
> - [architecture-review.md](./architecture-review.md)  
> - [classical-ml-capability-map.md](./classical-ml-capability-map.md)  
> - [ingest-engine-checkpoint-design.md](./ingest-engine-checkpoint-design.md)  
> - [phase-1-build-plan.md](./phase-1-build-plan.md)  
> - [deep-learning-phase-plan.md](./deep-learning-phase-plan.md)  
> - [rag-phase-plan.md](./rag-phase-plan.md)  
> - [llm-operator-phase-plan.md](./llm-operator-phase-plan.md) *(draft — pending M0 approval)*  
> - [development-environment.md](./development-environment.md)  
> Canvas twin:  
> `C:/Users/leona/.cursor/projects/c-Users-leona-Desktop-Github-Projects-BuildML/canvases/buildml-reconstruction-roadmap.canvas.tsx`

| Signal | Locked direction |
| --- | --- |
| Vision | Hybrid build platform (learners + professionals) |
| North star | Flexibility · depth · functionality |
| Release shape | 2.0 reconstruction (clean break) |
| Depth bar | Full workflow coverage — not mediocre / basics-only |
| Scale | Automated ingest + multi-engine from day one |
| Backends | Multi-option (auto-default + override); both Polars and DuckDB |
| Packaging | Lean core + expanding extras |
| Python | 3.10–3.13 |
| Expansion | Classical ML → Deep Learning → RAG → LLM operator (last) |
| Docs bar | Extensive docstrings + guides for every public method |
| LLM operator | Draft plan pending M0 approval ([llm-operator-phase-plan.md](./llm-operator-phase-plan.md)); not blocking core |
| Phase 0 | **Closed** |

**Plan before build:** Incremental patching cannot fix duplicated facade logic, leakage-by-default workflows, broken paths, and packaging/import failure.

---

## Locked product principles

**North-star triad:** flexibility · depth · functionality.

1. **Dual audience.** Simple enough for learners; deep enough for professionals. Progressive disclosure, not a shallow subset.
2. **Method-simple session API.** Users call methods on a session/dataset object (`drop_columns`, `eda`, `split`, `fit`, …) instead of assembling ad-hoc library glue.
3. **Delegate, don’t duplicate.** The session orchestrates; real implementations live in domain packages. No 6,600-line god object.
4. **Leakage safety is non-negotiable.** Fit on train (or declared partitions) only; transforms apply to other partitions; invalid orders fail clearly.
5. **Scale-ready ingest from day one.** Dataset is a handle over engines (Arrow/Parquet interchange; Polars **and** DuckDB for large ops; Pandas as a materialization bridge). Modes: memory / lazy / out-of-core.
6. **Multi-option flexibility.** When two approaches are both useful and efficient, support **both** behind one API — auto-select a sensible default, let users override. Do not artificially lock the product to a single backend/choice.
7. **Automate the mechanical bits.** Ingest should detect format, schema, scale/load characteristics, and recommend engine/mode; users decide only where judgment matters.
8. **Mid-loop exit and reentry.** Export checkpoints; work outside BuildML; reattach with validation. Professionals need escape hatches without silent corruption.
9. **Platform, not a single technique.** Start with complete classical ML; architect domains for deep learning, RAG, and future methods.
10. **Core + extras expand capability.** Lean installable core; extras unlock depth without forcing every dependency on every user. Missing an extra yields a clear install hint — never a silent dead end.
11. **Documentation is release-critical.** Every public method ships with purpose, args, returns, examples, leakage notes, scale notes, and related methods.
12. **AI operator is optional and last in sequence.** Natural-language control maps to real methods via tool-calling; core library must work without an LLM. See [llm-operator-phase-plan.md](./llm-operator-phase-plan.md) for full design (draft pending M0 approval).
13. **Depth is mandatory — no thin wrappers.** Every capability (EDA, preprocess, model, eval, dates, persistence, future domains) must be extensive: cover the full professional surface, adaptive behavior, creative high-impact UX where visual, and honest scale handling. “Core-only” means install weight, not shallow functionality.

---

## A. Product vision

| Option | Status |
| --- | --- |
| Educational-only | Rejected as sole identity |
| Production-only | Rejected as sole identity |
| **Hybrid build platform** | **Locked** |

BuildML is a unified **build session** for data work: classical ML first, then deep learning and RAG-style flows, always behind a simple method API with professional depth underneath.

---

## B. Keep vs discard (v1 → v2)

| Disposition | Item | Rationale |
| --- | --- | --- |
| Keep concept | End-to-end build workflow | Ingest → EDA → prepare → split → fit → evaluate → compare → export |
| Keep concept | OOP session surface | Method calls on owned data remain the product experience |
| Keep carefully | Estimator-agnostic training | Caller-supplied estimators/models with structured results |
| Discard / rewrite | Mutable god-object architecture | Primary design failure |
| Discard / rewrite | Duplicate facade vs module logic | One source of truth only |
| Discard | Broken date / sort / model-key paths | Replace with tested implementations |
| Discard | Fit-on-full-data preprocessing | Leakage by default |
| Discard | Eager optional imports | Root import must stay lean |
| Discard | Unpinned / wrong dependency names | Truthful packaging |
| Discard | Tracked generated noise | Hygiene policy |
| Locked | 2.0 clean break | Migration notes; shim only if later demanded |

---

## C. Platform architecture

```text
BuildML Session
├── ingest / data / checkpoint     shared spine
├── classical ml                   v2 focus (full depth)
├── deep learning                  later domain
├── rag / retrieval                later domain
└── future domains                 attach without rewriting core
```

| Package | Responsibility | Constraint |
| --- | --- | --- |
| `buildml.core` | Types, results, errors, validation | Lean baseline deps |
| `buildml.ingest` | Sources, schema, partitioning, engine selection | Arrow/Parquet canonical interchange |
| `buildml.data` | Dataset handle, roles, splits, modes | Forbids silent full-data fit |
| `buildml.checkpoint` | Export / reattach bundles + validation | Mid-loop exit is first-class |
| `buildml.preprocess` | Leakage-safe transformers | Fit on train only |
| `buildml.model` | Classical fit / evaluate / compare / probabilities | Honest validation for sweeps |
| `buildml.eda` | Summaries and plots | Lazy optional backends; sample-aware for large data |
| `buildml.pipeline` | Recipe + fitted artifact bundle | Persist preprocess + model together |
| `buildml.session` | OOP method API | Delegates only |
| `buildml.dl` | Deep learning domain (later) | Same session language, separate extras |
| `buildml.rag` | RAG / retrieval domain (later) | Same session language, separate extras |
| `buildml.ai` | LLM operator (later) | Maps NL → allowed methods; optional |

### Design rules

- One implementation path: session delegates or the feature does not exist.
- Fit on train partitions only; transform applies to validation/test/infer.
- Return typed result objects with stable field names.
- Optional extras imported lazily inside feature entrypoints.
- Domain expansion attaches to shared data/checkpoint/result/docs patterns.

### Non-goals for v2 foundation

- No server, CLI, or database required to use the library.
- No preserving byte-identical mutable flag machines from `automate/_automate.py`.
- No requiring profiling / DL / RAG stacks for `import buildml`.
- No claiming every sklearn/DL model trains a full terabyte in-process on day one — TB means ingest/transform/prepare without full RAM residency.

---

## D. Domain roadmap

| Horizon | Domain | Goal | Status |
| --- | --- | --- | --- |
| **v2 foundation** | Ingest, session, checkpoints, leakage-safe classical ML spine | Importable, tested, documented core | `2.0.0a1` |
| **v2 depth** | Complete classical ML turns (probs, calibration, thresholds, CV/search, persistence, rich eval) | Professionals not limited | `2.0.0a1` |
| **v3 (DL)** | Deep learning domain | Session-attached trainers/eval/export | `2.1.0a1` |
| **v3 (RAG)** | RAG / retrieval domain | Document → chunk → embed → index → retrieve → generate (+ eval) | `2.2.0a1` |
| **v3 (LLM operator)** | LLM operator domain | Tool-calling over real BuildML methods; E2E ML pipeline orchestration | Draft plan — [llm-operator-phase-plan.md](./llm-operator-phase-plan.md) |
| **Ongoing** | New methodologies as domains | Expand without rewriting the spine | — |

**Locked sequencing:** Classical ML → Deep Learning → RAG → **LLM operator last**.

**LLM operator plan:** See [llm-operator-phase-plan.md](./llm-operator-phase-plan.md) for the
full phase plan (M0–M3 milestones, API sketch, security/privacy requirements, open questions).
Status: draft pending M0 approval; proposed version `2.3.0a1` at M3 gate.

**Expansion rule:** New tools attach as domain packages + extras. Do not grow a second mega-facade.

---

## E. Reconstruction workstreams

| Workstream | Priority | Scope | Exit signal |
| --- | --- | --- | --- |
| Architecture rewrite | P0 | Layered packages, session delegation, ingest spine | Core import works; no duplicate transform bodies |
| Correctness / validation | P0 | Leakage, dates, sort, keys, samplers, honest search | Regression + leakage tests green |
| Cleaning / packaging | P1 | Hygiene, `pyproject.toml`, extras, Python matrix | Metadata truthful; generated files untracked |
| Depth / quality | P1 | Full classical capability map, typed results, docs-as-tests | Docs/examples CI green |
| Scale spine | P1 | Engines, modes, sample-aware EDA, materialization gates | Large-data path documented + tested at fixture scale |
| Checkpoint / reattach | P1 | Export bundle + reentry validation matrix | Mid-loop round-trip tests green |
| Platform expansion | P2 | DL, RAG, other domains after classical depth gates | Each domain meets docs/tests/extras bar |

### Audit defects that must die

1. Date extraction: missing `.dt`, unreachable branch, undefined `items`
2. `sort_index` unsupported `by` / `ignore_index`
3. Model result key mismatch (`Built Model` vs `Model`)
4. Invalid sampler paths should raise clear errors
5. Imputation/scaling/encoding/sampling/feature selection fitted on full data
6. Standalone prediction scalers never fit
7. KNN k on training score; polynomial CV on unexpanded X; feature sweeps mutate X
8. Eager optional imports and unsatisfiable requirements

---

## F. Validation strategy

| Layer | What it proves | Minimum bar |
| --- | --- | --- |
| Unit | Transformers, metrics, dates, samplers, schemas, errors | Every public function; P0 regressions |
| Integration | ingest→prepare→split→fit→predict→evaluate; session delegation | No leakage; artifacts travel with model |
| Checkpoint | Export → external edit cases → reattach | Validation matrix covered |
| Docs / examples | Sphinx + README as executable tests | Extensive docstrings; no stale examples |
| Smoke | Clean venv install core/extras; `import buildml` | Core import without heavy optionals |
| CI matrix | Supported Python × OS; lint, typecheck, tests, build | Fail merge on P0 / import failure |
| Release gates | Changelog, version single-source, trusted publish | No unpinned deps; extras documented; tags exist |

**Hard release blockers for v2.0:** core import smoke; P0 regressions; leakage fit-scope tests; checkpoint round-trip tests; wheel/sdist build+install smoke; docs standard met for shipped public API.

---

## G. Suggested release phases

| Phase | Focus | Exit criteria |
| --- | --- | --- |
| 0 · Decisions | Principles, domains, engines, checkpoints | Written and agreed (this doc + siblings) |
| 1 · Foundation | Packaging, import graph, core types, ingest handle, CI | `import buildml` on core install; CI skeleton green |
| 2 · Classical parity (correct) | Reimplement v1 capabilities with correct semantics | Parity checklist + P0/P1 regressions green |
| 3 · Classical depth | Probabilities, calibration, thresholds, rich eval, search, persistence | Capability map “v2 depth” items green |
| 4 · Scale hardening | Lazy/out-of-core paths, materialization gates | Engine modes tested; docs honest about limits |
| 5 · Platform expansion | DL domain (`2.1.0a1`), then RAG (`2.2.0a1`) | Per-domain validation + docs bar |
| 6 · LLM operator | Optional LLM method orchestration (proposed `2.3.0a1`) | Tool-calling, security, privacy — see [llm-operator-phase-plan.md](./llm-operator-phase-plan.md) |

---

## H. Packaging: core vs extras

**Intent:** Core stays reliably installable and importable. Extras expand engines, reporting, IO, and future domains. Users can combine extras (`pip install buildml[polars,duckdb,eda]`) or use meta-extras for common profiles.

### Core (always)

| Area | Contents |
| --- | --- |
| Runtime baseline | `numpy`, `pandas`, `pyarrow`, `scikit-learn` |
| Product surface | Session, Dataset contract, roles/splits, leakage guards |
| Ingest | DataFrame / CSV / Parquet / Arrow; schema + scale detection; mode recommendation |
| Classical spine | Preprocess + model fit/predict/metrics (sklearn-compatible) |
| Checkpoint | Save/reattach bundle (Parquet + metadata) |
| Bridge | Pandas/NumPy materialization path |

### Extras (expand on demand)

| Extra | Unlocks | Notes |
| --- | --- | --- |
| `polars` | Polars engine adapter | Large/lazy tabular path |
| `duckdb` | DuckDB engine adapter | SQL/scan ally; both engines supported |
| `engines` | `polars` + `duckdb` | Convenience meta-extra |
| `imbalanced` | `imbalanced-learn` | Resampling strategies |
| `viz` | `matplotlib`, `seaborn` | Standard plots |
| `reports` | Sweetviz / ydata-profiling (or successors) | Heavy HTML profiling |
| `eda` | `viz` + `reports` | Convenience meta-extra |
| `excel` | openpyxl / Excel IO | Optional spreadsheet path |
| `persist` | Preferred persistence helpers (e.g. skops) when adopted | May fold into core later if tiny |
| `all-classical` | `engines` + `imbalanced` + `eda` + `excel` + `persist` | Full classical workstation |
| `dl` | Deep learning stack (later) | Domain extra |
| `rag` | Retrieval / embedding stack (later) | Domain extra |
| `ai` | LLM client extras (later) | Operator extra |
| `all` | Everything published | Power users / CI matrices |

**Rules**

- `import buildml` succeeds on **core alone**.
- Calling a feature that needs a missing extra raises a clear error with the install extra name.
- Auto-ingest may *recommend* `polars`/`duckdb` when scale warrants it; it never hard-crashes the core path without saying how to unlock the engine.
- Prefer supporting multiple useful backends over premature single-backend lock-in.

---

## I. Decision log

| Topic | Status | Decision |
| --- | --- | --- |
| Product vision | **Locked** | Hybrid build platform for learners + professionals |
| API compatibility | **Locked** | 2.0 clean break + migration notes |
| Depth ambition | **Locked** | Full classical workflow turns — not basics-only |
| North-star triad | **Locked** | Flexibility · depth · functionality |
| Multi-option backends | **Locked** | Support both when useful; auto-default + user override |
| Automated ingest | **Locked** | Detect format/schema/scale; recommend engine/mode |
| Scale strategy | **Locked** | Engines from day one; Arrow/Parquet interchange |
| Engine baseline | **Locked** | Pandas bridge + **both** Polars and DuckDB adapters |
| Mid-loop flexibility | **Locked** | Checkpoint export/reattach with validation |
| Platform expansion | **Locked** | Classical → DL → RAG / modern methods |
| Documentation bar | **Locked** | Docs are release-critical |
| Python support | **Locked** | 3.10–3.13 for v2; revisit when DL domain lands |
| Core vs extras | **Locked** | Lean core + expanding extras (see §H) |
| Mutation semantics | **Locked** | Immutable pipeline outputs; optional explicit `inplace=` on session helpers |
| Who owns the split? | **Locked** | BuildML-owned default + explicit pre-split injection with leakage checks |
| Report privacy | **Locked** | Local-only defaults, warnings, opt-in sample/redaction guidance |
| Checkpoint format | **Locked** | Data files + sidecar metadata bundle |
| Phase-1 entry order | **Locked** | Types → Dataset(memory) → Session → splits → checkpoint → CI → engine adapters → classical depth |
| LLM operator | **Draft plan** | [llm-operator-phase-plan.md](./llm-operator-phase-plan.md) pending M0 approval; sequenced after RAG |
| Split membership storage | Open (impl) | Row IDs vs positions vs recipe+seed — decide at implementation |
| Artifact serializer | Open (impl) | joblib vs skops vs directory — decide at implementation |

---

## Phase 0 status

**Closed** for product direction.

## Phase 1 status

**Build plan:** [phase-1-build-plan.md](./phase-1-build-plan.md)

**In progress (2026-08-01):**

- `pyproject.toml` + version `2.0.0a1`
- Project `.venv/` + frozen `requirements.txt` / `requirements-dev.txt`
- Legacy 1.x in `buildml/_legacy/` (not exported)
- Packages: `core`, `ingest`, `data`, `checkpoint`, `session`
- `Session.ingest` / roles / split / inject_split / leakage guard / checkpoints
- Automated ingest report + large-file refusal heuristics
- Tests + GitHub Actions CI skeleton (pytest green)

**Phase-2 classical depth in progress:**

- Extensive adaptive EDA (`EDAReport` + optional viz)
- Deep evaluate diagnostics
- Date feature extraction, model persistence, multi-model compare
- Quality bar locked: [quality-bar.md](./quality-bar.md)

**Next implementation:** calibration/threshold tools, richer plot extras for eval, imbalance strategies, learning curves.
