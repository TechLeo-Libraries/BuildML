# RAG phase plan

Next-phase plan after classical alpha `2.0.0a1` and DL alpha `2.1.0a1`.  
Grounded in: [reconstruction-roadmap.md](./reconstruction-roadmap.md) ·
[deep-learning-phase-plan.md](./deep-learning-phase-plan.md) ·
[dl-m0-lock.md](./dl-m0-lock.md) ·
[classical-ml-capability-map.md](./classical-ml-capability-map.md) ·
[quality-bar.md](./quality-bar.md) · [editorial-standards.md](./editorial-standards.md)

**Status:** M0 locked · M1 in progress.  
**Sequencing (locked):** Classical ML → Deep Learning → RAG / modern methods → LLM operator last.  
**North star:** flexibility · depth · functionality.  
**M0 lock artifact:** [rag-m0-lock.md](./rag-m0-lock.md).

---

## 1. Goals and non-goals

### Goals (RAG phase / v1 domain)

1. Attach a **retrieval domain** (`buildml.rag`) that uses the same Session language
   (history, explain, checkpoints for data workflow) without turning `Session` into a
   second god-object.
2. Ship a **complete first RAG turn**: ingest corpus → chunk → embed → index → retrieve →
   (optional) generate → evaluate retrieval quality → persist a RAG bundle.
3. Keep **core install lean**: `import buildml` never requires embedding models, vector
   stores, or LLM clients. Missing extras raise `MissingExtraError` with an install hint.
4. Prefer **open / local-friendly defaults** (local embedders + on-disk or in-process
   indexes). Document GPU and optional API backends without requiring cloud keys for the
   happy path.
5. Meet the same quality bar as classical/DL: typed results, catalog coverage, tests,
   docstring standard, honest scale notes — not a thin wrapper around one vendor SDK.
6. Reuse Teaching Studio / explain **principles** (evidence, limitations, progressive
   disclosure) with RAG-appropriate surfaces: teach retrieval, chunking, and evals.
   Do **not** force classical EDA domain boards onto document corpora blindly.
7. Enforce **corpus hygiene**: train/index vs eval query sets stay separate; leakage of
   eval answers into the index is a hard failure mode with tests and catalog leakage notes.

### Non-goals (this phase)

| Non-goal | Rationale |
| --- | --- |
| Replacing classical `Session.fit` or Torch `fit_torch` | Those remain the supervised paths |
| Production multi-tenant vector DB / SaaS product | Library-first; local index + optional remote adapters later |
| Fine-tuning embedding or generative models as day-one identity | Use callable / local models; training belongs to DL or later |
| Agent frameworks / tool-calling product | That is `buildml.ai` — last in the locked sequence |
| Claiming cloud LLM quality without local fallback | Optional generate path; retrieval must work offline |
| Full enterprise document ETL (SharePoint, ACLs, OCR product) | Document limits; file/text/URL ingest first |
| Forcing classical tabular EDA Studio as the RAG cockpit | Separate disclosures; corpus stats ≠ VIF boards |
| Inverting sequencing to ship LLM operator before retrieval spine | Explicit reject without decision-log amendment |

### Relationship to classical Session and Torch DL

| Concern | Classical (`buildml.model`) | DL (`buildml.dl`) | RAG (`buildml.rag`) |
| --- | --- | --- | --- |
| Primary object | Tabular `Dataset` + estimator | `nn.Module` + loaders | Corpus / chunks + index |
| Fit / train API | `Session.fit` | `Session.fit_torch` | Index build / update (not sklearn fit) |
| Artifacts | pipeline / model bundle | `buildml.torch_bundle.v1` | `buildml.rag_bundle.v1` (proposed) |
| Shared spine | Dataset, roles, splits, checkpoint, history, explain, core errors | same | same patterns; corpus may be non-tabular |
| Embeddings | N/A | May reuse Torch for local models | Prefer local embedders; may depend on `torch` transitively via extra |
| Generate step | N/A | N/A | Optional; not required for retrieve-only v1 |

Classical and DL APIs stay authoritative for their domains. RAG methods are additive and
refuse cross-wiring (e.g. treating a Torch trainer bundle as a vector index) with clear
errors — never silent coercion.

---

## 2. Architecture (no god-object)

### Package boundaries

```text
buildml/
  session/          # thin delegates only — no chunk/embed/index bodies
  data/             # Dataset, roles, splits (shared; tabular prep still lives here)
  checkpoint/       # workflow data resume (shared); RAG index elsewhere
  explain/          # catalog + concepts (shared schemas; RAG ops register here)
  core/             # types, results, MissingExtraError, LeakageError
  model/            # classical sklearn only
  dl/               # Torch supervised trainers (separate; optional embed backend)
  rag/              # NEW — all retrieval-facing implementation
    __init__.py     # lazy public exports; no eager heavy imports at package root
    extras.py       # require_rag_stack() → MissingExtraError("rag", ...)
    types.py        # ChunkConfig, EmbedConfig, IndexConfig, RetrieveConfig
    corpus.py       # load documents / rows → CorpusHandle
    chunk.py        # splitters (fixed / recursive / structure-aware where tested)
    embed.py        # embedding backend protocol + local default
    store.py        # vector store protocol + default backend
    index.py        # build / update / delete-by-id
    retrieve.py     # dense (+ later hybrid) retrieval
    generate.py     # optional grounded generation behind a clear boundary
    evaluate.py     # retrieval metrics (recall@k, MRR, nDCG, …)
    checkpoint.py   # rag bundle save/load (≠ Session checkpoint)
    results.py      # IndexResult, RetrieveResult, RagEvalResult, …
    explain_hooks.py  # history summaries + catalog-facing result reading
```

Optional later submodules under the same extra (not v1 blockers): `buildml.rag.hybrid`,
`buildml.rag.rerank`, `buildml.rag.vision_docs`.

### Session attachment rules

1. **Delegate or do not exist.** `Session` methods call into `buildml.rag.*` and record
   history; they do not contain embedding loops or store SQL.
2. **Lazy import.** `import buildml` and `from buildml import Session` must succeed without
   sentence-transformers, FAISS/Chroma, or LLM clients. Heavy imports live inside
   `buildml.rag.extras.require_*` when an entrypoint runs.
3. **Separate result slots.** Prefer `session.rag_index_result` / `session.rag_retrieve_result`
   (names locked in M0) rather than overloading `fit_result` or `dl_train_result`.
4. **Separate artifact kinds.**
   - Session **checkpoint** = data + roles + splits + history (+ optional classical plans).
   - Torch **trainer bundle** = weights / opt / TrainConfig (`buildml.torch_bundle.v1`).
   - RAG **bundle** = chunk config, embedding model id + dim, index files, doc/chunk
     metadata, eval snapshots as configured. Do **not** embed index blobs in Session
     checkpoints (same law as DL: checkpoint ≠ domain bundle).
5. **One implementation path.** No parallel “handy” retriever in Session and another in
   `buildml.rag`.

### Integration sketch

```text
Session (or standalone CorpusHandle)
  → Session.rag_ingest_corpus(...)          # → buildml.rag.corpus
  → Session.rag_chunk(...)                  # → buildml.rag.chunk
  → Session.rag_embed_and_index(...)        # → embed + store/index
  → Session.rag_retrieve(query, k=...)      # → retrieve
  → Session.rag_evaluate(...)               # → evaluate (gold qrels)
  → [optional] Session.rag_generate(...)    # → generate (extra / API optional)
  → Session.save_rag_bundle(...) / load_rag_bundle(...)
  → session.explain("rag_retrieve") / workflow() sees RAG ops when registered
```

Naming is indicative; final public names land in M0. Prefer a consistent `rag_*` Session
prefix so classical `fit` / Torch `*_torch` remain unambiguous.

### Shared spine reuse

| Shared | Reuse how |
| --- | --- |
| `MissingExtraError` | Extra name `rag` (and component extras if split) |
| Operation history | Record RAG ops with typed `result_summary` dicts |
| Explain catalog schemas | New `OperationSpec` rows for RAG ops |
| Checkpoint save/load | Resume tabular/data workflow only; reload RAG via bundle |
| Editorial standards | Observation → finding → recommendation; no overclaim |
| DL / Torch (optional) | Local embedders may pull Torch via `rag` deps; not a Session DL call |

### Proposed extras layout

| Extra | Contents (indicative; pin in M0) | Notes |
| --- | --- | --- |
| `rag` | Local embedder + default vector store + light IO helpers | Canonical install: `pip install 'buildml[rag]'` |
| `rag-gpu` (optional later) | Same as `rag` with GPU wheel notes / CUDA index URL docs | Docs-first; not required for CPU CI |
| `rag-api` (optional later) | Optional remote embed/generate clients | Never required for retrieve-only path |
| `all` (future) | classical + dl + rag (+ ai when exists) | Do not fold RAG into `all-classical` |

**Default stack preference (open / local-friendly):**

- **Embeddings:** local sentence-transformer (or equivalent) with a small default model id
  documented in M0; callable/protocol escape hatch for custom embedders.
- **Vector store:** local on-disk or in-process backend chosen in M0 (candidates below).
- **Generate:** optional; local or API behind `rag_generate`; retrieval alpha must pass
  without any generate dependency.

**M0 store-backend candidates (pick one default + protocol):**

| Candidate | Pros | Cons |
| --- | --- | --- |
| FAISS (+ JSON/Parquet chunk sidecar) | Fast, local, common in RAG tutorials | Native wheels / platform notes |
| Chroma (persistent client) | Simple local persist API | Heavier; version churn |
| NumPy / sklearn NearestNeighbors + files | Tiny M1 surface; few deps | Scale ceiling; hybrid later harder |

Recommendation for planning: lock a **store protocol** in M0; ship M1 on the lightest
backend that meets save/load + kNN; allow a second backend in M2 if evidence warrants it
(roadmap multi-option rule: auto-default + override).

---

## 3. Capability map — RAG v1

Status tags: **M0** design · **M1** thin vertical slice · **M2** depth · **M3** docs/alpha · **L** later · **X** non-goal for RAG v1.

### 3.1 Ingest and chunk

| Capability | Tag | Notes |
| --- | --- | --- |
| Load text files / folder corpus | M1 | UTF-8; clear encoding errors |
| Load tabular text column → documents | M1 | Bridge from Session/Dataset rows |
| Chunk with size + overlap config | M1 | Deterministic IDs for resume/update |
| Structure-aware chunk (markdown / headings) | M2 | When fixtures prove value |
| PDF / OCR / HTML boilerplate cleanup product | L/X | Document limits; adapters later |
| Multi-tenant ACL / redaction product | X | Out of library scope |

### 3.2 Embeddings

| Capability | Tag | Notes |
| --- | --- | --- |
| Local default embedder behind protocol | M1 | Batch encode; dim recorded in bundle |
| Caller-supplied embed callable | M1 | Escape hatch; contract = list[str] → ndarray |
| GPU device for local embedder | M2 | Explicit device; CPU fallback with warning (mirror DL) |
| Remote API embedder | L | Optional `rag-api`; never block offline path |
| Train / fine-tune embedder in BuildML | X | Use DL or external tools |

### 3.3 Vector store and index

| Capability | Tag | Notes |
| --- | --- | --- |
| Build index from chunk embeddings | M1 | Typed `IndexResult` |
| Persist / load index + chunk metadata | M1 | Schema `buildml.rag_bundle.v1` |
| Delete / upsert by document or chunk id | M2 | Resume/update index |
| Metadata filters on retrieve | M2 | Declared fields only |
| Distributed / hosted vector DB product | X | Protocol may allow adapters later |

### 3.4 Retrieve

| Capability | Tag | Notes |
| --- | --- | --- |
| Dense top-k retrieve with scores | M1 | Query embed → search → ranked chunks |
| Hybrid dense + lexical (BM25-style) | M2 | Config-driven blend; document default |
| Rerank pass | M2 | Optional cross-encoder or score fusion |
| Multi-query / HyDE-style expansion | L | After eval harness exists |
| Agent tool loops | X | `buildml.ai` later |

### 3.5 Generate (optional)

| Capability | Tag | Notes |
| --- | --- | --- |
| Grounded generate from retrieved context | M2→L | Optional; alpha can be retrieve-only |
| Citation / source span attachments | M2 | When generate ships |
| Require cloud LLM for RAG alpha | X | Explicit non-goal |

### 3.6 Evaluate retrieval quality

| Capability | Tag | Notes |
| --- | --- | --- |
| Gold qrels → recall@k / MRR / nDCG@k | M1 | Minimal harness for the vertical slice |
| Chunk-level vs doc-level relevance modes | M2 | Document which mode is claimed |
| Faithfulness / answer grading | L | Needs generate + careful methodology |
| Compare embedders / chunk configs | M2 | Structured experiment table |

### 3.7 Checkpoint / bundle artifacts

| Capability | Tag | Notes |
| --- | --- | --- |
| Save/load RAG bundle | M1 | Distinct schema id; wrong-loader errors |
| Session checkpoint mid-loop (data only) | M1 | Existing API; document RAG resume recipe |
| Embed index inside Session checkpoint | X | Keep trust / size model honest |
| Version migration for rag_bundle.v1→v2 | L | After first stable alpha |

### 3.8 Explain / teaching hooks

| Capability | Tag | Notes |
| --- | --- | --- |
| Catalog entries for ingest/chunk/index/retrieve/eval/bundle | M1 | Prerequisites, leakage, result reading |
| Concept notes: chunk leakage, eval contamination, embedding drift | M1 | Link from operations |
| History + walkthrough awareness of RAG ops | M1–M2 | Status block when index attached |
| Retrieval diagnostic panels (hit lists, score gaps) | M2 | Structured results first; Studio later |
| Force classical EDA boards onto corpora | X | Teach retrieval/evals instead |

### 3.9 Leakage and data hygiene

| Capability | Tag | Notes |
| --- | --- | --- |
| Separate index corpus vs eval query set | M1 | Config + tests; refuse silent reuse |
| Refuse indexing labeled eval answers when declared | M1 | Catalog leakage field |
| Document train/index vs test query contamination | M1 | Teaching copy + anti-patterns |
| Time-aware corpus cutoff for queries | M2 | When timestamps exist |
| PII redaction product | L | Warnings + guidance first |

---

## 4. Explain catalog and Teaching Studio

### Reuse

- `OperationSpec` / prerequisites / leakage / anti-patterns / result_reading
- `CONCEPT_NOTES` registry and Concept Academy shape
- History records + `workflow()` / `walkthrough()` / `dry_run` resolution
- Editorial standards: observation → finding → recommendation; no overclaim
- Local-only HTML defaults; no network dependency for core reports

### Do not force classical EDA UI onto RAG

| Classical surface | RAG v1 approach |
| --- | --- |
| EDA domain boards (quality, VIF, drift, …) | Optional only when corpus is tabular text columns; not the primary RAG cockpit |
| Eval plot boards (ROC, residuals) | Retrieval metrics tables + ranked-hit diagnostics |
| Teaching Studio “preprocess scope” | RAG analogs: chunk scope, index corpus vs eval queries, embed model id/dim |
| `session.eda_app()` as primary RAG cockpit | Optional later; M1 ships catalog + structured results |

### Disclosure principles for RAG copy

- State **which corpus** was indexed and **which query set** was evaluated.
- State **embedder id**, dimension, and whether GPU/API was requested but unavailable.
- State **k** and metric definitions (recall@k is not “accuracy”).
- Never imply Session checkpoint contains the vector index.
- Never imply catalog “available” means the retrieval setup is adequate for production claims.
- Never imply generate step ran when only retrieve/eval ran.

---

## 5. Packaging, Python, CI

### Extras

| Extra | Contents | Notes |
| --- | --- | --- |
| `rag` | Local embedder + default store + declared helpers (pins in M0) | Primary install hint |
| `rag-gpu` / `rag-api` | Optional later | Docs + decision log; not M1 blockers |
| `all` (future) | classical + dl + rag (+ ai) | Keep `all-classical` free of RAG |

Rules (unchanged product law):

- `import buildml` on core alone.
- Missing stack → `MissingExtraError("rag", feature=...)` with
  `pip install 'buildml[rag]'`.
- No eager RAG imports in `buildml/__init__.py` or `session.py` module top-level.
- If the default embedder needs Torch wheels, document that `rag` may pull Torch
  transitively; still do **not** require `buildml[torch]` for classical-only users.
  Core CI must remain green without `rag`.

### Python support

| Item | Direction |
| --- | --- |
| Classical + DL alphas | 3.10–3.13 core; DL CI subset 3.11–3.12 |
| RAG phase | Keep 3.10–3.13 for core; RAG CI subset chosen in M0 from wheel reality (likely 3.11–3.12, mirror `torch` job) |
| Decision point | M0: pin embedder + store versions after checking current wheels |

### CI shape

| Job | Role |
| --- | --- |
| Existing `test` | Core + classical; **must stay green without RAG extras** |
| Existing `torch` | DL; unchanged by RAG unless shared fixtures require isolation |
| New `rag` | `pip install -e ".[dev,rag]"` (or lean pytest like `torch` job); RAG unit + integration smoke; Python subset matrix |
| Optional `rag-gpu` / API | Manual / scheduled; not a PR blocker |
| Import smoke | Assert `import buildml` still works in an env **without** RAG deps |

Mirror the `engines` / `optuna` / `torch` pattern: skip-friendly dedicated job, not weight on
every core matrix cell.

---

## 6. Phased milestones and exit criteria

### M0 — Design lock (docs + spikes)

**Deliverables**

- This plan approved (or amended with decision log entries).
- Short lock file `docs/rag-m0-lock.md`: public method names, store backend choice,
  bundle schema id, extra pins, Python matrix for RAG CI, version line.
- Spike notes: embed+index latency on a small fixture; CPU-only retrieve path;
  bundle directory layout.

**Exit**

- [x] Public API sketch agreed (method names + result types)
- [x] Default embedder + store backend chosen; protocol written
- [x] Bundle vs Session checkpoint vs Torch bundle boundary written and accepted
- [x] Extra pins + CI Python subset chosen
- [x] Version line locked (`2.2.0a1` at M3)
- [x] Lock file written (`docs/rag-m0-lock.md`); M1 implementation follows

### M1 — Thin vertical slice

**Deliverables**

- Package `buildml.rag` with lazy heavy imports
- Corpus → chunk → embed → index → retrieve → score (eval metrics) → RAG bundle save/load
- Session delegates + history recording
- Catalog entries for the slice operations
- Integration smoke test (CPU, local embedder) gated on `buildml[rag]`
- CI `rag` job

**Canonical smoke**

```text
rag_ingest_corpus (fixture docs)
  → rag_chunk
  → rag_embed_and_index
  → rag_retrieve(query, k=…)
  → rag_evaluate(qrels)   # recall@k / MRR at minimum
  → save_rag_bundle / load_rag_bundle
  → explain("rag_retrieve")  # catalog hit
```

**Exit**

- [ ] Core CI unchanged (no RAG required)
- [ ] RAG smoke green on CI CPU job
- [ ] Hygiene tests: eval queries/answers not silently indexed when marked eval-only
- [ ] `MissingExtraError` path tested without RAG installed
- [ ] Typed results + docstrings for public delegates
- [ ] Bundle schema id stable for the alpha line

### M2 — Depth

**Deliverables**

- Hybrid search and/or rerank (at least one depth path tested)
- Eval harness depth (doc vs chunk relevance; compare configs)
- Resume/update index (upsert/delete by id)
- Metadata filters where store supports them
- Walkthrough / workflow status for RAG ops
- Optional generate **only if** it stays optional and local/API boundaries are honest

**Exit**

- [ ] Capability map M2 rows green or explicitly deferred with reason
- [ ] Teaching disclosures for corpus vs eval set + embedder/device
- [ ] Quality bar: not “top-1 accuracy only”; structured retrieval diagnostics present

### M3 — Docs and RAG alpha gate

**Deliverables**

- Quickstart for RAG slice; glossary terms; known limits list
- RAG alpha gate doc (sibling to `classical-alpha-gate` / `dl-alpha-gate`) with must IDs
- README extras list includes `rag`
- Version bump: proposed **`2.2.0a1`** (classical `2.0.0a1`, DL `2.1.0a1`) — confirm in M0

**Exit**

- [ ] RAG alpha gate musts green
- [ ] Editorial lint clean for new user-facing strings
- [ ] Changelog + known limits do not claim LLM operator / agent product

---

## 7. Risks and what stays classical / DL-only

### Risks

| Risk | Mitigation |
| --- | --- |
| Session grows another mega-facade | Hard rule: no embed/index loops in `session.py`; review diffs for body length |
| RAG deps leak into core import graph | Lazy require + CI job that installs core-only and imports buildml |
| Eval contamination (answers in index) | Explicit corpus roles + tests + catalog leakage fields |
| Bundle / checkpoint / Torch confusion | Distinct schema ids; wrong-loader errors; docs boundary paragraph |
| Cloud-only defaults creep in | M0 locks local default; API path optional and documented as such |
| Scope creep into LLM operator / agents | Explicit sequencing (§8); reject feature PRs that invert order |
| Embedder wheel / Python matrix pain | RAG CI subset; classical keeps full matrix |
| “RAG” means generate-only demos | Alpha gate requires retrieve + eval + bundle without generate |

### Stays classical-only (for now)

- `Session.fit` / CV / search / Optuna / pipeline bundles
- Classical EDA Teaching Studio domain boards as the primary tabular exploration product
- `all-classical` extra contents

### Stays DL-only (for now)

- `Session.fit_torch` / `make_torch_loaders` / Torch trainer bundle resume
- Supervised net training curves as the DL teaching surface
- Architecture search / AMP / DDP (still DL non-goals or later)

### Composition (allowed)

- Tabular Session may supply a text column as corpus source.
- Local embedders may use Torch under the `rag` extra without calling `fit_torch`.
- Shared explain/history/checkpoint patterns — not shared god-object methods.

---

## 8. Explicit LLM-operator-later boundary

Locked vision (reconstruction roadmap): **Classical → DL → RAG → LLM operator**.

| Horizon | Package | Depends on | Starts when |
| --- | --- | --- | --- |
| Done / alpha | Classical Session spine | — | `2.0.0a1` |
| Done / alpha | `buildml.dl` + `torch`/`dl` | Shared data/checkpoint/explain | `2.1.0a1` |
| This plan | `buildml.rag` + extra `rag` | Stable Session/explain/checkpoint; local embed path | After this plan + M0 approval |
| Last | `buildml.ai` + extra `ai` | Broad, stable method catalog for tool-calling | After RAG method surface is documented (prefer after RAG M2) |

**RAG is not the LLM operator.** RAG may optionally call a generator for grounded answers;
it does not map natural language to arbitrary BuildML methods, does not own dry-run/execute
tool policy, and must not require API keys for the retrieve/eval alpha path.

**LLM operator constraints (preview, not this phase)**

- Maps natural language → **allowed existing methods** (dry-run / execute)
- Optional; core, DL, and RAG must work with no API keys
- No LLM dependency in `import buildml`
- Must not start as a fork of Session or as a second facade

---

## 9. Recommended immediate first implementation slice

After **M0 approval** (not before), implement **only** this vertical slice before hybrid
search, Studio redesign, or agent spikes:

1. Add `buildml/rag/` with `extras.require_rag_stack()` and typed result stubs.
2. `CorpusHandle` from fixture docs + optional tabular text column bridge.
3. Deterministic `chunk_documents(...)` (size/overlap) with stable chunk ids.
4. Embed protocol + one local default embedder (CPU).
5. Default store: build index → dense top-k retrieve with scores.
6. Minimal `evaluate_retrieval(...)` (recall@k + MRR on gold qrels).
7. `save_rag_bundle` / `load_rag_bundle` round-trip (`buildml.rag_bundle.v1`).
8. Thin `Session` `rag_*` delegates + history keys.
9. Catalog ops + concept notes (eval contamination; chunk/index vs query set).
10. Tests: missing-extra, contamination refusal, CPU smoke; CI job `rag`.

**Explicitly defer in that first PR series:** hybrid/BM25, rerank, generate/LLM calls,
PDF/OCR productization, hosted vector DBs, Teaching Studio redesign, `buildml.ai`.

---

## Decision log (locked in M0)

| Topic | Status | Notes |
| --- | --- | --- |
| Public method prefix (`rag_*` vs `*_rag`) | Locked | `rag_*` Session methods; result slots `rag_index_result` / `rag_retrieve_result` / `rag_eval_result` |
| Version line for RAG alpha | Locked | `2.2.0a1` at M3 (classical `2.0.0a1`, DL `2.1.0a1`) |
| Extra name canonical | Locked | `rag` = deps; optional later `rag-gpu` / `rag-api` |
| Default embedder + pin | Locked | Default `buildml.hashing_embed.v1` (HashingVectorizer, dim=384); extra pins `sentence-transformers>=2.2` for gate + optional ST backend |
| Default vector store | Locked | NumPy cosine top-k + `embeddings.npy` / `chunks.jsonl` sidecar |
| Bundle schema id | Locked | `buildml.rag_bundle.v1` |
| Bundle vs Session checkpoint vs Torch | Locked | Three distinct artifacts; see §2 and [rag-m0-lock.md](./rag-m0-lock.md) |
| Generate in M1 vs M2+ | Locked | Retrieve+eval+bundle in M1; generate deferred (M2→L) |
| RAG CI Python versions | Locked | 3.11 + 3.12 (mirror `torch` job) |
| Whether tabular text auto-ingests from current Session frame | Locked | Explicit `rag_ingest_corpus` only (no silent full-frame index) |

---

## References

- Architecture anti-pattern to avoid: mutable facade reimplementation
  ([architecture-review.md](./architecture-review.md))
- Domain attachment rule: [reconstruction-roadmap.md](./reconstruction-roadmap.md) §C–D, §H
- Classical completeness target (RAG marked X): [classical-ml-capability-map.md](./classical-ml-capability-map.md) §10
- DL precedent for domain attachment: [deep-learning-phase-plan.md](./deep-learning-phase-plan.md),
  [dl-m0-lock.md](./dl-m0-lock.md)
- DL alpha out-of-scope confirmation pattern: [dl-alpha-gate.md](./dl-alpha-gate.md)
