# LLM operator phase plan

Next-phase plan after classical alpha `2.0.0a1`, DL alpha `2.1.0a1`, and RAG alpha `2.2.0a1`.  
Grounded in: [reconstruction-roadmap.md](./reconstruction-roadmap.md) ·
[deep-learning-phase-plan.md](./deep-learning-phase-plan.md) ·
[rag-phase-plan.md](./rag-phase-plan.md) ·
[quality-bar.md](./quality-bar.md) · [editorial-standards.md](./editorial-standards.md)

**Status:** M0 LOCKED · M1 COMPLETE · M2 COMPLETE.  
**M0 lock artifact:** [llm-m0-lock.md](./llm-m0-lock.md).  
**Sequencing (locked):** Classical ML → Deep Learning → RAG / modern methods → **LLM operator last**.  
**North star:** flexibility · depth · functionality.  
**Proposed version line:** `2.3.0a1` at M3 gate (after RAG `2.2.0a1`).

---

## 1. Goals and non-goals

### Goals (LLM operator phase / v1 domain)

1. Attach an **LLM operator domain** (`buildml.ai` or `buildml.llm`) that maps natural
   language to existing Session operations without turning `Session` into a second
   god-object.
2. Ship an **end-to-end ML pipeline operator**: the LLM guides and orchestrates the full
   classical (and later DL/RAG) ML process through BuildML Session ops (ingest → EDA →
   prep → split → fit/select → eval → checkpoint/export), always via tools, always
   evidence-bound.
3. Keep **core install lean**: `import buildml` never requires LLM clients. Missing extras
   raise `MissingExtraError` with an install hint.
4. **Bring-your-own API key** — users supply their own provider API key. BuildML never
   ships, proxies, or embeds keys. Keys must not be logged, persisted in transcripts/
   checkpoints/bundles by default, or echoed in errors/walkthroughs.
5. **Security is paramount** — design for adversarial users and prompt injection:
   - Tool-only actions via typed allowlisted registry mapped to explain catalog
   - Propose → confirm → execute by default; autopilot opt-in later
   - Treat retrieved docs / user text / column values as untrusted
   - No silent arbitrary code execution in v1
   - Operator cannot bypass leakage guards
6. **User-controlled privacy before LLM egress** — before any data/schema/sample is sent
   to an external LLM, the **user** controls redaction: anonymize values, drop specific
   columns, strip/rename headers, choose whether to send schema-only vs samples vs
   aggregates-only.
7. Meet the same quality bar as classical/DL/RAG: typed results, catalog coverage, tests,
   docstring standard, honest scale notes — not a thin wrapper around one SDK.
8. Reuse Teaching Studio / explain **principles** (evidence, limitations, progressive
   disclosure) with operator-appropriate surfaces: teach tool-calling, context, and safety.

### Non-goals (this phase)

| Non-goal | Rationale |
| --- | --- |
| Replacing Teaching Studio as the primary teaching surface | AI operator supplements, not replaces |
| Training / fine-tuning LLMs inside BuildML | Use DL or external tools |
| Claiming chat fluency equals model quality | Operator guides evidence-bound workflows |
| RAG generate as the primary LLM path | RAG generate stays separable; operator calls RAG if instructed |
| Shipping or proxying API keys for users | Explicit non-goal; security-critical |
| Silent arbitrary code execution | Propose → confirm → execute; no hidden exec |
| Bypassing leakage guards via operator | Leakage checks fire regardless of invocation source |
| Full autonomous agent with multi-step planning in M1 | Start advisor → executor → planner progression |
| Closed-source LLM dependency as the only path | Provider protocol; local-only path documented as future |

### Relationship to classical Session, Torch DL, and RAG

| Concern | Classical (`buildml.model`) | DL (`buildml.dl`) | RAG (`buildml.rag`) | LLM Operator (`buildml.ai`) |
| --- | --- | --- | --- | --- |
| Primary object | Tabular `Dataset` + estimator | `nn.Module` + loaders | Corpus / chunks + index | Session + tool registry |
| Fit / train API | `Session.fit` | `Session.fit_torch` | Index build / update | Guides / calls existing fit APIs |
| Artifacts | pipeline / model bundle | `buildml.torch_bundle.v1` | `buildml.rag_bundle.v1` | Transcript (separate from bundles) |
| Shared spine | Dataset, roles, splits, checkpoint, history, explain, core errors | same | same | same + tool registry |
| Generate / LLM calls | N/A | N/A | Optional grounded generate | Primary path (user-supplied provider) |
| Security scope | Leakage guards | Leakage guards | Index/corpus hygiene | Full untrusted-input discipline |

Classical, DL, and RAG APIs stay authoritative for their domains. The operator **orchestrates**
them; it does not invent separate fit/train/index logic and must not bypass domain-level
guards (leakage, contamination).

---

## 2. Architecture (no god-object)

### Package boundaries

```text
buildml/
  session/          # thin delegates only — no LLM-call bodies
  data/             # Dataset, roles, splits (shared)
  checkpoint/       # workflow data resume (shared); transcript elsewhere
  explain/          # catalog + concepts (shared schemas; operator ops register here)
  core/             # types, results, MissingExtraError, LeakageError
  model/            # classical sklearn only
  dl/               # Torch supervised trainers
  rag/              # retrieval domain
  ai/               # NEW — LLM operator implementation
    __init__.py     # lazy public exports; no eager LLM client import at package root
    extras.py       # require_ai_stack() → MissingExtraError("ai", ...)
    types.py        # ProviderConfig, ToolSpec, ConfirmPolicy, EgressManifest
    provider.py     # provider protocol (OpenAI-compatible + extensible)
    tools.py        # tool registry: name → Session method → schema → confirm level
    executor.py     # propose → confirm → execute dispatch
    advisor.py      # advisory Q&A (M1 focus): explain, describe, suggest
    planner.py      # multi-step plan (M2→M3): propose → approve → batch-execute
    privacy.py      # egress helpers: column allow/deny, redact, schema-only, samples
    transcript.py   # transcript store (separate from checkpoint/bundle)
    security.py     # injection tests, boundary enforcement, destructive-op confirm
    results.py      # AdvisorResult, ExecutorResult, PlanResult, TranscriptEntry
    explain_hooks.py  # catalog entries for operator ops
```

Optional later submodules (not v1 blockers): `buildml.ai.multi_turn`, `buildml.ai.local_provider`.

### Session attachment rules

1. **Delegate or do not exist.** `Session` methods call into `buildml.ai.*` and record
   history; they do not contain prompt construction or LLM client code.
2. **Lazy import.** `import buildml` and `from buildml import Session` must succeed without
   LLM clients (openai, anthropic, etc.). Heavy imports live inside
   `buildml.ai.extras.require_ai_stack()` when an entrypoint runs.
3. **Separate result slots.** Prefer `session.ai_result` / `session.ai_transcript`
   (names locked in M0) rather than overloading classical/DL/RAG result slots.
4. **Separate artifact kinds.**
   - Session **checkpoint** = data + roles + splits + history (+ optional classical plans).
   - DL/RAG **bundles** = weights / index / embeddings / domain config.
   - AI **transcript** = conversation history, tool calls, confirmations — stored
     **separately** from Session checkpoint. Secrets (API keys, raw data) never persisted
     in transcripts by default; egress manifests recorded instead.
5. **One implementation path.** No parallel "handy" advisor in Session and another in
   `buildml.ai`.

### Integration sketch

```text
Session (or standalone)
  → Session.ai_advisor(question)           # → buildml.ai.advisor (Q&A / explain)
  → Session.ai_execute(tool, params)       # → propose → confirm → execute
  → Session.ai_plan(goal)                  # → multi-step plan (M2+)
  → Session.ai_set_provider(config)        # → provider config (key via env or session)
  → Session.ai_egress_preview(...)         # → show what will leave the machine
  → Session.save_ai_transcript(...)        # → separate file; secrets redacted
  → session.explain("ai_advisor")          # → catalog hit
```

Naming is indicative; final public names land in M0 lock. Prefer a consistent `ai_*` Session
prefix so classical `fit` / Torch `*_torch` / RAG `rag_*` remain unambiguous.

### Shared spine reuse

| Shared | Reuse how |
| --- | --- |
| `MissingExtraError` | Extra name `ai` (and component extras if split) |
| Operation history | Record operator ops with typed `result_summary` dicts |
| Explain catalog schemas | New `OperationSpec` rows for operator ops |
| Checkpoint save/load | Resume data workflow; transcript is separate |
| Leakage guards | Operator tool calls checked by normal Session guards |
| Editorial standards | Observation → finding → recommendation; no overclaim; no AI-slop |

---

## 3. Locked product decisions

### 3.1 Bring-your-own API key

| Requirement | Implementation |
| --- | --- |
| Users supply their own provider API key | `ProviderConfig` accepts key via env var (`BUILDML_OPENAI_API_KEY`, etc.) or session-scoped config |
| BuildML never ships or proxies keys | No embedded keys in library; no proxy endpoint |
| Keys not logged in transcripts/checkpoints/bundles | Transcripts record egress manifests, not raw keys or data by default |
| Keys not echoed in errors/walkthroughs | Errors show `"API key not set"` — never the key value |
| Documentation patterns | Env-var config, session-config, redaction requirements in quickstart |

### 3.2 Security is paramount (public release)

| Principle | Implementation |
| --- | --- |
| Tool-only actions via typed allowlist | `ToolRegistry` maps allowed methods to explain catalog; no arbitrary `exec` |
| Registry mapped to explain catalog | Each tool has an `OperationSpec`; prerequisites/leakage/limits disclosed |
| Propose → confirm → execute (default) | `ConfirmPolicy.DEFAULT` = propose → user confirm → execute |
| Autopilot opt-in later | `ConfirmPolicy.AUTO` requires explicit session config + per-tool allowlist |
| Retrieved docs / user text / column values untrusted | System prompt boundaries; data payloads marked as untrusted |
| No silent arbitrary code execution in v1 | Executor runs only allowlisted methods; no `eval`/`exec` |
| Operator cannot bypass leakage guards | Tool calls go through Session delegates that enforce guards |
| Prompt-injection hardening | System/tool boundary markers; ignore instructions in data; CI injection tests |
| Destructive ops require confirmation | `ConfirmPolicy.DESTRUCTIVE` = always confirm for drop/delete/overwrite |
| Separate transcript store | Transcripts ≠ Session checkpoint ≠ DL/RAG bundles |
| CI with mocked provider | Unit tests use `MockProvider`; CI never needs real keys |

### 3.3 E2E ML pipeline operator

| Capability | Milestone | Notes |
| --- | --- | --- |
| Advisory Q&A (describe, explain, suggest) | M1 | Foundation advisor; no execution |
| Single-tool confirmed execution | M1→M2 | Propose → confirm → execute one method |
| Multi-step plan (ingest → fit → eval chain) | M2 | Batch propose → approve → batch execute |
| Evidence-bound recommendations | M1+ | Operator cites Session history, EDA findings, eval metrics |
| Full classical workflow orchestration | M2→M3 | ingest → EDA → prep → split → fit/select → eval → checkpoint |
| DL/RAG orchestration when domains attached | M3→L | Operator can guide `fit_torch`, `rag_retrieve` if extras present |

### 3.4 User-controlled privacy before LLM egress

| Requirement | Implementation |
| --- | --- |
| User controls redaction before any egress | `EgressConfig` + `egress_preview()` before send |
| Column allow/deny lists | `EgressConfig.allow_columns`, `deny_columns` |
| Header scrubbing / renaming | `EgressConfig.rename_columns`, `strip_headers` |
| Fake / anonymized samples | `privacy.generate_fake_sample(...)` helper |
| PII-ish column warnings | `privacy.detect_pii_columns(...)` heuristic + disclosures |
| Schema-only vs samples vs aggregates-only | `EgressLevel.SCHEMA_ONLY`, `STATS_ONLY`, `REDACTED_SAMPLE`, `FULL_SAMPLE` |
| Default is conservative | Default = `STATS_ONLY` (aggregates/findings, not raw rows) |
| Never auto-send full datasets | `FULL_SAMPLE` requires explicit opt-in per call |
| Egress manifest shown before send | `ai_egress_preview(...)` returns manifest; user confirms |
| Threat model documented | API provider sees whatever egress payload the user approved |

---

## 4. Capability map — LLM operator v1

Status tags: **M0** design · **M1** thin vertical slice · **M2** depth · **M3** docs/alpha · **L** later · **X** non-goal for operator v1.

### 4.1 Provider and configuration

| Capability | Tag | Notes |
| --- | --- | --- |
| Provider protocol (OpenAI-compatible) | M1 | Typed `ProviderConfig`; callable contract |
| API key from env var or session config | M1 | `BUILDML_OPENAI_API_KEY`, etc. |
| Key redaction in logs/errors/transcripts | M1 | Never echo; error says "not set" |
| Multiple provider backends (OpenAI, Anthropic, etc.) | M2 | Protocol extensible |
| Local-only provider path | L | Optional; retrieve-only advisor first |
| Token / cost budget limits | M2 | Per-call and per-session budget caps |
| Rate limits / max tool iterations | M2 | Configurable; default safe limits |

### 4.2 Tool registry and executor

| Capability | Tag | Notes |
| --- | --- | --- |
| Typed tool registry (name → method → schema) | M1 | Each tool linked to explain catalog |
| Read-only tools (describe, explain, summary) | M1 | No confirmation required |
| Confirmed tools (prep, fit, split) | M1→M2 | Propose → confirm → execute |
| Destructive tools (drop, delete) | M2 | Always confirm |
| Allowlist per mode (advisor vs executor) | M2 | Advisor = read-only; executor = write tools allowed |
| Autopilot opt-in | M3→L | Per-tool auto-confirm after explicit enable |
| Tool call audit log | M1 | Recorded in transcript (no secrets) |

### 4.3 Advisor (Q&A / explain)

| Capability | Tag | Notes |
| --- | --- | --- |
| Describe dataset, columns, roles, splits | M1 | Schema/summary egress only |
| Explain operation results (fit, evaluate, EDA) | M1 | Reference explain catalog |
| Suggest next steps based on history | M1 | Evidence-bound; cites Session state |
| Advisor refuses to execute (read-only) | M1 | Clear boundary |
| Teaching integration (link to concepts) | M2 | Return concept note references |

### 4.4 Executor (confirmed tool calls)

| Capability | Tag | Notes |
| --- | --- | --- |
| Propose single tool call | M1 | Return `ExecutorProposal` |
| Confirm before execute | M1 | User confirms; then dispatch |
| Execute and record result | M1 | Tool output in transcript |
| Rollback / undo not guaranteed | M1 | Document; checkpoints are the safety net |
| Refuse tools not in allowlist | M1 | Error names the denied tool |

### 4.5 Planner (multi-step)

| Capability | Tag | Notes |
| --- | --- | --- |
| Goal → multi-step plan | M2 | Propose N tool calls for user approval |
| Batch approve / selective approve | M2 | User can reject individual steps |
| Batch execute after approval | M2 | Sequential dispatch; stop on error |
| E2E classical workflow | M2→M3 | ingest → EDA → prep → split → fit → eval |
| E2E DL / RAG workflow | M3→L | Guide `fit_torch`, `rag_embed_and_index`, etc. |
| Autonomous replanning on failure | L | Deferred; M2 stops on error |

### 4.6 Privacy and egress

| Capability | Tag | Notes |
| --- | --- | --- |
| Egress levels (schema / stats / sample) | M1 | `EgressLevel` enum |
| Column allow/deny lists | M1 | Per-call config |
| Header scrubbing | M1 | Rename or strip |
| Egress manifest preview | M1 | Show what leaves before send |
| Fake / anonymized sample generator | M2 | `generate_fake_sample` helper |
| PII column heuristic warnings | M2 | Non-blocking warnings |
| Local dry-run of prompt payload | M1 | Return payload without calling provider |
| Cost / token budget enforcement | M2 | Refuse if budget exceeded |

### 4.7 Transcript and audit

| Capability | Tag | Notes |
| --- | --- | --- |
| Transcript store (separate from checkpoint) | M1 | Conversation + tool calls |
| Secrets redacted from transcript | M1 | Keys/raw data not persisted by default |
| Egress manifest in transcript | M1 | What was sent, when, which columns |
| Tool call log in transcript | M1 | Tool name, params (redacted), result summary |
| Save / load transcript | M1 | `save_ai_transcript` / `load_ai_transcript` |
| Transcript ≠ bundle | M1 | Never embed in checkpoint/DL/RAG bundles |

### 4.8 Security hardening

| Capability | Tag | Notes |
| --- | --- | --- |
| System / tool prompt boundaries | M1 | Clear markers; instructions in data ignored |
| Injection CI tests (malicious columns) | M1 | CI suite with adversarial fixtures |
| Injection CI tests (malicious RAG chunks) | M2 | When RAG integration ships |
| Max tool iterations per call | M2 | Default 10; configurable |
| Rate limit enforcement | M2 | Per-minute / per-session |
| Confirmation policy matrix | M2 | Per-tool confirm levels |

### 4.9 Explain / teaching hooks

| Capability | Tag | Notes |
| --- | --- | --- |
| Catalog entries for advisor/executor/planner | M1 | Prerequisites, security notes |
| Concept notes: prompt injection, egress privacy, tool trust | M1 | Link from operations |
| History + walkthrough awareness of operator ops | M1–M2 | `ai_status` disclosures |
| No AI-slop in catalog copy | M1 | Editorial standards enforced |
| Teaching Studio integration | L | Optional panel; not M1 blocker |

---

## 5. Making privacy and security better

Beyond the locked requirements above, these concrete process improvements strengthen the
privacy and security posture:

### 5.1 Privacy profiles

| Profile | What is sent | Use case |
| --- | --- | --- |
| `SCHEMA_ONLY` | Column names, dtypes, row count | Safest; advisor can suggest ops |
| `STATS_ONLY` | Aggregates, percentiles, cardinality | EDA-driven suggestions without rows |
| `REDACTED_SAMPLE` | N rows with PII columns masked/faked | Balanced context |
| `FULL_SAMPLE` | Raw rows (user opt-in only) | Debugging; requires explicit confirm |

Default: `STATS_ONLY`. User escalates when needed; operator refuses `FULL_SAMPLE` without
explicit opt-in.

### 5.2 Egress manifest shown before send

```python
manifest = session.ai_egress_preview(question="suggest next prep step")
# EgressManifest:
#   columns_sent: ["age", "income"]  (headers only, values aggregated)
#   rows_sent: 0
#   egress_level: "STATS_ONLY"
#   estimated_tokens: 320
#   cost_estimate: "$0.0012"
# User inspects and confirms before actual send.
```

### 5.3 Local dry-run of prompt payload

`ai_dry_run(...)` returns the full prompt payload (system, user, tool schemas) as a string
without calling the provider. Users can inspect exactly what would be sent.

### 5.4 Token / cost budgets

| Budget | Scope | Effect |
| --- | --- | --- |
| `max_tokens_per_call` | Single call | Refuse / truncate if exceeded |
| `max_tokens_per_session` | Cumulative | Refuse new calls when exhausted |
| `cost_limit_usd` | Cumulative | Track estimated cost; warn/refuse |

### 5.5 Allowlist of tools per mode

| Mode | Allowed tools | Confirm policy |
| --- | --- | --- |
| Advisor (read-only) | `describe`, `explain`, `summary`, `suggest` | None (no state change) |
| Executor (write) | All registry tools | Per-tool: read → auto, write → confirm, destructive → always |
| Autopilot | Configurable allowlist | Auto-confirm only for listed tools |

### 5.6 Audit log of egress + tool calls (without secrets)

Transcript records:
- Timestamp, tool name, params (column names, not values)
- Egress manifest (what was sent, level, columns, row count)
- Result summary (success/failure, metrics)
- Never: API keys, raw row values (unless `FULL_SAMPLE` and explicitly opted in)

### 5.7 Optional local-only provider path (later)

Document a protocol for local LLM providers (llama.cpp, ollama, etc.) so users who cannot
send data externally can still use the advisor/executor with a local model. Not M1; protocol
extensibility in M1 enables this later.

### 5.8 Injection tests in CI

| Test category | Fixtures | What it proves |
| --- | --- | --- |
| Malicious column names | `"; DROP TABLE users; --"` | Column names don't execute |
| Malicious RAG chunks | Chunk text with `Ignore previous instructions...` | System boundaries hold |
| Malicious user prompts | Prompt injection attempts | Tool registry limits execution |
| Nested injection | Data that looks like tool calls | Parser rejects |

CI runs with `MockProvider`; tests assert no unauthorized tool calls, no leaked secrets,
no boundary bypass.

### 5.9 Rate limits / max tool iterations

| Limit | Default | Rationale |
| --- | --- | --- |
| Max tool calls per `ai_plan` | 10 | Prevent runaway loops |
| Max retries per tool | 2 | Fail fast on repeated errors |
| API calls per minute | 20 | Respect provider limits; prevent abuse |

### 5.10 Confirmation policy matrix

| Tool category | Default policy | Autopilot policy |
| --- | --- | --- |
| Read-only (describe, explain) | Auto | Auto |
| State-changing (prep, fit) | Confirm | User allowlist |
| Destructive (drop, delete) | Always confirm | Never auto (hardcoded) |
| Egress-heavy (full sample) | Always confirm | Never auto (hardcoded) |

---

## 6. Packaging, Python, CI

### Extras

| Extra | Contents | Notes |
| --- | --- | --- |
| `ai` | Provider clients (openai), types, registry | Primary install hint |
| `llm` | Alias meta-extra → `buildml[ai]` | Alternative name if preferred |
| `all` (future) | classical + dl + rag + ai | Full workstation |

Rules (unchanged product law):

- `import buildml` on core alone.
- Missing stack → `MissingExtraError("ai", feature=...)` with
  `pip install 'buildml[ai]'`.
- No eager LLM imports in `buildml/__init__.py` or `session.py` module top-level.
- Core, DL, and RAG CI must remain green without `ai`.

### Python support

| Item | Direction |
| --- | --- |
| Classical + DL + RAG alphas | 3.10–3.13 core; DL/RAG CI subset 3.11–3.12 |
| AI phase | Keep 3.10–3.13 for core; AI CI subset chosen in M0 from client wheel reality |
| Decision point | M0: pin provider client versions after checking current wheels |

### CI shape

| Job | Role |
| --- | --- |
| Existing `test` | Core + classical; **must stay green without AI extras** |
| Existing `torch` | DL; unchanged by AI |
| Existing `rag` | RAG; unchanged by AI |
| New `ai` | `pip install -e ".[dev,ai]"`; AI unit + injection tests; **mocked provider only** |
| Import smoke | Assert `import buildml` still works without AI deps |

**CI never needs real API keys.** All AI tests use `MockProvider` that returns canned
responses. Injection tests run against the mock to verify boundary enforcement.

---

## 7. Phased milestones and exit criteria

### M0 — Design lock (docs + spikes)

**Deliverables**

- This plan approved (or amended with decision log entries).
- Short lock file `docs/llm-m0-lock.md`: public method names, provider protocol shape,
  transcript schema, extra pins, confirmation policy matrix, version line.
- Spike notes: provider round-trip latency; tool registry serialization; injection test
  fixtures.

**Exit (M0 lock checklist)**

- [ ] Public API sketch agreed (method names + result types)
- [ ] Provider protocol documented (OpenAI-compatible shape)
- [ ] Transcript vs checkpoint vs bundle boundary written and accepted
- [ ] Egress levels and default (`STATS_ONLY`) confirmed
- [ ] Confirmation policy matrix approved
- [ ] Injection test categories listed
- [ ] Extra pins + CI Python subset chosen
- [ ] Version line locked (`2.3.0a1` at M3)
- [ ] No production code required beyond optional spikes in a branch

### M1 — Thin vertical slice (advisor + single-tool executor)

**Deliverables**

- Package `buildml.ai` with lazy provider imports
- Provider protocol + OpenAI default backend
- Tool registry with read-only tools (describe, explain, summary)
- Advisor Q&A path (no execution)
- Single-tool executor with propose → confirm → execute
- Egress preview + levels (`SCHEMA_ONLY`, `STATS_ONLY`)
- Transcript save/load (secrets redacted)
- Session delegates + history recording
- Catalog entries for advisor/executor ops
- Injection test suite (mocked provider)
- CI `ai` job (mocked; no real keys)

**Canonical smoke**

```text
ai_set_provider(config)   # key from env
  → ai_advisor("describe the dataset")   # read-only
  → ai_egress_preview(...)               # manifest
  → ai_execute("set_roles", {...})       # propose → confirm → execute
  → save_ai_transcript(...)              # secrets redacted
  → explain("ai_advisor")                # catalog hit
```

**Exit**

- [ ] Core CI unchanged (no AI required)
- [ ] AI smoke green on CI CPU job (mocked provider)
- [ ] Injection tests green (malicious columns, prompts)
- [ ] `MissingExtraError` path tested without AI installed
- [ ] Typed results + docstrings for public delegates
- [ ] Transcript schema stable for the alpha line

### M2 — Depth (multi-step planner + security hardening)

**Deliverables**

- Multi-step planner: goal → plan → batch approve → batch execute
- E2E classical workflow (ingest → fit → eval) via planner
- Multiple provider backends (Anthropic, etc.)
- Token / cost budgets
- Rate limits / max iterations
- Confirmation policy matrix (per-tool levels)
- Fake sample generator + PII warnings
- Injection tests for RAG chunks (if RAG attached)
- Walkthrough / workflow status for AI ops (`ai_status`)

**Exit**

- [ ] Capability map M2 rows green or explicitly deferred with reason
- [ ] E2E classical workflow planner demo on a fixture
- [ ] Security hardening tests (rate limits, iteration caps)
- [ ] Teaching disclosures for egress privacy + tool trust

### M3 — Docs and AI alpha gate

**Deliverables**

- Quickstart for AI operator; glossary terms; known limits list
- Security guide: threat model, egress, injection, key handling
- AI alpha gate doc (sibling to `classical-alpha-gate` / `dl-alpha-gate` / `rag-alpha-gate`)
- README extras list includes `ai`
- Version bump: **`2.3.0a1`** (classical `2.0.0a1`, DL `2.1.0a1`, RAG `2.2.0a1`)

**Exit**

- [ ] AI alpha gate musts green
- [ ] Editorial lint clean for new user-facing strings (no AI-slop)
- [ ] Changelog + known limits do not claim autonomous agent / production safety
- [ ] Security doc reviewed

---

## 8. Risks and what stays classical / DL / RAG-only

### Risks

| Risk | Mitigation |
| --- | --- |
| Session grows another mega-facade | Hard rule: no LLM call code in `session.py`; review diffs for body length |
| AI deps leak into core import graph | Lazy require + CI job that installs core-only and imports buildml |
| Prompt injection leads to data exfil | System boundaries + injection tests + egress preview + user confirm |
| Keys logged in errors/transcripts | Redaction by default; tests verify no key echo |
| Users trust AI suggestions blindly | Evidence-bound recommendations; disclosures; no overclaim |
| Operator bypasses leakage guards | Tool calls go through normal Session delegates with guards |
| Cost runaway | Token/cost budgets; rate limits; max iterations |
| Security theater (checks without substance) | Injection CI tests; mocked provider; audit log review |
| Scope creep into autonomous agent | Explicit M1 boundary: advisor + single-tool; planner in M2 with approval |

### Stays classical-only

- `Session.fit` / CV / search / Optuna / pipeline bundles (operator can *call* these)
- Classical EDA Teaching Studio domain boards as the primary tabular exploration product
- `all-classical` extra contents (no AI in this meta-extra)

### Stays DL-only

- `Session.fit_torch` / `make_torch_loaders` / Torch trainer bundle resume
- Supervised net training curves as the DL teaching surface

### Stays RAG-only

- `Session.rag_*` methods / RAG bundle persistence / retrieval evals
- Corpus hygiene and index management

### Composition (allowed)

- Operator may *call* classical fit, Torch fit_torch, RAG retrieve via tool registry.
- Operator may use RAG retrieve to ground answers (but RAG generate stays separable).
- Shared explain/history/checkpoint patterns — not shared god-object methods.

---

## 9. Recommended immediate first implementation slice

After **M0 approval** (not before), implement **only** this vertical slice before multi-step
planner, multi-provider, or Teaching Studio integration:

1. Add `buildml/ai/` with `extras.require_ai_stack()` and typed result stubs.
2. `ProviderProtocol` + OpenAI default backend (key from env).
3. `ToolRegistry` with read-only tools: `describe_dataset`, `explain_result`, `summary`.
4. `AdvisorResult` for Q&A responses.
5. `ExecutorProposal` + `ExecutorResult` for propose → confirm → execute.
6. `EgressLevel` enum + `egress_preview(...)` returning manifest.
7. `TranscriptStore` with `save_ai_transcript` / `load_ai_transcript` (secrets redacted).
8. Thin `Session` `ai_*` delegates + history keys.
9. Catalog ops + concept notes (prompt injection; egress privacy; tool trust).
10. Injection test suite + mocked provider fixtures.
11. CI job `ai` (mocked; no real keys).

**Explicitly defer in that first PR series:** multi-step planner, autonomous replanning,
multiple provider backends, Teaching Studio redesign, local provider path, full RAG
integration, `buildml.ai.multi_turn`.

---

## 10. Decision log (M0 lock checklist — UNLOCKED)

| Topic | Status | Notes |
| --- | --- | --- |
| Public method prefix (`ai_*` vs `llm_*`) | Open | Propose `ai_*`; awaiting approval |
| Version line for AI alpha | Open | Propose `2.3.0a1` at M3 |
| Extra name canonical (`ai` vs `llm` vs both) | Open | Propose `ai` = deps; `llm` = alias |
| Provider protocol shape | Open | Propose OpenAI-compatible callable |
| Default egress level | Open | Propose `STATS_ONLY` |
| Confirmation policy matrix | Open | Propose read=auto, write=confirm, destructive=always |
| Transcript schema id | Open | Propose `buildml.ai_transcript.v1` |
| Transcript vs checkpoint vs bundle | Open | Propose three distinct artifacts |
| AI CI Python versions | Open | Propose 3.11 + 3.12 (mirror torch/rag) |
| Injection test categories | Open | Propose malicious columns, prompts, RAG chunks, nested |
| Max tool iterations default | Open | Propose 10 |
| Token budget default | Open | Propose no default (user sets) |

**Status:** All items open pending user M0 approval. Once approved, this section becomes the
locked decision log.

---

## References

- Architecture anti-pattern to avoid: mutable facade reimplementation
  ([architecture-review.md](./architecture-review.md))
- Domain attachment rule: [reconstruction-roadmap.md](./reconstruction-roadmap.md) §C–D, §H
- Classical completeness target: [classical-ml-capability-map.md](./classical-ml-capability-map.md)
- DL precedent for domain attachment: [deep-learning-phase-plan.md](./deep-learning-phase-plan.md),
  [dl-m0-lock.md](./dl-m0-lock.md)
- RAG precedent for domain attachment: [rag-phase-plan.md](./rag-phase-plan.md),
  [rag-m0-lock.md](./rag-m0-lock.md)
- Quality bar: [quality-bar.md](./quality-bar.md)
- Editorial standards (no AI-slop): [editorial-standards.md](./editorial-standards.md)
