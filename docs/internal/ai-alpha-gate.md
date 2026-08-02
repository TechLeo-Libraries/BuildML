# AI alpha gate

Concrete exit criteria for declaring BuildML **2.3.0a1** AI operator alpha.
Sibling to [classical-alpha-gate.md](./classical-alpha-gate.md),
[dl-alpha-gate.md](./dl-alpha-gate.md), and
[rag-alpha-gate.md](./rag-alpha-gate.md). This is a release checklist, not a
capability wishlist.

Related docs: [quickstart-ai-alpha.md](../quickstart-ai-alpha.md) ·
[llm-m0-lock.md](./llm-m0-lock.md) · [llm-operator-phase-plan.md](./llm-operator-phase-plan.md) ·
[glossary.md](../glossary.md) · [editorial-standards.md](./editorial-standards.md)

---

## Verdict rubric

| Status | Meaning |
| --- | --- |
| **Pass** | Every **must** criterion below is green in CI or explicitly verified |
| **Fail** | Any **must** criterion is red, missing, or contradicted by docs |
| **Conditional** | Musts pass, but a listed known limit blocks a claimed workflow |

Assess readiness after CI: **Pass** when all must IDs are green; otherwise
**Fail** or **Conditional** per the known-limits section.

---

## Must criteria

### Security and key handling

| ID | Criterion | Evidence |
| --- | --- | --- |
| AS1 | API keys never appear in repr/str of `ProviderConfig` | `tests/unit/test_ai_slice.py` key tests |
| AS2 | API keys never persisted in transcripts (redacted before save) | Transcript unit tests |
| AS3 | API keys never echoed in error messages | Error message tests |
| AS4 | Tool calls go through typed allowlist; unknown tools raise `ValidationError` | Tool registry tests |
| AS5 | Injection patterns detected in user prompts, column names, and tool arguments | Injection hardening tests |
| AS6 | `FULL_SAMPLE` / `REDACTED_SAMPLE` egress requires explicit `confirm=True` | Egress confirmation tests |

### Privacy and egress

| ID | Criterion | Evidence |
| --- | --- | --- |
| AP1 | Default egress level is `STATS_ONLY` (no raw rows) | Privacy tests |
| AP2 | `ai_egress_preview` returns accurate manifest before any API call | Egress preview tests |
| AP3 | `ai_dry_run` returns full prompt payload without calling provider | Dry run tests |
| AP4 | Column deny/allow lists filter egress payload correctly | Column filter tests |
| AP5 | Transcript records egress manifests (what was sent), not raw data | Transcript schema tests |

### End-to-end smoke

| ID | Criterion | Evidence |
| --- | --- | --- |
| AE1 | Path: configure → egress_preview → dry_run → advisor → plan → execute (confirmed) → save/load transcript | `tests/unit/test_ai_slice.py` integration tests |
| AE2 | Smoke runs with `MockProvider` (no real API keys required) | CI `ai` job |
| AE3 | Write tools require confirmation; read-only tools do not | Tool confirmation tests |
| AE4 | Destructive tools (drop_columns) always require confirmation | Destructive tool tests |
| AE5 | Budget tracker enforces token/cost limits | Budget tests |

### Docs and catalog

| ID | Criterion | Evidence |
| --- | --- | --- |
| AD1 | Public Session `ai_*` methods have catalog entries | `buildml.explain.catalog` + AI unit tests |
| AD2 | Quickstart covers configure → egress → advisor → plan → execute → transcript and known limits | `docs/quickstart-ai-alpha.md` |
| AD3 | Concept notes exist for `ai-egress-privacy`, `ai-tool-trust`, `ai-prompt-injection` | Concept tests |
| AD4 | Editorial / user-copy lint clean | `scripts/lint_user_copy.py` in CI |
| AD5 | README documents `buildml[ai]` without claiming autonomous agent or production safety | `README.md` |

### CI and packaging

| ID | Criterion | Evidence |
| --- | --- | --- |
| AC1 | `import buildml` succeeds without AI extras | Core CI import smoke |
| AC2 | Dedicated `ai` CI job on Python 3.11–3.12 with AI unit tests (MockProvider only) | `.github/workflows/ci.yml` |
| AC3 | Missing openai raises `MissingExtraError("ai", ...)` with install hint | Missing-extra unit tests |
| AC4 | Version is `2.3.0a1` in `pyproject.toml` and `buildml/_version.py` | Packaging files |

---

## Should criteria (alpha-tolerant)

| ID | Criterion | Notes |
| --- | --- | --- |
| AW1 | Multi-step plan with batch approve / selective approve | M2 depth tests |
| AW2 | Token/cost budget tracking in `ai_status` | Budget display tests |
| AW3 | PII column detection warnings | Privacy heuristic tests |
| AW4 | Max iterations enforcement (default 10) | Iteration limit tests |
| AW5 | Walkthrough `ai_status` disclosures | Status display tests |

---

## Known limits (do not claim as done)

1. **Bring-your-own API key.** BuildML never ships, proxies, or embeds keys.
2. **Default egress is STATS_ONLY.** Raw rows require explicit opt-in and
   confirmation. Provider sees whatever egress payload the user approved.
3. **Propose → confirm → execute.** No autonomous agent, auto-execution, or
   autopilot mode in this alpha.
4. **Tool registry is the trust boundary.** The operator cannot execute
   arbitrary code or tools not in the allowlist.
5. **Transcript ≠ checkpoint ≠ bundle.** Three distinct artifacts; transcripts
   are separate from Session checkpoints and DL/RAG bundles.
6. **Not a replacement for Teaching Studio.** The operator supplements, not
   replaces, the explain catalog and structured results.
7. **Not fine-tuning LLMs inside BuildML.** Use DL domain or external tools.
8. **Advice must be verified.** Evidence-bound recommendations are not
   infallible. The operator can be wrong.
9. **No local-only provider path.** OpenAI-compatible protocol only in this
   alpha; local LLM support is later.
10. **CI runs with MockProvider only.** No real API keys in CI; integration
    tests with real providers are user responsibility.
11. **Public AI APIs and transcript formats may change before a stable release.**

---

## Smoke path (canonical)

```text
Session.ai_configure(provider="mock")
  → ai_egress_preview()            # manifest
  → ai_dry_run("question")         # payload without API call
  → ai_advisor("describe data")    # read-only Q&A
  → ai_plan("build model")         # structured steps
  → ai_execute(tool, params)       # proposal
  → ai_execute(tool, params, confirm=True)  # confirmed execution
  → save_ai_transcript / load_ai_transcript
  → explain("ai_advisor")          # catalog hit
```

CI entry: `pytest tests/unit/test_ai_slice.py -q`

---

## Sign-off checklist

Copy into a release note when cutting an AI alpha tag (see also
[release-checklist-ai-a1.md](./release-checklist-ai-a1.md)):

- [ ] AS1–AS6 green
- [ ] AP1–AP5 green
- [ ] AE1–AE5 green on CI `ai` job
- [ ] AD1–AD5 green
- [ ] AC1–AC4 green
- [ ] Known limits reviewed; README/quickstart/`CHANGELOG.md` do not contradict them
- [ ] Version is `2.3.0a1` in `pyproject.toml` and `buildml/_version.py`
- [ ] Changelog / history notes name this gate document
- [ ] Docs do not claim autonomous agent, auto-execution, or production safety

Tag only after remote CI is green on the release candidate push. Do not tag from
this checklist alone.
