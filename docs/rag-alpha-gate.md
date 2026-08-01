# RAG alpha gate

Concrete exit criteria for declaring BuildML **2.2.0a1** retrieval alpha.
Sibling to [classical-alpha-gate.md](./classical-alpha-gate.md) and
[dl-alpha-gate.md](./dl-alpha-gate.md). This is a release checklist, not a
capability wishlist.

Related docs: [quickstart-rag-alpha.md](./quickstart-rag-alpha.md) ·
[rag-m0-lock.md](./rag-m0-lock.md) · [rag-phase-plan.md](./rag-phase-plan.md) ·
[glossary.md](./glossary.md) · [editorial-standards.md](./editorial-standards.md)

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

### Hygiene and boundaries

| ID | Criterion | Evidence |
| --- | --- | --- |
| RL1 | Index corpus and eval query/qrel sets stay separate; indexing `eval_only` docs raises `LeakageError` | `tests/unit/test_rag_slice.py` hygiene tests |
| RL2 | Session checkpoint, Torch trainer bundle, and RAG bundle remain distinct artifacts | Bundle schema tests + RS1; `CHECKPOINT_BOUNDARY` |
| RL3 | Wrong-loader paths raise clear errors naming the expected schema id | Bundle unit tests |
| RL4 | Docs and catalog never imply generate/LLM operator shipped in this alpha | Quickstart, README, CHANGELOG, catalog copy |

### End-to-end smoke

| ID | Criterion | Evidence |
| --- | --- | --- |
| RS1 | Path: ingest → chunk → embed/index → retrieve (dense + hybrid) → evaluate → upsert/delete light → save/load bundle | `tests/integration/test_rag_alpha_smoke.py` |
| RS2 | Smoke runs under the CI `rag` job (Python 3.11–3.12); hashing path does not require GPU or cloud keys | CI `rag` job |
| RS3 | Default hashing embedder id and NumPy store are recorded on `IndexResult` / bundle meta | RS1 + unit slice |

### Docs and catalog

| ID | Criterion | Evidence |
| --- | --- | --- |
| RD1 | Public Session `rag_*` methods have catalog entries | `buildml.explain.catalog` + RAG unit tests |
| RD2 | Quickstart covers ingest → retrieve/eval → upsert/delete → bundle and known limits | `docs/quickstart-rag-alpha.md` |
| RD3 | Glossary covers RAG terms used in the alpha path | `docs/glossary.md` |
| RD4 | Editorial / user-copy lint clean | `scripts/lint_user_copy.py` in CI |
| RD5 | README documents `buildml[rag]` and Session `rag_*` APIs without claiming generate/LLM | `README.md` |

### CI and packaging

| ID | Criterion | Evidence |
| --- | --- | --- |
| RC1 | `import buildml` succeeds without RAG extras | Core CI import smoke |
| RC2 | Dedicated `rag` CI job on Python 3.11–3.12 with RAG unit + integration + alpha smoke | `.github/workflows/ci.yml` |
| RC3 | Missing sentence-transformers raises `MissingExtraError("rag", ...)` with install hint when ST/rerank backends are requested | Missing-extra unit tests |
| RC4 | Version is `2.2.0a1` in `pyproject.toml` and `buildml/_version.py` | Packaging files |

---

## Should criteria (alpha-tolerant)

| ID | Criterion | Notes |
| --- | --- | --- |
| RW1 | Hybrid dense + BM25 with RRF (default) or weighted fusion | Covered by M2 depth tests |
| RW2 | Optional cross-encoder rerank behind `buildml[rag]` | Missing-extra gate when absent |
| RW3 | Eval depth: nDCG@k, hit-rate@k, document\|chunk relevance, config compare | M2 unit tests |
| RW4 | Walkthrough / workflow `rag_status` disclosures | Index, embedder, store, last eval |
| RW5 | Metadata equality filters on retrieve | M2 unit tests |

---

## Known limits (do not claim as done)

1. **Hashing default is lexical/hashed, not semantic.** Disclose embedder id; do not
   market hashing as sentence-transformer quality.
2. **Local-first NumPy store.** No FAISS/Chroma/hosted vector-DB product path in
   this alpha.
3. **No generate / LLM operator.** No `rag_generate`; no `buildml.ai` tool-calling.
4. **No Teaching Studio RAG cockpit redesign.** Catalog + structured results +
   `rag_status` only.
5. **CPU merge gate.** RAG CI is CPU on Python 3.11–3.12; GPU embed/rerank is
   optional when available.
6. **PDF / OCR / HTML cleanup product** and multi-tenant ACL redaction are out of
   library scope for this alpha.
7. **Session checkpoints never embed the vector index.** Use `buildml.rag_bundle.v1`.
8. **Classical and Torch APIs are unchanged.** Do not treat a Torch trainer bundle
   as a vector index (or the reverse).

---

## Smoke path (canonical)

```text
Session.rag_ingest_corpus
  → rag_chunk
  → rag_embed_and_index          # hashing default
  → rag_retrieve (dense)
  → rag_retrieve (hybrid)
  → rag_evaluate (qrels)
  → rag_upsert / rag_delete      # light update
  → save_rag_bundle / load_rag_bundle
  → explain("rag_retrieve")
```

CI entry: `pytest tests/integration/test_rag_alpha_smoke.py -q`

---

## Sign-off checklist

Copy into a release note when cutting a RAG alpha tag (see also
[release-checklist-rag-a1.md](./release-checklist-rag-a1.md)):

- [ ] RL1–RL4 green
- [ ] RS1–RS3 green on CI `rag` job
- [ ] RD1–RD5 green
- [ ] RC1–RC4 green
- [ ] Known limits reviewed; README/quickstart/`CHANGELOG.md` do not contradict them
- [ ] Version is `2.2.0a1` in `pyproject.toml` and `buildml/_version.py`
- [ ] Changelog / history notes name this gate document
- [ ] Docs do not claim generate / LLM operator as shipped features

Tag only after remote CI is green on the release candidate push. Do not tag from
this checklist alone.
