# RAG M0 design lock

Approved lock for the retrieval thin slice.  
Parent plan: [rag-phase-plan.md](./rag-phase-plan.md).

**Status:** M0 locked · M1 in progress  
**Approved:** 2026-08-01

---

## Public API (Session delegates)

| Method | Role |
| --- | --- |
| `Session.rag_ingest_corpus(...)` | Load text files / folder, or a tabular text column, into a `CorpusHandle` |
| `Session.rag_chunk(...)` | Split corpus documents with size + overlap; stable chunk ids |
| `Session.rag_embed_and_index(...)` | Embed chunks and build the default vector index |
| `Session.rag_retrieve(query, k=...)` | Dense top-k retrieve with scores |
| `Session.rag_evaluate(...)` | Retrieval metrics on gold qrels (recall@k, MRR) |
| `Session.save_rag_bundle(path)` | Persist RAG bundle (`buildml.rag_bundle.v1`) |
| `Session.load_rag_bundle(path)` | Restore a RAG bundle into the Session |

Result slots (distinct from classical `fit_result` and Torch `dl_train_result`):

| Slot | Type |
| --- | --- |
| `session.rag_index_result` | `IndexResult` |
| `session.rag_retrieve_result` | `RetrieveResult` |
| `session.rag_eval_result` | `RagEvalResult` |

Prefix rule: `rag_*` keeps classical `fit` / Torch `*_torch` unambiguous.

Canonical smoke:

```text
rag_ingest_corpus → rag_chunk → rag_embed_and_index
  → rag_retrieve → rag_evaluate → save_rag_bundle / load_rag_bundle
  → explain("rag_retrieve")
```

---

## RAG bundle vs Session checkpoint vs Torch bundle

| Artifact | Schema id | Contains | Does not contain |
| --- | --- | --- | --- |
| Session checkpoint | existing checkpoint formats | data, roles, splits, history, optional classical plans | vector index, chunk embeddings, Torch weights |
| Torch trainer bundle | `buildml.torch_bundle.v1` | module/optimizer state, TrainConfig, history, feature contract | dataset rows, RAG index |
| RAG bundle | `buildml.rag_bundle.v1` | chunk config, embedder id + dim, embeddings, chunk metadata, index knobs, optional eval snapshot | Session dataset rows, Torch trainer weights |

Data resume: keep or `checkpoint_load` the Session for tabular workflow; reload retrieval via `load_rag_bundle`. Do not embed the index inside a Session checkpoint.

Layout:

```text
<path>/
  meta.json         # format, buildml_version, configs, embedder id/dim
  chunks.jsonl      # one chunk record per line (id, doc_id, text, …)
  embeddings.npy    # float32 matrix [n_chunks, dim]
```

Wrong-loader errors: loading a Session checkpoint or Torch bundle with `load_rag_bundle` (or the reverse) raises a clear `ValidationError` naming the expected schema id.

---

## Default embedder and vector store (M1)

| Concern | Lock |
| --- | --- |
| Embed protocol | `callable[[list[str]], ndarray]` or an object with `.encode(texts) -> ndarray` |
| Default embedder | `buildml.hashing_embed.v1` — sklearn `HashingVectorizer` (`n_features=384`, `alternate_sign=False`, L2-normalized). Deterministic, CPU-only, no model download. |
| Semantic override | Optional `SentenceTransformerEmbedder` (model id e.g. `sentence-transformers/all-MiniLM-L6-v2`) when the `rag` extra’s sentence-transformers pin is present; or any caller-supplied embed callable |
| Store protocol | Build from embeddings + chunk records; `query(vector, k) -> ranked hits`; save/load with the bundle |
| Default store | In-process NumPy matrix + cosine similarity top-k (sklearn-free distance via L2-normalized matmul). Persist as `embeddings.npy` + `chunks.jsonl` |
| Deferred stores | FAISS / Chroma — M2+ if evidence warrants a second backend |

M1 chooses the lightest stack that meets save/load + dense kNN. Hashing embeddings are lexical/hashed, not semantic; disclosures and catalog copy must say so. Promote a sentence-transformer default only when CI model caching and honesty notes are ready (decision log).

---

## Packaging and CI

| Decision | Lock |
| --- | --- |
| Canonical extra | `rag` → `pip install 'buildml[rag]'` |
| Extra pins (M1) | `sentence-transformers>=2.2` (install gate + optional semantic embedder; hashing path still requires the extra so the install contract is uniform) |
| Later extras | `rag-gpu` / `rag-api` — docs-first; not M1 blockers |
| RAG CI Python | 3.11 and 3.12 (mirror `torch` job) |
| Core import | Must succeed without RAG extras; no eager `buildml.rag` imports in `buildml/__init__.py` or Session module top-level |
| Version line for RAG alpha (M3) | `2.2.0a1` (classical `2.0.0a1`, DL `2.1.0a1`) — do not bump package version until the M3 gate |

Missing stack → `MissingExtraError("rag", feature=...)` with `pip install 'buildml[rag]'`.

---

## Explicit non-goals (M1)

| Non-goal | Notes |
| --- | --- |
| Generate / LLM operator | No `rag_generate`; no `buildml.ai` |
| Hybrid dense+lexical / BM25 | M2 |
| Rerank / cross-encoder | M2 |
| Teaching Studio redesign / RAG cockpit | Catalog + structured results only in M1 |
| Hosted vector DB product | Out of library scope |
| PDF / OCR / HTML cleanup product | Later adapters |
| Embedding fine-tuning inside BuildML | Use DL or external tools |
| Folding RAG into `all-classical` | Keep classical extras free of RAG |

---

## Other M0 locks

| Topic | Decision |
| --- | --- |
| Tabular ingest | Explicit `rag_ingest_corpus(..., text_column=...)` only — never silently index the full Session frame |
| Eval hygiene | Index corpus and eval query/qrel sets are separate; indexing documents marked eval-only raises `LeakageError` |
| Metrics (M1) | `recall_at_k` and `mrr` on gold qrels; nDCG depth in M2 |
| Relevance mode (M1) | Document-level relevance via `doc_id` (chunk hits count as a hit for their parent doc) |
| Session attachment | Thin delegates only; no embed/index loops in `session.py` |
