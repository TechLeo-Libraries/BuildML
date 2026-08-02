# Release checklist — 2.2.0a1 (RAG alpha)

Use this when cutting the first retrieval alpha tag. Do **not** tag until
remote CI is green on the commit you intend to release.

Related: [rag-alpha-gate.md](./rag-alpha-gate.md) ·
[CHANGELOG.md](../CHANGELOG.md) · [README.md](../README.md) ·
[rag-m0-lock.md](./rag-m0-lock.md)

---

## Pre-tag

1. [ ] Version strings agree: `pyproject.toml`, `buildml/_version.py` → `2.2.0a1`
2. [ ] `CHANGELOG.md` has a `2.2.0a1` section with known limits for that tag
   (retrieve-first alpha). Current HEAD (`2.3.0a1`) documents `rag_generate`
   and the AI operator separately — do not paste stale “no generate” claims
   into newer release notes.
3. [ ] README RAG alpha status matches gate known limits
4. [ ] `rag-alpha-gate.md` sign-off criteria reviewed against current APIs
5. [ ] Local verification green:
   - `ruff check buildml tests scripts docs/conf.py`
   - `python scripts/lint_user_copy.py`
   - `pytest --cov=buildml --cov-report=term-missing` (core; RAG ST-only tests
     skip if sentence-transformers absent; hashing path still runs)
   - `pip install -e ".[rag]"` then:
     - `pytest tests/unit/test_rag_slice.py tests/unit/test_rag_m2_depth.py tests/integration/test_rag_smoke.py tests/integration/test_rag_alpha_smoke.py -q`
6. [ ] Push release candidate branch/commit to GitHub
7. [ ] Remote CI green for all jobs: `test` (3.10–3.13), `engines`, `optuna`,
   `torch` (3.11–3.12), `rag` (3.11–3.12), `extras`

## Tag and publish (only after CI green)

```bash
# On the exact commit CI passed:
git tag -a v2.2.0a1 -m "BuildML 2.2.0a1 RAG alpha"
git push origin v2.2.0a1
```

Optional follow-ups (not required for the alpha gate):

- [ ] Build and upload wheel/sdist to the intended index (`python -m build`, then Twine)
- [ ] GitHub Release notes: paste gate summary + link to CHANGELOG + known limits
- [ ] Announce that RAG APIs/bundle formats may still change before a stable RAG line

## Post-tag

- [ ] Confirm `import buildml; buildml.__version__ == "2.2.0a1"` from the published artifact
- [ ] Confirm hashing retrieve path works without sentence-transformers
- [ ] Confirm `MissingExtraError` still guides `pip install 'buildml[rag]'` for ST/rerank
- [ ] Open issues for any Conditional gate items that remain product gaps
- [ ] Do not mark RAG alpha “stable” until a non-`aN` release on the RAG line

## Do not claim at tag time (2.2.0a1 retrieve-first tag)

- For the **2.2.0a1** tag only: do not claim grounded generate or the LLM operator
  (those ship on the later `2.3.0a1` line — see CHANGELOG).
- That the hashing default is semantic retrieval quality
- Hosted vector DB, FAISS/Chroma product path, or PDF/OCR cleanup product
- Teaching Studio RAG cockpit redesign
- That Session checkpoints contain the vector index
- That classical or Torch APIs were replaced by RAG
- Multimodal fusion or nested Torch search (still out of scope on current HEAD)
