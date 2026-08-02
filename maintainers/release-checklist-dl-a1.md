# Release checklist — 2.1.0a1 (DL alpha)

Use this when cutting the first deep-learning alpha tag. Do **not** tag until
remote CI is green on the commit you intend to release.

Related: [dl-alpha-gate.md](./dl-alpha-gate.md) ·
[CHANGELOG.md](../CHANGELOG.md) · [README.md](../README.md) ·
[dl-m0-lock.md](./dl-m0-lock.md)

---

## Pre-tag

1. [ ] Version strings agree: `pyproject.toml`, `buildml/_version.py` → `2.1.0a1`
2. [ ] `CHANGELOG.md` has a `2.1.0a1` section with known limits (no RAG/LLM claims)
3. [ ] README DL alpha status matches gate known limits
4. [ ] `dl-alpha-gate.md` sign-off criteria reviewed against current APIs
5. [ ] Local verification green:
   - `ruff check buildml tests scripts docs/conf.py`
   - `python scripts/lint_user_copy.py`
   - `pytest --cov=buildml --cov-report=term-missing` (core; Torch tests skip if absent)
   - `pip install -e ".[torch]"` then:
     - `pytest tests/unit/test_dl_torch_slice.py tests/unit/test_dl_m2_depth.py tests/integration/test_dl_torch_smoke.py tests/integration/test_dl_alpha_smoke.py -q`
6. [ ] Push release candidate branch/commit to GitHub
7. [ ] Remote CI green for all jobs: `test` (3.10–3.13), `engines`, `optuna`,
   `torch` (3.11–3.12), `extras`

## Tag and publish (only after CI green)

```bash
# On the exact commit CI passed:
git tag -a v2.1.0a1 -m "BuildML 2.1.0a1 DL alpha"
git push origin v2.1.0a1
```

Optional follow-ups (not required for the alpha gate):

- [ ] Build and upload wheel/sdist to the intended index (`python -m build`, then Twine)
- [ ] GitHub Release notes: paste gate summary + link to CHANGELOG + known limits
- [ ] Announce that Torch APIs/bundle formats may still change before a stable DL line

## Post-tag

- [ ] Confirm `import buildml; buildml.__version__ == "2.1.0a1"` from the published artifact
- [ ] Confirm `MissingExtraError` still guides `pip install 'buildml[torch]'` without Torch
- [ ] Open issues for any Conditional gate items that remain product gaps
- [ ] Do not mark DL alpha “stable” until a non-`aN` release on the DL line

## Do not claim at tag time (2.1.0a1 DL-first tag)

Historical lock for the **2.1.0a1** tag only. Phase C on later Unreleased / `2.3`
HEAD deepened several of these — do not paste this list onto current docs:

- For **2.1.0a1 only:** no text/sequence loaders, no built-in MLP zoo, no
  fold-local Torch CV (those ship in Phase C — see `dl-m0-lock.md` / CHANGELOG)
- Multi-node cluster DDP as of **2.1.0a1** only (single-node DDP / AMP / export
  shipped in Pass G; torchrun multi-node join is on later HEAD via Pass O —
  still not Kubernetes multi-cluster orchestration)
- Polars/DuckDB zero-copy into DataLoaders (still out of scope)
- RAG / LLM operator as shipped features of the **2.1.0a1** tag (later lines)
- That Session checkpoints contain Torch weights
- Speech foundation models / Whisper-scale ASR as of **2.1.0a1** / Pass L only
  (later HEAD Pass O adds ASR integration + finetune-lite — still not FM-from-scratch)
