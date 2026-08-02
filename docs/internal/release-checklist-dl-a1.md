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

## Do not claim at tag time

- Image / sequence / multimodal loaders or a built-in model zoo
- Fold-local Torch CV, DDP, or mixed-precision product path
- Polars/DuckDB zero-copy into DataLoaders
- RAG / LLM operator as shipped features
- That Session checkpoints contain Torch weights
