# Release checklist — 2.0.0a1 (classical alpha)

Use this when cutting the first classical alpha tag. Do **not** tag until remote
CI is green on the commit you intend to release.

Related: [classical-alpha-gate.md](./classical-alpha-gate.md) ·
[CHANGELOG.md](../CHANGELOG.md) · [README.md](../README.md)

---

## Pre-tag

1. [ ] Version strings agree: `pyproject.toml`, `buildml/_version.py` → `2.0.0a1`
2. [ ] `CHANGELOG.md` has a `2.0.0a1` section with known limits
3. [ ] README alpha status matches gate known limits (no overclaims)
4. [ ] `docs/classical-alpha-gate.md` sign-off criteria reviewed against current APIs
5. [ ] Local verification green:
   - `ruff check buildml tests scripts docs/conf.py`
   - `python scripts/lint_user_copy.py`
   - `pytest --cov=buildml --cov-report=term-missing`
   - `pytest tests/integration/test_classical_alpha_smoke.py -q`
6. [ ] Push release candidate branch/commit to GitHub
7. [ ] Remote CI green for all jobs: `test` (3.10–3.13), `engines`, `optuna`, `extras`

## Tag and publish (only after CI green)

```bash
# On the exact commit CI passed:
git tag -a v2.0.0a1 -m "BuildML 2.0.0a1 classical alpha"
git push origin v2.0.0a1
```

Optional follow-ups (not required for the alpha gate):

- [ ] Build and upload wheel/sdist to the intended index (`python -m build`, then Twine)
- [ ] GitHub Release notes: paste gate summary + link to CHANGELOG + known limits
- [ ] Announce that APIs/formats may still change before stable 2.0

## Post-tag

- [ ] Confirm `import buildml; buildml.__version__ == "2.0.0a1"` from the published artifact
- [ ] Open issues for any Conditional gate items that remain product gaps
- [ ] Do not mark classical alpha “stable” until a non-`aN` 2.0.0 release

## Do not claim at tag time

- Out-of-core sklearn training via Polars/DuckDB
- Fold-local custom transforms or resample
- Fairness / SHAP / deep learning / RAG as shipped classical alpha features
