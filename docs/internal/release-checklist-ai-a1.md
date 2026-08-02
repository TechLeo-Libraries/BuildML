# Release checklist — 2.3.0a1 (AI alpha)

Use this when cutting the first AI operator alpha tag. Do **not** tag until
remote CI is green on the commit you intend to release.

Related: [ai-alpha-gate.md](./ai-alpha-gate.md) ·
[CHANGELOG.md](../CHANGELOG.md) · [README.md](../README.md) ·
[llm-m0-lock.md](./llm-m0-lock.md)

---

## Pre-tag

1. [ ] Version strings agree: `pyproject.toml`, `buildml/_version.py` → `2.3.0a1`
2. [ ] `CHANGELOG.md` has a `2.3.0a1` section with known limits (**no autonomous
   agent / auto-execute / production safety claims**)
3. [ ] README AI alpha status matches gate known limits
4. [ ] `ai-alpha-gate.md` sign-off criteria reviewed against current APIs
5. [ ] Local verification green:
   - `ruff check buildml tests scripts docs/conf.py`
   - `python scripts/lint_user_copy.py`
   - `pytest --cov=buildml --cov-report=term-missing` (core; AI tests skip if
     openai absent; MockProvider path still runs)
   - `pip install -e ".[ai]"` then:
     - `pytest tests/unit/test_ai_slice.py -q`
6. [ ] Push release candidate branch/commit to GitHub
7. [ ] Remote CI green for all jobs: `test` (3.10–3.13), `engines`, `optuna`,
   `torch` (3.11–3.12), `rag` (3.11–3.12), `ai` (3.11–3.12), `extras`

## Tag and publish (only after CI green)

```bash
# On the exact commit CI passed:
git tag -a v2.3.0a1 -m "BuildML 2.3.0a1 AI operator alpha"
git push origin v2.3.0a1
```

Optional follow-ups (not required for the alpha gate):

- [ ] Build and upload wheel/sdist to the intended index (`python -m build`, then Twine)
- [ ] GitHub Release notes: paste gate summary + link to CHANGELOG + known limits
- [ ] Announce that AI APIs/transcript formats may still change before a stable AI line

## Post-tag

- [ ] Confirm `import buildml; buildml.__version__ == "2.3.0a1"` from the published artifact
- [ ] Confirm MockProvider path works without openai
- [ ] Confirm `MissingExtraError` still guides `pip install 'buildml[ai]'`
- [ ] Open issues for any Conditional gate items that remain product gaps
- [ ] Do not mark AI alpha "stable" until a non-`aN` release on the AI line

## Do not claim at tag time

- Autonomous agent, auto-execution, or autopilot mode
- Production-ready safety or security guarantees
- That the operator replaces Teaching Studio as the primary teaching surface
- Training or fine-tuning LLMs inside BuildML
- Local-only LLM provider path
- That transcripts are equivalent to Session checkpoints or bundles
- That advice from the operator is infallible or a substitute for domain expertise
