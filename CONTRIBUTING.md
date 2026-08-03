# Contributing to BuildML

Thanks for helping improve BuildML. This is a short contributor guide for the
2.x Session line.

## Setup

```bash
git clone https://github.com/TechLeo-Libraries/BuildML.git
cd BuildML
pip install -e ".[dev]"
```

Python 3.10–3.13. Optional extras (`torch`, `rag`, `ai`, `serve`, …) match
`README.md` / `docs/installation.rst`.

**Install honesty:** PyPI `buildml` is still legacy 1.x (`1.0.9`). Use a GitHub
or editable install for Session 2.x until a 2.x wheel is published.

## Checks before opening a PR

```bash
ruff check buildml tests scripts docs/conf.py
python scripts/lint_user_copy.py
python scripts/sync_teaching_surface.py --check
python scripts/audit_docstrings.py --check
mypy --follow-imports=silent buildml/core buildml/_version.py \
  buildml/explain/schemas.py buildml/explain/history.py buildml/explain/sync.py \
  buildml/explain/concepts buildml/explain/capability_status.py \
  buildml/explain/glossary.py buildml/explain/prerequisites.py
pytest -q --cov=buildml --cov-report=term-missing
```

Coverage `fail_under` lives in `pyproject.toml` (`[tool.coverage.report]`) and is a
ratchet — raise it when classical/core coverage improves; do not lower it to
silence a regression. `requirements.txt` / `requirements-dev.txt` are convenience
mirrors of `pyproject.toml` ranges; prefer `pip install -e ".[dev]"`.

CI also runs a **Windows classical-only** job (`windows-classical`: import smoke,
ruff, classical alpha smoke). Torch / PyG / industry extras stay Linux-only.

After Session / AI tool / overlay changes, regenerate teaching surfaces:

```bash
python scripts/sync_teaching_surface.py --write
python scripts/sync_teaching_surface.py --check
```

## Docstring standard

BuildML's promise is that someone who does not already know a technique can
still apply it correctly. Docstrings are the only documentation that travels
with the code, so they carry that promise — they are documentation, not
labels. `python scripts/audit_docstrings.py --check` enforces the structural
half of this on the packages listed in that script's `ENFORCED_PREFIXES`; the
teaching half is on the author.

Format is **NumPy style**, rendered by `sphinx.ext.napoleon`. Every public
class, function, method, and property gets:

1. **A one-line summary** in the imperative mood that a beginner can parse
   without knowing the domain. "Fit a gradient-boosted ranker on grouped
   query data" — not "Fit ranker".
2. **A description paragraph** saying what the operation actually does, where
   it sits in the workflow, and what state it changes. Name the step before
   and the step after so the reader can place it.
3. **`Parameters`** — every argument, with what it means and what changes when
   you change it. Give the practical default reasoning, not a restatement of
   the type annotation. "`n_estimators`: number of trees" is useless;
   "`n_estimators`: how many boosting rounds to run. More trees fit the
   training data more closely and cost proportionally more time" is the bar.
4. **`Returns`** — what the object *means* and what you do with it next, not
   just its class name. Link the result class with `:class:` so readers can
   follow through.
5. **`Raises`** — every exception raised directly, with the condition that
   triggers it, phrased so the reader can avoid it.
6. **`Notes`** — the intuition. When to reach for this versus the obvious
   alternative, the assumption that has to hold, the leakage or scale trap.
   BuildML already uses bolded `**Leakage:**` and `**Scale:**` leads for the
   two recurring hazards; keep them.
7. **`Examples`** — for anything non-trivial. Use `>>>` doctest form with real
   BuildML calls. Illustrative examples are fine (they are not run by CI), but
   they must use APIs that genuinely exist.

Additional conventions:

- Properties need a summary and, when the annotation is `... | None`, a
  sentence saying what `None` means and which call populates the value.
- Conceptual teaching material lives in `buildml/explain/` and the `docs/`
  guides. Docstrings describe the API and link outward with `:meth:`,
  `:class:`, and `See Also` rather than duplicating a guide.
- `scripts/lint_user_copy.py` bans marketing language. Write plainly.
- When a package finishes its depth pass, add it to `ENFORCED_PREFIXES` in
  `scripts/audit_docstrings.py`. That list is a ratchet — entries are added,
  never removed.

Use `python scripts/audit_docstrings.py --report` to see per-package coverage
and pick the next target, and `--path buildml/<pkg>` to audit work in progress.

## Release checklist

1. Bump `pyproject.toml` + `buildml/_version.py` together.
2. Move `[Unreleased]` notes into a dated CHANGELOG section.
3. Keep install honesty accurate (GitHub vs PyPI) until 2.x is published.
4. Ensure remote CI is green (`test`, `windows-classical`, `engines`, `optuna`,
   `torch`, `rag`, `ai`, `extras`, `benchmarks`).
5. Publish to PyPI only when credentials are available and intentional; otherwise
   document the gap and leave the honesty banner.
6. Refresh Read the Docs after the tag / docs change.

## Pull requests

- Prefer focused branches off updated `main`.
- Do not force-push `main`.
- Keep user-facing copy honest about alpha boundaries (no false PyPI / product claims).
