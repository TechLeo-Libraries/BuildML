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
pytest -q
```

After Session / AI tool / overlay changes, regenerate teaching surfaces:

```bash
python scripts/sync_teaching_surface.py --write
python scripts/sync_teaching_surface.py --check
```

## Release checklist

1. Bump `pyproject.toml` + `buildml/_version.py` together.
2. Move `[Unreleased]` notes into a dated CHANGELOG section.
3. Keep install honesty accurate (GitHub vs PyPI) until 2.x is published.
4. Ensure remote CI is green (`test`, `engines`, `optuna`, `torch`, `rag`, `ai`,
   `extras`).
5. Publish to PyPI only when credentials are available and intentional; otherwise
   document the gap and leave the honesty banner.
6. Refresh Read the Docs after the tag / docs change.

## Pull requests

- Prefer focused branches off updated `main`.
- Do not force-push `main`.
- Keep user-facing copy honest about alpha boundaries (no false PyPI / product claims).
