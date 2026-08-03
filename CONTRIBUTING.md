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

## Session architecture

`buildml.Session` is the public facade. Domain orchestration and **canonical
docstrings** (full Parameters / Raises / Notes / Examples) live in
`buildml/session/*_ops.py`. Public method signatures live in
`buildml/session/mixins/` (one mixin per domain) as thin delegates with a short
summary, Returns, and a `:func:` pointer to the ops function.
`scripts/audit_docstrings.py` allows that facade shape under
`buildml/session/mixins/` while still requiring summary + description + Returns
+ ops pointer. `session.py` assembles the mixins and owns `__init__` /
context-manager / core state glue. Do not add fat logic to mixin method bodies.

Bundle / checkpoint loaders that deserialize pickle/joblib/torch require
keyword-only `trusted=True` (default `False`). Thread that flag through Session
ops and mixins; see `buildml.core.serialization`.

## Domain maturity index

BuildML has many domains; depth is uneven by design and is **machine-checked**
so the unevenness is governed rather than accidental:

```bash
python scripts/domain_maturity_index.py --report
python scripts/domain_maturity_index.py --check
```

`--check` fails CI when a domain listed as claimed-complete is missing required
artifacts or falls below the domain floor (artifact score `< 6`). Industry wheel
availability is orthogonal: matrices and `scripts/probe_industry_extras.py`
report that at runtime.

### Domain floor (every claimed-complete domain)

Equal LOC across 42 domains is **not** the goal. Every claimed-complete domain
must still meet this quality floor:

| Requirement | Notes |
| --- | --- |
| `catalog.py` + capability matrix | Honest backends / methods / non-goals |
| `Session.<domain>_capability_matrix()` | Mixin static method |
| `capability_status` wiring | Walkthrough / audit embed the live matrix |
| `explain_hooks.py` | Status summaries for history / walkthrough |
| `checkpoint.py` | Fitted-plan bundle save/load: **except** analysis-only surfaces (e.g. timeseries analysis) |
| Guide under `guides/` | Quickstart pointer |
| Proof under `proofs/` **or** proof mention | Thin README ok when the domain is analysis-only |
| Tests under `tests/` | Unit and/or alpha smoke |
| Artifact score ≥ 6 | Ratchet in `scripts/domain_maturity_index.py` (`MIN_CLAIMED_SCORE`) |

Discovery paths for humans:

1. **Capability matrices**: every refined domain exposes
   `Session.<domain>_capability_matrix()` (and catalog helpers under
   `buildml/<domain>/catalog.py`). Matrices report which backends import on
   *this* machine (including `platform_markers` / `skipped_by_marker` when an
   environment marker skips a wheel); treat them as the honesty source of truth.
2. **Explain / walkthrough**: `Session.walkthrough()` and
   `buildml.explain.capability_status` attach live matrices to status payloads.
3. **Guides**: `guides/` deep guides and `docs/` RST mark classical / DL / RAG
   / AI as primary paths; industry adapters (`*-industry` extras) are depth
   overlays that may skip when wheels are missing.
4. **Rough depth bands (honest, not marketing):**
   - **Deep:** classical supervised + preprocess, checkpoint/pipeline, explain
     teaching surface, DL (torch), RAG, AI operator (closed tools).
   - **Solid domain shape (factory + matrix + Session + proofs):** anomaly,
     forecasting, unsupervised, ranking, recommenders, causal, NLP, ensemble,
     AutoML, and most R1–R6 refinement domains.
   - **Thinner / adapter-led:** some industry backends (LightFM, learn2learn,
     giotto-tda, …) where upstream wheels gate availability: matrix will say so.

## Checks before opening a PR

```bash
ruff check buildml tests scripts docs/conf.py
python scripts/lint_user_copy.py
python scripts/sync_teaching_surface.py --check
python scripts/audit_docstrings.py --check
python scripts/domain_maturity_index.py --check
mypy --follow-imports=silent buildml/core buildml/_version.py \
  buildml/explain/schemas.py buildml/explain/history.py buildml/explain/sync.py \
  buildml/explain/concepts buildml/explain/capability_status.py \
  buildml/explain/glossary.py buildml/explain/prerequisites.py \
  buildml/session
pytest -q --cov=buildml --cov-report=term-missing
python scripts/probe_industry_extras.py --artifact industry-probe.json
```

Coverage `fail_under` lives in `pyproject.toml` (`[tool.coverage.report]`) and
`scripts/coverage_ratchet.json`. It is a one-way ratchet (**60** active, next
**70**): raise only from a full-suite measure
(`python scripts/run_full_coverage.py --update-ratchet` or CI
`pytest tests --cov=buildml`); do not lower it to silence a regression.
`requirements.txt` / `requirements-dev.txt` are convenience mirrors of
`pyproject.toml` ranges; prefer `pip install -e ".[dev]"`.

Proof CI gate: `python -m proofs._lib.run_all --smoke` (expanded Tier A subset,
never skips existing results, fails on `skipped_missing_extra` / `partial`
unless `--allow-skip`). Full suites may use `--skip-existing` locally only.

Surface stability: see [`docs/stability.md`](docs/stability.md).

CI also runs a **Windows classical-only** job (`windows-classical`: import smoke,
ruff, classical alpha smoke). Torch / PyG / industry extras stay Linux-only.
`scripts/probe_industry_extras.py` prints which industry modules import vs skip
on the current platform and can write a JSON artifact; it must not fail the
build when wheels are missing.

After Session / AI tool / overlay changes, regenerate teaching surfaces:

```bash
python scripts/sync_teaching_surface.py --write
python scripts/sync_teaching_surface.py --check
```

## Docstring standard

BuildML's promise is that someone who does not already know a technique can
still apply it correctly. Docstrings are the only documentation that travels
with the code, so they carry that promise: they are documentation, not
labels. `python scripts/audit_docstrings.py --check` enforces the structural
half of this on the packages listed in that script's `ENFORCED_PREFIXES`; the
teaching half is on the author.

Format is **NumPy style**, rendered by `sphinx.ext.napoleon`. Every public
class, function, method, and property gets:

1. **A one-line summary** in the imperative mood that a beginner can parse
   without knowing the domain. "Fit a gradient-boosted ranker on grouped
   query data": not "Fit ranker".
2. **A description paragraph** saying what the operation actually does, where
   it sits in the workflow, and what state it changes. Name the step before
   and the step after so the reader can place it.
3. **`Parameters`**: every argument, with what it means and what changes when
   you change it. Give the practical default reasoning, not a restatement of
   the type annotation. "`n_estimators`: number of trees" is useless;
   "`n_estimators`: how many boosting rounds to run. More trees fit the
   training data more closely and cost proportionally more time" is the bar.
4. **`Returns`**: what the object *means* and what you do with it next, not
   just its class name. Link the result class with `:class:` so readers can
   follow through.
5. **`Raises`**: every exception raised directly, with the condition that
   triggers it, phrased so the reader can avoid it.
6. **`Notes`**: the intuition. When to reach for this versus the obvious
   alternative, the assumption that has to hold, the leakage or scale trap.
   BuildML already uses bolded `**Leakage:**` and `**Scale:**` leads for the
   two recurring hazards; keep them.
7. **`Examples`**: for anything non-trivial. Use `>>>` doctest form with real
   BuildML calls. Illustrative examples are fine (they are not run by CI), but
   they must use APIs that genuinely exist.

Additional conventions:

- Properties need a summary and, when the annotation is `... | None`, a
  sentence saying what `None` means and which call populates the value.
- **Session mixin facades** may omit full Parameters / Raises essays when they
  include a real summary, a description, Returns (when annotated), and an
  explicit pointer to the canonical `buildml.session.*_ops` function. Put the
  full pedagogical Parameters / Notes / Examples on the ops function.
- Conceptual teaching material lives in `buildml/explain/` and the `docs/`
  guides. Docstrings describe the API and link outward with `:meth:`,
  `:class:`, and `See Also` rather than duplicating a guide.
- `scripts/lint_user_copy.py` bans marketing language and Unicode em dashes
  (U+2014). Write plainly; use ASCII punctuation (`:`, `;`, `,`, `.`, or `-`).
- When a package finishes its depth pass, add it to `ENFORCED_PREFIXES` in
  `scripts/audit_docstrings.py`. That list is a ratchet: entries are added,
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
