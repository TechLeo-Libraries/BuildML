# PyPI 2.x publish checklist

**Package:** `buildml`  
**Repo version:** `2.4.0a3` (see `buildml/_version.py` + `pyproject.toml`)  
**Current PyPI reality:** `pip install buildml` still resolves the legacy **1.x**
line (`1.0.9`, MIT). That project is outside the Session 2.x tree until a 2.x
wheel ships under the same name (or the package is deliberately renamed).

The packaging side of this repo is ready. Publishing still needs human PyPI /
GitHub ownership — credentials and publisher config that CI cannot invent.

## Decision: keep the name `buildml` (preferred)

| Option | When | Action |
| --- | --- | --- |
| **Keep `buildml`** | Publisher controls (or can reclaim) the PyPI project `buildml` | Publish `2.4.0a3` with Trusted Publishing below |
| **Rename** | PyPI `buildml` cannot be claimed for Session 2.x | Change `[project].name` in `pyproject.toml`, update README install lines, publish the new name, document the rename in CHANGELOG |

Do **not** ship a second silent name without updating README install honesty.

## Before tagging / publishing `2.4.0a3`

### Facades (2.4.x)

- [x] Facades are the supported public API for domains
- [x] Flat domain aliases supported-but-deprecated until **3.0** (warnings only)
- [x] Classical core dual first-class (no flat warnings)
- [x] Docs / examples / proofs prefer facades
- [x] See [`session-facade-migration.md`](session-facade-migration.md)

### Quality gates

- [x] Coverage floor **≥ 70** (`pyproject.toml` `fail_under` + `scripts/coverage_ratchet.json`)
- [x] Mypy widened in CI beyond session / core / explain (see `.github/workflows/ci.yml`)
- [x] Industry extras probe job present (`scripts/probe_industry_extras.py`)
- [x] Real-dataset proofs smoke gate (`python -m proofs._lib.run_all --smoke`)
- [x] Domain maturity check (`scripts/domain_maturity_index.py --check`)

### Packaging

- [x] Version `2.4.0a3` consistent in `_version.py` / `pyproject.toml` / docs
- [x] Release workflow ready (OIDC Trusted Publishing; dry-run by default)

## Verify the build locally

```bash
python -m pip install --upgrade build twine
python -m build
twine check dist/*
```

Expected:

- sdist + wheel under `dist/`
- `twine check` passes
- version string is `2.4.0a3` in both artifacts
- classifiers include `Development Status :: 3 - Alpha`

Workflow: [`.github/workflows/release.yml`](../.github/workflows/release.yml)

- Builds on `workflow_dispatch` (default **dry_run=true**) and on tags `v2.*` / `v3.*`
- Publish job uses **OIDC Trusted Publishing** (`id-token: write` + `pypa/gh-action-pypi-publish`)
- No long-lived PyPI token is stored in the repo

## Human one-shot checklist (credentials required)

1. **Confirm PyPI project ownership** for `buildml` (or take the rename path).
2. On [PyPI → Publishing](https://pypi.org/manage/account/publishing/): add a
   Trusted Publisher for GitHub:
   - Owner: `TechLeo-Libraries`
   - Repository: `BuildML`
   - Workflow: `release.yml`
   - Environment: leave empty unless you create a matching GitHub Environment
3. Confirm GitHub Actions can write `id-token` (the workflow already requests it).
4. Tag and push (publishes) **or** run the Release workflow with `dry_run=false`:
   ```bash
   git tag v2.4.0a3
   git push origin v2.4.0a3
   ```
   Dry-run first:
   - Actions → Release → Run workflow → `dry_run: true` → inspect artifacts
5. Verify:
   ```bash
   pip index versions buildml
   pip install "buildml==2.4.0a3"
   python -c "import buildml; print(buildml.__version__)"
   ```
6. Update README install honesty only after the wheel is live (remove the
   “PyPI is still 1.x” callout; keep git / editable as alternates).

## If publish fails

| Symptom | Likely blocker |
| --- | --- |
| 403 / ownership error on upload | PyPI project `buildml` not owned by the TechLeo publisher |
| Trusted Publishing rejected | Publisher config mismatch (owner / repo / workflow / environment) |
| Tag push did not publish | Tag pattern not `v2.*` / `v3.*`, or `dry_run` dispatch only |
| Users still get 1.0.9 | 2.x not uploaded, or pip cache / index lag |

CI cannot finish steps that need PyPI account credentials. Everything else for
`2.4.0a3` is prepared in-repo.
