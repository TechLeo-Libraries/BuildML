# PyPI 2.x publish notes

**Package:** `buildml`  
**Repo version:** `2.6.1` (Apache-2.0). GitHub Release / tag `v2.6.1`.  
**PyPI latest stable:** [`2.6.1`](https://pypi.org/project/buildml/2.6.1/)
([`https://pypi.org/pypi/buildml/json`](https://pypi.org/pypi/buildml/json)).  
**Prior stable on index:** `2.6.0` · **Prior pre-release:** `2.4.0a3`  
**Legacy line:** `1.0.9` (MIT; pin only)

`2.4.0`–`2.6.0` wheels omit `operation_index.json` and cannot
`import buildml`. Do not install those; use `2.6.1` or a source checkout.

## Install for users

```bash
pip install buildml
```

That resolves to the latest **non-pre-release** Session 2.x on PyPI. To force legacy 1.x:

```bash
pip install "buildml==1.0.9"
```

## Why `2.4.0a3` did not win over `1.0.9`

PEP 440 treats `a3` as a **pre-release**. Pip’s default install ignores
pre-releases, so `1.0.9` stayed the default until a non-pre-release `2.4.0`
shipped.

## How to cut the next release

1. Bump `buildml/_version.py` + `pyproject.toml`
2. Update CHANGELOG + install pins if needed
3. `python -m build && python scripts/check_wheel_contents.py dist`
   (must include `buildml/explain/generated/operation_index.json`)
4. Publish (pick one path):

**A — GitHub Actions Trusted Publishing (preferred)**

1. On PyPI → project `buildml` → Publishing → Add a new pending publisher:
   - Owner: `TechLeo-Libraries`
   - Repository: `BuildML`
   - Workflow name: `release.yml`
   - Environment name: *(leave blank: workflow does not use a GitHub Environment)*
2. Tag and/or dispatch:

```bash
git tag -a v2.6.1 -m "BuildML 2.6.1"
git push origin v2.6.1
# or:
gh workflow run release.yml --ref v2.6.1 -f dry_run=false
```

`release.yml` also runs on `release: published` so `gh release create` works.

**B — API token fallback** (what `release.yml` uses today)

Set the repo secret `PYPI_API_TOKEN` (a PyPI token scoped to `buildml`).
The workflow reads it on tag push. `skip-existing: true` so a second
upload of the same files does not fail the job.

```bash
gh secret set PYPI_API_TOKEN  # paste pypi-... token (scope: upload to buildml)
gh workflow run release.yml --ref v2.6.1 -f dry_run=false
```

**C - Local build + twine** (how `2.4.0`, `2.5.0`, and `2.6.0` landed when OIDC was not configured):

```bash
python -m build
python scripts/check_wheel_contents.py dist
python -m twine check dist/*
python -m twine upload dist/buildml-<version>*
```

5. Verify: `pip index versions buildml` shows the new version as latest, and
   `https://pypi.org/pypi/buildml/<version>/` returns 200. Confirm a
   clean venv can `import buildml` from the uploaded wheel.
6. Flip install honesty in this file, `docs/stability.md`, `docs/installation.rst`,
   `README.md`, and `docs/index.rst` so they no longer say PyPI still serves
   the previous version.

### Known failure mode

Tag-push used to fail with Trusted Publishing `invalid-publisher` when
PyPI had no matching publisher claims for `TechLeo-Libraries/BuildML` +
`release.yml`. That red check does **not** mean PyPI is missing the
release if path B or C already uploaded it. `release.yml` now uses
`PYPI_API_TOKEN` (path B) with `skip-existing`.
`2.4.0`, `2.5.0`, and `2.6.0` were uploaded via local twine when the OIDC job did not
have a matching publisher. Those three wheels omitted
`operation_index.json`; `2.6.1` is the packaging fix.
