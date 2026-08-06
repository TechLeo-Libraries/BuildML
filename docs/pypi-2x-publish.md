# PyPI 2.x publish notes

**Package:** `buildml`  
**Repo version / GitHub Release:** [`2.5.0`](https://github.com/TechLeo-Libraries/BuildML/releases/tag/v2.5.0) (Apache-2.0)  
**PyPI latest stable:** [`2.5.0`](https://pypi.org/project/buildml/2.5.0/) (uploaded; confirm via [`https://pypi.org/pypi/buildml/json`](https://pypi.org/pypi/buildml/json))  
**Prior stable on index:** `2.4.0` · **Prior pre-release:** `2.4.0a3`  
**Legacy line:** `1.0.9` (MIT; pin only)

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
3. Publish (pick one path):

**A — GitHub Actions Trusted Publishing (preferred)**

1. On PyPI → project `buildml` → Publishing → Add a new pending publisher:
   - Owner: `TechLeo-Libraries`
   - Repository: `BuildML`
   - Workflow name: `release.yml`
   - Environment name: *(leave blank — workflow does not use a GitHub Environment)*
2. Tag and/or dispatch:

```bash
git tag -a v2.5.0 -m "BuildML 2.5.0"
git push origin v2.5.0
# or:
gh workflow run release.yml --ref v2.5.0 -f dry_run=false
```

`release.yml` also runs on `release: published` so `gh release create` works.

**B — API token fallback**

```bash
gh secret set PYPI_API_TOKEN  # paste pypi-... token (scope: upload to buildml)
gh workflow run release.yml --ref v2.5.0 -f dry_run=false
```

**C — Local build + twine** (how `2.4.0` and `2.5.0` landed when OIDC was not configured):

```bash
python -m build
python -m twine check dist/*
python -m twine upload dist/buildml-<version>*
```

4. Verify: `pip index versions buildml` shows the new version as latest, and
   `https://pypi.org/pypi/buildml/<version>/` returns 200.

### Known failure mode

Tag-push / workflow publish fails with Trusted Publishing
`invalid-publisher` when PyPI has no matching publisher claims for
`TechLeo-Libraries/BuildML` + `release.yml`. Fix with path A or B/C above.
`2.4.0` and `2.5.0` were uploaded via local twine when the OIDC job did not
have a matching publisher.
