# PyPI 2.x publish notes

**Package:** `buildml`  
**Current stable:** [`2.4.0`](https://pypi.org/project/buildml/2.4.0/) (Apache-2.0)  
**Prior pre-release:** `2.4.0a3` (still on the index; pip prefers `2.4.0`)  
**Legacy line:** `1.0.9` (MIT) — pin only

## Install for users

```bash
pip install buildml
```

That resolves to Session **2.4.x**. To force legacy 1.x:

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
3. Build + upload:

```bash
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

Or tag `v2.*` / run `release.yml` with Trusted Publishing configured.

4. Verify: `pip index versions buildml` shows the new version as latest
