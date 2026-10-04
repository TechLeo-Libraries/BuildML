# BuildML 2.x API stability policy

BuildML **2.6.4** continues the stable Session 2.x line (first stable was
**2.4.0**). This repo is **2.6.4**. `pip install buildml` selects the latest
published stable release on [PyPI](https://pypi.org/project/buildml/). Wheels
**2.4.0a3–2.6.0** omit `operation_index.json` and cannot import; install
**2.6.1 or later**. This policy describes supported APIs,
deprecation rules, and dependency availability.

## What “stable” means here

- **`pip install buildml`** installs the latest non-pre-release Session 2.x on
  PyPI (**2.6.x**; not legacy 1.0.9).
- Public Session / facade APIs in 2.6.x follow SemVer: breaking removals wait
  for a major bump (see facades → 3.0 below).
- Optional industry extras remain **best-effort** across platforms; capability
  matrices and runtime probes report backend availability. For subprocess use-case
  checks (`ok` / `crash`), see `guides/safe-install-and-runtime.md` and
  `scripts/verify_runtime_stability.py`.
- The serving API runs a local application. Hosting and tenant isolation require
  deployment infrastructure outside BuildML.

## Rules

1. **Additive by default.** New domains and methods may land without removing
   existing ones.
2. **Breaking changes need a CHANGELOG entry** under `Removed` / `Changed`, plus
   a migration note in `docs/legacy.rst` or the affected guide.
3. **Bundle schemas are versioned** (`buildml.<domain>_bundle.v1`, …). Bump the
   version string when the on-disk layout changes. Loaders must refuse unknown
   versions with a clear error.
4. **Capability matrices report backend availability.** Prefer reporting
   `available: false` over deleting a public method when an optional backend is
   withdrawn.
5. **Supported API groups.** Classical ingest / roles / split / preprocess /
   fit / evaluate / CV / search, checkpoint / pipeline bundles, domain facades,
   and `*_capability_matrix` names are supported in 2.6.x.
6. **Proofs and CI smoke** (`python -m proofs._lib.run_all --smoke`) must stay
   green for these API groups. Smoke fails on unexpected `skipped_missing_extra` /
   `partial` result statuses (use `--allow-skip` only for local investigation).
7. **Namespaced Session facades.** For domains, facades are the supported public
   API (`session.<domain>.*`). Flat domain actions still work and emit
   `DeprecationWarning` until **BuildML 3.0**. Classical core stays dual and
   first-class with no warnings. Details:
   [`docs/session-facade-migration.md`](session-facade-migration.md).
   Discovery exposes `stability_tier` (`core` | `domain` | `experimental`) and
   `preferred_path`.

## Coverage requirements

See `scripts/coverage_ratchet.json` and `pyproject.toml` `fail_under`.
Active floor **66** (Linux CI monolith measured 66.67% on the 2.6.0 cut).
Measure the complete suite with `python scripts/run_full_coverage.py`; subset
runs are not comparable to the project coverage threshold.
