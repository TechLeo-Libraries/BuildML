# BuildML 2.x surface stability policy

BuildML **2.4.0** is the first stable Session 2.x release on PyPI. The Session
surface is large on purpose. This note is how I keep that surface usable.

## What “stable” means here

- **`pip install buildml`** installs Session **2.4.x** (not legacy 1.0.9).
- Public Session / facade APIs in 2.4.x follow SemVer: breaking removals wait
  for a major bump (see facades → 3.0 below).
- Optional industry extras remain **best-effort** across platforms; capability
  matrices + runtime probes are the honesty layer.
- Local serve is a single-deploy path, not a multi-tenant SaaS product.

## Rules

1. **Additive by default.** New domains and methods may land without removing
   existing ones.
2. **Breaking changes need a CHANGELOG entry** under `Removed` / `Changed`, plus
   a migration note in `docs/legacy.rst` or the affected guide.
3. **Bundle schemas are versioned** (`buildml.<domain>_bundle.v1`, …). Bump the
   version string when the on-disk layout changes. Loaders must refuse unknown
   versions with a clear error.
4. **Capability matrices are the honesty layer.** Prefer reporting
   `available: false` over deleting a public method when an optional backend is
   withdrawn.
5. **Supported freeze set.** Classical ingest / roles / split / preprocess /
   fit / evaluate / CV / search, checkpoint / pipeline bundles, domain facades,
   and `*_capability_matrix` names are supported in 2.4.x.
6. **Proofs and CI smoke** (`python -m proofs._lib.run_all --smoke`) must stay
   green on the freeze set. Smoke fails on unexpected `skipped_missing_extra` /
   `partial` result statuses (use `--allow-skip` only for local investigation).
7. **Namespaced Session facades.** For domains, facades are the supported public
   API (`session.<domain>.*`). Flat domain actions still work and emit
   `DeprecationWarning` until **BuildML 3.0**. Classical core stays dual and
   first-class with no warnings. Details:
   [`docs/session-facade-migration.md`](session-facade-migration.md).
   Discovery exposes `stability_tier` (`core` | `domain` | `experimental`) and
   `preferred_path`.

## Coverage ratchet

See `scripts/coverage_ratchet.json` and `pyproject.toml` `fail_under`.
Active floor **70** (full-suite measure ~70.7%). Measure with
`python scripts/run_full_coverage.py` — never from a tiny subset.
