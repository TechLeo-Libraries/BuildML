# BuildML 2.x surface stability policy

BuildML `2.4.0a*` is alpha. The Session surface is large on purpose. This note
is how I limit churn before a stable 2.x cut.

## Rules

1. **Additive by default.** New domains and methods may land in alpha without
   removing existing ones.
2. **Breaking changes need a CHANGELOG entry** under `Removed` / `Changed`, plus
   a migration note in `docs/legacy.rst` or the affected guide.
3. **Bundle schemas are versioned** (`buildml.<domain>_bundle.v1`, …). Bump the
   version string when the on-disk layout changes. Loaders must refuse unknown
   versions with a clear error.
4. **Capability matrices are the honesty layer.** Prefer reporting
   `available: false` over deleting a public method when an optional backend is
   withdrawn.
5. **Freeze candidates for 2.4.0.** Classical ingest / roles / split /
   preprocess / fit / evaluate / CV / search, checkpoint / pipeline bundles, and
   domain `*_capability_matrix` names are freeze candidates. Experimental
   Torch / AI operator details may still move until beta.
6. **Proofs and CI smoke** (`python -m proofs._lib.run_all --smoke`) must stay
   green on the freeze set. Smoke fails on unexpected `skipped_missing_extra` /
   `partial` result statuses (use `--allow-skip` only for local investigation).
7. **Namespaced Session facades (2.4.0a3+).** For domains, facades are the
   supported public API (`session.<domain>.*`). Flat domain actions still work
   and emit `DeprecationWarning` until **BuildML 3.0**. Classical core stays
   dual and first-class with no warnings. Details:
   [`docs/session-facade-migration.md`](session-facade-migration.md).
   Discovery exposes `stability_tier` (`core` | `domain` | `experimental`) and
   `preferred_path`.

## Coverage ratchet

See `scripts/coverage_ratchet.json` and `pyproject.toml` `fail_under`.
Active floor **70** (full-suite measure ~70.7%). Measure with
`python scripts/run_full_coverage.py` — never from a tiny subset.
