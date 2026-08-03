# BuildML 2.x surface stability policy

BuildML `2.4.0a*` is alpha. The Session public surface is large on purpose
(hundreds of methods and domain bundles). This policy limits churn risk before
a stable 2.x release.

## Rules

1. **Additive by default.** New domains and methods may land in alpha without
   removing existing ones.
2. **Breaking changes require a CHANGELOG entry** under `Removed` / `Changed`
   and a migration note in `docs/legacy.rst` or the affected guide.
3. **Bundle schemas are versioned** (`buildml.<domain>_bundle.v1`, …). Bump the
   version string when the on-disk layout changes; keep loaders able to refuse
   unknown versions with a clear error.
4. **Capability matrices are the honesty layer.** Prefer reporting
   `available: false` over deleting a public method when an optional backend
   is withdrawn.
5. **Freeze candidates for 2.4.0.** Classical ingest/roles/split/preprocess/
   fit/evaluate/CV/search, checkpoint/pipeline bundles, and domain
   `*_capability_matrix` names are freeze candidates. Experimental Torch/AI
   operator details may still move until beta.
6. **Proofs and CI smoke** (`python -m proofs._lib.run_all --smoke`) must stay
   green on the freeze set. Smoke fails on unexpected
   `skipped_missing_extra` / `partial` result statuses (use `--allow-skip`
   only for local investigation).

## Coverage ratchet

See `scripts/coverage_ratchet.json` and `pyproject.toml` `fail_under`.
Active floor **60** (full-suite measure ~70.7%); next planned **70**.
Measure with `python scripts/run_full_coverage.py` (never a tiny subset).
