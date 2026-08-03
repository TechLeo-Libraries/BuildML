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
4. **Capability matrices are the honesty layer.** When an optional backend is
   withdrawn, report `available: false` rather than deleting the public method.
5. **Stable-surface candidates for 2.4.0.** Classical ingest/roles/split/
   preprocess/fit/evaluate/CV/search, checkpoint/pipeline bundles, and domain
   `*_capability_matrix` names are freeze candidates. Experimental Torch/AI
   operator details may still move until beta.
6. **Proofs and CI smoke** (`python -m proofs._lib.run_all --smoke`) must stay
   green on the freeze set.

## Coverage floor

See `scripts/coverage_ratchet.json` and `pyproject.toml` `fail_under`.
The active floor is **60** (full-suite measure about 70.7%); the next planned
floor is **70**. Measure with `python scripts/run_full_coverage.py` (full
suite only, never a tiny subset).

## Runtime stability / use-case probe

Run a machine-local honesty check (subprocess-isolated so native access
violations cannot kill the parent process):

```bash
python scripts/verify_runtime_stability.py
python scripts/probe_industry_extras.py --artifact industry-probe.json
```

Statuses: `ok` / `fail` / `crash` / `skip`. **Core/gate `ok`** means classical
Session paths are safe in your environment. **`crash` on optional-native**
means an installed wheel hard-kills that OS/Python combo (Torch / hnswlib /
cvxpy, and similar); treat that surface as unsupported there even if
`pip install` succeeded. Windows CI gates classical-only; Torch and industry
release gates stay on Linux CI.

Staged install (venv → classical → optional extras one group at a time):
[`guides/safe-install-and-runtime.md`](../guides/safe-install-and-runtime.md).
