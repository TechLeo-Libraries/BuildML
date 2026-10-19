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
Active floor **70** (corrected thread/process tracing measured 71.28-71.29%
across Linux Python 3.10-3.13, run 38589015544, commit 2837fb3).
Measure the complete suite with `python scripts/run_full_coverage.py`; subset
runs are not comparable to the project coverage threshold.
The separate expanded Linux job runs the same complete test directory with
`requirements-expanded-coverage.txt` and an **84%** floor, raised after passing
remote CPU coverage above 84%. Remote CPU at `18f0199` passed at 84.96%
(52,528 / 61,824 lines; 3,104 passed, 85 skipped). Local Linux at `d9af9b6`
completed with 3,096 passed and 64 skipped, measuring 84.88% (52,448 / 61,792).
That local result is qualified: installed FLAML skipped after an availability
probe, and an unreadable child coverage-file warning occurred. It does not set
floors. Strict installation-based FLAML gating and a bounded 90-second scientific
import probe are new repairs; the original probe failure's cause is unestablished.
The complete CI, backend
evidence and release acceptance workflows passed. The Linux Gloo forkserver
workaround passed 50 repetitions on each remote Python 3.11/3.12 stack and
50 locally on Python 3.11.17/Torch 2.14.1 CPU, without retries.
Remote CPU results are recorded separately
with their source revision and installed package list. The requested 90% coverage
target remains open.

Current expanded coverage evidence (2026-10-19, source `3d35017`): remote Linux
passed at **85.0410%** (52,580 / 61,829 lines), with 3,125 passed and 91 skipped.
[CI run 39326055971](https://github.com/TechLeo-Libraries/BuildML/actions/runs/39326055971)
passed, as did backend evidence and release acceptance. Tested merge `a75c6d26`
and source trees are equal. This supports expanded floor **84%** with the required
one-point headroom. Scope and 566 excluded lines are unchanged; core floor stays 70%.
Local Linux at the same source passed at **85.1397%** (52,641 / 61,829 lines),
3,153 passed and 63 skipped in 24:29, without coverage collection warnings.
Installed FLAML integration passed. Six new ONNX consumer cases skip remotely
because that profile lacks their dependencies; they pass locally. These records
supersede earlier latest-measurement statements above; historical results retain
their original floors and qualifications. Reaching 90% requires 3,067 further
covered lines at the current remote denominator; subset percentages are not added.
