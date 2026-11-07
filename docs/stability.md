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
The separate expanded Linux job runs the complete test directory with
`requirements-expanded-coverage.txt` and an **88%** floor. Remote Linux source `eab0f3e` passed **89.6724%** (56,942/63,500 lines),
with 19,329 passed and 173 skipped. [CI42346610505](https://github.com/TechLeo-Libraries/BuildML/actions/runs/42346610505),
backend42346610507, acceptance42346610519 and CodeQL42346609353 all passed.
Tested merge `4697de85` and source whole trees match. The run uses coverage.py
7.16.2 across 800 files and 566 exclusions. All 5,452 required consumer cases
executed. Python 3.12.16, every installed distribution, profile inputs and all
173 skips across 37 messages match `1b29e88`. The warning count fell by 52 to
3,412 as existing NMF convergence warnings disappeared. The same Starlette
and AnyIO deprecation moved to the first new TestClient import. There were
no failures, errors or collection warnings.
Full coverage rose by 22 covered lines with eight more statements. The
serving app gained 21 covered lines, and one unchanged plot board line
was exercised again. Core 70 and expanded 88 remain in force. The next
expanded floor of 89 requires at least 90% complete coverage to preserve
one percentage point of headroom. At `eab0f3e`, 208 more covered lines
would reach 90% at the measured statement count.

The subsequent `8594200` gate passed its tests, but coverage.py upgraded from
7.16.2 to7.17.0 and reported607 excluded lines instead of566. Its raw86.5151%
is not accepted as a comparable coverage gain. Development coverage tooling is
now pinned to7.16.2; the subsequent91b34d6 gate passed with566 exclusions
with the same exclusion accounting used by the current accepted measurement above.

The accepted expanded profile includes scikit-activeml1.0.1, DoWhy0.14 and ONNX Runtime1.31.0 using
NumPy2.4.1/SciPy1.15.3/mlxtend0.23.4. A prior same-source local comparison
passed85.3947% (original) and85.5192% (broader), gained80/lost3 lines, net77,
without combining databases or collection warnings. Local and remote results
are recorded separately; they do not establish coverage for every backend.

The Linux Gloo forkserver workaround passed50 repetitions on each remote
Python3.11/3.12 stack and50 locally on Python3.11.17/Torch2.14.1 CPU, without
retries. This bounded evidence does not establish multi-GPU/NCCL reliability
or a long-term production SLA. The requested90% coverage target remains open.
