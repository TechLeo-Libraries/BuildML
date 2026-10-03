# Safe install and runtime verification

Use isolated environments to reduce conflicts between optional native
packages and a working classical setup, and verify which workflows run in your environment before
you rely on them.

This guide pairs with `scripts/verify_runtime_stability.py` (subprocess
use-case probes: `ok` / `fail` / `crash` / `skip`). That is different from
`scripts/probe_industry_extras.py`, which only checks whether industry modules
**import**.

## Why staged install matters

`pip install` can succeed while a later `import torch` or industry ANN call
hard-crashes the process (Windows access violation or DLL initialization
failure). BuildML cannot catch those faults inside the same Python process.
Use a clean virtual environment, install in stages, and run the runtime probe
after each stage.

## Recommended platform matrix

| Goal | Python | OS | Notes |
| --- | --- | --- | --- |
| Classical / most sklearn domains | **3.11 or 3.12** | Windows, Linux, macOS | Matches the Windows CI classical gate |
| Torch / DL / heavy industry | **3.11 or 3.12** | **Linux preferred** | Linux CI is the release gate for Torch and industry extras |
| Python 3.13 | 3.13 | any | Core works; many industry wheels are marker-skipped; check optional backend support for the selected platform |

Always use a project virtual environment. On Windows, avoid mixing BuildML with
packages from the user site-packages tree (`%APPDATA%\Python\...`).

## Stage A: clean classical environment

PowerShell (Windows):

```powershell
# Prefer 3.12 (or 3.11).
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
$env:PYTHONNOUSERSITE = "1"   # block AppData user-site leakage
python -m pip install --upgrade pip setuptools wheel
pip install "buildml[dev,shap]"
# Or from a source checkout: pip install -e ".[dev,shap]"
```

POSIX:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
export PYTHONNOUSERSITE=1
python -m pip install --upgrade pip setuptools wheel
pip install "buildml[dev,shap]"
# Or from a source checkout: pip install -e ".[dev,shap]"
```

Run `verify_runtime_stability.py` from a BuildML source checkout (the script lives under `scripts/`).

Verify before adding optional native extras:

```bash
python scripts/verify_runtime_stability.py \
  --artifact runtime-stability-core.json \
  --markdown runtime-stability-core.md
```

**Stage A pass criteria:** every probe with tier `gate` or `core` reports `ok`.
Torch / industry ANN rows may report `skip` until you install those extras.

Inspect the probe results for each workflow below. A successful probe
checks its executed example; validate additional APIs and your own data
separately:

- Classical fit / evaluate / pipeline / checkpoint
- Fairness (`session.fairness.evaluate`)
- SHAP (`session.explain_shap` via `buildml[shap]`)
- Ensembles, sklearn anomaly, classical forecast, CBR with `backend="sklearn"`
- Native AutoML

## Stage B: optional native stacks (one family at a time)

After Stage A passes, Install one extra group, re-run the probe, then
keep or remove that group based on the result.

### B1: Torch / DL

```bash
pip install -e ".[torch]"
# If the default wheel fails, try the official CPU index, for example:
# pip install torch --index-url https://download.pytorch.org/whl/cpu
python scripts/verify_runtime_stability.py \
  --artifact runtime-stability-torch.json \
  --markdown runtime-stability-torch.md
```

Require `torch_import` and `dl_tiny_mlp_fit` = `ok` before trusting DL workflows.
If either reports `fail` or `crash`, uninstall Torch and stay on Stage A, or
move DL work to Linux.

### B2: CBR industry ANN (`hnswlib`)

```bash
pip install -e ".[cbr-industry]"
python scripts/verify_runtime_stability.py \
  --artifact runtime-stability-cbr.json \
  --markdown runtime-stability-cbr.md
```

- Prefer `fit_cbr(backend="sklearn")` unless `cbr_industry_ann` reports `ok`.
- If `hnswlib_build` is `ok` but `cbr_industry_ann` is `crash`, do not use
  industry ANN in that environment; sklearn CBR remains the supported path.

### B3: other industry extras

```bash
python scripts/probe_industry_extras.py \
  --artifact industry-probe.json \
  --markdown industry-probe.md
```

Import `ok` is necessary but not sufficient. Before deployment, exercise
it with `verify_runtime_stability.py` or the matching example or end-to-end check.

## How to read probe statuses

| Status | Meaning | What to do |
| --- | --- | --- |
| `ok` | Use case completed in an isolated subprocess | Probe passed; validate your actual workload |
| `skip` | Extra not installed | Install only if you need that API |
| `fail` | Python exception (often catchable) | Fix the dependency or avoid that API |
| `crash` | Native hard-kill / access violation | Treat that API as unsupported here |

## What CI checks

- **Windows CI:** classical-only (`pip install -e ".[dev]"`), not full Torch/industry.
- **Linux CI:** Torch, RAG, optional-backend tests, and full-suite coverage.
- **Release acceptance:** core wheel installation and representative workflows
  on Windows, Linux, and macOS with Python 3.10 through 3.13.
- `buildml[production]` is **best-effort**; environment markers skip known-broken
  wheels (especially on Python 3.13 / Windows).

## Checklist

1. Create a **venv** on **Python 3.11 or 3.12**.
2. On Windows, set **`PYTHONNOUSERSITE=1`**.
3. Install **`[dev]` / classical (+ `[shap]`)** first; verify with
   `scripts/verify_runtime_stability.py`.
4. Add Torch or industry extras **one group at a time**; re-verify after each.
5. If a probe returns `crash`, fall back to sklearn backends or Linux for that
   surface.
6. Prefer Linux for production Torch and heavy industry workloads.

## Interpret results from your environment

The probe reports the status of each use case and the versions installed
in that environment. Treat `skip` as unverified and investigate every
`fail` or `crash`. Installing an extra does not establish that all of its
methods work; run the workflow you intend to use with representative data.

Mixing user-site packages with a project environment can introduce
incompatible native libraries. Keep optional backends isolated and record
their versions with the probe results.

## Redirecting teaching output on Windows

Teaching text includes Unicode symbols. If redirected output raises
`UnicodeEncodeError` on Windows, enable Python's UTF-8 mode. For an example
saved as `example.py`, run:

```bash
python -X utf8 example.py > explanation.txt
```

## Related links

- [Installation (Sphinx)](../docs/installation.rst)
- [API stability policy](../docs/stability.md)
- `scripts/verify_runtime_stability.py`
- `scripts/probe_industry_extras.py`
- `scripts/run_full_coverage.py` (full-suite coverage measure)
