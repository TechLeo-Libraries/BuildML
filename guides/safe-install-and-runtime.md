# Safe install and runtime verification

Install BuildML so optional native stacks cannot break a working classical
setup, and verify which Session surfaces are safe in your environment before
you rely on them.

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
| Python 3.13 | 3.13 | any | Core works; many industry wheels are marker-skipped; Torch is often fragile on Windows |

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
pip install -e ".[dev,shap]"
```

POSIX:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
export PYTHONNOUSERSITE=1
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev,shap]"
```

Verify before adding optional native extras:

```bash
python scripts/verify_runtime_stability.py \
  --artifact runtime-stability-core.json \
  --markdown runtime-stability-core.md
```

**Stage A pass criteria:** every probe with tier `gate` or `core` reports `ok`.
Torch / industry ANN rows may report `skip` until you install those extras.

When Stage A is green, these Session paths are safe to use:

- Classical fit / evaluate / pipeline / checkpoint
- Fairness (`evaluate_fairness`)
- SHAP (`explain_shap` via `buildml[shap]`)
- Ensembles, sklearn anomaly, classical forecast, CBR with `backend="sklearn"`
- Native AutoML

## Stage B: optional native stacks (one family at a time)

Only after Stage A is green. Install one extra group, re-run the probe, then
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

Import `ok` is necessary but not sufficient. Before you ship a surface, exercise
it with `verify_runtime_stability.py` or the matching alpha smoke / proof.

## How to read probe statuses

| Status | Meaning | What to do |
| --- | --- | --- |
| `ok` | Use case completed in an isolated subprocess | Safe to use in this environment |
| `skip` | Extra not installed | Install only if you need that surface |
| `fail` | Python exception (often catchable) | Fix the dependency or avoid that API |
| `crash` | Native hard-kill / access violation | Treat that surface as unsupported here |

## What CI already guarantees

- **Windows CI:** classical-only (`pip install -e ".[dev]"`), not full Torch/industry.
- **Linux CI:** Torch / RAG / industry jobs and the coverage ratchet.
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

## Example: clean Windows 3.12 venv

On a clean Windows 11 + Python 3.12.8 virtual environment with
`PYTHONNOUSERSITE=1`, expect results in this shape:

| Stage | Install | Probe summary |
| --- | --- | --- |
| A | `pip install -e ".[dev,shap]"` | gate/core **ok**; Torch/ANN **skip** |
| B1 | `+[torch]` | `torch_import` + `dl_tiny_mlp_fit` **ok** (Torch 2.13 CPU) |
| B2 | `+[cbr-industry]` | `cbr_industry_ann` + `hnswlib_build` **ok** |

A system Python 3.13 install that also pulls user-site Torch often fails DLL
load and industry ANN. Prefer the clean 3.12 venv path on Windows.

## Related

- [Installation (Sphinx)](../docs/installation.rst)
- [Surface stability policy](../docs/stability.md)
- `scripts/verify_runtime_stability.py`
- `scripts/probe_industry_extras.py`
- `scripts/run_full_coverage.py` (full-suite coverage measure)
