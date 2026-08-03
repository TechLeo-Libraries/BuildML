# Synthetic-data systems: deep guide

Session path for **train-fitted tabular generators**. Industry depth: native fallback, SDV when installed, honest capability matrix.

## What this is / is not

| Is | Is not |
| --- | --- |
| Native bootstrap / copula / SMOTE | Differential privacy product |
| SDV CTGAN/TVAE/CopulaGAN (`buildml[synthetic-industry]`) | Drop-in replacement for `Session.resample` |
| Built-in fidelity + TSTR; SDMetrics when installed | Membership-inference / anonymization audit |
| `validate_synthetic` built-in checks (+ optional GE lite) | Silent poisoning of roles / holdouts |
| Explicit `extend_train` merge with provenance | Required SDV stack in core |

Dependency policy: core stays light (numpy / scipy / sklearn). Native SMOTE
reuses `buildml[imbalanced]`. SDV + SDMetrics use `buildml[synthetic-industry]`.

## Capability matrix

```python
from buildml import Session

Session.synthetic_capability_matrix()
# backends: native (always), sdv (when SDV installed)
# evaluation: builtin fidelity/TSTR; sdmetrics when installed
```

Use `backend=` on `fit_synthesizer` / `sample_synthetic` / `evaluate_synthetic`.
When `backend=None`, method name resolves the backend (`gaussian_copula` → native,
`ctgan` → sdv). Default backend when SDV is installed: **sdv** (see matrix).

## Cross-link: `Session.resample`

| | `resample` | `fit_synthesizer` |
| --- | --- | --- |
| Goal | Class rebalance | General tabular generation |
| Mutates train? | Yes (rebuilds split) | Only if `merge_mode='extend_train'` |
| Persists generator? | Lineage `ResamplePlan` | `SynthesizerPlan` bundle |
| Extra | `buildml[imbalanced]` | Core native; SDV → `synthetic-industry` |

## Backends and methods

### Native (`backend='native'`)

**bootstrap**: row resample (+ optional `smooth_sigma` noise).

**gaussian_copula**: mixed-type empirical CDF + correlation latent; optional
`condition={col: value}` rejection sampling.

**smote**: reusable imblearn wrap (`buildml[imbalanced]`).

### SDV (`backend='sdv'`, `buildml[synthetic-industry]`)

**ctgan**, **tvae**, **copulagan**: SDV single-table deep synthesizers.
Knobs: `epochs`, `batch_size`. Train-only fit; not differential privacy.
Small train sets (n<100) may underfit: disclosures warn accordingly.

## Leakage

- Fit **always** on train (`assert_fit_partition`).
- Holdouts never estimate schema / joints.
- `extend_train` rebuilds indices; validation/test values unchanged.
- `evaluate_synthetic` never refits the generator on the eval partition.

## Evaluation

### Built-in (`eval_backend='builtin'` or `'auto'` without SDMetrics)

**fidelity**: KS / total variation / correlation L1.

**tstr**: train-on-synthetic, test-on-real sklearn utility + TRTR baseline.

### SDMetrics (`eval_backend='sdmetrics'` or `'auto'` when installed)

Appends SDMetrics QualityReport scores (`sdmetrics_overall`, property breakdown)
alongside built-in fidelity metrics.

## Validation

```python
session.sample_synthetic(n=100, validate=True)  # built-in checks on sample
# or:
from buildml.synthetic import validate_synthetic
validate_synthetic(session.synthesizer_plan, frame)
```

Built-in: column presence, null-rate tolerance, categorical vocabulary, numeric
range slack. Optional Great Expectations lite column-presence expectations when
`great_expectations` is separately installed.

## Merge provenance

```python
session.sample_synthetic(n=100, merge_mode="extend_train", provenance_column="_synthetic")
```

Provenance column role = `ignore`. Default `merge_mode='none'` returns Frame only.

## Bundle boundary

`buildml.synthetic_bundle.v1` = `meta.json` + `synthetic_plan.joblib`.
Session checkpoints do **not** embed `SynthesizerPlan`.

## Benchmark

`benchmarks/synthetic/tstr_quality.py`: TSTR vs native copula baseline; SDV
methods run when `buildml[synthetic-industry]` is installed.

## Privacy

Not differential privacy. Do not ship synthetic samples as an anonymization
control without a dedicated privacy review.

