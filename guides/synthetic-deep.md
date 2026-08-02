# Synthetic-data systems — deep guide

Session path for **train-fitted tabular generators**. Phase-1 bar: Session API,
leakage discipline, tests, explain/catalog, guides, bundle, AI allowlist,
honest docs.

## What this is / is not

| Is | Is not |
| --- | --- |
| Bootstrap / smoothed bootstrap | Differential privacy product |
| Gaussian copula (mixed types, empirical CDF) | SDV / CTGAN / TVAE stack in core |
| Optional SMOTE wrap (`buildml[imbalanced]`) | Drop-in replacement for `Session.resample` |
| Fidelity metrics + TSTR utility eval | Membership-inference / anonymization audit |
| Explicit `extend_train` merge with provenance | Silent poisoning of roles / holdouts |

Dependency policy: core stays light (numpy / scipy / sklearn). No
`buildml[synthetic]` extra — SMOTE reuses the existing `imbalanced` extra.

## Cross-link: `Session.resample`

| | `resample` | `fit_synthesizer` |
| --- | --- | --- |
| Goal | Class rebalance | General tabular generation |
| Mutates train? | Yes (rebuilds split) | Only if `merge_mode='extend_train'` |
| Persists generator? | Lineage `ResamplePlan` | `SynthesizerPlan` bundle |
| Extra | `buildml[imbalanced]` | Core (smote method → same extra) |

Prefer `resample` when the only goal is imbalance handling before `fit`.
Prefer the synthetic path when you need reusable sampling, fidelity/TSTR
evaluation, or controlled augmentation with provenance.

## Methods

### `bootstrap`

Row resampling with replacement from train. Set `smooth_sigma > 0` to add
Gaussian noise (`smooth_sigma × train column std`) on continuous/integer
columns (smoothed bootstrap). Categoricals are copied as-is from donor rows.

**Honesty:** plain bootstrap can emit near-duplicates of train rows.

### `gaussian_copula`

1. Infer column kinds (continuous / integer / categorical).
2. Map each column to a Gaussian latent via empirical CDF (categoricals use
   frequency-bin midpoints so proportions participate in the joint).
3. Estimate correlation (+ ridge / PSD projection).
4. Sample MVN → inverse transform to original domains.

Optional `condition={col: value}` uses rejection sampling (copula only).

**Honesty:** models rank correlations + empirical marginals — not a deep
generative model. Nulls re-introduced at train rates.

### `smote`

Reusable imblearn SMOTE wrap. Requires numeric features, a target, and
`buildml[imbalanced]`. Does **not** mutate Session until an explicit merge.
Prefer `Session.resample(sampler='smote')` for one-shot class balancing.

## Leakage

- Fit **always** on train (`assert_fit_partition`).
- Holdouts never estimate schema / joints.
- `extend_train` rebuilds indices; validation/test values unchanged
  (internal equality guard).
- `evaluate_synthetic` never refits the generator on the eval partition.

## Evaluation

### `mode='fidelity'`

- Continuous/integer: two-sample KS statistic (mean across columns).
- Categorical: total variation distance.
- Continuous pairwise: mean absolute correlation difference (`corr_l1`).

### `mode='tstr'`

Train-on-Synthetic, Test-on-Real with a simple sklearn pipeline
(impute → scale/one-hot → LogisticRegression or Ridge). Also reports a TRTR
baseline when real train is available (`tstr_gap_vs_trtr`).

**Disclosure:** utility proxy, not a generative quality certificate or privacy proof.

## Merge provenance

```python
session.sample_synthetic(n=100, merge_mode="extend_train", provenance_column="_synthetic")
```

- Appends only to train.
- Provenance column role = `ignore` (cannot silently become a feature).
- Clears classical `FitResult` (train membership changed).

Default `merge_mode='none'` returns `SyntheticSampleResult.frame` only.

## Bundle boundary

`buildml.synthetic_bundle.v1` = `meta.json` + `synthetic_plan.joblib`.
Session checkpoints do **not** embed `SynthesizerPlan`.

## Privacy

Not differential privacy. Do not ship synthetic samples as an anonymization
control without a dedicated privacy review. Disclosures are attached to fit /
sample / evaluate / bundles.

## Phase tracker

Phase 1–2 complete. Phase 3 application systems:

- Recommenders **PASS**
- Search / LTR **PASS**
- Knowledge graphs **PASS**
- Optimisation / decisions **PASS**
- **Synthetic-data systems PASS** (this guide)

Residuals after Phase 3 synthetic: NLP/CV deepenings vs existing Torch
multimodal / speech / vision hooks — see package tracker notes.
