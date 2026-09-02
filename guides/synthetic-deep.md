# Synthetic tables

```bash
pip install buildml
# native SMOTE: pip install "buildml[imbalanced]"
# SDV CTGAN / TVAE / CopulaGAN + SDMetrics:
# pip install "buildml[synthetic-industry]"
```

You want extra rows that look like train, or a reusable generator you
can sample from later. That is `session.synthetic.*`. It is not
differential privacy. It is not `session.resample` (class rebalance
that rewrites train in place). Samples can still memorize train
structure. Do not ship them as an anonymization control.

Default `method` is `gaussian_copula`, which forces the native backend
even when SDV is installed. `backend=None` follows the method:
`bootstrap` / `gaussian_copula` / `smote` → native; `ctgan` / `tvae` /
`copulagan` → SDV (needs the extra). Fit is always train-only. Default
`sample` merge is `"none"`: you get a frame, the Session does not
change. `merge_mode="extend_train"` appends rows, marks them with
`_synthetic` (`ignore` role), rebuilds split indices, and **clears a
classical `FitResult`** because train membership changed. Validation
and test values stay as they were.

You choose method, sample size, and whether to merge. The API refuses
fit before `split`, SDV methods without the extra, and sample/evaluate
without a synthesizer plan.

Short on-ramp: [synthetic quickstart](quickstart-synthetic.md). Proof:
[synthetic-privacy-utility](../proofs/synthetic-privacy-utility/).

## Fit, sample, evaluate

```python
import pandas as pd
from sklearn.datasets import make_classification

from buildml import Session

x, y = make_classification(
    n_samples=400,
    n_features=6,
    n_informative=4,
    weights=[0.7, 0.3],
    random_state=0,
)
frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
frame["y"] = y
frame["grp"] = pd.Series(y).map({0: "A", 1: "B"})

session = (
    Session.ingest(frame)
    .set_roles(
        {
            **{c: "feature" for c in frame.columns if c.startswith("f")},
            "grp": "feature",
            "y": "target",
        }
    )
    .split(test_size=0.25, validation_size=0.25, random_state=0)
)

fit = session.synthetic.fit(method="gaussian_copula", random_state=0)
print(fit.backend, fit.method)

sample = session.synthetic.sample(n=200, random_state=1, validate=True)
print(sample.frame.shape)

fid = session.synthetic.evaluate(mode="fidelity", partition="test")
tstr = session.synthetic.evaluate(mode="tstr", partition="test")
print(fid.metrics, tstr.metrics)
```

`n=None` samples as many rows as train. `validate=True` runs built-in
schema checks (columns present, null-rate slack, categorical
vocabulary, numeric range slack) and attaches warnings. Great
Expectations column-presence checks run too if that package is
installed separately; they are not required.

`evaluate` never refits the generator on the eval partition. Default
mode is `fidelity`. Default `eval_backend` is `"auto"`: SDMetrics
QualityReport when `sdmetrics` is installed, otherwise built-in KS /
total variation / correlation L1. TSTR trains a sklearn estimator on
synthetic rows and scores real holdout, with a train-on-real baseline
for the gap.

## Backends and methods

| Backend | Extra | Methods |
| --- | --- | --- |
| `native` | none (`smote` needs `imbalanced`) | `bootstrap`, `gaussian_copula`, `smote` |
| `sdv` | `synthetic-industry` | `ctgan`, `tvae`, `copulagan` |

**bootstrap**: resample train rows, optional `smooth_sigma` noise.

**gaussian_copula**: mixed-type empirical CDF plus a correlation
latent (`correlation_ridge=1e-3`). Optional
`condition={col: value}` rejection sampling on `sample`.

**smote**: imblearn wrap. Needs `buildml[imbalanced]`. Target from
`target_column` or the Session target. `k_neighbors` default 5.

**SDV**: single-table deep synthesizers. Knobs: `epochs` (default
300), `batch_size` (default 500). Small train sets (n < 100) may
underfit; disclosures warn. Not DP.

```python
# pip install "buildml[synthetic-industry]"
session.synthetic.fit(backend="sdv", method="ctgan", epochs=100, batch_size=256)
session.synthetic.sample(n=300)
session.synthetic.evaluate(mode="fidelity", eval_backend="auto")
```

Calling `session.synthetic.fit()` with no arguments is native copula,
not SDV, even on a machine where SDV is the catalog's
`default_backend_when_installed`.

## Merge vs resample

`session.resample` rebalances **train** for a class mix and stays a
preprocess step. `session.synthetic.fit` stores a generator. Merge is
explicit:

```python
session.synthetic.sample(
    n=100,
    merge_mode="extend_train",
    provenance_column="_synthetic",
)
# Classical FitResult is cleared. Refit before session.evaluate.
```

Default `merge_mode="none"` leaves the Session alone. Do not treat
synthetic labels as extra holdout.

## Leakage

- Fit always on train (`assert_fit_partition`).
- Holdouts never estimate schema, marginals, or joints.
- `extend_train` rebuilds indices; validation/test cells are unchanged.
- `session.synthetic.evaluate` scores a frozen plan.

## Bundle

`session.synthetic.save_bundle` writes `buildml.synthetic_bundle.v1`
(`meta.json` + `synthetic_plan.joblib`). Session checkpoints do not
embed `SynthesizerPlan`.

```python
session.synthetic.save_bundle("artifacts/synthetic_bundle")
```

## Benchmark

`benchmarks/synthetic/tstr_quality.py` compares TSTR against the
native copula baseline. SDV methods run when
`buildml[synthetic-industry]` is installed.
