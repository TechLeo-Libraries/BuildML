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
structure. Generating synthetic rows does not itself provide an anonymization guarantee.

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

Quickstart: [synthetic quickstart](quickstart-synthetic.md). Proof:
[synthetic-privacy-utility](https://github.com/TechLeo-Libraries/BuildML/tree/main/proofs/synthetic-privacy-utility).

## Fit, sample, evaluate

```python
import numpy as np
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
frame["grp"] = np.random.default_rng(17).choice(["A", "B"], size=len(frame))

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
Conditions compare values without converting them to text: `1` differs from
`"1"`, and `None` selects missing values rather than a literal `"None"` label.
Keys must name fitted columns, and values must be finite numbers, strings,
booleans or `None`. All conditions must hold. Invalid requests and conditions
not satisfied within the rejection-sampling limit raise without appending rows.
An empty dictionary or `None` requests unconditional sampling. AI sampling tools
carry the same `condition` argument through confirmed plans and explicit autonomy;
the proposed conditions appear before confirmation. Other synthesizers refuse
non-empty conditions. Post-sample validation checks the full fitted marginals,
so a deliberately conditioned sample may trigger distribution warnings.

**smote**: imblearn wrap. Needs `buildml[imbalanced]`. Target from
`target_column` or the Session target. `k_neighbors` default 5.

**SDV**: single-table deep synthesizers. Parameters: `epochs` (default
300), `batch_size` (default 500). Small train sets (n < 100) may
underfit; disclosures warn. Not DP.

```python
# Requires: pip install "buildml[synthetic-industry]" (SDV).
import numpy as np
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
frame["grp"] = np.random.default_rng(17).choice(["A", "B"], size=len(frame))

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

# pip install "buildml[synthetic-industry]"
session.synthetic.fit(backend="sdv", method="ctgan", epochs=100, batch_size=200)
session.synthetic.sample(n=300)
session.synthetic.evaluate(mode="fidelity", eval_backend="auto")
```

Calling `session.synthetic.fit()` with no arguments is native copula,
not SDV, even on a machine where SDV is the catalog's
`default_backend_when_installed`.

Builtin fidelity uses typed scalar categories, with one separate category for
missing values. Numeric `1`, string `"1"` and boolean `True` remain distinct
when the input column preserves those types. NumPy scalar wrappers compare as
their corresponding Python values. This avoids a perfect total-variation score
caused by converting distinct categories to the same text.

Classification TSTR and TRTR fit their probes on training labels represented
by integer codes. Holdout labels never add training examples or model outputs;
an unseen class counts as an error and participates in macro-F1. Missing
classification targets raise before storing the result. Categorical feature
preprocessing still uses the existing sklearn contract; this change does not
extend the feature types that it accepts. Saved generators and their sampled
rows remain unchanged. Rerun evaluation to correct a historical report.

## Merge vs resample

`session.resample` rebalances **train** for a class mix and stays a
preprocess step. `session.synthetic.fit` stores a generator. Merge is
explicit:

```python
import numpy as np
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
frame["grp"] = np.random.default_rng(17).choice(["A", "B"], size=len(frame))

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
import numpy as np
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
frame["grp"] = np.random.default_rng(17).choice(["A", "B"], size=len(frame))

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

session.synthetic.save_bundle("artifacts/synthetic_bundle")
```

## Benchmark

`benchmarks/synthetic/tstr_quality.py` compares TSTR against the
native copula baseline. SDV methods run when
`buildml[synthetic-industry]` is installed.
