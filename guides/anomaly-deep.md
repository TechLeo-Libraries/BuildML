# Anomaly / fraud (deep)

```bash
pip install buildml
# PyOD / GBDT: pip install "buildml[anomaly-industry]"
# autoencoder: pip install "buildml[torch]"
```

A detector on the same Session. Default is sklearn IsolationForest,
unsupervised. A target is only required for supervised mode, threshold
tuning, and labeled eval. `method="isolation_forest"` stays sklearn even
if PyOD is installed. Tuning on test is refused unless
`allow_test_tuning=True`.

Higher `anomaly_score` means more anomalous. Clustering and stream processing use separate workflows.

Quickstart: [anomaly quickstart](quickstart-anomaly.md).

## Backends

| Backend | Extra | Methods | Modes |
| --- | --- | --- | --- |
| `sklearn` | core | `isolation_forest`, `lof`, `one_class_svm` | unsupervised, novelty |
| `pyod` | `anomaly-industry` | `hbos`, `copod`, `ecod`, `deepsvdd` | unsupervised, novelty |
| `torch` | `torch` | `autoencoder` | unsupervised, novelty |
| supervised | core / industry | `supervised_hgb`, `supervised_xgb`, `supervised_lgbm` | supervised |

Score calibration is disclosed on every `AnomalyPlan`: sklearn inverts
`score_samples`; PyOD uses `decision_function`; torch AE uses train-only
MSE reconstruction error; supervised scorers emit positive-class
probability (not guaranteed calibrated under extreme imbalance).

```python
import numpy as np
import pandas as pd

from buildml import Session

rng = np.random.default_rng(0)
n_normal, n_fraud = 200, 20
normal = rng.normal(0.0, 1.0, size=(n_normal, 2))
fraud = rng.normal(4.0, 0.6, size=(n_fraud, 2))
frame = pd.DataFrame(np.vstack([normal, fraud]), columns=["x", "y"])
frame["is_fraud"] = [0] * n_normal + [1] * n_fraud

session = (
    Session.ingest(frame)
    .set_roles({"x": "feature", "y": "feature", "is_fraud": "target"})
    .split(test_size=0.25, validation_size=0.15, stratify=True, random_state=0)
    .scale(method="standard")
)

matrix = session.anomaly.capability_matrix()
print(matrix["backends"]["sklearn"]["methods"])
print(matrix["backends"]["pyod"]["available"])
```

## The loop

1. Split first (`assert_can_fit("train")`).
2. Fit the detector, and usually a threshold, on train only.
3. Optionally `session.anomaly.tune_threshold` on **validation**.
4. Score / flag / evaluate holdout with the frozen plan.
5. Persist `buildml.anomaly_bundle.v1`. That is not a Session checkpoint.

## Modes

| Mode | Fit rows | Label during fit |
| --- | --- | --- |
| `unsupervised` | All train rows | None |
| `novelty` | Normal-only train subset | Selects the fit subset |
| `supervised` | All labeled train rows | Binary target required |

## Thresholds

| Policy | Meaning |
| --- | --- |
| `contamination` | Train score quantile at `1 - contamination` |
| `quantile` | Same, with an explicit `quantile` |
| `score_threshold` | Absolute cut |
| `decision_zero` | One-Class SVM convenience (cut at 0) |
| `validation_tuned` | Set by `tune_threshold` after fit |

```python
import numpy as np
import pandas as pd

from buildml import Session

rng = np.random.default_rng(0)
n_normal, n_fraud = 200, 20
normal = rng.normal(0.0, 1.0, size=(n_normal, 2))
fraud = rng.normal(4.0, 0.6, size=(n_fraud, 2))
frame = pd.DataFrame(np.vstack([normal, fraud]), columns=["x", "y"])
frame["is_fraud"] = [0] * n_normal + [1] * n_fraud

session = (
    Session.ingest(frame)
    .set_roles({"x": "feature", "y": "feature", "is_fraud": "target"})
    .split(test_size=0.25, validation_size=0.15, stratify=True, random_state=0)
    .scale(method="standard")
)

session.anomaly.fit(backend="pyod", method="copod", contamination=0.08)
session.anomaly.tune_threshold(partition="validation", metric="fbeta", fbeta=2.0)
ev = session.anomaly.evaluate(partition="test")
```

## Evaluate

Always: `threshold`, `alert_rate`, score summary. When labels exist:
`average_precision` (PR-AUC), `roc_auc`, thresholded precision / recall /
f1, and precision/recall at k. Under rare positives, prefer PR-AUC and
@k over accuracy. These scores do not establish whether fraud occurred or identify its cause.

## What usually goes wrong

- Fit without a split: `LeakageError`.
- Supervised mode without a target: `ValidationError`.
- Tuning on test without `allow_test_tuning=True`: refused.
- Treating this as `session.unsupervised.fit`.
- Graph analysis and stream processing require their respective workflows.

[Unsupervised](unsupervised-deep.md) · [Leakage](leakage-cv-recipes.md)
