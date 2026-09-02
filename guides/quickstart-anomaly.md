# Anomaly / fraud quickstart

```bash
pip install buildml
# PyOD: pip install "buildml[anomaly-industry]"
# torch AE: pip install "buildml[torch]"
```

Default is sklearn IsolationForest, unsupervised. A target is only required
for supervised mode, threshold tuning, and labeled eval. Tuning on test is
refused unless `allow_test_tuning=True`. This is not clustering and not a
streaming fraud platform.

[Anomaly deep](anomaly-deep.md) ·
[network-intrusion-anomaly](../proofs/network-intrusion-anomaly/)

Classical `Session.fit` stays unchanged. Anomaly methods are
`session.anomaly.fit` / `session.anomaly.score` / `session.anomaly.evaluate` /
`session.anomaly.tune_threshold` plus bundle save/load.

EDA IsolationForest screens and `handle_outliers` fences are **not** this API.

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

print(session.anomaly.capability_matrix()["backends"].keys())

fit = session.anomaly.fit(
    backend="sklearn",
    method="isolation_forest",
    mode="unsupervised",
    contamination=0.1,
)
session.anomaly.tune_threshold(partition="validation", metric="f1")
metrics = session.anomaly.evaluate(partition="test", positive_label=1)
print(metrics.alert_rate, metrics.labeled_metrics)
```

PyOD industry backend (when `buildml[anomaly-industry]` installed):

```python
session.anomaly.fit(backend="pyod", method="ecod", contamination=0.1)
session.anomaly.evaluate(partition="test")
```

Torch autoencoder reconstruction error (when `buildml[torch]` installed):

```python
session.anomaly.fit(
    backend="torch",
    method="autoencoder",
    ae_epochs=30,
    contamination=0.1,
)
```

Supervised fraud scorers:

```python
session.anomaly.fit(method="supervised_hgb", mode="supervised")  # core
# session.anomaly.fit(method="supervised_xgb", mode="supervised")  # industry
session.anomaly.evaluate(partition="test", k=10)
```

## Honesty limits

- Higher `anomaly_score` means more anomalous. Score calibration differs by
  backend: compare detectors with ranking metrics (PR-AUC), not raw score scale.
- Tune thresholds on validation via `session.anomaly.tune_threshold`; reserve test for
  final claims (same discipline as `Session.tune_threshold`).
- Not a full fraud platform (no graph fraud, no online streaming product).
- No causal fraud claims. Under labels, prefer PR-AUC and precision/recall@k.
