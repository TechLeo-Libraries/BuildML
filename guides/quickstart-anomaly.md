# Anomaly / fraud quickstart

> **Install first (GitHub):** PyPI `buildml` is still legacy 1.x and does **not**
> install Session 2.x. Install 2.x from GitHub (or an editable checkout).
> Core sklearn detectors need no extra. Industry depth uses optional extras below.
> See [installation](../docs/installation.rst).

Leakage-safe anomaly scoring on the same `Session` as classical ML: history,
explain catalog, capability matrix, and a distinct anomaly bundle. Thresholds
and alert rates are always disclosed.

**Go deeper:** [Anomaly deep](anomaly-deep.md) ·
[Artifacts](artifacts-checkpoints-bundles.md) ·
[Unsupervised](quickstart-unsupervised.md) (separate clustering API).

```bash
# After a GitHub / editable 2.x install:
pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"

# Industry PyOD + XGBoost/LightGBM fraud scorers:
pip install "buildml[anomaly-industry]"

# Torch tabular autoencoder path:
pip install "buildml[torch]"
```

Classical `Session.fit` stays unchanged. Anomaly methods are
`fit_anomaly` / `score_anomalies` / `evaluate_anomaly` /
`tune_anomaly_threshold` plus bundle save/load.

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

print(Session.anomaly_capability_matrix()["backends"].keys())

fit = session.fit_anomaly(
    backend="sklearn",
    method="isolation_forest",
    mode="unsupervised",
    contamination=0.1,
)
session.tune_anomaly_threshold(partition="validation", metric="f1")
metrics = session.evaluate_anomaly(partition="test", positive_label=1)
print(metrics.alert_rate, metrics.labeled_metrics)
```

PyOD industry backend (when `buildml[anomaly-industry]` installed):

```python
session.fit_anomaly(backend="pyod", method="ecod", contamination=0.1)
session.evaluate_anomaly(partition="test")
```

Torch autoencoder reconstruction error (when `buildml[torch]` installed):

```python
session.fit_anomaly(
    backend="torch",
    method="autoencoder",
    ae_epochs=30,
    contamination=0.1,
)
```

Supervised fraud scorers:

```python
session.fit_anomaly(method="supervised_hgb", mode="supervised")  # core
# session.fit_anomaly(method="supervised_xgb", mode="supervised")  # industry
session.evaluate_anomaly(partition="test", k=10)
```

## Honesty limits

- Higher `anomaly_score` means more anomalous. Score calibration differs by
  backend — compare detectors with ranking metrics (PR-AUC), not raw score scale.
- Tune thresholds on validation via `tune_anomaly_threshold`; reserve test for
  final claims (same discipline as `Session.tune_threshold`).
- Not a full fraud platform (no graph fraud, no online streaming product).
- No causal fraud claims. Under labels, prefer PR-AUC and precision/recall@k.
