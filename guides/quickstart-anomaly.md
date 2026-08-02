# Anomaly / fraud quickstart

> **Install first (GitHub):** PyPI `buildml` is still legacy 1.x and does **not**
> install Session 2.x. Install 2.x from GitHub (or an editable checkout).
> Anomaly detectors use core sklearn — no optional extra is required.
> See [installation](../docs/installation.rst).

Leakage-safe anomaly scoring on the same `Session` as classical ML: history,
explain catalog, and a distinct anomaly bundle. Thresholds and alert rates are
always disclosed.

**Go deeper:** [Anomaly deep](anomaly-deep.md) ·
[Artifacts](artifacts-checkpoints-bundles.md) ·
[Unsupervised](quickstart-unsupervised.md) (separate clustering API).

```bash
# After a GitHub / editable 2.x install:
pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"
```

Classical `Session.fit` stays unchanged. Anomaly methods are
`fit_anomaly` / `score_anomalies` / `evaluate_anomaly` plus
`save_anomaly_bundle` / `load_anomaly_bundle`.

EDA IsolationForest screens and `handle_outliers` fences are **not** this API.
`fit_clusters` remains a separate structure path — cluster labels are not
anomaly flags.

```python
import numpy as np
import pandas as pd

from buildml import Session

rng = np.random.default_rng(0)
n_normal, n_fraud = 200, 20
normal = rng.normal(0.0, 1.0, size=(n_normal, 2))
fraud = rng.normal(4.0, 0.6, size=(n_fraud, 2))
frame = pd.DataFrame(
    np.vstack([normal, fraud]),
    columns=["x", "y"],
)
frame["is_fraud"] = [0] * n_normal + [1] * n_fraud

session = (
    Session.ingest(frame)
    # Keep labels as target so scale does not transform them; anomaly fit
    # excludes protected target roles from features automatically.
    .set_roles({"x": "feature", "y": "feature", "is_fraud": "target"})
    .split(test_size=0.25, stratify=True, random_state=0)
    .scale(method="standard")
)

fit = session.fit_anomaly(
    method="isolation_forest",
    mode="unsupervised",
    contamination=0.1,
)
print(fit.threshold, fit.train_alert_rate, fit.threshold_policy)

scores = session.score_anomalies(partition="test")
print(scores.n_flagged, scores.alert_rate, scores.threshold)

metrics = session.evaluate_anomaly(partition="test", positive_label=1)
print(metrics.alert_rate, metrics.labeled_metrics)

bundle = session.save_anomaly_bundle("artifacts/anomaly_bundle")
fresh = Session.ingest(session.to_pandas()).set_roles(
    {"x": "feature", "y": "feature", "is_fraud": "target"}
)
fresh.split(test_size=0.25, stratify=True, random_state=0).scale(method="standard")
fresh.load_anomaly_bundle(bundle)
again = fresh.score_anomalies(partition="test")
print(again.flags[:5])
```

Novelty mode (normal-only train fit):

```python
session = (
    Session.ingest(frame)
    .set_roles({"x": "feature", "y": "feature", "is_fraud": "target"})
    .split(test_size=0.25, stratify=True, random_state=0)
    .scale(method="standard")
)
session.fit_anomaly(
    method="lof",
    mode="novelty",
    normal_label_value=0,  # uses target role as normal_label_column
    contamination=0.1,
)
session.evaluate_anomaly(partition="test")
```

Supervised fraud-like mode (binary target role):

```python
session = (
    Session.ingest(frame)
    .set_roles({"x": "feature", "y": "feature", "is_fraud": "target"})
    .split(test_size=0.25, stratify=True, random_state=0)
    .scale(method="standard")
)
session.fit_anomaly(method="supervised_hgb", mode="supervised")
session.evaluate_anomaly(partition="test", k=10)
```

Explain catalog coverage:

```python
print(session.explain("fit_anomaly", moment="before").operation)
print(session.explain("evaluate_anomaly", moment="before").concept_links)
```

## Honesty limits

- Higher `anomaly_score` means more anomalous. Flags use a disclosed threshold;
  always report `alert_rate` with operational claims.
- Not a full fraud platform (no graph fraud, no online streaming product).
- No causal fraud claims. Under labels, prefer PR-AUC and precision/recall@k
  when positives are rare — accuracy alone misleads.
- Anomaly bundles are complementary to Session checkpoints and to
  Torch/RAG/unsupervised bundles — not interchangeable.
- EDA IsolationForest / `handle_outliers` / `fit_clusters` are neighboring
  surfaces with different contracts; do not promote screens to `AnomalyPlan`.
