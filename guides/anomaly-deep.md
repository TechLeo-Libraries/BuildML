# Anomaly / fraud deep

> **Install (GitHub 2.x):**
> `pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"`
> Core sklearn only — no optional extra. See [installation](../docs/installation.rst).

Depth guide for the Session anomaly path: modes, thresholds, imbalance-honest
metrics, bundles, and clear boundaries vs EDA / clustering / classical `fit`.

**Related:** [Anomaly quickstart](quickstart-anomaly.md) ·
[Artifacts](artifacts-checkpoints-bundles.md) ·
[Unsupervised deep](unsupervised-deep.md) ·
[Leakage](leakage-cv-recipes.md).

Phase 1 depth order (complete): unsupervised → ensembles → AutoML →
forecasting → **anomaly**. Phase 2 kicks off with semi/self-supervised hooks
(not implemented in this guide).

---

## Contract

1. Require a `SplitPlan` (`session.assert_can_fit("train")`).
2. Fit detector (+ usually threshold) on **train only**.
3. Score / flag / evaluate holdout partitions with a frozen `AnomalyPlan`.
4. Disclose threshold policy, threshold value, and alert rate every time.
5. Persist via `buildml.anomaly_bundle.v1` (not a Session checkpoint).

Score orientation: **higher `anomaly_score` = more anomalous**.

---

## Modes

| Mode | Fit rows | Typical methods | Label role during fit |
| --- | --- | --- | --- |
| `unsupervised` | All train rows | IsolationForest, LOF, One-Class SVM | None |
| `novelty` | Train rows where `normal_label_column == normal_label_value` | IsolationForest, LOF, One-Class SVM | Selects normal-only subset |
| `supervised` | All labeled train rows | `supervised_hgb` | Binary target required |

Novelty is **semi-supervised in the normal-only sense** — not Phase-2
representation learning. Do not call it “fully unlabeled” when normal labels
selected the fit subset.

---

## Threshold policies

| Policy | Meaning |
| --- | --- |
| `contamination` | τ ≈ train score quantile at `1 - contamination` |
| `quantile` | Same idea with explicit `quantile` (defaults to contamination) |
| `score_threshold` | Absolute cut on anomaly scores |
| `decision_zero` | One-Class SVM convenience (score threshold 0) |

Holdout `alert_rate` is **not** guaranteed to equal the contamination prior.
Report both.

```python
session.fit_anomaly(
    method="one_class_svm",
    mode="unsupervised",
    threshold_policy="decision_zero",
    nu=0.05,
)
session.score_anomalies(partition="test", override_threshold=0.1)  # call-local only
```

---

## Integration notes

- **PCA:** `prefer_reduce_components=True` consumes `Session.reduce_dimensions`
  components without refitting PCA (same contract as clustering).
- **Clustering:** `fit_clusters` / `ClusterPlan` are complementary structure
  signals. Do not treat cluster ids as anomaly flags; do not fork a second
  clustering API inside anomaly.
- **EDA IsolationForest:** descriptive dashboard/HTML screen only.
- **`handle_outliers`:** train-fitted IQR/z-score fences for preprocess — not
  anomaly scoring.
- **Classical `fit`:** still the general supervised path; anomaly
  `mode='supervised'` is a focused fraud-like wrapper with imbalance-honest
  `evaluate_anomaly` metrics.

---

## Evaluation honesty

Always: `threshold`, `alert_rate`, score summary stats.

When labels exist (`label_column` or target role):

- `average_precision` (PR-AUC), `roc_auc`
- thresholded `precision` / `recall` / `f1`
- `precision_at_k` / `recall_at_k` (default `k ≈ positive_rate * n`)

Under rare positives, prefer PR-AUC and @k over accuracy. No causal fraud
claims.

---

## Bundles

```python
path = session.save_anomaly_bundle("artifacts/anomaly_bundle")
# meta.json + anomaly_plan.joblib  (format buildml.anomaly_bundle.v1)

other = Session.ingest(frame).set_roles(...).split(...).scale(...)
other.load_anomaly_bundle(path)
other.score_anomalies(partition="test")
```

Session checkpoints do **not** embed `AnomalyPlan`.

---

## Failure modes

- Fitting without a split (refused).
- Novelty without `normal_label_column` / target defining normals.
- Supervised without a binary target.
- Null / non-numeric features (impute/scale first).
- Publishing flags without threshold or alert rate.
- Equating EDA `anomaly_rate` with a production `AnomalyPlan`.
- Expecting graph fraud, streaming alerts, or causal attribution.

---

## Worked loops

### Unsupervised IsolationForest → labeled holdout metrics

```python
import numpy as np
import pandas as pd
from buildml import Session

rng = np.random.default_rng(1)
frame = pd.DataFrame(
    {
        "x": np.concatenate([rng.normal(0, 1, 300), rng.normal(5, 0.5, 30)]),
        "y": np.concatenate([rng.normal(0, 1, 300), rng.normal(-4, 0.5, 30)]),
        "is_fraud": [0] * 300 + [1] * 30,
    }
)
session = (
    Session.ingest(frame)
    .set_roles({"x": "feature", "y": "feature", "is_fraud": "target"})
    .split(test_size=0.25, stratify=True, random_state=0)
    .scale(method="standard")
)
session.fit_anomaly(method="isolation_forest", contamination=0.08)
ev = session.evaluate_anomaly(partition="test")
assert "average_precision" in ev.labeled_metrics
assert ev.threshold_policy == "contamination"
```

Keep fraud/event labels on the **target** role (or otherwise unscaled).
`Session.scale` transforms numeric non-target columns — an `ignore` label that
has been z-scored will break novelty filters and `positive_label` matching.

### Novelty LOF

```python
session.fit_anomaly(
    method="lof",
    mode="novelty",
    normal_label_value=0,
    n_neighbors=15,
    contamination=0.1,
)
print(session.anomaly_plan.n_fit_rows, session.anomaly_plan.n_train_rows)
```

### Supervised HGB

```python
session = (
    Session.ingest(frame)
    .set_roles({"x": "feature", "y": "feature", "is_fraud": "target"})
    .split(test_size=0.25, stratify=True, random_state=0)
    .scale(method="standard")
)
session.fit_anomaly(method="supervised_hgb", mode="supervised")
print(session.evaluate_anomaly(partition="test", k=20).labeled_metrics)
```

---

## Teaching surface

```python
session.explain("fit_anomaly", moment="before")
session.explain("evaluate_anomaly", moment="after")
wt = session.walkthrough()
print(wt.anomaly_status["has_anomaly_plan"], wt.anomaly_status["disclosures"][:2])
```

AI allowlist tools: `fit_anomaly`, `score_anomalies`, `evaluate_anomaly`,
`save_anomaly_bundle`, `load_anomaly_bundle`.

---

## Out of scope (Phase 1)

- Graph fraud / entity networks
- Online / streaming detectors as a product
- Causal fraud attribution
- Full autoencoder / Torch anomaly zoo (core sklearn path only)
- Semi/self/active/online Session paths (Phase 2 items 1–4 done; next: multi-task)
