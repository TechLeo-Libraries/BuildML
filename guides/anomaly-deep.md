# Anomaly / fraud deep

> **Install:**
> `pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"`
> Core sklearn only: no optional extra. Industry depth:
> `pip install 'buildml[anomaly-industry]'` (PyOD + XGB/LGBM) and/or
> `pip install 'buildml[torch]'` (autoencoder). See [installation](../docs/installation.rst).

Depth guide for the Session anomaly path: backends, modes, thresholds,
imbalance-honest metrics, validation threshold tuning, bundles, and boundaries
vs EDA / clustering / classical `fit`.

**Related:** [Anomaly quickstart](quickstart-anomaly.md) ·
[Artifacts](artifacts-checkpoints-bundles.md) ·
[Unsupervised deep](unsupervised-deep.md) ·
[Leakage](leakage-cv-recipes.md).

---

## Backends and capability matrix

```python
import pandas as pd

from buildml import Session

# Preferred namespaced form on any Session instance.
# Flat Session.*_capability_matrix classmethods still work for discoverability.
session = Session.ingest(pd.DataFrame({"x": [0.0]}))  # placeholder; use your frame
matrix = session.anomaly.capability_matrix()
print(matrix["backends"]["sklearn"]["methods"])
print(matrix["backends"]["pyod"]["available"])
print(matrix["supervised_scorers"])
```

| Backend | Extra | Methods | Modes |
| --- | --- | --- | --- |
| `sklearn` | core | `isolation_forest`, `lof`, `one_class_svm` | unsupervised, novelty |
| `pyod` | `anomaly-industry` | `hbos`, `copod`, `ecod`, `deepsvdd` | unsupervised, novelty |
| `torch` | `torch` | `autoencoder` | unsupervised, novelty |
| supervised | core / industry | `supervised_hgb`, `supervised_xgb`, `supervised_lgbm` | supervised |

Score calibration disclosures are recorded on every `AnomalyPlan`:
sklearn inverts `score_samples`; PyOD uses `decision_function`; torch AE uses
train-only MSE reconstruction error; supervised scorers emit positive-class
probability (not guaranteed calibrated under extreme imbalance).

---

## Contract

1. Require a `SplitPlan` (`session.assert_can_fit("train")`).
2. Fit detector (+ usually threshold) on **train only**.
3. Optionally tune threshold on **validation** (`session.anomaly.tune_threshold`).
4. Score / flag / evaluate holdout partitions with a frozen `AnomalyPlan`.
5. Disclose threshold policy, threshold value, and alert rate every time.
6. Persist via `buildml.anomaly_bundle.v1` (not a Session checkpoint).

Score orientation: **higher `anomaly_score` = more anomalous**.

---

## Modes

| Mode | Fit rows | Typical methods | Label role during fit |
| --- | --- | --- | --- |
| `unsupervised` | All train rows | IF, LOF, OCSVM, PyOD, AE | None |
| `novelty` | Normal-only train subset | Same unsupervised catalog | Selects fit subset |
| `supervised` | All labeled train rows | HGB / XGB / LGBM | Binary target required |

---

## Threshold policies and validation tuning

| Policy | Meaning |
| --- | --- |
| `contamination` | τ ≈ train score quantile at `1 - contamination` |
| `quantile` | Same with explicit `quantile` |
| `score_threshold` | Absolute cut on anomaly scores |
| `decision_zero` | One-Class SVM convenience (score threshold 0) |
| `validation_tuned` | Set by `session.anomaly.tune_threshold` after fit |

```python
session.anomaly.fit(backend="pyod", method="copod", contamination=0.08)
session.anomaly.tune_threshold(partition="validation", metric="fbeta", fbeta=2.0)
ev = session.anomaly.evaluate(partition="test")  # untouched test
```

Refuses test-partition tuning unless `allow_test_tuning=True` (exploratory only).

---

## Evaluation honesty

Always: `threshold`, `alert_rate`, score summary stats.

When labels exist:

- `average_precision` (PR-AUC), `roc_auc`
- thresholded `precision` / `recall` / `f1`
- `precision_at_k` / `recall_at_k`

Under rare positives, prefer PR-AUC and @k over accuracy. No causal fraud claims.

---

## Benchmark

```bash
python benchmarks/anomaly/detector_comparison.py
# writes benchmarks/anomaly/results/detector_comparison.json
```

Compares sklearn, PyOD (when installed), torch AE (when installed), and
supervised HGB with validation threshold tuning.

---

## Out of scope

- Graph fraud / entity networks
- Online / streaming detectors as a product
- Causal fraud attribution
- Full PyOD algorithm zoo beyond catalog methods
