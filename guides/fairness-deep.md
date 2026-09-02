# Fairness (observational)

```bash
pip install buildml
```

You fitted a binary classifier and you want group rates on a holdout
you trust. `session.fairness.evaluate` reports selection rate,
demographic parity, disparate impact, equalized odds, and per-group
classical metrics. That is an observational audit on one split. It
does not certify legal compliance, prove causal discrimination, or
change the model.

You name `sensitive_column`. BuildML will not infer protected class
from the rest of the table. Default evaluate partition is `test`.
Default `positive_label` is `1`; string labels need an explicit
value or the call raises instead of inventing zero rates. Stability
bands are off until `bootstrap_samples > 1`.
`session.fairness.suggest_thresholds` and
`session.fairness.suggest_reweighing` return suggestions only. They
are not applied.

The API refuses a missing fit, a missing split, a missing sensitive
column, an empty partition, and a `positive_label` that never appears
in `y_true`. You still decide which column is sensitive, which
partition to quote, and whether to act on a suggestion.

Short on-ramp: [fairness quickstart](quickstart-fairness.md). Proof:
[loan-fairness-observational](../proofs/loan-fairness-observational/).

## Report after fit

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from buildml import Session

rng = np.random.default_rng(0)
n = 400
group = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
x = rng.normal(size=n)
logits = x + np.where(group == "B", -0.7, 0.0)
y = np.where(logits > 0, "approved", "denied")
frame = pd.DataFrame({"x": x, "group": group, "decision": y})

session = (
    Session.ingest(frame)
    .set_roles({"x": "feature", "group": "ignore", "decision": "target"})
    .split(test_size=0.25, validation_size=0.2, stratify=True, random_state=0)
    .fit(LogisticRegression(max_iter=500), task="classification")
)

report = session.fairness.evaluate(
    sensitive_column="group",
    partition="test",
    positive_label="approved",
)
print(report.demographic_parity_difference)
print(report.selection_rate_by_group)
print(report.classical_metrics_by_group["A"]["f1"])
print(report.to_markdown().splitlines()[0])
```

Give the sensitive column role `ignore` (or leave it out of the
design matrix) so the classifier is not trained on the group id you
later audit. Read `report.warnings` and `report.scope` before you
quote a gap. Gaps describe one split. They do not prove
discrimination and they do not excuse the model.

Bridge from classical evaluate without shrinking that API:

```python
session.evaluate(partition="test")
report = session.fairness.attach_to_last_eval(
    sensitive_column="group",
    positive_label="approved",
)
```

`attach_to_last_eval` uses the partition of the latest
`session.evaluate`, or `test` if none exists. It does not rewrite
classical metrics. The fairness report lives on
`session.fairness.last_report`.

## What the report contains

Native metrics (always, given a fitted binary classifier):

- selection rate by group
- demographic parity difference
- disparate impact ratio
- equalized odds ΔTPR / ΔFPR
- per-group accuracy / precision / recall / F1
- per-group ROC-AUC when scores exist and both classes appear in
  that group's labels (`include_classical_metrics=False` turns the
  classical block off)

Also: `groups`, `support_by_group`, `stability` (or `None`),
`scope` (`legal_audit=False`, `mitigation_applied=False`, …),
`warnings`, `disclosures`, plus `to_markdown()` / `to_dict()`.

This path is binary classification only. Multi-class and regression
fairness suites are out of scope. SHAP (`session.explain_shap`) is
attribution, not a group disparity metric.

## Intersectional keys

Pass a list of columns. Keys are joined as `group|region`:

```python
report = session.fairness.evaluate(
    sensitive_column=["group", "region"],
    partition="test",
    positive_label="approved",
)
print(report.intersectional)
print(report.support_by_group)
```

Sparse cells are expected. Support under 30 emits warnings. Prefer
stability bands before a strong claim on a thin slice.

## Stability bands

Set `bootstrap_samples > 1`. Methods: `bootstrap` (default) or
`stratified_subsample`. These describe sampling variability of
observational gaps on one partition. They are not causal uncertainty.

```python
report = session.fairness.evaluate(
    sensitive_column="group",
    positive_label="approved",
    bootstrap_samples=200,
    stability_method="bootstrap",
    confidence_level=0.95,
    random_state=0,
)
band = report.stability.metrics["demographic_parity_difference"]
print(band["point"], band["ci_low"], band["ci_high"])
```

## Suggestions that stay suggestions

Threshold equalization defaults to `partition="validation"` so you
are not fishing on test. Reweighing defaults to `train`. Neither
call rewrites predictions or refits.

```python
thr = session.fairness.suggest_thresholds(
    sensitive_column="group",
    partition="validation",
    positive_label="approved",
    target="demographic_parity",  # or "equal_opportunity"
)
weights = session.fairness.suggest_reweighing(
    sensitive_column="group",
    partition="train",
    positive_label="approved",
)
```

Applying those thresholds on the same test rows you headline is
optimistic. Reweighing is a statistical adjustment, not a
certificate. If you use the weights, pass them into a future
`session.fit` yourself.

## Leakage

Prefer validation for threshold selection and test for one-shot
reporting. Do not retune thresholds, reweigh, and re-fit against
the same test rows, then claim an unbiased fairness number.
Intersectional sparsity is a statistics problem: keep support
visible.

`error_slices` is a segment error table, not this report. Causal
ML estimates under declared assumptions are a different product.
`session.decision.fit` is cost-sensitive operating points, not a
fairness certificate.
