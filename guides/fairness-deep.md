# Fairness (observational) deep

BuildML’s fairness path is an **honest observational audit** on holdout
predictions from a fitted binary classifier. It reports group disparity gaps,
optional stability bands, and per-group classical metrics. It does **not**
certify legal compliance, identify causal discrimination, or silently “fix”
a model.

**Quickstart:** [quickstart-fairness.md](quickstart-fairness.md) ·
**Proof:** [loan-fairness-observational](../proofs/loan-fairness-observational/) ·
**Capability matrix:** `session.fairness.capability_matrix()`

## Mental model

1. Fit a classical classifier on Session train (`session.fit`).
2. Declare sensitive column(s) yourself — BuildML never infers protected class.
3. `session.fairness.evaluate(...)` scores a partition (default `test`):
   - selection rate by group
   - demographic parity difference
   - disparate impact ratio
   - equalized odds ΔTPR / ΔFPR
   - per-group accuracy / precision / recall / F1 (and ROC-AUC when scores exist)
   - optional bootstrap / stratified-subsample stability bands
4. Read `FairnessReport.to_markdown()` / `to_dict()` including **warnings** and
   **scope** disclosures.
5. Optionally explore post-hoc helpers (`suggest_thresholds`,
   `suggest_reweighing`) — they return suggestions only.

Bridge from classical evaluate without shrinking that API:

```python
session.evaluate(partition="test")
report = session.fairness.attach_to_last_eval(sensitive_column="group")
```

## Intersectional groups

Pass a list/tuple of columns to compose composite keys (`group|region`):

```python
report = session.fairness.evaluate(
    sensitive_column=["group", "region"],
    partition="test",
    positive_label=1,
)
assert report.intersectional
print(report.support_by_group)
```

Sparse intersectional cells are expected. Support `< 30` emits warnings;
prefer stability bands before strong claims.

## Stability bands

Set `bootstrap_samples > 1` (method `bootstrap` or `stratified_subsample`):

```python
report = session.fairness.evaluate(
    sensitive_column="group",
    bootstrap_samples=200,
    stability_method="bootstrap",
    confidence_level=0.95,
    random_state=0,
)
band = report.stability.metrics["demographic_parity_difference"]
print(band["point"], band["ci_low"], band["ci_high"])
```

Bands describe **sampling variability of observational gaps** on one
partition. They are not causal uncertainty and do not prove fairness.

## Classical metrics bridge

Each group gets accuracy / precision / recall / F1. When the estimator exposes
`predict_proba`, per-group ROC-AUC is attached when both classes appear in
that group’s truth labels. Disable with
`include_classical_metrics=False`.

## Opt-in mitigation helpers (not washing)

Under `buildml.fairness.mitigation` and Session facades:

| Helper | Facade | Returns | Default partition |
| --- | --- | --- | --- |
| Threshold equalization | `session.fairness.suggest_thresholds` | per-group thresholds | `validation` |
| Kamiran–Calders reweighing | `session.fairness.suggest_reweighing` | sample weights | `train` |

```python
thr = session.fairness.suggest_thresholds(
    sensitive_column="group",
    partition="validation",
    target="demographic_parity",  # or "equal_opportunity"
)
weights = session.fairness.suggest_reweighing(
    sensitive_column="group",
    partition="train",
)
```

**Hard honesty rules:**

- Helpers never rewrite Session predictions or auto-refit.
- Applying thresholds on the same test rows you headline is leakage / optimism.
- Reweighing is a statistical adjustment, not a fairness certificate.
- Catalog `non_goals` explicitly refuse legal certification and silent washing.

## Report contract

`FairnessReport` fields of note:

- `groups`, `support_by_group`, rate / gap metrics
- `classical_metrics_by_group`
- `stability` (`FairnessStability` or `None`)
- `scope` (`legal_audit=False`, `mitigation_applied=False`, …)
- `warnings`, `disclosures`
- `to_markdown()`, `to_dict()`

## Leakage discipline

- Prefer **validation** for threshold selection; **test** for one-shot reporting.
- Do not retune thresholds / reweigh / re-fit against the same test rows and
  then claim an unbiased fairness number.
- Intersectional sparsity is a statistical problem, not a UI omission — keep
  support visible.

## Relation to other paths

| Path | Role vs fairness |
| --- | --- |
| Classical `evaluate` | Predictive metrics; attach fairness afterward |
| `error_slices` | Segment error tables; not disparity certification |
| Causal ML | Counterfactual / ATE under declared assumptions — different product |
| Decision / optimize | Cost-sensitive thresholds; complementary, not a fairness certificate |
| SHAP (`explain_shap`) | Attribution; not a group disparity metric |

## Non-goals (explicit)

- Legal disparate-impact certification / regulator filings
- Causal fair representation learning
- Multi-class / regression fairness suites
- Automatic silent bias mitigation
- Inferring protected class membership

## Scope notes

Shipped for observational use: intersectional keys, stability bands,
classical bridge, richer reports, opt-in mitigation suggestions, Session
facade paths (`evaluate`, `attach_to_last_eval`, `suggest_*`). Limits:
binary classification only; no causal fairness; no legal product.
