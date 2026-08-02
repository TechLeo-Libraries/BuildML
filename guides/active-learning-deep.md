# Active learning deep guide

> **Install (GitHub 2.x):**
> `pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"`
> See [installation](../docs/installation.rst).

Pool-based active learning on the Session train partition: scarce seed labels,
uncertainty / committee queries, human `label_rows`, budget caps, labeled
holdout eval, and `buildml.activelearning_bundle.v1`.

**Related:** [Quickstart](quickstart-active-learning.md) ·
[Semi-supervised](semisupervised-deep.md) ·
[Artifacts](artifacts-checkpoints-bundles.md).

---

## What this is / is not

| Is | Is not |
| --- | --- |
| Human-in-the-loop labeling loop on **train** | Semi-supervised graph propagation |
| Uncertainty / committee query strategies | A built-in oracle that peeks at truths |
| Budget-capped `suggest_query` → `label_rows` | Querying validation/test |
| Distinct AL bundle + explain catalog | Online / continual `partial_fit` (done — ``buildml.online``) |

Honesty: **labels come from the user**. Library core never invents an oracle.
Examples and tests may simulate one — always disclose that.

---

## Pool convention (aligned with semi-supervised)

After a normal stratified split on fully labeled data, blank a fraction of
**train** targets to `NaN`. Those rows become the unlabeled pool. Holdout stays
labeled so `evaluate_active_learning` can score honestly.

`unlabeled_marker` overrides the default NaN convention (same helper as
semi-supervised).

---

## API loop

1. `fit_active_learner(strategy=..., label_budget=...)` — fit on labeled train
2. `suggest_query(batch_size=...)` — ranked train-pool indices (no labels)
3. `label_rows(indices=..., labels=...)` — user labels; auto-refit by default
4. Repeat until budget exhausted or pool empty
5. `evaluate_active_learning(partition="test")` — labeled holdout only
6. `save_active_learning_bundle(...)` — model + pool indices + query history

Leakage guards:

- Fit requires a split (`assert_can_fit`)
- Pool ⊆ train indices
- `label_rows` refuses validation/test indices
- Eval scores only labeled holdout rows

---

## Strategies

| Strategy | Score (higher = query first) |
| --- | --- |
| `least_confidence` | `1 - max p(y\|x)` |
| `margin` | `-(p_(1) - p_(2))` |
| `entropy` | `-∑ p log p` |
| `committee` | Bagged vote entropy (requires `strategy='committee'` at fit) |
| `expected_model_change_lite` | `‖x‖ (1 - p_max)` gradient-magnitude proxy |

All uncertainty strategies need `predict_proba` (logistic regression and
hist gradient boosting both provide it).

---

## Budget and disclosures

- `label_budget` caps how many labels `label_rows` may incorporate
- Exhausted budgets → `suggest_query` returns empty indices with a warning
- Fit / query / label / eval disclosures state that labels are user-supplied
- Walkthrough exposes `activelearning_status`

---

## Bundle boundary

`buildml.activelearning_bundle.v1` stores `ActiveLearningPlan` (estimator,
encoder, labeled/pool indices, query history, budget). Session checkpoints do
**not** embed the learner. See [Artifacts](artifacts-checkpoints-bundles.md).

---

## Failure modes

- Fitting before split
- Blanking holdout targets and treating them as the pool
- Expecting `suggest_query` to return labels
- Reporting train accuracy after each query as holdout performance
- Exceeding `label_budget` without raising it

---

## Phase tracker

Phase 2 items 1–4 (semi / self / active / online) are done. **Next:** multi-task
learning.
