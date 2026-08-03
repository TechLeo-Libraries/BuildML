# Active learning deep guide

> **Install (GitHub 2.x):**
> `pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"`
> See [installation](../docs/installation.rst).

Pool-based active learning on the Session train partition: scarce seed labels,
uncertainty / committee / CoreSet / BALD queries, human `label_rows`, budget
caps, labeled holdout eval, and `buildml.activelearning_bundle.v1`.

**Related:** [Quickstart](quickstart-active-learning.md) ·
[Semi-supervised](semisupervised-deep.md) ·
[Artifacts](artifacts-checkpoints-bundles.md).

---

## What this is / is not

| Is | Is not |
| --- | --- |
| Human-in-the-loop labeling loop on **train** | Semi-supervised graph propagation |
| Uncertainty / committee / CoreSet / BALD query strategies | A built-in oracle that peeks at truths |
| Budget-capped `suggest_query` → `label_rows` | Querying validation/test |
| Distinct AL bundle + explain catalog | Passive NaN-label propagation (`fit_semisupervised`) |

Honesty: **labels come from the user**. Library core never invents an oracle.
Examples and tests may simulate one: always disclose that.

**vs semi-supervised:** Active learning is an *interactive* query loop
(`suggest_query` → human `label_rows` → refit). Semi-supervised learning uses
*passive* missing labels and propagates/pseudo-labels without an oracle loop.

---

## Backends and install

| Backend | Extra | Strategies |
| --- | --- | --- |
| `sklearn` (default) | none | `least_confidence`, `margin`, `entropy`, `committee`, `expected_model_change_lite` |
| `industry` | `buildml[activelearning-industry]` | `core_set`, `qbc_kl`, `qbc_variation_ratios` (scikit-activeml) |
| `torch` | `buildml[torch]` | `bald`, `mc_dropout` (MC-dropout tabular MLP) |

```python
from buildml.activelearning import activelearning_capability_matrix
activelearning_capability_matrix()
```

When extras are installed, industry/torch backends become the honest defaults
for their strategy families. Sklearn remains the fallback when extras are absent.

---

## Pool convention (aligned with semi-supervised)

After a normal stratified split on fully labeled data, blank a fraction of
**train** targets to `NaN`. Those rows become the unlabeled pool. Holdout stays
labeled so `evaluate_active_learning` can score honestly.

`unlabeled_marker` overrides the default NaN convention (same helper as
semi-supervised).

---

## API loop

1. `fit_active_learner(backend=..., strategy=..., label_budget=...)`: fit on labeled train
2. `suggest_query(batch_size=...)`: ranked train-pool indices (no labels)
3. `label_rows(indices=..., labels=...)`: **user** labels; auto-refit by default
4. Repeat until budget exhausted or pool empty
5. `evaluate_active_learning(partition="test")`: labeled holdout only
6. `save_active_learning_bundle(...)`: model + pool indices + query history

`label_rows` is **Session-primary** and **not AI-allowlisted**: humans (or test
harnesses that disclose simulation) supply labels.

Leakage guards:

- Fit requires a split (`assert_can_fit`)
- Pool ⊆ train indices
- `label_rows` refuses validation/test indices
- Eval scores only labeled holdout rows

---

## Strategies

### Sklearn backend

| Strategy | Score (higher = query first) |
| --- | --- |
| `least_confidence` | `1 - max p(y\|x)` |
| `margin` | `-(p_(1) - p_(2))` |
| `entropy` | `-∑ p log p` |
| `committee` | Bagged vote entropy |
| `expected_model_change_lite` | `‖x‖ (1 - p_max)` gradient-magnitude proxy |

### Industry backend (scikit-activeml)

| Strategy | Notes |
| --- | --- |
| `core_set` | k-center / CoreSet diversity on the pool |
| `qbc_kl` | Query-by-committee with KL divergence |
| `qbc_variation_ratios` | QBC variation-ratio disagreement |

### Torch backend

| Strategy | Notes |
| --- | --- |
| `bald` | Bayesian Active Learning by Disagreement via MC dropout |
| `mc_dropout` | Predictive entropy from MC-dropout samples |

---

## Budget and disclosures

- `label_budget` caps how many labels `label_rows` may incorporate
- Exhausted budgets → `suggest_query` returns empty indices with a warning
- Fit / query / label / eval disclosures state that labels are user-supplied
- Walkthrough exposes `activelearning_status` + capability matrix

---

## Benchmark

```bash
python benchmarks/activelearning/query_efficiency.py
```

Produces a label-budget vs test-accuracy curve across sklearn/industry/torch
backends when extras are installed.

---

## Bundle boundary

`buildml.activelearning_bundle.v1` stores `ActiveLearningPlan` (estimator,
encoder, labeled/pool indices, query history, budget, backend). Session
checkpoints do **not** embed the learner. See [Artifacts](artifacts-checkpoints-bundles.md).

---

## Failure modes

- Fitting before split
- Blanking holdout targets and treating them as the pool
- Expecting `suggest_query` to return labels
- Using `fit_semisupervised` when you need a human query loop
- Reporting train accuracy after each query as holdout performance
- Exceeding `label_budget` without raising it

---

