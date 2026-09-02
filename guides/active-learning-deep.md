# Active learning

```bash
pip install buildml
# BALD / MC-dropout: pip install "buildml[torch]"
# scikit-activeml host path: pip install "buildml[activelearning-industry]"
```

You have a labeled seed and a larger unlabeled train pool. Labels cost
money or time. You want the next batch of indices to show a human, not a
guessed label from the library.

`session.active_learning.fit` defaults to strategy `margin` on a sklearn
logistic. That pairing stays sklearn even if torch or scikit-activeml is
installed. Pass `strategy="core_set"` (or another industry name) with
`backend=None` to take the industry path. Pass `strategy="bald"` with
`backend=None` to take torch, which refuses without `buildml[torch]`.
`backend="industry"` with the default `margin` strategy is refused:
the strategy has to belong to that backend.

The API refuses to query validation or test, invent an oracle, or score
unlabeled holdout rows as truth. You decide who labels, when the budget
stops, and whether each round refits.

Short on-ramp: [active learning quickstart](quickstart-active-learning.md).
Proof: [active-labeling-budget](../proofs/active-labeling-budget/).
This is not [semi-supervised](semisupervised-deep.md): that path
propagates missing labels without a human loop.

## The loop

1. Split on fully labeled data so holdout stays labeled for eval.
2. Blank a fraction of **train** targets to NaN (or your `unlabeled_marker`).
3. `session.active_learning.fit` on labeled train only.
4. `session.active_learning.suggest_query` returns ranked train-pool indices.
5. A human (or a test harness you disclose as simulated) supplies labels to
   `session.active_learning.label_rows`.
6. Repeat until the pool is empty or `label_budget` is spent.
7. `session.active_learning.evaluate` scores **labeled** holdout rows only.
8. `session.active_learning.save_bundle` writes `buildml.activelearning_bundle.v1`.

`label_budget` defaults to 50. `batch_size` defaults to 5. `auto_refit`
defaults to True, so `label_rows` refits unless you pass `refit=False`.

```python
import numpy as np
import pandas as pd

from buildml import Session
from buildml.data.dataset import Dataset
from buildml.ingest.detect import schema_from_dataframe

rng = np.random.default_rng(0)
x0 = rng.normal([-1.0, -1.0], 0.55, size=(140, 2))
x1 = rng.normal([1.2, 1.0], 0.55, size=(140, 2))
frame = pd.DataFrame(np.vstack([x0, x1]), columns=["x", "y"])
frame["label"] = [0] * 140 + [1] * 140
# Hidden copy for this example's simulated oracle only. The library never sees it.
truth = frame["label"].copy()

session = (
    Session.ingest(frame)
    .set_roles({"x": "feature", "y": "feature", "label": "target"})
    .split(test_size=0.25, stratify=True, random_state=0)
    .scale(method="standard")
)

full = session.to_pandas().copy()
train_idx = list(session.split_plan.train_indices)
blank = rng.choice(train_idx, size=int(0.85 * len(train_idx)), replace=False)
full.loc[blank, "label"] = np.nan
session._dataset = Dataset.from_transformed(
    session.dataset,
    full,
    schema=schema_from_dataframe(full),
    roles=dict(session.dataset.roles),
)

fit = session.active_learning.fit(
    strategy="margin",
    base_estimator="logistic_regression",
    batch_size=8,
    label_budget=24,
)
print(fit.n_labeled_train, fit.n_unlabeled_pool, fit.strategy)

for round_i in range(3):
    q = session.active_learning.suggest_query(batch_size=8)
    if not q.indices:
        break
    human_labels = [int(truth.loc[i]) for i in q.indices]
    labeled = session.active_learning.label_rows(
        indices=q.indices, labels=human_labels
    )
    print(round_i, labeled.n_newly_labeled, labeled.budget_remaining)

ev = session.active_learning.evaluate(partition="test")
print(ev.n_labeled_eval, ev.metrics)
session.active_learning.save_bundle("artifacts/activelearning_bundle")
```

Fit requires a split. The unlabeled pool is train-target missingness
(NaN unless you set `unlabeled_marker`). `suggest_query` never returns
labels. `label_rows` refuses validation/test indices, length mismatches,
and a spent budget.

## Pool convention

Do the split first, then blank **train** only. If you blank holdout
targets and treat them as the pool, eval has nothing honest to score.

Production data may already arrive with missing train labels. Same
contract: holdout should stay labeled if you want `evaluate` to mean
anything.

## Backends and strategies

| Backend | Extra | Strategies |
| --- | --- | --- |
| `sklearn` | none (this is the `margin` default) | `least_confidence`, `margin`, `entropy`, `committee`, `expected_model_change_lite` |
| `industry` | none for native CoreSet/QBC; `buildml[activelearning-industry]` for the scikit-activeml host path | `core_set`, `qbc_kl`, `qbc_variation_ratios` |
| `torch` | `buildml[torch]` | `bald`, `mc_dropout` |

Industry CoreSet and QBC scoring runs in-tree on numpy/sklearn. That
backend is usable without the extra. The extra is an optional
scikit-activeml host path. If that import is broken, query scoring
falls back to the native scorer and says so. Seeing the package name
on disk is not a promise that skactiveml imports cleanly.

To see what this machine actually has:

```python
session.active_learning.capability_matrix()
```

### Sklearn scores (higher = query first)

| Strategy | Score |
| --- | --- |
| `least_confidence` | `1 - max p(y\|x)` |
| `margin` | `-(p_(1) - p_(2))` |
| `entropy` | `-∑ p log p` |
| `committee` | Bagged vote entropy |
| `expected_model_change_lite` | `‖x‖ (1 - p_max)` gradient-magnitude proxy |

### Industry

| Strategy | Notes |
| --- | --- |
| `core_set` | k-center / CoreSet diversity on the pool |
| `qbc_kl` | Query-by-committee with KL divergence |
| `qbc_variation_ratios` | QBC variation-ratio disagreement |

### Torch

| Strategy | Notes |
| --- | --- |
| `bald` | Bayesian Active Learning by Disagreement via MC dropout |
| `mc_dropout` | Predictive entropy from MC-dropout samples |

Torch uses a tabular MLP. `epochs` defaults to 60, `mc_samples` to 20,
`device` to `"cpu"`.

## Budget and eval

When the budget is exhausted, `suggest_query` returns empty indices and
a warning. Raise `label_budget` on a new `fit` if you meant a larger
cap.

`evaluate` defaults to `partition="validation"`. Metrics on labeled
rows: accuracy, macro/weighted F1, macro precision and recall. The pool
is never scored as if those rows had truth.

Do not quote train accuracy after each query as holdout performance.

## Bundles

`buildml.activelearning_bundle.v1` stores the `ActiveLearningPlan`:
estimator, encoder, labeled and pool indices, query history, budget,
backend. A Session checkpoint does not embed the learner. Loaders that
deserialize pickle default to `trusted=False`. Pass `trusted=True` only
for a file you made.

[Artifacts](artifacts-checkpoints-bundles.md)

## When it refuses

| What you see | What happened |
| --- | --- |
| `ValidationError: No split exists` (or `assert_can_fit`) | `fit` before a split |
| Query or label on holdout indices | Pool must be train |
| `MissingExtraError` for torch | `bald` / `mc_dropout` without `buildml[torch]` |
| Empty `suggest_query` indices | Budget spent or pool empty |
| `label_rows` length mismatch | `indices` and `labels` are not 1:1 |

`session.semisupervised.fit` is the wrong tool for a human query loop.
[Semi-supervised deep](semisupervised-deep.md) ·
[Active learning quickstart](quickstart-active-learning.md)
