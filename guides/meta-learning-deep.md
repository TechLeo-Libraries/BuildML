# Meta-learning deep guide

> **Install:**
> `pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"`
> See [installation](../docs/installation.rst).

Practical Session-facing meta-learning for tabular few-shot / episodic
protocols. This is **not** a foundation-model meta-learning platform, **not**
MAML-at-scale, and **not** EconML-style causal meta (see [Causal deep](causal-deep.md)).

**Related:** [Quickstart](quickstart-meta-learning.md) ·
[Multi-task](multi-task-deep.md) ·
[Artifacts](artifacts-checkpoints-bundles.md).

---

## What this is / is not

| Is | Is not |
| --- | --- |
| Episodic few-shot on a task/group column | Single-task `Session.fit` |
| sklearn / torch / industry backend routing | Vision Mini-ImageNet benchmarks |
| Held-out task ids + novel-task eval disclosure | Silent overlap between meta-train and holdout tasks |
| `buildml.metalearning_bundle.v1` persistence | Session checkpoint embedding the meta-learner |

---

## Backends and install

| Backend | Extra | Methods |
| --- | --- | --- |
| `sklearn` (default) | none | `prototypical`, `warm_start` |
| `torch` | `buildml[torch]` | `prototypical_torch` (MLP encoder + prototype loss) |
| `industry` | `buildml[metalearning-industry,torch]` | `maml`, `reptile` (learn2learn first-order tabular) |

```python
from buildml.metalearning import metalearning_capability_matrix
metalearning_capability_matrix()
```

When extras are installed, industry/torch become honest defaults for their
method families. Sklearn remains the fallback when extras are absent.

```bash
pip install "buildml[torch]"
pip install "buildml[metalearning-industry,torch]"
pip install "buildml[production]"  # includes metalearning-industry + torch
```

---

## Mental model

1. A **task** is the set of rows sharing a task/group id.
2. **Meta-train** runs on the train partition only (optionally holding out train
   task ids → `held_out_task_ids`).
3. An **episode** samples balanced support (`k_shot` per class) + query rows.
4. **Adapt** freezes the meta-train plan and fits only on a support set.
5. **Evaluate** prefers novel task ids on holdout partitions; overlapping ids
   are disclosed.

---

## Algorithms

### `prototypical` (sklearn)

Nearest-centroid few-shot on tabular features: no learned neural embedding.

### `warm_start` (sklearn)

Pooled `logistic_regression` / `sgd_classifier` meta-init → clone + refit on
support. Honest warm initialization: **not** second-order MAML.

### `prototypical_torch` (torch)

Small MLP encoder trained with episodic prototype cross-entropy. Adapt/eval use
embedding-space nearest prototypes. Small-scale tabular ProtoNet: not vision
ProtoNet claims.

### `maml` / `reptile` (industry)

First-order tabular MAML/Reptile with learn2learn when installed. Inner-loop
SGD on support, meta-update across episodic tasks. Honest small-scale task
adaptation: not second-order MAML-at-scale.

---

## Session API

| Method | Role |
| --- | --- |
| `session.metalearning.fit` | Meta-train on train tasks (`backend=`, `method=`) |
| `session.metalearning.adapt` | Fast adapt to one task support set |
| `session.metalearning.evaluate` | Episodic holdout metrics |
| `session.metalearning.save_bundle` / `session.metalearning.load_bundle` | `buildml.metalearning_bundle.v1` |

Properties: `session.metalearning.plan`, `session.metalearning.fit_result`,
`session.metalearning.adapt_result`, `session.metalearning.eval_result`.

```python
session.metalearning.fit(
    backend="torch",
    method="prototypical_torch",
    k_shot=5,
    n_episodes=20,
    meta_epochs=40,
)
session.metalearning.evaluate(partition="validation", prefer_novel_tasks=True)
```

---

## Episodic metrics

| Metric | Where |
| --- | --- |
| `meta_train_accuracy` | Fit result: episodic query accuracy during meta-train |
| `mean_accuracy`, `mean_f1_macro` | Eval result: holdout episodic aggregate |
| `held_out_task_ids` | Plan: internal train task holdout from fit |
| `novel_task_ids` / `overlapping_task_ids` | Eval disclosure |

Benchmark: `python benchmarks/metalearning/few_shot_adaptation.py`

---

## Leakage discipline

- Meta-train requires a split and uses **train only**.
- Validation/test are never used for meta-training.
- `session.metalearning.evaluate` discloses overlapping task ids.
- Null features are refused (impute/scale first).

---

## Bundles vs checkpoints

`buildml.metalearning_bundle.v1` stores `MetaLearningPlan` (backend, protocol,
feature/task contract, label encoder, optional warm-start init or torch/industry
meta-learner). Session checkpoints do **not** embed the meta-learner.

---

## AI allowlist

Teaching-critical tools: `session.metalearning.fit`, `session.metalearning.adapt`,
`session.metalearning.evaluate`, plus save/load bundle.

---

## Known limits

- Classification-focused surface (shared global label space).
- Tabular-only; no vision/audio episodic suites.
- Industry MAML is first-order / small-scale: not full second-order MAML.
- Random row splits may place the same task id in train and holdout: disclosed.
- Causal meta / EconML-style estimation lives in `buildml.causal`, not here.

Related next: symbolic / neuro-symbolic learning.
