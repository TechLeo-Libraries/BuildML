# Semi-supervised deep guide

> **Install (GitHub 2.x):**
> `pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"`

Industry depth (R6.1): scarce-label classification with unlabeled train features :
not anomaly novelty, not self-supervised pretext, not active learning.

## Contract

| Concern | Rule |
| --- | --- |
| Fit partition | Train only (`assert_can_fit("train")`) |
| Unlabeled convention | Target NaN/NA/None by default (`unlabeled_marker` optional) |
| Internal encoding | sklearn `-1` unlabeled + LabelEncoder for observed classes |
| Holdout labels | Evaluation-only; never invent selection labels from unlabeled holdout |
| Bundle | `buildml.semisupervised_bundle.v1` ≠ Session checkpoint |

## Backends and methods

| Backend | Extra | Methods | Modality |
| --- | --- | --- | --- |
| `sklearn` (fallback) | core | `label_propagation`, `label_spreading`, `self_training` | tabular |
| `industry` | `buildml[semisupervised-industry]` | `pseudo_label_xgb`, `pseudo_label_lgbm` | tabular |
| `torch` | `buildml[torch]` | `fixmatch_tabular`, `mixmatch_tabular` | tabular |
| `hf` | `buildml[ssl]` | `text_pseudo_label` | text |

Inspect honest availability:

```python
from buildml.semisupervised import semisupervised_capability_matrix

print(semisupervised_capability_matrix()["default_backend_when_installed"])
```

When industry extras are installed, prefer `backend="industry"` or `backend="torch"`
for tabular partial-label tasks; sklearn remains the honest fallback.

## Session API

`fit_semisupervised` → `predict_semisupervised` → `evaluate_semisupervised` →
`save_semisupervised_bundle` / `load_semisupervised_bundle`.

Industry example:

```python
session.fit_semisupervised(
    backend="industry",
    method="pseudo_label_xgb",
    threshold=0.75,
    max_self_train_iter=10,
)
```

Torch consistency example:

```python
session.fit_semisupervised(
    backend="torch",
    method="fixmatch_tabular",
    epochs=40,
    threshold=0.75,
)
```

## SSL integration (documented pipeline)

Self-supervised pretext learns representations on **all** train rows (labels optional).
Semi-supervised fit then uses **partial labels** on those representations:

```python
session.fit_ssl_pretext(method="simclr_tabular", latent_dim=16, epochs=30)
session.transform_ssl(attach=True, partition="all")
session.fit_semisupervised(
    method="self_training",
    columns=list(session.ssl_plan.representation_columns),
    prefer_reduce_components=False,
)
```

- `finetune_ssl_head`: labeled train rows only (supervised head).
- `fit_semisupervised`: uses unlabeled train rows via propagation/pseudo-labels.

## Leakage discipline

1. Split first (prefer fully labeled data if you need `stratify=True`).
2. Blank **train** targets only when simulating scarce labels.
3. Never use validation/test unlabeled rows to invent labels for selection.
4. Read `n_labeled_*` / `n_unlabeled_*` beside every metric.

## Benchmark

```bash
python benchmarks/semisupervised/partial_labels.py
```

Writes `benchmarks/semisupervised/results/partial_labels.json`.

## Failure modes

- `<2` labeled train rows or a single class among labels
- Null feature columns (impute/scale first)
- Missing extra for non-sklearn backend (`MissingExtraError`)
- Confusing this API with `fit_anomaly(mode="novelty")`
- Expecting a Session checkpoint to embed `SemiSupervisedPlan`

## Related

- [Quickstart](quickstart-semisupervised.md)
- [Self-supervised](quickstart-selfsupervised.md) (pretext → head / embeddings)
- [Anomaly](anomaly-deep.md) (novelty is a different metaphor)
- Next Phase 2 item: **active learning** (`buildml.activelearning`)
