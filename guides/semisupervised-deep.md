# Semi-supervised (deep)

```bash
pip install buildml
# GBDT pseudo-label: pip install "buildml[semisupervised-industry]"
# FixMatch: pip install "buildml[torch]"
```

Scarce labels on train, unlabeled train features still used. Unlabeled
means target NaN (or your `unlabeled_marker`). Internally that is sklearn
`-1`. Default `method="label_propagation"` stays sklearn even if XGBoost
is installed. Holdout labels are for evaluation only.

This is not novelty detection (`session.anomaly`), not pretext
(`session.ssl`), and not an oracle (`session.active_learning`).

Short on-ramp: [semi-supervised quickstart](quickstart-semisupervised.md).

## Backends

| Backend | Extra | Methods |
| --- | --- | --- |
| `sklearn` | core | `label_propagation`, `label_spreading`, `self_training` |
| `industry` | `semisupervised-industry` | `pseudo_label_xgb`, `pseudo_label_lgbm` |
| `torch` | `torch` | `fixmatch_tabular`, `mixmatch_tabular` |
| `hf` | `ssl` | `text_pseudo_label` |

```python
session.semisupervised.fit(
    backend="industry",
    method="pseudo_label_xgb",
    threshold=0.75,
    max_self_train_iter=10,
)
```

## The loop

Split first. If you need `stratify=True`, start from fully labeled data,
then blank **train** targets to simulate scarce labels. Fit on train.
Predict / evaluate holdout. Bundle: `buildml.semisupervised_bundle.v1`.
A Session checkpoint does not embed the plan.

Read `n_labeled_*` / `n_unlabeled_*` beside every metric.

## With a pretext

Self-supervised pretext can run on all train rows (labels optional).
Semi-supervised fit then uses partial labels on those representations:

```python
session.ssl.fit_pretext(method="simclr_tabular", latent_dim=16, epochs=30)
session.ssl.transform(attach=True, partition="all")
session.semisupervised.fit(
    method="self_training",
    columns=list(session.ssl.plan.representation_columns),
    prefer_reduce_components=False,
)
```

`session.ssl.finetune_head` is labeled train only. `session.semisupervised.fit`
uses unlabeled train rows via propagation or pseudo-labels.

## What usually goes wrong

- Fewer than two labeled train rows, or a single class among labels.
- Null feature columns: impute/scale first.
- Missing extra for a non-sklearn backend: `MissingExtraError`.
- Using validation/test unlabeled rows to invent labels for selection.

[Self-supervised](selfsupervised-deep.md) · [Active learning](active-learning-deep.md)
