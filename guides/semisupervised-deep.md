# Semi-supervised deep guide

> **Install (GitHub 2.x):**
> `pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"`

Phase 2 first item after Phase 1
(unsupervised → ensembles → AutoML → forecasting → anomaly). This path is
**scarce-label classification** with unlabeled train features — not anomaly
novelty, not self-supervised pretext, not active learning.

## Contract

| Concern | Rule |
| --- | --- |
| Fit partition | Train only (`assert_can_fit("train")`) |
| Unlabeled convention | Target NaN/NA/None by default (`unlabeled_marker` optional) |
| Internal encoding | sklearn `-1` unlabeled + LabelEncoder for observed classes |
| Holdout labels | Evaluation-only; never invent selection labels from unlabeled holdout |
| Bundle | `buildml.semisupervised_bundle.v1` ≠ Session checkpoint |

## Methods

- `label_propagation` / `label_spreading` — graph-based (sklearn)
- `self_training` — pseudo-labeling around `logistic_regression` or
  `hist_gradient_boosting`

## Session API

`fit_semisupervised` → `predict_semisupervised` → `evaluate_semisupervised` →
`save_semisupervised_bundle` / `load_semisupervised_bundle`.

## Leakage discipline

1. Split first (prefer fully labeled data if you need `stratify=True`).
2. Blank **train** targets only when simulating scarce labels.
3. Never use validation/test unlabeled rows to invent labels for selection.
4. Read `n_labeled_*` / `n_unlabeled_*` beside every metric.

## Failure modes

- `<2` labeled train rows or a single class among labels
- Null feature columns (impute/scale first)
- Confusing this API with `fit_anomaly(mode="novelty")`
- Expecting a Session checkpoint to embed `SemiSupervisedPlan`

## Related

- [Quickstart](quickstart-semisupervised.md)
- [Self-supervised](quickstart-selfsupervised.md) (pretext → head)
- [Anomaly](anomaly-deep.md) (novelty is a different metaphor)
- Phase 2 online / continual is done (`buildml.online`); next: **multi-task learning**
