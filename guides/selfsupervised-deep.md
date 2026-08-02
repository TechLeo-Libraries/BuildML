# Self-supervised deep guide

> **Install (GitHub 2.x):**
> `pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"`

Phase 2 second item: honest Session-shaped SSL hooks. Complete smaller surface
shipped: **masked tabular autoencoder lite** + representation export + supervised
head. Not BERT/SimCLR product training.

## Story

```text
fit_ssl_pretext (train features; labels ignored)
        ↓
transform_ssl (optional attach of ssl_emb_* columns)
        ↓
finetune_ssl_head (labeled train only; NaN targets skipped)
        ↓
evaluate_ssl (labeled holdout only)
```

You can also `transform_ssl(attach=True)` and continue with classical
`Session.fit` / `fit_semisupervised` on the embedding columns.

## Contract

| Concern | Rule |
| --- | --- |
| Pretext fit | Train features only |
| Labels during pretext | Ignored |
| Head fit | Labeled train rows only |
| Holdout | Frozen encoder + head; unlabeled holdout excluded from metrics |
| Bundle | `buildml.selfsupervised_bundle.v1` |

## Torch backbone transfer (related, not duplicated)

Vision/audio/speech freeze/finetune remains:

- `Session.load_pretrained_backbone`
- `Session.attach_backbone_head`

under `buildml[torch]` / `buildml[speech]`. That path loads published (or mock)
weights — it does not train tabular masked AEs.

## Honesty / non-goals

- No contrastive foundation-model zoo
- No training BERT/Whisper from scratch
- Reconstruction MAE is a pretext diagnostic, not predictive utility
- Active learning and online / continual are done; next Phase 2 item: **multi-task learning**

## Related

- [Quickstart](quickstart-selfsupervised.md)
- [Semi-supervised](semisupervised-deep.md)
- [Pretrained backbones](pretrained-backbones.md)
- [Artifacts](artifacts-checkpoints-bundles.md)
