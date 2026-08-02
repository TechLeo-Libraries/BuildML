# Self-supervised deep guide (Phase R1 — Torch industry depth)

> **Install:**
> `pip install "buildml[torch]"` for tabular contrastive/generative SSL defaults.
> `pip install "buildml[ssl]"` adds sentence-transformers for text SSL.
> `pip install "buildml[vision]"` for vision SSL backbones.

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

## Method catalog (Session `method=`)

| Method | Modality | Backend | Notes |
| --- | --- | --- | --- |
| `simclr_tabular` | tabular | Torch | **Default** when torch installed |
| `byol_tabular` | tabular | Torch | Bootstrap-your-own-latent |
| `vicreg_tabular` | tabular | Torch | Variance-invariance-covariance |
| `mae_tabular` | tabular | Torch | Masked autoencoder |
| `vae_tabular` | tabular | Torch | Variational AE |
| `hf_text_ssl` | text | sentence-transformers | Pass `text_column=` |
| `vision_ssl` | vision | torchvision + projector | Pass `image_column=` |
| `masked_tabular` | tabular | sklearn | **Deprecated** — use Torch methods |

## Contract

| Concern | Rule |
| --- | --- |
| Pretext fit | Train features only |
| Labels during pretext | Ignored |
| Head fit | Labeled train rows only |
| Holdout | Frozen encoder + head |
| Bundle | `buildml.ssl_bundle.v2` (v1 legacy loadable) |

## Migration from legacy sklearn SSL

`method="masked_tabular"` still works but emits `DeprecationWarning`.
Replace with:

```python
session.fit_ssl_pretext(method="simclr_tabular", latent_dim=16, epochs=40)
```

Bundles saved after Torch fit use `buildml.ssl_bundle.v2`. Old
`buildml.selfsupervised_bundle.v1` bundles load unchanged.

## Torch backbone transfer (related)

Vision/audio/speech freeze/finetune for downstream supervised heads:

- `Session.load_pretrained_backbone`
- `Session.attach_backbone_head`

`vision_ssl` trains a projector on image columns inside the SSL Session path;
backbone transfer remains the path for published-weight linear probes.

## Benchmarks

```bash
python benchmarks/ssl/linear_probe_tabular.py --epochs 25
```

Compares linear-probe accuracy across Torch methods vs legacy sklearn.

## Related

- [Quickstart](quickstart-selfsupervised.md)
- [Pretrained backbones](pretrained-backbones.md)
- [Artifacts](artifacts-checkpoints-bundles.md)
