# Self-supervised (deep)

```bash
pip install "buildml[torch]"
# text SSL: pip install "buildml[ssl]"
# vision SSL: pip install "buildml[vision]"
```

Pretext on train features. Labels are ignored at that step. Then a head
on labeled train only. Holdout scores the frozen encoder plus head.
Default when torch is installed is `simclr_tabular`. Evaluate needs
both pretext and head.

This is not semi-supervised pseudo-labelling and not
`session.dl.load_backbone` (that path is published-weight probes).

Quickstart: [self-supervised quickstart](quickstart-selfsupervised.md).

## The order

```text
session.ssl.fit_pretext   (train features; labels ignored)
        ↓
session.ssl.transform     (optional ssl_emb_* columns)
        ↓
session.ssl.finetune_head (labeled train; NaN targets skipped)
        ↓
session.ssl.evaluate      (labeled holdout)
```

## Methods

| Method | Modality | Backend | Notes |
| --- | --- | --- | --- |
| `simclr_tabular` | tabular | Torch | Default when torch is installed |
| `byol_tabular` | tabular | Torch | |
| `vicreg_tabular` | tabular | Torch | |
| `mae_tabular` | tabular | Torch | Masked autoencoder |
| `vae_tabular` | tabular | Torch | |
| `hf_text_ssl` | text | sentence-transformers | Pass `text_column=` |
| `vision_ssl` | vision | torchvision + projector | Pass `image_column=` |
| `masked_tabular` | tabular | sklearn | Deprecated; use Torch methods |

```python
import pandas as pd
from sklearn.datasets import make_classification
from buildml import Session

X, y = make_classification(n_samples=160, n_features=2, n_informative=2,
                           n_redundant=0, random_state=0)
frame = pd.DataFrame(X, columns=["x1", "x2"])
frame["y"] = y
session = (Session.ingest(frame)
    .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
    .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0))

session.scale(method="standard")
session.ssl.fit_pretext(method="simclr_tabular", latent_dim=16, epochs=40)
```

`method="masked_tabular"` still runs and emits `DeprecationWarning`.
Bundles after Torch fit use `buildml.ssl_bundle.v2`. Old
`buildml.selfsupervised_bundle.v1` bundles still load.

## Backbone transfer is a different path

`session.dl.load_backbone` / `session.dl.attach_head` freeze or finetune
published weights. `vision_ssl` trains a projector on image columns
inside this SSL Session path.

[Pretrained backbones](pretrained-backbones.md) ·
[Semi-supervised](semisupervised-deep.md)
