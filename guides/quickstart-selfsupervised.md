# Self-supervised quickstart

> **Install first (GitHub):** PyPI `buildml` is still legacy 1.x and does **not**
> install Session 2.x. Install 2.x from GitHub (or an editable checkout).
> The masked-tabular path uses core sklearn — no optional extra is required.
> Vision/audio/speech freeze/finetune still uses `buildml[torch]` /
> `buildml[speech]` via `load_pretrained_backbone`.
> See [installation](../docs/installation.rst).

Honest Session-shaped SSL: **pretext on train features → export representations →
supervised head on labeled train**. Not BERT-from-scratch.

**Go deeper:** [Self-supervised deep](selfsupervised-deep.md) ·
[Semi-supervised](quickstart-semisupervised.md) ·
[Pretrained backbones](pretrained-backbones.md) ·
[Artifacts](artifacts-checkpoints-bundles.md).

```bash
pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"
```

```python
import numpy as np
import pandas as pd

from buildml import Session

rng = np.random.default_rng(0)
x0 = rng.normal([-1.0, -1.0], 0.7, size=(100, 2))
x1 = rng.normal([1.5, 1.2], 0.7, size=(100, 2))
frame = pd.DataFrame(np.vstack([x0, x1]), columns=["x", "y"])
frame["label"] = [0] * 100 + [1] * 100

session = (
    Session.ingest(frame)
    .set_roles({"x": "feature", "y": "feature", "label": "target"})
    .split(test_size=0.25, stratify=True, random_state=0)
    .scale(method="standard")
)

pre = session.fit_ssl_pretext(
    method="masked_tabular",
    latent_dim=8,
    mask_ratio=0.2,
    max_iter=120,
)
print(pre.latent_dim, pre.reconstruction_mae)

# Optional: attach embedding columns for classical Session.fit
session.transform_ssl(partition="all", attach=True)

head = session.finetune_ssl_head(estimator="logistic_regression")
print(head.n_labeled_train, head.estimator_name)

ev = session.evaluate_ssl(partition="test")
print(ev.metrics)

bundle = session.save_ssl_bundle("artifacts/ssl_bundle")
```

Vision/audio/speech transfer (separate optional path — not tabular masked AE):

```python
# requires buildml[torch] / torchvision as documented
session.load_pretrained_backbone("resnet18", weight_mode="mock", freeze=True)
session.attach_backbone_head(n_classes=2)
```

**Not this API:** contrastive foundation-model zoos, training BERT from scratch,
or semi-supervised label propagation (`fit_semisupervised`).
