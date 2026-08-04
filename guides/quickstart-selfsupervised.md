# Self-supervised quickstart

> **Install:**
> `pip install "git+https://github.com/TechLeo-Libraries/BuildML.git#egg=buildml[torch]"`

Industry-default SSL uses **Torch** when installed (`simclr_tabular` default).
Legacy sklearn `masked_tabular` remains as deprecated fallback.

**Go deeper:** [Self-supervised deep](selfsupervised-deep.md) ·

**Proof:** [ssl-representation-probe](../proofs/ssl-representation-probe/) (+ Tier C PCA probe). Cross-domain: [atlas-label-studio](../proofs/atlas-label-studio/).
[Pretrained backbones](pretrained-backbones.md) ·
[Artifacts](artifacts-checkpoints-bundles.md).

```bash
pip install "buildml[torch]"
# optional text SSL:
pip install "buildml[ssl]"
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

# Default: simclr_tabular when torch installed
pre = session.ssl.fit_pretext(latent_dim=8, epochs=30, batch_size=32)
print(pre.method, pre.pretext_loss)

head = session.ssl.finetune_head(estimator="logistic_regression")
ev = session.ssl.evaluate(partition="test")
print(ev.metrics)

bundle = session.ssl.save_bundle("artifacts/ssl_bundle")  # buildml.ssl_bundle.v2
```

Other tabular methods: `byol_tabular`, `vicreg_tabular`, `mae_tabular`, `vae_tabular`.

Text SSL (`buildml[ssl]`):

```python
session.ssl.fit_pretext(method="hf_text_ssl", text_column="text", latent_dim=384)
```

Vision SSL (`buildml[vision]`):

```python
session.ssl.fit_pretext(
    method="vision_ssl",
    image_column="path",
    backbone="resnet18",
    weight_mode="mock",
    epochs=5,
)
```

**Deprecated:** `method="masked_tabular"` (sklearn MLP): migrate to Torch methods.
