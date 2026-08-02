# Torch quickstart

Optional tabular Torch path on the same `Session` as classical ML: roles,
splits, history, and explain. Install the Torch extra; core `import buildml`
never requires it.

```bash
pip install "buildml[torch]"
# alias: pip install "buildml[dl]"
```

Classical `Session.fit` stays the default sklearn path. Torch methods use the
`*_torch` prefix and store results in `session.dl_train_result`.

```python
import pandas as pd
import torch
from torch import nn

from buildml import Session

frame = pd.DataFrame(
    {
        "a": [0.1, 0.4, 0.2, 0.8, 0.3, 0.7, 0.5, 0.9, 0.15, 0.65],
        "b": [1.0, 0.2, 0.9, 0.1, 0.8, 0.3, 0.6, 0.4, 0.75, 0.25],
        "label": [0, 1, 0, 1, 0, 1, 1, 0, 0, 1],
    }
)


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


session = (
    Session.ingest(frame)
    .set_roles({"a": "feature", "b": "feature", "label": "target"})
    .split(
        test_size=0.25,
        validation_size=0.25,
        stratify=True,
        random_state=42,
    )
)

# Optional classical prep first (not auto-applied before loaders).
# session.impute(strategy="median").scale(method="standard")

session.make_torch_loaders(batch_size=4, normalize=True, seed=42)
session.fit_torch(
    TinyMLP(),
    epochs=6,
    learning_rate=5e-3,
    device="cpu",
    early_stopping_patience=3,
    scheduler="none",
)

# Prefer validation while iterating; reserve test for a fixed recipe.
validation = session.evaluate_torch(partition="validation")
test = session.evaluate_torch(partition="test")
print(test.metrics)

curve = session.torch_training_curve()
print(curve.disclosures)

bundle = session.save_torch_bundle("artifacts/torch_bundle")
```

Reload and optionally resume:

```python
restored = (
    Session.ingest(frame)
    .set_roles({"a": "feature", "b": "feature", "label": "target"})
    .split(
        test_size=0.25,
        validation_size=0.25,
        stratify=True,
        random_state=42,
    )
)
restored.load_torch_bundle(bundle, TinyMLP(), map_location="cpu")
restored.make_torch_loaders(batch_size=4, normalize=True, seed=42)
restored.evaluate_torch(partition="test")
# Additional epochs; optimizer/scheduler state restored when compatible.
restored.fit_torch(TinyMLP(), epochs=2, resume=True, device="cpu")
```

Explain catalog coverage:

```python
before = session.explain("fit_torch", moment="before")
print(before.operation, before.prerequisites)
```

## Artifacts

| Artifact | Schema | Contains | Does not contain |
| --- | --- | --- | --- |
| Session checkpoint | existing checkpoint formats | data, roles, splits, history | Torch weights / optimizer |
| Torch trainer bundle | `buildml.torch_bundle.v1` | weights, optimizer (+ scheduler), config, history, feature contract | dataset rows, split indices |

Layout: `<path>/meta.json` + `<path>/trainer.pt`.

## Known limits (honest)

- **CPU-first merge gate.** CI runs Torch on CPU (Python 3.11–3.12). CUDA/MPS
  are supported when available with explicit fallback warnings; GPU CI is not a
  PR blocker.
- **Tabular numeric features first.** Image / sequence / multimodal loaders are
  later. Caller supplies the `nn.Module`; BuildML does not ship a model zoo.
- **Materialization.** Partition rows become tensors via the current Session
  frame (Pandas/NumPy bridge). No Polars/DuckDB zero-copy into DataLoaders.
- **No auto classical preprocess.** Call impute/encode/scale (or other Session
  prep) explicitly before `make_torch_loaders` if you need it.
- **No fold-local Torch CV** in this alpha. Holdout + validation early stopping
  is the tested discipline; do not tune early stop on test.
- **Not RAG / LLM.** Those domains are sequenced after DL.

See [glossary.md](./glossary.md).
