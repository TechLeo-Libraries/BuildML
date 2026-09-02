"""Mirror of guides/quickstart-torch.md — TinyMLP on a split table.

Skips with an install hint when Torch is not importable. That is the same
honesty as the proof: this path needs ``buildml[torch]``.
"""

from __future__ import annotations

import pandas as pd

from pathlib import Path

from buildml import Session


def main() -> None:
    artifacts = Path(__file__).resolve().parent / '.artifacts'
    artifacts.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        from torch import nn
    except ImportError:
        print("skip: pip install 'buildml[torch]'")
        return

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

    session.dl.make_loaders(batch_size=4, normalize=True, seed=42)
    session.dl.fit(
        TinyMLP(),
        epochs=6,
        learning_rate=5e-3,
        device="cpu",
        early_stopping_patience=3,
        scheduler="none",
    )
    validation = session.dl.evaluate(partition="validation")
    test = session.dl.evaluate(partition="test")
    print("validation", validation.metrics)
    print("test", test.metrics)
    print(session.dl.training_curve().disclosures)
    session.dl.save_bundle(Path(__file__).resolve().parent / ".artifacts" / "torch_bundle")


if __name__ == "__main__":
    main()
    raise SystemExit(0)
