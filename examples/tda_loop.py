"""Session TDA loop: fit → transform → evaluate → bundle.

Requires: pip install "buildml[tda]"
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session
from buildml.tda.extras import tda_available


def main() -> None:
    if not tda_available():
        raise SystemExit(
            "buildml[tda] required (ripser + persim). "
            'Install with: pip install "buildml[tda]"'
        )

    rng = np.random.default_rng(0)
    a = rng.normal(size=(140, 4))
    b = rng.normal(size=(140, 4)) * 1.6 + np.array([2.5, 0.0, 0.0, 0.0])
    x = np.vstack([a, b])
    y = np.array([0] * 140 + [1] * 140)
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(4)])
    frame["y"] = y

    session = (
        Session.ingest(frame)
        .set_roles({**{f"f{i}": "feature" for i in range(4)}, "y": "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=0, stratify=True)
        .scale(method="standard")
    )

    fit = session.fit_tda(
        vectorization="persistence_image",
        knn=12,
        n_bins=12,
        head="logistic_regression",
        random_state=0,
    )
    print("fit", fit.to_dict())

    tr = session.transform_tda(partition="test")
    print("transform shape", tr.features.shape)

    ev = session.evaluate_tda(partition="validation")
    print("eval", ev.metrics)

    out = Path("artifacts/tda_demo_bundle")
    session.save_tda_bundle(out)
    other = (
        Session.ingest(frame)
        .set_roles({**{f"f{i}": "feature" for i in range(4)}, "y": "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=0, stratify=True)
        .scale(method="standard")
    )
    other.load_tda_bundle(out)
    print("reloaded eval", other.evaluate_tda(partition="test").metrics)


if __name__ == "__main__":
    main()
