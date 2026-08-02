"""Native ensemble loop: voting → stacking → blending → evaluate → bundle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from buildml import Session


def main() -> None:
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (x1 - 0.5 * x2 + rng.normal(scale=0.4, size=n) > 0).astype(int)
    frame = pd.DataFrame({"x1": x1, "x2": x2, "y": y})

    bases = {
        "lr": LogisticRegression(max_iter=500),
        "rf": RandomForestClassifier(n_estimators=50, random_state=0),
    }

    session = (
        Session.ingest(frame)
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=0, stratify=True)
        .scale(method="standard")
    )

    session.fit_voting(bases, voting="soft").show()
    print(session.evaluate_ensemble(partition="validation").metrics)

    session.fit_stacking(bases, cv=3).show()
    print(session.evaluate_ensemble(partition="test").metrics)

    session.fit_blending(bases, holdout_fraction=0.2, random_state=0).show()
    print(session.evaluate_ensemble(partition="test").metrics)

    out = Path(".buildml-artifacts") / "ensemble_demo_bundle"
    session.save_ensemble_bundle(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
