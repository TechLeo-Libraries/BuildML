"""AutoML loop: family + recipe search → evaluate → bundle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session
from buildml.automl.types import AutoMLBudget


def main() -> None:
    rng = np.random.default_rng(0)
    n = 220
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    cat = rng.choice(["a", "b", "c"], size=n)
    y = (x1 - 0.5 * x2 + rng.normal(scale=0.4, size=n) > 0).astype(int)
    frame = pd.DataFrame({"x1": x1, "x2": x2, "cat": cat, "y": y})

    session = (
        Session.ingest(frame)
        .set_roles({"x1": "feature", "x2": "feature", "cat": "feature", "y": "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=0, stratify=True)
    )

    result = session.run_automl(
        method="randomized",
        selection="cv",
        n_trials=12,
        cv=3,
        include_recipe_search=True,
        include_ensembles=True,
        families=("logistic", "random_forest", "gradient_boosting"),
        budget=AutoMLBudget(max_trials=12, max_recipe_strategies=6),
        random_state=0,
    )
    result.show()
    print(session.evaluate_automl(partition="validation").metrics)
    print(session.evaluate_automl(partition="test").metrics)

    out = Path(".buildml-artifacts") / "automl_demo_bundle"
    session.save_automl_bundle(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
