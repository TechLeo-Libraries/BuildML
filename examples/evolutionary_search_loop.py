"""Evolutionary GA HPO: evolve hyperparams → evaluate → save pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from buildml import Session
from buildml.preprocess import PreprocessRecipe


def main() -> None:
    rng = np.random.default_rng(0)
    n = 160
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    y = (x1 + 0.4 * x2 - 0.2 * x3 + rng.normal(scale=0.35, size=n) > 0).astype(int)
    frame = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})

    session = (
        Session.ingest(frame)
        .set_roles({"x1": "feature", "x2": "feature", "x3": "feature", "y": "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=0, stratify=True)
    )

    recipe = PreprocessRecipe(scale="standard", select="variance", select_k=2)
    result = session.evolutionary_search(
        DecisionTreeClassifier(random_state=0),
        param_space={
            "max_depth": {"type": "int", "low": 2, "high": 8},
            "min_samples_leaf": [1, 2, 4],
            "criterion": ["gini", "entropy"],
        },
        recipe_space={"select_k": {"type": "int", "low": 1, "high": 3}},
        preprocess=recipe,
        population_size=8,
        n_generations=4,
        elite_size=2,
        max_evaluations=24,
        cv=3,
        random_state=0,
        refit=True,
    )
    result.show()
    print("best_params:", result.best_params)
    print("best_recipe_knobs:", result.best_recipe_knobs)
    print("generation_best:", result.study["generation_best"] if result.study else None)

    print("validation:", session.evaluate(partition="validation").metrics)
    print("test:", session.evaluate(partition="test").metrics)

    out = Path(".buildml-artifacts") / "evolutionary_demo_pipeline"
    session.save_pipeline(out, evaluate_partition="test")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
