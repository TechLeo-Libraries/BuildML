"""Mirror of guides/leakage-cv-recipes.md — fold-local CV + hard-refuse demo."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.core.errors import LeakageError
from buildml.preprocess import PreprocessRecipe


def main() -> None:
    frame = pd.DataFrame(
        {
            "age": [21, None, 35, 40, 29, 33, 52, 47, 31, 44, 38, 27],
            "income": [40, 55, 60, 80, 50, 70, 90, 65, 48, 88, 61, 72],
            "city": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C", "B", "A"],
            "approved": [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0],
        }
    )

    session = (
        Session.ingest(frame)
        .set_roles(
            {
                "age": "feature",
                "income": "feature",
                "city": "feature",
                "approved": "target",
            }
        )
        .split(test_size=0.25, stratify=True, random_state=42)
    )

    recipe = PreprocessRecipe(impute="median", encode="onehot", scale="standard")
    cv = session.cv_score(
        LogisticRegression(max_iter=500),
        cv=4,
        preprocess=recipe,
    )
    print(
        "honest cv:",
        cv.mean_metrics[cv.scoring_metric],
        "±",
        cv.std_metrics[cv.scoring_metric],
    )

    # Poison the frame, then show the hard refuse.
    session.impute(strategy="median")
    session.scale(method="standard")
    try:
        session.cv_score(
            LogisticRegression(max_iter=500),
            cv=4,
            preprocess=recipe,
        )
    except LeakageError as exc:
        print("refused as expected:", exc)


if __name__ == "__main__":
    main()
