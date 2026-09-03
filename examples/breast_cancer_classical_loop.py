"""Pasteable Wisconsin breast cancer loop. Public sklearn table, no proofs._lib.

The proof at proofs/breast-cancer-classical/ needs a checkout (it imports
proofs._lib). This file is the public-data loop you can copy.
"""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

from buildml import Session


def _wisconsin_table() -> pd.DataFrame:
    bunch = load_breast_cancer(as_frame=True)
    frame = bunch.frame.copy()
    frame = frame.rename(columns={"target": "malignant"})
    rename = {c: c.replace(" ", "_") for c in frame.columns if c != "malignant"}
    return frame.rename(columns=rename)


def main() -> None:
    frame = _wisconsin_table()
    features = [c for c in frame.columns if c != "malignant"]
    print(f"wisconsin breast cancer n={len(frame)} features={len(features)}")

    session = Session.ingest(frame)
    session.set_roles({**{c: "feature" for c in features}, "malignant": "target"})
    session.split(
        test_size=0.2,
        validation_size=0.2,
        stratify=True,
        random_state=42,
    )
    session.impute(strategy="median")
    session.scale(method="standard")
    session.fit(LogisticRegression(max_iter=2000, random_state=42), task="classification")

    session.calibration()
    session.tune_threshold(fp_cost=1.0, fn_cost=5.0)
    val = session.evaluate(partition="validation")
    test = session.evaluate(partition="test")
    print("validation:", val.metrics)
    print("test:", test.metrics)


if __name__ == "__main__":
    main()
