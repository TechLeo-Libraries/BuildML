import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import LeakageError


def test_drop_columns_preserves_split_membership() -> None:
    frame = pd.DataFrame(
        {
            "keep": [1, 2, 3, 4, 5, 6],
            "drop_me": [9, 9, 9, 9, 9, 9],
            "y": [0, 1, 0, 1, 0, 1],
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"keep": "feature", "y": "target"})
        .split(test_size=0.33, random_state=0)
    )
    train_idx = session.split_plan.train_indices  # type: ignore[union-attr]
    session.drop_columns(["drop_me"])
    assert "drop_me" not in session.dataset.columns
    assert session.split_plan.train_indices == train_idx  # type: ignore[union-attr]


def test_impute_requires_split_and_uses_train_stats() -> None:
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "y": [0, 1, 0, 1, 0, 1],
        }
    )
    session = Session.ingest(frame).set_roles({"x": "feature", "y": "target"})
    with pytest.raises(LeakageError):
        session.impute(columns=["x"], strategy="mean")

    session.split(test_size=0.33, random_state=0)
    train = session.partition("train")
    expected = float(train["x"].mean())
    session.impute(columns=["x"], strategy="mean")
    assert session.impute_plan is not None
    assert session.impute_plan.statistics_["x"] == pytest.approx(expected)
    assert session.to_pandas()["x"].isna().sum() == 0
