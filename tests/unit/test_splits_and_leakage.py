import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import LeakageError, ValidationError


def _binary_frame(n: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": list(range(n)),
            "y": [i % 2 for i in range(n)],
        }
    )


def test_stratified_split_roughly_preserves_balance() -> None:
    session = Session.ingest(_binary_frame(40)).set_roles({"x": "feature", "y": "target"})
    session.split(test_size=0.25, stratify=True, random_state=0)
    train = session.partition("train")
    test = session.partition("test")
    assert set(train.index).isdisjoint(test.index)
    train_rate = train["y"].mean()
    test_rate = test["y"].mean()
    assert abs(train_rate - 0.5) <= 0.15
    assert abs(test_rate - 0.5) <= 0.15


def test_stratify_requires_target_role() -> None:
    session = Session.ingest(_binary_frame(20))
    with pytest.raises(ValidationError, match="exactly one target"):
        session.split(stratify=True)


def test_partitions_are_disjoint_with_validation() -> None:
    session = Session.ingest(_binary_frame(50)).set_roles({"x": "feature", "y": "target"})
    session.split(test_size=0.2, validation_size=0.25, random_state=1)
    train_idx = set(session.split_plan.train_indices)  # type: ignore[union-attr]
    valid_idx = set(session.split_plan.validation_indices)  # type: ignore[union-attr]
    test_idx = set(session.split_plan.test_indices)  # type: ignore[union-attr]
    assert train_idx.isdisjoint(valid_idx)
    assert train_idx.isdisjoint(test_idx)
    assert valid_idx.isdisjoint(test_idx)
    assert train_idx and valid_idx and test_idx


def test_assert_can_fit_blocks_full_data_and_non_train() -> None:
    session = Session.ingest(_binary_frame(20)).set_roles({"x": "feature", "y": "target"})
    with pytest.raises(LeakageError, match="full data"):
        session.assert_can_fit()
    session.split(test_size=0.25, random_state=2)
    session.assert_can_fit("train")
    with pytest.raises(LeakageError, match="Fit only on 'train'"):
        session.assert_can_fit("test")


def test_inject_split() -> None:
    session = Session.ingest(_binary_frame(10))
    session.inject_split(train_indices=[0, 1, 2, 3, 4, 5], test_indices=[6, 7, 8, 9])
    assert len(session.partition("train")) == 6
    assert len(session.partition("test")) == 4
