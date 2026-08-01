import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import LeakageError, MissingExtraError, ValidationError
from buildml.preprocess.imbalance import list_resample_strategies


def _has_imblearn() -> bool:
    try:
        import imblearn  # noqa: F401
    except ImportError:
        return False
    return True


def test_list_resample_strategies_metadata() -> None:
    rows = list_resample_strategies()
    names = {r["name"] for r in rows}
    expected = {
        "smote",
        "random_oversample",
        "random_undersample",
        "adasyn",
        "borderline_smote",
    }
    assert expected <= names
    for row in rows:
        assert row["extra"] == "imbalanced"
        assert row["description"]
        assert row["when_to_use"]


def test_resample_requires_split() -> None:
    frame = pd.DataFrame({"x": list(range(20)), "y": [0] * 15 + [1] * 5})
    session = Session.ingest(frame).set_roles({"x": "feature", "y": "target"})
    with pytest.raises(LeakageError):
        session.resample(sampler="random_oversample")


@pytest.mark.skipif(not _has_imblearn(), reason="imbalanced extra not installed")
def test_resample_train_only_preserves_holdout() -> None:
    frame = pd.DataFrame(
        {
            "x": list(range(80)),
            "y": [0] * 64 + [1] * 16,
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
    )
    test_before = session.partition("test").reset_index(drop=True)
    train_before_n = len(session.split_plan.train_indices)  # type: ignore[union-attr]
    session.resample(sampler="random_oversample", random_state=0)
    plan = session.resample_plan
    assert plan is not None
    assert plan.n_train_after >= plan.n_train_before
    assert plan.class_counts_before
    assert plan.class_counts_after
    assert plan.n_test_unchanged == len(test_before)
    test_after = session.partition("test").reset_index(drop=True)
    pd.testing.assert_frame_equal(test_before, test_after)
    assert len(session.split_plan.train_indices) >= train_before_n  # type: ignore[union-attr]
    assert "Imbalance ratio" in " ".join(plan.notes)


@pytest.mark.skipif(not _has_imblearn(), reason="imbalanced extra not installed")
def test_smote_rejects_non_numeric_features() -> None:
    frame = pd.DataFrame(
        {
            "city": ["a", "b"] * 30,
            "y": [0] * 50 + [1] * 10,
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"city": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
    )
    with pytest.raises(ValidationError, match="numeric"):
        session.resample(sampler="smote")


@pytest.mark.skipif(not _has_imblearn(), reason="imbalanced extra not installed")
def test_unknown_sampler_errors_clearly() -> None:
    frame = pd.DataFrame({"x": list(range(40)), "y": [0] * 30 + [1] * 10})
    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
    )
    with pytest.raises(ValidationError, match="Unknown sampler"):
        session.resample(sampler="not_a_real_sampler")  # type: ignore[arg-type]


def test_resample_missing_extra() -> None:
    if _has_imblearn():
        pytest.skip("imbalanced-learn installed")
    frame = pd.DataFrame({"x": list(range(40)), "y": [0] * 30 + [1] * 10})
    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
    )
    with pytest.raises(MissingExtraError):
        session.resample(sampler="random_oversample")
