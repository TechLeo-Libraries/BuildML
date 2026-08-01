import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.core.errors import MissingExtraError


def test_calibration_threshold_importance_learning_curve() -> None:
    frame = pd.DataFrame(
        {
            "x1": list(range(40)),
            "x2": [i * 0.5 for i in range(40)],
            "y": [0] * 20 + [1] * 20,
        }
    )
    est = LogisticRegression(max_iter=500)
    session = (
        Session.ingest(frame)
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
        .fit(est, task="classification")
    )
    cal = session.calibration()
    assert cal.kind == "calibration"
    assert "ece" in cal.payload
    assert cal.interpretation
    thr = session.tune_threshold()
    assert thr.payload["best_f1_threshold"]["threshold"] > 0
    assert "roc_auc" in thr.payload
    assert thr.interpretation
    imp = session.feature_importance(n_repeats=3)
    assert imp.payload["rows"]
    assert imp.interpretation
    lc = session.learning_curve(LogisticRegression(max_iter=500), task="classification", cv=3)
    assert lc.payload["train_sizes"]
    assert "final_gap" in lc.payload


def test_resample_train_only_or_missing_extra() -> None:
    frame = pd.DataFrame(
        {
            "x": list(range(60)),
            "y": [0] * 50 + [1] * 10,
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
    )
    try:
        import imblearn  # noqa: F401
    except ImportError:
        with pytest.raises(MissingExtraError):
            session.resample(sampler="random_oversample")
        return

    test_before = session.partition("test").reset_index(drop=True)
    before = len(session.split_plan.train_indices)  # type: ignore[union-attr]
    session.resample(sampler="random_oversample")
    after = len(session.split_plan.train_indices)  # type: ignore[union-attr]
    assert after >= before
    assert session.resample_plan is not None
    assert session.resample_plan.class_counts_after
    pd.testing.assert_frame_equal(test_before, session.partition("test").reset_index(drop=True))
