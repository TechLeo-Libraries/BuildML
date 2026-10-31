"""Public sampling budgets must not silently change requested row counts."""

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError


def _session(method):
    frame = pd.DataFrame({"x": np.arange(40, dtype=float), "y": [0, 1] * 20})
    session = Session.ingest(frame).set_roles({"x": "feature", "y": "target"}).split(
        test_size=0.2, validation_size=0.2, random_state=11
    )
    session.synthetic.fit(method=method, random_state=3, **(
        {"smooth_sigma": 0} if method == "bootstrap" else {}
    ))
    return session


@pytest.mark.parametrize("method", ["bootstrap", "gaussian_copula"])
@pytest.mark.parametrize("merge_mode", ["none", "extend_train"])
@pytest.mark.parametrize("budget", [True, np.bool_(True), 2.7, "3", float("nan"), float("inf"), 0, -1])
def test_invalid_budget_refused_without_mutating_session(method, merge_mode, budget):
    session = _session(method)
    frame = session.dataset.frame.copy(deep=True)
    split = session.split_plan
    plan = session.synthesizer_plan
    with pytest.raises(ValidationError, match="n must"):
        session.synthetic.sample(n=budget, merge_mode=merge_mode, random_state=17)
    pd.testing.assert_frame_equal(session.dataset.frame, frame)
    assert session.split_plan is split
    assert session.synthesizer_plan is plan


@pytest.mark.parametrize("budget", [None, 1, np.int64(7)])
@pytest.mark.parametrize("merge_mode", ["none", "extend_train"])
def test_bootstrap_rows_match_independent_draws_and_preserve_holdouts(budget, merge_mode):
    session = _session("bootstrap")
    frame = session.dataset.frame.copy(deep=True)
    split = session.split_plan
    train = frame.iloc[list(split.train_indices)].reset_index(drop=True)
    count = len(train) if budget is None else int(budget)
    draws = np.random.default_rng(17).integers(0, len(train), size=count)
    expected = train.iloc[draws].reset_index(drop=True)
    result = session.synthetic.sample(n=budget, merge_mode=merge_mode, random_state=17)
    pd.testing.assert_frame_equal(result.frame, expected)
    assert result.n_rows == count
    if merge_mode == "extend_train":
        merged = session.dataset.frame
        pd.testing.assert_frame_equal(merged.iloc[len(train):len(train) + count].drop(
            columns="_synthetic"
        ).reset_index(drop=True), expected)
        for partition in ("validation", "test"):
            before = frame.iloc[list(getattr(split, f"{partition}_indices"))].reset_index(drop=True)
            after = merged.iloc[list(getattr(session.split_plan, f"{partition}_indices"))].drop(
                columns="_synthetic"
            ).reset_index(drop=True)
            pd.testing.assert_frame_equal(after, before)
    else:
        pd.testing.assert_frame_equal(session.dataset.frame, frame)
