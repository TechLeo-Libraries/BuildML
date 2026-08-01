"""Leakage-audited Optuna study warm-start across nested CV outer folds."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.model import selection as selection_mod


def _cls_frame(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (x1 + 0.3 * x2 + rng.normal(scale=0.4, size=n) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def _has_optuna() -> bool:
    try:
        import optuna  # noqa: F401

        return True
    except ImportError:
        return False


def test_warm_start_studies_requires_optuna_inner() -> None:
    session = (
        Session.ingest(_cls_frame(80))
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
    )
    with pytest.raises(ValidationError, match="warm_start_studies"):
        session.nested_cv_score(
            DecisionTreeClassifier(random_state=0),
            param_grid={"max_depth": [2, 3]},
            outer_cv=2,
            inner_cv=2,
            warm_start_studies=True,
        )


@pytest.mark.skipif(not _has_optuna(), reason="optuna not installed")
def test_warm_start_studies_shares_study_without_test_peeking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = (
        Session.ingest(_cls_frame(140))
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.15, stratify=True, random_state=0)
    )
    test_idx = set(session.split_plan.test_indices)  # type: ignore[union-attr]
    valid_idx = set(session.split_plan.validation_indices)  # type: ignore[union-attr]
    seen_train_partitions: list[set[int]] = []
    studies: list[Any] = []

    real_optuna = selection_mod.optuna_search

    def _spy_optuna(*args: Any, **kwargs: Any) -> Any:
        plan = args[1]
        train = set(plan.train_indices)
        eval_idx = set(plan.test_indices)
        assert not (train & test_idx)
        assert not (train & valid_idx)
        assert not (eval_idx & test_idx)
        assert not (eval_idx & valid_idx)
        assert not (train & eval_idx)
        seen_train_partitions.append(train)
        result = real_optuna(*args, **kwargs)
        studies.append(result.study)
        return result

    monkeypatch.setattr(selection_mod, "optuna_search", _spy_optuna)

    result = session.nested_cv_score(
        DecisionTreeClassifier(random_state=0),
        inner_search="optuna",
        param_space={"max_depth": {"type": "int", "low": 2, "high": 4}},
        n_trials=2,
        outer_cv=2,
        inner_cv=2,
        random_state=0,
        warm_start_studies=True,
    )
    assert result.warm_start_studies is True
    assert result.search_method == "optuna"
    assert "test" in result.held_out_partitions
    assert any("warm_start_studies=True" in note for note in result.limitations)
    assert len(seen_train_partitions) == 2
    assert len(studies) == 2
    assert studies[0] is studies[1]
    # Second outer fold continues the same study (prior trials retained).
    assert len(studies[1].trials) >= 4
    assert set(session.split_plan.test_indices) == test_idx  # type: ignore[union-attr]
    assert set(session.split_plan.validation_indices) == valid_idx  # type: ignore[union-attr]
    assert session.fit_result is None


@pytest.mark.skipif(not _has_optuna(), reason="optuna not installed")
def test_warm_start_default_off_uses_independent_studies(monkeypatch: pytest.MonkeyPatch) -> None:
    session = (
        Session.ingest(_cls_frame(100))
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.2, stratify=True, random_state=1)
    )
    studies: list[Any] = []
    real_optuna = selection_mod.optuna_search

    def _spy(*args: Any, **kwargs: Any) -> Any:
        assert kwargs.get("study") is None
        result = real_optuna(*args, **kwargs)
        studies.append(result.study)
        return result

    monkeypatch.setattr(selection_mod, "optuna_search", _spy)
    result = session.nested_cv_score(
        DecisionTreeClassifier(random_state=0),
        inner_search="optuna",
        param_space={"max_depth": {"type": "int", "low": 2, "high": 3}},
        n_trials=2,
        outer_cv=2,
        inner_cv=2,
        random_state=1,
    )
    assert result.warm_start_studies is False
    assert len(studies) == 2
    assert studies[0] is not studies[1]
