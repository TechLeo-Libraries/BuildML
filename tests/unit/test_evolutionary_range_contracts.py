"""Malformed mathematical search domains must fail before any model fit."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from buildml import Session
from buildml.core.errors import ValidationError


@pytest.mark.parametrize("spec", [
    {"type": "float", "high": 1},
    {"type": "float", "low": 0},
    {"type": "float", "low": "not-a-number", "high": 1},
    {"type": "float", "low": 0, "high": None},
    {"type": "float", "low": float("nan"), "high": 1},
    {"type": "float", "low": 0, "high": float("inf")},
    {"type": "float", "low": 0, "high": 1, "log": True},
    {"type": "float", "low": -1, "high": 1, "log": True},
    {"type": "float", "low": 0.1, "high": 1, "log": "false"},
    {"type": "int", "high": 4},
    {"type": "int", "low": 1, "high": None},
    {"type": "int", "low": 1.5, "high": 4},
    {"type": "int", "low": 1, "high": 4.5},
    {"type": "int", "low": True, "high": 4},
    {"type": "int", "low": 1, "high": float("inf")},
    {"type": "int", "low": float("nan"), "high": 4},
])
def test_invalid_numeric_domain_never_starts_search_fitting(spec, monkeypatch):
    session = (Session.ingest(pd.DataFrame({"x": np.arange(40), "y": [0, 1] * 20}))
               .set_roles({"x": "feature", "y": "target"}).split(test_size=0.2, random_state=1))
    fitting = Mock(side_effect=AssertionError("Malformed domain started fitting"))
    monkeypatch.setattr(DecisionTreeClassifier, "fit", fitting)
    with pytest.raises(ValidationError):
        session.evolutionary_search(DecisionTreeClassifier(), param_space={"max_depth": spec},
                                    population_size=2, elite_size=1, n_generations=1,
                                    max_evaluations=2, cv=2, refit=False)
    fitting.assert_not_called()
