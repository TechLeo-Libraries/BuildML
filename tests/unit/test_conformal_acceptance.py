"""Finite-sample coverage boundaries must not silently shrink intervals."""

import numpy as np
import pytest

from buildml.core.errors import ValidationError
from buildml.probabilistic.conformal import conformal_quantile


def test_unattainable_finite_coverage_is_rejected():
    with pytest.raises(ValidationError, match="requires an unbounded"):
        conformal_quantile(np.arange(5.0), alpha=0.01)


def test_smallest_supported_alpha_uses_largest_score():
    assert conformal_quantile(np.arange(5.0), alpha=1 / 6) == 4.0
    assert conformal_quantile(np.arange(9.0), alpha=0.1) == 8.0


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_calibration_scores_are_rejected(invalid):
    with pytest.raises(ValidationError, match="must all be finite"):
        conformal_quantile(np.array([0.1, invalid, 0.3]), alpha=0.5)
