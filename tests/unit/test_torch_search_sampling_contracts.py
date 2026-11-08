"""Randomized search stays seeded and never hides a sampler's internal failure."""
from __future__ import annotations

import pytest
from scipy.stats import loguniform, randint

from buildml.core.errors import ValidationError
from buildml.dl.search import _sample_distributions


def test_real_scipy_sampling_is_reproducible_and_within_declared_domains():
    space = {"learning_rate": loguniform(1e-5, 1e-2), "epochs": randint(1, 4), "dropout": [0., .2]}
    first = _sample_distributions(space, n_iter=32, seed=18)
    assert first == _sample_distributions(space, n_iter=32, seed=18)
    assert first != _sample_distributions(space, n_iter=32, seed=19)
    assert all(1e-5 <= trial["learning_rate"] <= 1e-2 for trial in first)
    assert all(1 <= trial["epochs"] < 4 for trial in first)
    assert all(trial["dropout"] in [0., .2] for trial in first)


def test_internal_sampler_typeerror_is_not_retried_without_the_seed():
    class BrokenSampler:
        def __init__(self):
            self.calls = []

        def rvs(self, random_state=None):
            self.calls.append(random_state)
            if random_state is not None:
                raise TypeError("internal sampler failure")
            return .123

    sampler = BrokenSampler()
    with pytest.raises(TypeError, match="internal sampler failure"):
        _sample_distributions({"learning_rate": sampler}, n_iter=1, seed=9)
    assert len(sampler.calls) == 1 and isinstance(sampler.calls[0], int)


def test_legacy_sampler_without_seed_parameter_is_called_once_per_trial():
    class LegacySampler:
        def __init__(self):
            self.calls = 0

        def rvs(self):
            self.calls += 1
            return self.calls

    sampler = LegacySampler()
    trials = _sample_distributions({"epochs": sampler}, n_iter=3, seed=8)
    assert trials == [{"epochs": 1}, {"epochs": 2}, {"epochs": 3}]
    assert sampler.calls == 3


@pytest.mark.parametrize("space,n_iter,message", [
    ({}, 1, "non-empty mapping"),
    ({"epochs": []}, 1, "Empty distribution"),
    ({"epochs": object()}, 1, "sequence or scipy-like"),
    ({"unknown_parameter": [1]}, 1, "Unsupported Torch search keys"),
    ({"epochs": [1]}, 0, "n_iter must be >= 1"),
])
def test_invalid_search_spaces_fail_before_trials(space, n_iter, message):
    with pytest.raises(ValidationError, match=message):
        _sample_distributions(space, n_iter=n_iter, seed=3)
