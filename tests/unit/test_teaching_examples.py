"""Execute complete teaching examples and check their resulting state."""
from __future__ import annotations

import numpy as np
import pytest

from buildml.explain.beginner.activelearning import ACTIVELEARNING_BEGINNER
from buildml.explain.beginner.classical import CLASSICAL_BEGINNER
from buildml.explain.beginner.probabilistic import PROBABILISTIC_BEGINNER
from scripts.check_teaching_examples import check_examples


def test_authored_examples_bind_and_use_real_result_fields():
    count, errors = check_examples()
    assert count >= 430
    assert not errors, "\n".join(errors)


@pytest.fixture(autouse=True)
def _example_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize("key", [
    "cross-validation", "probability-calibration", "thresholds",
    "feature-importance", "mutual-information", "normality-screens",
])
def test_classical_teaching_fragment(key):
    namespace = {}
    exec("\n".join(CLASSICAL_BEGINNER[key].mini_example), namespace)


@pytest.mark.parametrize("key", [
    "probabilistic-uncertainty", "probabilistic-bayesian-ridge",
    "probabilistic-split-conformal",
])
def test_probabilistic_teaching_fragment(key):
    namespace = {}
    exec("\n".join(PROBABILISTIC_BEGINNER[key].mini_example), namespace)
    session = namespace["session"]
    result = session.probabilistic.predict_interval(alpha=0.1, partition="test")
    assert len(result.lower) == len(result.upper) == len(session.split_plan.test_indices)
    assert np.all(np.asarray(result.lower) <= np.asarray(result.upper))


def test_active_learning_query_and_label_example():
    namespace = {}
    exec("\n".join(ACTIVELEARNING_BEGINNER["activelearning-train-pool"].mini_example), namespace)
    assert namespace["indices"]


def test_custom_transform_teaching_fragment():
    namespace = {}
    exec("\n".join(CLASSICAL_BEGINNER["custom-transforms"].mini_example), namespace)
    original = namespace["frame"]["spend"]
    transformed = namespace["session"].to_pandas()["spend"]
    np.testing.assert_allclose(transformed, np.sign(original) * np.log1p(np.abs(original)))


@pytest.mark.parametrize("domain", ["probabilistic", "rl", "anomaly", "unsupervised", "ensemble", "automl", "activelearning"])
def test_bundle_teaching_roundtrip(domain, tmp_path, monkeypatch):
    """Fresh Sessions have no split unless the example explicitly restores it."""
    import importlib

    monkeypatch.chdir(tmp_path)
    module = importlib.import_module(f"buildml.explain.beginner.{domain}")
    layers = getattr(module, f"{domain.upper()}_BEGINNER")
    key = "imitation-bundle-boundary" if domain == "rl" else f"{domain}-bundle-boundary"
    namespace = {}
    exec("\n".join(layers[key].mini_example), namespace)
    session = namespace["session"]
    restored = next(namespace[name] for name in ["job", "service", "later", "restored", "resumed"] if name in namespace)
    if domain in {"ensemble", "automl", "activelearning"}:
        assert restored.split_plan == session.split_plan or restored.split_plan.train_indices == session.split_plan.train_indices
    else:
        assert restored.split_plan is None
