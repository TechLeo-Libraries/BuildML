"""Execute teaching fragments against representative data, not mocked facades."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.explain.beginner.activelearning import ACTIVELEARNING_BEGINNER
from buildml.explain.beginner.classical import CLASSICAL_BEGINNER
from buildml.explain.beginner.probabilistic import PROBABILISTIC_BEGINNER
from scripts.check_teaching_examples import check_examples


def test_authored_examples_bind_and_use_real_result_fields():
    count, errors = check_examples()
    assert count >= 430
    assert not errors, "\n".join(errors)


def _session(regression=False):
    if regression:
        x, y = make_regression(n_samples=100, n_features=4, noise=5, random_state=7)
    else:
        x, y = make_classification(n_samples=100, n_features=4, random_state=7)
    frame = pd.DataFrame(x, columns=["a", "b", "c", "d"])
    frame["target"] = y
    session = Session.ingest(frame)
    session.set_roles({"target": "target"})
    session.split(test_size=0.2, validation_size=0.2, random_state=7)
    return session


@pytest.mark.parametrize("key", [
    "cross-validation", "probability-calibration", "thresholds",
    "feature-importance", "mutual-information", "normality-screens",
])
def test_classical_teaching_fragment(key):
    session = _session()
    session.fit(LogisticRegression(max_iter=1000))
    namespace = {"session": session, "LogisticRegression": LogisticRegression}
    exec("\n".join(CLASSICAL_BEGINNER[key].mini_example), namespace)


@pytest.mark.parametrize("key", [
    "probabilistic-uncertainty", "probabilistic-bayesian-ridge",
    "probabilistic-split-conformal",
])
def test_probabilistic_teaching_fragment(key):
    session = _session(regression=True)
    namespace = {"session": session}
    exec("\n".join(PROBABILISTIC_BEGINNER[key].mini_example), namespace)
    result = session.probabilistic.predict_interval(alpha=0.1, partition="test")
    assert len(result.lower) == len(result.upper) == 20
    assert np.all(np.asarray(result.lower) <= np.asarray(result.upper))


def test_active_learning_query_and_label_example():
    session = _session()
    frame = session.to_pandas()
    labels = frame["target"].to_dict()
    frame.loc[frame.index[::3], "target"] = np.nan
    session = Session.ingest(frame)
    session.set_roles({"target": "target"})
    namespace = {"session": session, "labels": labels}
    exec("\n".join(ACTIVELEARNING_BEGINNER["activelearning-train-pool"].mini_example), namespace)
    assert namespace["indices"]


def test_custom_transform_teaching_fragment():
    frame = pd.DataFrame({"spend": [10., 20., 5., 8., 9.], "visits": [2., 0., 1., 4., 3.]})
    session = Session.ingest(frame)
    session.split(test_size=0.2, random_state=0)
    exec("\n".join(CLASSICAL_BEGINNER["custom-transforms"].mini_example), {"session": session})
    assert "spend_per_visit" in session.to_pandas()


@pytest.mark.parametrize("domain", ["probabilistic", "rl", "anomaly", "unsupervised", "ensemble", "automl", "activelearning"])
def test_bundle_teaching_roundtrip(domain, tmp_path, monkeypatch):
    """Fresh Sessions have no split unless the example explicitly restores it."""
    import importlib

    monkeypatch.chdir(tmp_path)
    session = _session(regression=domain == "probabilistic")
    frame = session.to_pandas()
    if domain == "probabilistic":
        session.probabilistic.fit(estimator="bayesian_ridge")
    elif domain == "rl":
        session.rl.fit_imitation(estimator="logistic_regression")
    elif domain == "anomaly":
        session.anomaly.fit(method="isolation_forest", n_estimators=10)
    elif domain == "ensemble":
        session.ensemble.fit_voting({"first": LogisticRegression(), "second": LogisticRegression(C=0.5)})
    elif domain == "automl":
        session.automl.run(families=["logistic"], n_trials=1, cv=2, include_industry_families=False)
    elif domain == "activelearning":
        frame.loc[frame.index[::3], "target"] = np.nan
        session = Session.ingest(frame).set_roles({"target": "target"})
        session.split(test_size=0.2, random_state=0)
        session.active_learning.fit()
    module = importlib.import_module(f"buildml.explain.beginner.{domain}")
    layers = getattr(module, f"{domain.upper()}_BEGINNER")
    key = "imitation-bundle-boundary" if domain == "rl" else f"{domain}-bundle-boundary"
    namespace = {"Session": Session, "session": session, "frame": frame,
                 "new_frame": frame, "today_frame": frame, "state_frame": frame}
    exec("\n".join(layers[key].mini_example), namespace)
    restored = next(namespace[name] for name in ["job", "service", "later", "restored", "resumed"] if name in namespace)
    if domain in {"ensemble", "automl", "activelearning"}:
        assert restored.split_plan == session.split_plan or restored.split_plan.train_indices == session.split_plan.train_indices
    else:
        assert restored.split_plan is None
