"""Regression checks for leakage, statistical interpretation, and example semantics."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer

from buildml import Session
from buildml.core.errors import LeakageError, ValidationError
from buildml.eda.analyzers.target import _adjust_pvalues


def _session(classification=False):
    rng = np.random.default_rng(17)
    x = rng.normal(size=240)
    y = (x > 0).astype(int) if classification else x + rng.normal(size=240)
    return Session.ingest(pd.DataFrame({"x": x, "y": y})).set_roles(
        {"x": "feature", "y": "target"}
    ).split(test_size=.2, random_state=7)


@pytest.mark.parametrize("preparation", ["scale", "impute", "reduce"])
def test_conformal_rejects_calibration_influenced_preprocessing(preparation):
    s = _session()
    if preparation == "scale":
        s.scale(method="standard")
    elif preparation == "impute":
        s.impute(strategy="median")
    else:
        s.reduce_dimensions(n_components=1)
    with pytest.raises(LeakageError, match="calibration rows"):
        s.probabilistic.fit()
    assert s.probabilistic_plan is None


def test_mapie_cannot_bypass_preprocessing_guard_with_overlay_disabled():
    s = _session().scale(method="standard")
    with pytest.raises(LeakageError):
        s.probabilistic.fit(backend="mapie", estimator="split", conformal=False)


def test_normalized_mapie_alias_cannot_bypass_guard():
    with pytest.raises(LeakageError):
        _session().scale(method="standard").probabilistic.fit(estimator="cv-plus", conformal=False)


def test_rare_class_carve_refuses_instead_of_selecting_memberships(monkeypatch):
    s = _session(True)
    indices = list(s.split_plan.train_indices)
    frame = s.to_pandas().copy()
    frame.loc[indices[-1], "y"] = 2
    s = Session.ingest(frame).set_roles({"x": "feature", "y": "target"}).split(test_size=.2, random_state=7)
    monkeypatch.setattr("buildml.probabilistic.fit.split_train_for_conformal",
                        lambda *a, **k: (indices[:-10], indices[-10:]))
    with pytest.raises(ValidationError, match="class absent"):
        s.probabilistic.fit(estimator="gaussian_nb", task="classification")


def test_posterior_without_conformal_remains_available_after_scaling():
    s = _session().scale(method="standard")
    s.probabilistic.fit(conformal=False, interval_method="posterior_std")
    assert s.probabilistic.predict_interval().method == "posterior_std"


def test_calibration_membership_does_not_depend_on_labels():
    a, b = _session(True), _session(True)
    # Fit an independent classification task on the same features and memberships.
    frame = b.to_pandas().copy()
    frame["y"] = np.random.default_rng(88).integers(0, 2, len(frame))
    b = Session.ingest(frame).set_roles({"x": "feature", "y": "target"}).split(test_size=.2, random_state=7)
    for s in (a, b):
        s.probabilistic.fit(estimator="gaussian_nb", task="classification", random_state=3)
    assert a.probabilistic_plan.conformal_calib_indices_ == b.probabilistic_plan.conformal_calib_indices_
    assert set(a.probabilistic_plan.conformal_fit_indices_).isdisjoint(a.probabilistic_plan.conformal_calib_indices_)


def test_bh_adjustment_matches_known_family_and_is_monotone():
    rows = [{"pvalue": p} for p in (.01, .04, .03, .002)]
    _adjust_pvalues(rows)
    assert [r["qvalue_bh"] for r in rows] == pytest.approx([.02, .04, .04, .008])
    assert all(r["tested_family_size"] == 4 for r in rows)


def test_eda_interpretation_and_adjustments_are_exported():
    report = _session(True).eda(partition="train")
    assert report.overview["interpretation"]["purpose"] == "exploratory"
    assert report.target["interpretation"]["task_source"].startswith("observed-label")
    assert report.target["numeric_separation_tests"][0]["qvalue_bh"] >= report.target["numeric_separation_tests"][0]["pvalue"]
    assert any("multivariate shift" in warning for warning in report.warnings)


def test_example_positive_class_means_malignant():
    path = Path(__file__).parents[2] / "examples/breast_cancer_classical_loop.py"
    spec = importlib.util.spec_from_file_location("example_target_semantics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = (load_breast_cancer(as_frame=True).target == 0).astype(int)
    np.testing.assert_array_equal(module._wisconsin_table()["malignant"], expected)
