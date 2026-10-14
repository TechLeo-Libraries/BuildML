"""Probability columns and policy costs must retain the actual positive label."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from buildml import Session


class ScoreClassifier(ClassifierMixin, BaseEstimator):
    """Valid sklearn classifier with explicit probabilities for a policy oracle."""
    def fit(self, x, y):
        self.classes_ = np.unique(y)
        self.n_features_in_ = x.shape[1]
        return self

    def predict_proba(self, x):
        score = np.asarray(x)[:, 0]
        return np.column_stack([1 - score, score])

    def predict(self, x):
        return self.classes_[(self.predict_proba(x)[:, 1] >= .5).astype(int)]


@pytest.mark.parametrize("labels", [["benign", "malignant"], [-3, 8]])
def test_cost_minimum_includes_observed_probabilities_below_the_display_grid(labels):
    scores = [.1, .9, .2, .8, .01, .02, .001, .005, .03, .004, .3, .7]
    codes = [0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1]
    s = (Session.ingest(pd.DataFrame({"score": scores, "label": np.asarray(labels)[codes]}))
         .set_roles({"score": "feature", "label": "target"})
         .inject_split(train_indices=[0, 1, 2, 3], validation_indices=list(range(4, 10)), test_indices=[10, 11])
         .fit(ScoreClassifier(), task="classification"))
    report = s.tune_threshold(fp_cost=1., fn_cost=100.)
    assert report.payload["positive_class"] == str(labels[1])
    assert report.payload["expected_cost_at_recommended"] == pytest.approx(0.)
    point = report.payload["recommended_threshold"]
    assert point["tp"] == 3 and point["tn"] == 3
    assert point["fp"] == 0 and point["fn"] == 0
    cal = s.calibration()
    prob = np.asarray(scores[4:10])
    truth = np.asarray(codes[4:10])
    assert cal.payload["positive_class"] == str(labels[1])
    assert cal.payload["brier_score"] == pytest.approx(np.square(prob - truth).mean())


@pytest.mark.parametrize("costs", [
    {"fp_cost": 100., "fn_cost": 0.},
    {"fp_cost": 0., "fn_cost": 100.},
    {"fp_cost": 2., "fn_cost": 7., "tp_benefit": 3., "tn_benefit": 4.},
])
def test_empirical_minimum_and_all_row_counts_match_independent_policy_enumeration(costs):
    scores = np.asarray([0., .001, .001, .25, .8, 1.])
    truth = np.asarray([1, 0, 1, 0, 1, 0])
    frame = pd.DataFrame({"score": [.1, .9, *scores, .3, .7], "label": [0, 1, *truth, 0, 1]})
    s = (Session.ingest(frame).set_roles({"score": "feature", "label": "target"})
         .inject_split(train_indices=[0, 1], validation_indices=list(range(2, 8)), test_indices=[8, 9])
         .fit(ScoreClassifier(), task="classification"))
    report = s.tune_threshold(**costs)
    totals = []
    for row in report.payload["rows"]:
        chosen = scores >= row["threshold"]
        expected = {"tp": int((chosen & (truth == 1)).sum()),
                    "fp": int((chosen & (truth == 0)).sum()),
                    "tn": int((~chosen & (truth == 0)).sum()),
                    "fn": int((~chosen & (truth == 1)).sum())}
        assert {name: row[name] for name in expected} == expected
        cost = (costs["fp_cost"] * expected["fp"] + costs["fn_cost"] * expected["fn"]
                - costs.get("tp_benefit", 0.) * expected["tp"] - costs.get("tn_benefit", 0.) * expected["tn"])
        assert row["expected_cost_total"] == pytest.approx(cost)
        totals.append(cost)
    assert report.payload["expected_cost_at_recommended"] == pytest.approx(min(totals))
    if costs["fn_cost"] == 0:
        assert report.payload["recommended_threshold"]["predicted_positive_rate"] == 0.
    elif costs["fp_cost"] == 0:
        assert report.payload["recommended_threshold"]["predicted_positive_rate"] == 1.
