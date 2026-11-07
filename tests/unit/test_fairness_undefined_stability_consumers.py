"""Undefined fairness gaps retain unknown bands and disclose omitted draws."""

import numpy as np
import pytest

from buildml.fairness.evaluate import evaluate_fairness


@pytest.mark.parametrize("method", ["bootstrap", "stratified_subsample"])
@pytest.mark.parametrize("predicted", [0, 1])
def test_missing_negative_outcomes_and_absent_groups_do_not_create_zero_uncertainty(method, predicted):
    truth = np.ones(8, dtype=int)
    predictions = np.repeat(predicted, 8)
    groups = ["majority"]*7 + ["rare"]
    report = evaluate_fairness(truth, predictions, groups, bootstrap_samples=11,
                               stability_method=method, random_state=17)
    bands = report.stability.metrics
    for name in ["demographic_parity_difference", "equalized_odds_tpr_difference"]:
        assert bands[name] == {"point": 0., "ci_low": 0., "ci_high": 0., "std": 0., "n_finite_draws": 11}
    unknown = {"point": None, "ci_low": None, "ci_high": None, "std": None, "n_finite_draws": 0}
    assert bands["equalized_odds_fpr_difference"] == unknown
    if predicted == 0:
        assert bands["disparate_impact_ratio"] == unknown
    else:
        rng = np.random.default_rng(17)
        finite = (sum(len({groups[i] for i in rng.integers(0, 8, size=8)}) > 1 for _ in range(11))
                  if method == "bootstrap" else 11)
        assert bands["disparate_impact_ratio"] == {
            "point": 1., "ci_low": 1., "ci_high": 1., "std": 0., "n_finite_draws": finite,
        }
    assert report.equalized_odds_fpr_difference is None
    assert all(np.isnan(value) for value in report.fpr_by_group.values())
    assert any("Undefined gap draws" in line for line in report.disclosures)
    serialized = report.to_dict()
    assert serialized["stability"]["metrics"] == bands
    assert "n=11" in report.to_markdown()
