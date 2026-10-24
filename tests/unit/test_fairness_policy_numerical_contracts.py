"""Independent opt-in policy calculations and refusal of ambiguous inputs."""

from __future__ import annotations

import numpy as np
import pytest

from buildml.core.errors import ValidationError
from buildml.fairness.mitigation import (
    apply_group_thresholds,
    selection_rates_from_scores,
    suggest_group_thresholds,
    suggest_reweighing_weights,
)


@pytest.mark.parametrize("target", ["demographic_parity", "equal_opportunity"])
@pytest.mark.parametrize("grid_size", [5, np.int64(21)])
def test_thresholds_match_exhaustive_scalar_policy(target, grid_size):
    labels = ["yes", "no", "yes", "no", "no", "no", "yes", "yes", "no"]
    scores = [0.91, 0.72, 0.42, 0.16, 0.1, 0.81, 0.31, 0.63, 0.45]
    groups = ["a", "a", "a", "a", "b", "b", "b", "b", "empty-positive"]
    before = (labels.copy(), scores.copy(), groups.copy())
    reference_rows = [
        i for i, label in enumerate(labels) if target == "demographic_parity" or label == "yes"
    ]
    reference = sum(scores[i] >= 0.5 for i in reference_rows) / len(reference_rows)
    expected = {}
    for group in sorted(set(groups)):
        rows = [i for i, value in enumerate(groups) if value == group]
        eligible = [i for i in rows if target == "demographic_parity" or labels[i] == "yes"]
        expected[group] = (
            min(
                (k / (int(grid_size) - 1) for k in range(int(grid_size))),
                key=lambda threshold: (
                    abs(sum(scores[i] >= threshold for i in eligible) / len(eligible) - reference),
                    threshold,
                ),
            )
            if eligible
            else 0.5
        )
    result = suggest_group_thresholds(
        labels, scores, groups, positive_label="yes", target=target, grid_size=grid_size
    )
    assert result.thresholds_by_group == pytest.approx(expected)
    actions = apply_group_thresholds(
        scores, groups, result.thresholds_by_group, positive_label="yes", negative_label="no"
    )
    expected_actions = [
        "yes" if score >= expected[group] else "no"
        for score, group in zip(scores, groups, strict=True)
    ]
    assert actions.tolist() == expected_actions
    for group in expected:
        rows = [i for i, value in enumerate(groups) if value == group]
        positive_rows = [i for i in rows if labels[i] == "yes"]
        rate = sum(expected_actions[i] == "yes" for i in rows) / len(rows)
        assert result.achieved_selection_rate_by_group[group] == pytest.approx(rate)
        assert result.support_by_group[group] == len(rows)
        if positive_rows:
            tpr = sum(expected_actions[i] == "yes" for i in positive_rows) / len(positive_rows)
            assert result.achieved_tpr_by_group[group] == pytest.approx(tpr)
        else:
            assert np.isnan(result.achieved_tpr_by_group[group])
    assert (
        selection_rates_from_scores(
            scores, groups, expected, positive_label="yes", negative_label="no"
        )
        == result.achieved_selection_rate_by_group
    )
    if target == "equal_opportunity":
        assert any("no positives" in warning for warning in result.warnings)
    assert (labels, scores, groups) == before


@pytest.mark.parametrize("bad", [5.5, True, float("nan"), "21"])
def test_grid_resolution_requires_an_integer(bad):
    with pytest.raises(ValidationError, match="grid_size"):
        suggest_group_thresholds([0, 1], [0.2, 0.8], ["a", "b"], grid_size=bad)


@pytest.mark.parametrize(
    "scores", [[np.nan, 0.8], [np.inf, 0.8], [0.2 + 1j, 0.8], ["bad", 0.8], [[0.2], [0.8]], 0.2]
)
@pytest.mark.parametrize("operation", ["suggest", "apply"])
def test_nonfinite_or_nonvector_scores_are_refused(scores, operation):
    with pytest.raises(ValidationError, match="y_score"):
        if operation == "suggest":
            suggest_group_thresholds([0, 1], scores, ["a", "b"])
        else:
            apply_group_thresholds(scores, ["a", "b"], {"a": 0.5, "b": 0.5})


@pytest.mark.parametrize("threshold", [np.nan, np.inf, "invalid", 1j, np.complex128(0.5 + 1j)])
def test_invalid_threshold_cannot_silently_choose_negative_class(threshold):
    with pytest.raises(ValidationError, match="threshold"):
        apply_group_thresholds([0.9, 0.2], ["a", "a"], {"a": threshold})


@pytest.mark.parametrize("field", ["y_true", "sensitive"])
def test_policy_inputs_must_be_vectors(field):
    labels = [[0], [1]] if field == "y_true" else [0, 1]
    groups = [["a"], ["b"]] if field == "sensitive" else ["a", "b"]
    with pytest.raises(ValidationError, match=field):
        suggest_group_thresholds(labels, [0.2, 0.8], groups)
    with pytest.raises(ValidationError, match=field):
        suggest_reweighing_weights(labels, groups)


def test_reweighing_matches_joint_counts_and_reports_missing_cells():
    labels = [0, 0, 0, 1, 0, 1, 1, 1, 0]
    groups = ["a"] * 4 + ["b"] * 4 + ["c"]
    result = suggest_reweighing_weights(labels, groups)
    raw = [
        groups.count(group)
        * labels.count(label)
        / (
            len(labels)
            * sum(g == group and y == label for g, y in zip(groups, labels, strict=True))
        )
        for group, label in zip(groups, labels, strict=True)
    ]
    expected = np.array(raw) / np.mean(raw)
    np.testing.assert_allclose(result.weights, expected, rtol=1e-14)
    assert any("zero rows" in warning.lower() for warning in result.warnings)
    assert result.to_dict()["weights"] == pytest.approx(expected.tolist())


def test_unknown_group_uses_explicit_default_midpoint():
    assert apply_group_thresholds([0.5, 0.49], ["new", "new"], {}).tolist() == [1, 0]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"target": "unknown"}, "Unknown threshold target"),
        ({"grid_size": 4}, "grid_size"),
        ({"target": "equal_opportunity", "positive_label": "absent"}, "no positives"),
    ],
)
def test_unusable_policy_configuration_is_refused(kwargs, message):
    with pytest.raises(ValidationError, match=message):
        suggest_group_thresholds([0, 1], [0.2, 0.8], ["a", "b"], **kwargs)


def test_empty_and_misaligned_policy_populations_are_refused():
    with pytest.raises(ValidationError, match="at least one row"):
        suggest_group_thresholds([], [], [])
    with pytest.raises(ValidationError, match="equal length"):
        suggest_group_thresholds([0, 1], [0.2], ["a", "b"])
    with pytest.raises(ValidationError, match="equal length"):
        apply_group_thresholds([0.2], ["a", "b"], {})
    with pytest.raises(ValidationError, match="sensitive"):
        apply_group_thresholds([0.2, 0.8], [["a"], ["b"]], {})
    with pytest.raises(ValidationError, match="at least one row"):
        suggest_reweighing_weights([], [])
    with pytest.raises(ValidationError, match="equal length"):
        suggest_reweighing_weights([0, 1], ["a"])
