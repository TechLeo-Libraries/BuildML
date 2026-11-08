"""Fairness populations and resampling bands match explicit row calculations."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.fairness.evaluate import evaluate_fairness
from buildml.fairness.groups import compose_group_keys
from buildml.fairness.mitigation import (
    apply_group_thresholds,
    suggest_group_thresholds,
    suggest_reweighing_weights,
)
from buildml.fairness.stability import estimate_gap_stability


def _gaps(truth, predicted, groups):
    rates, tprs, fprs = [], [], []
    for group in sorted(set(groups)):
        rows = [i for i, value in enumerate(groups) if value == group]
        rates.append(sum(predicted[i] for i in rows)/len(rows))
        positives = [i for i in rows if truth[i]]
        negatives = [i for i in rows if not truth[i]]
        if positives:
            tprs.append(sum(predicted[i] for i in positives)/len(positives))
        if negatives:
            fprs.append(sum(predicted[i] for i in negatives)/len(negatives))
    return {
        "demographic_parity_difference": max(rates)-min(rates),
        "disparate_impact_ratio": min(rates)/max(rates) if len(rates)>1 and max(rates)>0 else None,
        "equalized_odds_tpr_difference": max(tprs)-min(tprs) if tprs else None,
        "equalized_odds_fpr_difference": max(fprs)-min(fprs) if fprs else None,
    }


@pytest.mark.parametrize("method", ["bootstrap", "stratified_subsample"])
@pytest.mark.parametrize("confidence", [.8, .95])
@pytest.mark.parametrize("encoding", ["numbers", "strings"])
def test_stability_matches_independent_row_draws_and_scalar_gaps(method, confidence, encoding):
    truth = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0])
    predicted = np.array([0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1])
    groups = np.array(["a"]*5 + ["b"]*6 + ["rare"])
    labels = truth if encoding == "numbers" else np.where(truth, "yes", "no")
    predictions = predicted if encoding == "numbers" else np.where(predicted, "yes", "no")
    positive = 1 if encoding == "numbers" else "yes"
    point = _gaps(truth, predicted, groups)
    draws = {name: [] for name in point}
    rng = np.random.default_rng(721)
    for _ in range(19):
        if method == "bootstrap":
            indices = rng.integers(0, len(truth), size=len(truth))
        else:
            parts = []
            for group in sorted(set(groups)):
                positions = np.where(groups == group)[0]
                parts.extend(rng.choice(positions, size=max(1, round(.6*len(positions))), replace=False))
            indices = np.array(parts)
            rng.shuffle(indices)
        for name, value in _gaps(truth[indices], predicted[indices], groups[indices]).items():
            if value is not None:
                draws[name].append(value)
    options = dict(positive_label=positive, n_resamples=np.int64(19), confidence_level=confidence,
                   method=method, subsample_fraction=.6, random_state=721)
    result = estimate_gap_stability(labels, predictions, groups, **options)
    report = evaluate_fairness(labels, predictions, groups, positive_label=positive,
                               bootstrap_samples=np.int64(19), confidence_level=confidence,
                               stability_method=method, subsample_fraction=.6, random_state=721)
    for name, value in point.items():
        expected = {"point": value, "ci_low": float(np.quantile(draws[name], (1-confidence)/2)),
                    "ci_high": float(np.quantile(draws[name], 1-(1-confidence)/2)),
                    "std": float(np.std(draws[name], ddof=1)), "n_finite_draws": len(draws[name])}
        assert result.metrics[name] == pytest.approx(expected)
        assert report.stability.metrics[name] == pytest.approx(expected)
    assert result.n_resamples == 19
    assert "not causal" in " ".join(result.disclosures)


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA, pd.NaT])
def test_nullable_sensitive_values_share_an_explicit_missing_population(missing):
    truth = [0, 1, 0, 1, 0, 1]
    predicted = [0, 1, 1, 1, 0, 0]
    groups = pd.Series([missing, "observed", missing, "observed", missing, "observed"], dtype=object)
    report = evaluate_fairness(truth, predicted, groups, bootstrap_samples=7)
    assert report.groups == ("<NA>", "observed")
    assert report.support_by_group == {"<NA>": 3, "observed": 3}
    assert report.selection_rate_by_group == pytest.approx({"<NA>": 1/3, "observed": 2/3})
    weights = suggest_reweighing_weights(truth, groups)
    assert set(weights.weight_table) == {"<NA>", "observed"}
    suggestion = suggest_group_thresholds(truth, [.1, .9, .8, .7, .2, .4], groups, grid_size=5)
    actions = apply_group_thresholds([.1, .9, .8, .7, .2, .4], groups, suggestion.thresholds_by_group)
    expected = [int(s >= suggestion.thresholds_by_group["<NA>" if i%2 == 0 else "observed"])
                for i, s in enumerate([.1, .9, .8, .7, .2, .4])]
    assert actions.tolist() == expected


COLLISIONS = [(["A|B", "A"], ["C", "B|C"]),
              ([None, "<NA>"], ["same", "same"]),
              (pd.Series([1, "1"], dtype=object), ["same", "same"])]


@pytest.mark.parametrize("parts", COLLISIONS)
def test_group_composition_refuses_distinct_populations_with_the_same_display_key(parts):
    with pytest.raises(ValidationError, match="ambiguous|collision"):
        compose_group_keys(*parts)


def _session(parts):
    n = 48
    x = np.linspace(-2, 2, n)
    frame = pd.DataFrame({"x": x, "y": (np.sin(np.arange(n))+.4*x > 0).astype(int),
                          "group": list(parts[0])*24, "region": list(parts[1])*24})
    session = Session.ingest(frame).set_roles({"x": "feature", "y": "target",
                                               "group": "ignore", "region": "ignore"})
    session.inject_split(train_indices=list(range(32)), validation_indices=list(range(32, 40)),
                         test_indices=list(range(40, 48)))
    return session.fit(LogisticRegression(), task="classification")


@pytest.mark.parametrize("parts", COLLISIONS)
@pytest.mark.parametrize("operation", ["evaluate", "suggest_thresholds", "suggest_reweighing"])
def test_session_refuses_ambiguous_populations_without_changing_prior_outputs(parts, operation):
    session = _session(parts)
    fingerprint = joblib.hash(session.fit_result)
    before = list(session._history)
    with pytest.raises(ValidationError, match="ambiguous|collision"):
        getattr(session.fairness, operation)(sensitive_column=["group", "region"])
    assert session._history == before and joblib.hash(session.fit_result) == fingerprint
    assert session._fairness_report is None and session._fairness_mitigation_suggestion is None


@pytest.mark.parametrize("operation,field", [(op,field) for op in ["evaluate", "stability"]
                                           for field in ["y_true", "y_pred"]]
                         + [("thresholds", "y_true"), ("reweighing", "y_true")])
@pytest.mark.parametrize("malformed", ["matrix", "missing"])
def test_audit_labels_require_aligned_nonmissing_vectors(operation, field, malformed):
    truth, predicted, groups = [0, 1, 0, 1], [0, 1, 1, 0], ["a", "a", "b", "b"]
    bad = [[0], [1], [0], [1]] if malformed == "matrix" else [0, 1, pd.NA, 1]
    if field == "y_true":
        truth = bad
    else:
        predicted = bad
    with pytest.raises(ValidationError, match=field):
        if operation == "evaluate":
            evaluate_fairness(truth, predicted, groups)
        elif operation == "stability":
            estimate_gap_stability(truth, predicted, groups, n_resamples=3)
        elif operation == "thresholds":
            suggest_group_thresholds(truth, [.1, .8, .6, .2], groups)
        else:
            suggest_reweighing_weights(truth, groups)


@pytest.mark.parametrize("bad", [True, np.bool_(True), 2.5, "5", -1, 1, np.nan, np.inf])
@pytest.mark.parametrize("operation", ["evaluate", "stability"])
def test_resampling_counts_cannot_be_silently_disabled_or_truncated(bad, operation):
    with pytest.raises(ValidationError, match="integer"):
        if operation == "evaluate":
            evaluate_fairness([0, 1, 0, 1], [0, 1, 1, 0], ["a"]*2+["b"]*2, bootstrap_samples=bad)
        else:
            estimate_gap_stability([0, 1, 0, 1], [0, 1, 1, 0], ["a"]*2+["b"]*2, n_resamples=bad)


@pytest.mark.parametrize("field", ["confidence_level", "subsample_fraction"])
@pytest.mark.parametrize("bad", [True, "0.8", np.nan, np.inf, 0, 1.1])
def test_resampling_real_controls_have_clear_boundary_errors(field, bad):
    with pytest.raises(ValidationError, match=field):
        estimate_gap_stability([0, 1, 0, 1], [0, 1, 1, 0], ["a"]*2+["b"]*2,
                               n_resamples=3, method="stratified_subsample", **{field: bad})


@pytest.mark.parametrize("operation", ["evaluate", "stability"])
@pytest.mark.parametrize("field", ["y_pred", "sensitive"])
def test_resampling_requires_one_shared_row_population(operation, field):
    truth, predicted, groups = [0, 1, 0, 1], [0, 1, 1, 0], ["a"]*2+["b"]*2
    if field == "y_pred":
        predicted = predicted[:2]
    else:
        groups = groups[:2]
    with pytest.raises(ValidationError, match="length"):
        if operation == "evaluate":
            evaluate_fairness(truth, predicted, groups)
        else:
            estimate_gap_stability(truth, predicted, groups, n_resamples=3)


@pytest.mark.parametrize("scores", [0.5, [[.1], [.9], [.3], [.7]], [.1, .9],
                                    [np.nan, .9, .3, .7], [np.inf, .9, .3, .7],
                                    [1j, .9, .3, .7], ["bad", .9, .3, .7]])
@pytest.mark.parametrize("classical", [False, True])
def test_optional_auc_scores_still_require_a_finite_aligned_vector(scores, classical):
    with pytest.raises(ValidationError, match="y_score"):
        evaluate_fairness([0, 1, 0, 1], [0, 1, 1, 0], ["a"]*2+["b"]*2,
                          y_score=scores, include_classical_metrics=classical)


@pytest.mark.parametrize("operation,canonical", [("evaluate", "evaluate_fairness"),
                                                ("suggest_thresholds", "suggest_fairness_thresholds"),
                                                ("suggest_reweighing", "suggest_fairness_reweighing")])
def test_real_session_audits_are_recorded_and_survive_checkpoint_replay(operation, canonical, tmp_path):
    session = _session((["a", "b"], ["urban", "rural"]))
    fingerprint = joblib.hash(session.fit_result)
    result = getattr(session.fairness, operation)(sensitive_column=["group", "region"])
    records = [r for r in session.history if r["operation_id"] == canonical]
    assert len(records) == 1
    assert records[0]["parameters"]["sensitive_column"] == ["group", "region"]
    assert records[0]["result_summary"]["n_rows"] == result.n_rows
    assert joblib.hash(session.fit_result) == fingerprint
    session.checkpoint_save(tmp_path / "audited")
    restored = Session.checkpoint_load(tmp_path / "audited", trusted=True)
    stored = [r for r in restored.history if r["operation_id"] == canonical]
    assert stored == records


@pytest.mark.parametrize("bad", [True, 2.5, "7"])
def test_invalid_session_stability_controls_preserve_history_and_prior_report(bad):
    session = _session((["a", "b"], ["urban", "rural"]))
    prior = session.fairness.evaluate(sensitive_column="group")
    before = list(session._history)
    with pytest.raises(ValidationError, match="integer"):
        session.fairness.evaluate(sensitive_column="group", bootstrap_samples=bad)
    assert session._history == before and session._fairness_report is prior


@pytest.mark.parametrize("missing", [None, pd.NA])
def test_unobserved_holdout_labels_refuse_before_prediction_history_changes(missing):
    session = _session((["a", "b"], ["urban", "rural"]))
    prior = session.fairness.evaluate(sensitive_column="group")
    session.dataset.frame["y"] = session.dataset.frame.y.astype("Int64")
    session.dataset.frame.loc[47, "y"] = missing
    before = list(session._history)
    with pytest.raises(ValidationError, match="y_true.*missing"):
        session.fairness.evaluate(sensitive_column="group")
    assert session._history == before and session._fairness_report is prior


@pytest.mark.parametrize("parts", COLLISIONS)
def test_raw_intersectional_report_refuses_colliding_keys(parts):
    values = np.column_stack([np.asarray(part, dtype=object) for part in parts])
    with pytest.raises(ValidationError, match="Ambiguous.*collision"):
        evaluate_fairness([0, 1], [0, 1], values, sensitive_column=["group", "region"])


@pytest.mark.parametrize("groups", [pd.Series([None, "<NA>"], dtype=object),
                                    pd.Series([1, "1"], dtype=object)])
@pytest.mark.parametrize("operation", ["evaluate", "thresholds", "reweighing", "apply"])
def test_raw_one_attribute_consumers_do_not_pool_ambiguous_labels(groups, operation):
    with pytest.raises(ValidationError, match="Ambiguous.*collision"):
        if operation == "evaluate":
            evaluate_fairness([0, 1], [0, 1], groups)
        elif operation == "thresholds":
            suggest_group_thresholds([0, 1], [.1, .9], groups)
        elif operation == "reweighing":
            suggest_reweighing_weights([0, 1], groups)
        else:
            apply_group_thresholds([.1, .9], groups, {})
