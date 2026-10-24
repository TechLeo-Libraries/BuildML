"""Population, segment-identity and exported diagnostic consumer contracts."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.data.splits import SplitPlan
from buildml.model.diagnostics import calibration_report, segment_error_report, threshold_report
from buildml.model.supervised import FitResult


def fitted_case(task='classification', labels=None, groups=None):
    x = np.arange(12, dtype=float)
    y = np.array([0, 1] * 6) if labels is None else np.asarray(labels)
    if task == 'regression':
        y = 2 * x + 3
    frame = pd.DataFrame({'x': x, 'y': y})
    if groups:
        for key, value in groups.items():
            frame[key] = value
    session = Session.ingest(frame).set_roles({'x': 'feature', 'y': 'target'})
    split = SplitPlan('injected', None, None, None, None, tuple(range(6)),
                      tuple(range(6, 12)), ())
    estimator = LogisticRegression() if task == 'classification' else LinearRegression()
    estimator.fit(frame.iloc[:6][['x']], frame.iloc[:6]['y'])
    fit = FitResult(estimator, task, ('x',), 'y', 6)
    return session.dataset, split, fit


@pytest.mark.parametrize('report', [calibration_report, threshold_report])
@pytest.mark.parametrize('string_labels', [False, True])
def test_unknown_heldout_class_is_not_counted_as_a_negative(report, string_labels):
    labels = [0, 1] * 3 + [0, 1, 9, 0, 1, 9]
    if string_labels:
        labels = [f'class {value}' for value in labels]
    dataset, split, fit = fitted_case(labels=labels)
    with pytest.raises(ValidationError, match='class'):
        report(dataset, split, fit)


@pytest.mark.parametrize('task', ['classification', 'regression'])
@pytest.mark.parametrize('kind', ['null_literal', 'delimiter', 'quoted'])
def test_distinct_slice_values_never_merge_into_one_report_row(task, kind):
    if kind == 'null_literal':
        groups = {'a': [None, '<NA>'] * 6}
        by = ['a']
    elif kind == 'delimiter':
        groups = {'a': ['left | b=right', 'left'] * 6,
                  'b': ['end', 'right | b=end'] * 6}
        by = ['a', 'b']
    else:
        groups = {'a': ['"<NA>"', '<NA>'] * 6}
        by = ['a']
    dataset, split, fit = fitted_case(task=task, groups=groups)
    report = segment_error_report(dataset, split, fit, by=by,
                                  partition='validation', min_segment_n=1)
    rows = report.payload['segments']
    assert report.payload['n_unique_segments'] == 2
    assert len(rows) == 2
    assert sorted(row['n'] for row in rows) == [3, 3]
    # Each alternating group has one class: the rates expose accidental merging.
    if task == 'classification':
        assert sorted(row['support_positive'] for row in rows) == [0, 3]
    else:
        assert all(row['mae'] < 1e-12 for row in rows)


def test_multiclass_calibration_matches_direct_squared_probability_errors():
    labels = [0, 1, 2] * 4
    dataset, split, fit = fitted_case(labels=labels)
    report = calibration_report(dataset, split, fit)
    frame = dataset._ensure_pandas().iloc[6:]
    probabilities = fit.estimator.predict_proba(frame[['x']])
    for index, row in enumerate(report.payload['per_class_brier']):
        actual = (frame.y.to_numpy() == fit.estimator.classes_[index]).astype(float)
        assert row['brier_score'] == pytest.approx(np.square(actual - probabilities[:, index]).mean())


def test_learning_curve_linear_population_exports_real_png_and_html(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    from buildml.model.diagnostics import learning_curve_report

    x = np.arange(100, dtype=float)
    session = Session.ingest(pd.DataFrame({'x': x, 'y': 2*x+3})).set_roles(
        {'x': 'feature', 'y': 'target'}).split(test_size=.2, random_state=3)
    estimator = LinearRegression()
    report = learning_curve_report(session.dataset, session.split_plan, estimator,
        task='regression', cv=3, export_figures=tmp_path / 'figures',
        export_html=tmp_path / 'curve.html')
    np.testing.assert_allclose(report.payload['train_scores_mean'], 1, atol=1e-12)
    np.testing.assert_allclose(report.payload['valid_scores_mean'], 1, atol=1e-12)
    np.testing.assert_allclose(report.payload['valid_scores_std'], 0, atol=1e-12)
    assert report.payload['final_gap'] == pytest.approx(0, abs=1e-12)
    assert not hasattr(estimator, 'coef_')
    png = tmp_path / 'figures' / 'learning_curve.png'
    assert png.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert (tmp_path / 'curve.html').is_file()


@pytest.mark.parametrize('report', [calibration_report, threshold_report])
def test_equivalent_numeric_label_dtypes_retain_positive_class_identity(report):
    dataset, split, fit = fitted_case()
    # sklearn class codes are float while the evaluation frame retains ints.
    fit.estimator.classes_ = fit.estimator.classes_.astype(float)
    result = report(dataset, split, fit)
    frame = dataset._ensure_pandas().iloc[6:]
    probabilities = fit.estimator.predict_proba(frame[['x']])[:, 1]
    if report is calibration_report:
        assert result.payload['brier_score'] == pytest.approx(
            np.square(frame.y.to_numpy() - probabilities).mean())
        assert result.payload['prevalence'] == .5
    else:
        assert result.payload['prevalence'] == .5


def test_slice_accuracy_uses_numeric_class_identity_not_display_spelling():
    dataset, split, fit = fitted_case(groups={'site': ['east', 'west'] * 6})
    fit.estimator.classes_ = fit.estimator.classes_.astype(float)
    report = segment_error_report(dataset, split, fit, by='site',
                                  partition='validation', min_segment_n=1)
    frame = dataset._ensure_pandas().iloc[6:]
    prediction = fit.estimator.predict(frame[['x']])
    for row in report.payload['segments']:
        site = row['segment'].split('=', 1)[1]
        mask = (frame.site == site).to_numpy()
        assert row['accuracy'] == pytest.approx(np.mean(frame.y.to_numpy()[mask] == prediction[mask]))
        assert row['support_positive'] == int((frame.y.to_numpy()[mask] == 1).sum())
