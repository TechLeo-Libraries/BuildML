"""Diagnostic boundaries and stationarity interpretation with real numerical consumers."""

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.timeseries.diagnostics import compute_diagnostics


@pytest.mark.parametrize('argument', ['acf_lags', 'pacf_lags'])
@pytest.mark.parametrize('value', [True, np.bool_(False), 2.5, '3', 0, -1, np.nan, np.inf])
def test_lags_require_positive_integers(argument, value):
    with pytest.raises(ValidationError, match=argument):
        compute_diagnostics(np.arange(32.), **{argument: value})


@pytest.mark.parametrize('values', [np.ones((4, 8)), np.arange(32.) + 1j,
                                  [0., 1., np.nan, 3.], [0., 1., np.inf, 3.],
                                  [0., 1., -np.inf, 3.], ['bad'] * 8, np.ones(16)])
def test_invalid_series_are_refused_without_fabricated_statistics(values):
    with pytest.raises(ValidationError, match='series|Series'):
        compute_diagnostics(values)


@pytest.mark.parametrize('argument', ['adf_regression', 'kpss_regression'])
@pytest.mark.parametrize('value', ['invalid', '', None])
def test_invalid_deterministic_terms_are_not_swallowed_as_numerical_failures(argument, value):
    with pytest.raises(ValidationError, match=argument):
        compute_diagnostics(np.arange(32.), **{argument: value})


def test_short_window_retains_correlations_when_adf_is_unavailable():
    pytest.importorskip('statsmodels')
    result = compute_diagnostics(np.array([0., 1., 3., 2.]),
                                 acf_lags=20, pacf_lags=20, adf_regression='ctt')
    assert result.acf_lags == 2 and result.pacf_lags == 1
    assert np.isfinite(result.acf_values).all()
    assert np.isfinite(result.pacf_values).all()
    assert result.adf_statistic is None and result.adf_pvalue is None
    assert result.adf_used_lags is None and result.adf_critical_values == {}
    assert any('ADF test failed' in warning for warning in result.warnings)


def test_too_short_window_is_refused_before_backend_work():
    with pytest.raises(ValidationError, match='at least 4 points'):
        compute_diagnostics(np.array([0., 1., 2.]))


@pytest.mark.parametrize('adf_term', ['n', 'c', 'ct', 'ctt'])
@pytest.mark.parametrize('kpss_term', ['c', 'ct'])
def test_stationarity_terms_and_correlations_match_real_consumers(adf_term, kpss_term):
    pytest.importorskip('statsmodels')
    from statsmodels.tsa.stattools import adfuller, kpss

    rng = np.random.default_rng(99)
    y = rng.normal(size=160) + .015 * np.arange(160)
    for i in range(1, len(y)):
        y[i] += .3 * y[i - 1]
    result = compute_diagnostics(y, acf_lags=np.int64(8), pacf_lags=np.int64(5),
                                 adf_regression=adf_term, kpss_regression=kpss_term)
    # Dense covariance and linear solves do not call the adapter's ACF/PACF backend.
    centered = y - y.mean()
    covariance = np.array([centered[k:] @ centered[:len(y)-k] for k in range(9)])
    np.testing.assert_allclose(result.acf_values, covariance / covariance[0], atol=1e-12)
    expected_pacf = [1.]
    for order in range(1, 6):
        positions = np.arange(order)
        expected_pacf.append(np.linalg.solve(
            covariance[np.abs(positions[:, None]-positions[None, :])],
            covariance[1:order+1])[-1])
    np.testing.assert_allclose(result.pacf_values, expected_pacf, atol=1e-12)
    expected_adf = adfuller(y, regression=adf_term, autolag='AIC')
    expected_kpss = kpss(y, regression=kpss_term, nlags='auto')
    assert result.adf_statistic == pytest.approx(expected_adf[0])
    assert result.adf_pvalue == pytest.approx(expected_adf[1])
    assert result.adf_used_lags == expected_adf[2]
    assert result.adf_critical_values == pytest.approx(expected_adf[4])
    assert result.kpss_statistic == pytest.approx(expected_kpss[0])
    assert result.kpss_pvalue == pytest.approx(expected_kpss[1])
    assert result.kpss_critical_values == pytest.approx(expected_kpss[3])
    disclosure = ' '.join(result.disclosures)
    assert f'ADF regression={adf_term}' in disclosure
    assert f'KPSS regression={kpss_term}' in disclosure
    assert ('level stationarity' if kpss_term == 'c' else 'trend stationarity') in disclosure


def test_session_diagnostics_use_training_population_and_preserve_input():
    pytest.importorskip('statsmodels')
    rng = np.random.default_rng(8)
    frame = pd.DataFrame({'clock': pd.date_range('2025-01-01', periods=100),
                          'value': rng.normal(size=100)})

    def analyze(data):
        session = (Session.ingest(data).set_roles({'clock': 'time', 'value': 'target'})
                   .time_split(test_size=.2, validation_size=.2))
        result = session.timeseries.diagnostics(acf_lags=5, pacf_lags=4)
        pd.testing.assert_frame_equal(session.to_pandas(), data)
        return session, result.diagnostics

    session, original = analyze(frame)
    changed = frame.copy()
    heldout = [*session.split_plan.validation_indices, *session.split_plan.test_indices]
    changed.loc[heldout, 'value'] += 10000
    _, perturbed = analyze(changed)
    assert original.n_points == len(session.split_plan.train_indices)
    assert original.to_dict() == perturbed.to_dict()
    np.testing.assert_allclose(original.acf_values, perturbed.acf_values)
    np.testing.assert_allclose(original.pacf_values, perturbed.pacf_values)
