"""Core PACF matches independent dense Yule-Walker solutions."""
import numpy as np
import pytest

from buildml.timeseries import diagnostics


def independent_correlations(series, lags):
    centered = np.asarray(series, dtype=float) - np.mean(series)
    covariance = np.array([sum(centered[t] * centered[t - lag] for t in range(lag, len(series)))
                           for lag in range(lags + 1)])
    acf = covariance / covariance[0]
    pacf = [1.0]
    for order in range(1, lags + 1):
        indices = np.arange(order)
        matrix = covariance[np.abs(indices[:, None] - indices[None, :])]
        coefficients = np.linalg.solve(matrix, covariance[1:order + 1])
        pacf.append(coefficients[-1])
    return acf, np.asarray(pacf)


@pytest.mark.parametrize("seed", [4, 29])
@pytest.mark.parametrize("lags", [3, 8])
def test_core_diagnostics_matches_dense_yule_walker(monkeypatch, seed, lags):
    rng = np.random.default_rng(seed)
    series = rng.normal(size=128)
    for t in range(2, len(series)):
        series[t] += 0.6 * series[t - 1] - 0.3 * series[t - 2]
    expected_acf, expected_pacf = independent_correlations(series, lags)
    # Exercise the actual core backend while retaining the installed host runtime.
    monkeypatch.setattr(diagnostics, "statsmodels_available", lambda: False)
    result = diagnostics.compute_diagnostics(series, acf_lags=lags, pacf_lags=lags)
    np.testing.assert_allclose(result.acf_values, expected_acf, atol=1e-12)
    np.testing.assert_allclose(result.pacf_values, expected_pacf, atol=1e-12)
    assert result.acf_confint == result.pacf_confint == ()
    assert result.adf_statistic is None and result.kpss_statistic is None
    assert any("numpy" in warning for warning in result.warnings)
    # Correlations are invariant to translation and positive rescaling.
    shifted = diagnostics.compute_diagnostics(7 * series + 18, acf_lags=lags, pacf_lags=lags)
    np.testing.assert_allclose(shifted.pacf_values, expected_pacf, atol=1e-12)


def test_zero_lag_pacf_returns_identity():
    np.testing.assert_array_equal(diagnostics._numpy_pacf(np.arange(12, dtype=float), 0), [1.])
