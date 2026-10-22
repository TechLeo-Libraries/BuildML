"""Real changepoint consumers respect the documented penalty and numeric scope."""
import numpy as np
import pytest

from buildml.core.errors import ValidationError
from buildml.timeseries.changepoints import detect_changepoints


@pytest.mark.parametrize("method", ["pelt", "binseg"])
@pytest.mark.parametrize("penalty", [0.2, 5., 1000.])
def test_penalized_breaks_and_segment_means_match_real_ruptures(method, penalty):
    rpt = pytest.importorskip("ruptures")
    rng = np.random.default_rng(31)
    series = np.repeat([0., 3., -2., 1.], 20) + rng.normal(0, 0.05, 80)
    algorithm = (rpt.Pelt(model="rbf", min_size=2, jump=1) if method == "pelt"
                 else rpt.Binseg(model="l2", min_size=2, jump=1))
    expected = tuple(algorithm.fit(series.reshape(-1, 1)).predict(pen=penalty)[:-1])
    result = detect_changepoints(series, method=method, penalty=penalty, target_column="signal")
    assert result.changepoint_indices == expected
    bounds = [0, *expected, len(series)]
    expected_means = [sum(series[start:end]) / (end - start) for start, end in zip(bounds, bounds[1:])]
    np.testing.assert_allclose(result.segment_means, expected_means, atol=1e-12)
    assert result.n_segments == len(expected) + 1
    assert result.target_column == "signal"
    if penalty == 1000:
        assert expected == ()


@pytest.mark.parametrize("penalty", [float("nan"), float("inf"), float("-inf"), -1., True, "3", None,
                                     pytest.param(10 ** 400, id="overflowing_real")])
def test_invalid_penalties_refused(penalty):
    with pytest.raises(ValidationError, match="penalty"):
        detect_changepoints(np.arange(12, dtype=float), method="cusum", penalty=penalty)


@pytest.mark.parametrize("series", [
    [0., 1., 2., float("nan")], [0., 1., 2., float("inf")],
    [0., 1., 2., float("-inf")], np.arange(8).reshape(4, 2),
    np.array([0., 1., 2., 3.], dtype=complex) + 1j, ["x"] * 4,
])
def test_invalid_series_refused(series):
    with pytest.raises(ValidationError, match="series"):
        detect_changepoints(series, method="cusum")


def test_cusum_constant_and_high_threshold_have_no_breaks():
    for series in [np.zeros(20), np.repeat([0., 4.], 10)]:
        result = detect_changepoints(series, method="cusum", penalty=1e9)
        assert result.changepoint_indices == ()
        assert result.segment_means == (float(sum(series) / len(series)),)
