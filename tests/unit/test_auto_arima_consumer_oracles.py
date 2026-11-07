"""Real statsmodels selection/forecasts distinguish searched and supplied orders."""
import warnings
from itertools import product

import numpy as np
import pytest

from buildml.forecasting.backends import fit_industry_backend


def series():
    rng=np.random.default_rng(92)
    y=np.zeros(36)
    for i in range(1,len(y)):
        y[i]=.65*y[i-1]+rng.normal(scale=.4)
    return y


def test_auto_arima_matches_independent_finite_aic_grid():
    pytest.importorskip('statsmodels')
    from statsmodels.tsa.arima.model import ARIMA
    y=series()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        candidates=[]
        for order in product(range(3),range(2),range(3)):
            if order==(0,0,0):
                continue
            try:
                fitted=ARIMA(y,order=order).fit()
            except Exception:
                continue
            if np.isfinite(fitted.aic):
                candidates.append((fitted.aic,order))
        expected=min(candidates,key=lambda v:v[0])[1]
        result=fit_industry_backend(y,method='auto_arima')
        direct=ARIMA(y,order=expected).fit()
    assert result.extra['order']==expected
    assert any('AIC grid search' in x for x in result.disclosures)
    np.testing.assert_allclose(result.estimator['result'].forecast(5),direct.forecast(5),atol=1e-10)


@pytest.mark.parametrize('order',[(1,0,0),(0,1,1)])
def test_explicit_order_is_not_described_as_searched(order):
    pytest.importorskip('statsmodels')
    from statsmodels.tsa.arima.model import ARIMA
    y=series()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result=fit_industry_backend(y,method='auto_arima',order=order)
        direct=ARIMA(y,order=order).fit()
    assert result.extra['order']==order
    assert any('explicit' in x.lower() for x in result.disclosures)
    assert not any('selected' in x.lower() and 'AIC' in x for x in result.disclosures)
    np.testing.assert_allclose(result.estimator['result'].forecast(5),direct.forecast(5),atol=1e-10)
