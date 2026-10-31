"""Public Session attributions match independently selected partition rows."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from buildml import Session
from buildml.core.errors import ValidationError


@pytest.fixture(params=['linear','tree'])
def trained(request):
    pytest.importorskip('shap')
    rng=np.random.default_rng(119)
    frame=pd.DataFrame(rng.normal(size=(90,2)),columns=['signal','noise'],index=['duplicate']*90)
    frame['target']=4*frame.signal+.25*frame.noise+3
    s=Session.ingest(frame).set_roles({'signal':'feature','noise':'feature','target':'target'}).split(test_size=.2,validation_size=.2,random_state=13)
    model=LinearRegression() if request.param=='linear' else RandomForestRegressor(n_estimators=8,max_depth=3,random_state=7)
    s.fit(model)
    return s,request.param


@pytest.mark.parametrize('partition',['train','validation','test'])
@pytest.mark.parametrize('budget',[7,1000])
def test_session_explanations_match_partition_oracle(trained,partition,budget):
    shap=pytest.importorskip('shap')
    s,kind=trained
    before=s.to_pandas().copy(deep=True)
    result=s.explain_shap(partition=partition,max_samples=budget,random_state=23)
    positions=getattr(s.split_plan,partition+'_indices')
    frame=before.iloc[list(positions)][['signal','noise']]
    selected=frame.sample(n=budget,random_state=23) if len(frame)>budget else frame
    model=s.fit_result.estimator
    if kind=='linear':
        expected=(selected-selected.mean()).abs().mean()*pd.Series([4.,.25],index=selected.columns)
        assert result.backend=='explainer'
    else:
        values=shap.TreeExplainer(model)(selected.to_numpy())
        np.testing.assert_allclose(values.base_values+values.values.sum(axis=1),model.predict(selected),atol=1e-6)
        expected=pd.Series(np.abs(values.values).mean(axis=0),index=selected.columns)
        assert result.backend=='tree_explainer'
    assert result.feature_names==('signal','noise')
    assert result.n_rows==min(budget,len(frame))
    for name in expected.index:
        assert result.mean_abs_shap[name]==pytest.approx(expected[name],rel=1e-6)
    assert any('not causal' in x for x in result.disclosures)
    json.dumps(result.to_dict(),allow_nan=False)
    pd.testing.assert_frame_equal(s.to_pandas(),before)


def test_unknown_partition_refused(trained):
    s,_=trained
    with pytest.raises(ValidationError,match='Unknown partition'):
        s.explain_shap(partition='holdout')
