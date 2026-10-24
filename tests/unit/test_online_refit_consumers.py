"""Independent batch-model oracles for disclosed online refit fallback."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.online.fit import fit_online
from buildml.online.update import partial_fit_online


@pytest.fixture(params=['regression', 'classification'])
def legacy(request):
    rng = np.random.default_rng(19)
    x = rng.normal(size=(90,2))
    y = 2*x[:,0]-x[:,1]
    if request.param == 'classification':
        y = (y > 0).astype(int)
    s = Session.ingest(pd.DataFrame({'a':x[:,0],'b':x[:,1],'y':y})).set_roles({'a':'feature','b':'feature','y':'target'}).split(test_size=.2,validation_size=.2,random_state=5)
    p,_ = fit_online(s.dataset,s.split_plan,estimator='sgd_regressor' if request.param == 'regression' else 'sgd_classifier',task=request.param,n_init=12,chunk_size=8,allow_refit_fallback=True)
    model = Ridge(alpha=.3) if request.param == 'regression' else LogisticRegression(C=.7,random_state=3)
    rows=s.to_pandas().loc[list(p.seen_train_indices)]
    model.fit(rows[['a','b']].to_numpy(),rows.y.to_numpy())
    return s, replace(p,estimator_name='legacy_batch_model',estimator_=model)


@pytest.mark.parametrize('overlap',[False,True])
def test_cumulative_train_refit_matches_independent_fit(legacy,overlap):
    s,p = legacy
    original_predictions=p.estimator_.predict(s.to_pandas()[['a','b']].to_numpy()).copy()
    indices=list(s.split_plan.train_indices)[8:22] if overlap else list(s.split_plan.train_indices)[12:26]
    updated,result=partial_fit_online(s.dataset,p,s.split_plan,indices=indices)
    seen=list(dict.fromkeys([*p.seen_train_indices,*indices]))
    rows=s.to_pandas().loc[seen]
    expected=clone(p.estimator_).fit(rows[['a','b']].to_numpy(),rows.y.to_numpy())
    probe=s.to_pandas()[['a','b']].to_numpy()
    np.testing.assert_allclose(updated.estimator_.predict(probe),expected.predict(probe),atol=1e-12)
    np.testing.assert_allclose(p.estimator_.predict(probe),original_predictions,atol=1e-12)
    assert result.n_seen_rows == len(seen)
    assert result.used_refit_fallback and result.update_mode == 'refit_fallback'
    assert any('not incremental' in x for x in result.disclosures)
    assert set(updated.seen_train_indices).isdisjoint(s.split_plan.test_indices)


@pytest.mark.parametrize('split_present',[False,True])
def test_external_refit_refused_before_losing_stream_history(legacy,split_present):
    s,p=legacy
    external=pd.DataFrame({'a':[100.,101.],'b':[50.,51.],'y':[0,1]})
    before=p.estimator_.predict(external[['a','b']].to_numpy()).copy()
    with pytest.raises(ValidationError,match='external.*refit|refit.*external'):
        partial_fit_online(s.dataset,p,s.split_plan if split_present else None,frame=external)
    np.testing.assert_array_equal(p.estimator_.predict(external[['a','b']].to_numpy()),before)
    assert p.n_updates == 0
