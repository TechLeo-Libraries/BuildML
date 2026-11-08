"""Dashboard gates preserve measured zeros instead of inventing fallback values."""
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.dashboard.gates import build_gate_context, build_gates_payload


@pytest.fixture
def report():
    n=40
    frame=pd.DataFrame({'a':np.where(np.arange(n)%2==0,np.arange(n),np.nan),
                        'b':np.where(np.arange(n)%2!=0,np.arange(n),np.nan),
                        'y':np.linspace(-19.5,19.5,n)})
    s=Session.ingest(frame).set_roles({'a':'feature','b':'feature','y':'target'})
    result=s.eda(include_plots=False,show=False).to_dict()
    return result,frame


def test_complete_case_zero_matches_independent_frame_count(report):
    data,frame=report
    expected=int(frame[['a','b']].notna().all(axis=1).sum())
    assert expected==0
    data.setdefault('multivariate',{})['complete_case_rows']=expected
    context=build_gate_context(data)
    assert context['completeRows']==expected
    payload=build_gates_payload(data)
    assert payload['counts']['total']>0


def test_zero_regression_median_is_not_replaced_by_legacy_alias(report):
    data,frame=report
    data['target']={'column':'y','summary':{'type':'regression','median':float(frame.y.median()),'p50':99}}
    assert frame.y.median()==0
    assert build_gate_context(data)['target']['stats']['median']==0


@pytest.mark.parametrize('scored',[0,40])
def test_zero_flagged_anomalies_are_a_real_measurement(report,scored):
    data,_=report
    data['outliers']={'multivariate':{'flagged_count':0,'n_scored':scored,'contamination':0}}
    actual=build_gate_context(data)['anomalies']
    assert actual is not None
    assert actual['flagged']==0
    assert actual['scored']==scored
    assert actual['contamination']==0


def test_missing_values_still_use_legacy_aliases(report):
    data,_=report
    data['target']={'column':'y','summary':{'type':'regression','median':None,'p50':4}}
    data['outliers']={'multivariate':{'flagged_count':None,'n_flagged':2,'n_scored':None,'scored':30}}
    actual=build_gate_context(data)
    assert actual['target']['stats']['median']==4
    assert actual['anomalies']['flagged']==2
    assert actual['anomalies']['scored']==30


def test_zero_analysis_rows_is_not_full_report_scope(report):
    data,_=report
    data['overview']['analysis_rows']=0
    actual=build_gate_context(data)
    assert actual['rows']==0
    assert actual['rowsTotal']==40
    assert actual['sampled'] is True
