"""Synthetic quality checks use explicit invalid-value and frequency oracles."""
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.synthetic.validation import validate_synthetic


@pytest.fixture
def plan():
    frame=pd.DataFrame({'x':np.linspace(0,10,100),'count':np.tile(np.arange(10),10),'cat':['a','b']*50})
    s=Session.ingest(frame).set_roles({'x':'feature','count':'feature','cat':'feature'}).split(test_size=.2,validation_size=.2,random_state=4)
    s.synthetic.fit(method='bootstrap',random_state=2)
    return s.synthesizer_plan


def sample():
    return pd.DataFrame({'x':[2.0]*20,'count':[2]*20,'cat':['a']*20})


@pytest.mark.parametrize('column,value',[('x','broken'),('x',np.inf),('x',-np.inf),('count',2.5)])
def test_invalid_numeric_sample_cannot_pass(plan,column,value):
    frame=sample().astype({'x':object,'count':object})
    frame.loc[0,column]=value
    result=validate_synthetic(plan,frame,run_great_expectations=False)
    assert not result.passed
    assert result.n_failed == sum(not v for v in result.checks.values())
    assert any(not v for k,v in result.checks.items() if column in k)
    assert all(np.isfinite(v) for v in result.metrics.values())


@pytest.mark.parametrize('name',['null_rate_tolerance','numeric_range_slack','categorical_oov_tolerance'])
@pytest.mark.parametrize('value',[np.nan,np.inf,-.1,True,'0.1'])
def test_invalid_tolerance_controls(plan,name,value):
    with pytest.raises(ValidationError,match=name):
        validate_synthetic(plan,sample(),run_great_expectations=False,**{name:value})


@pytest.mark.parametrize('name',['null_rate_tolerance','categorical_oov_tolerance'])
def test_probability_tolerances_bounded_by_one(plan,name):
    with pytest.raises(ValidationError,match=name):
        validate_synthetic(plan,sample(),run_great_expectations=False,**{name:1.1})


def test_frequency_metrics_match_independent_counts(plan):
    frame=sample()
    frame.loc[:3,'cat']='outside'
    frame.loc[:1,'x']=100
    frame.loc[0,'count']=np.nan
    result=validate_synthetic(plan,frame,run_great_expectations=False)
    assert result.metrics['oov_rate::cat'] == pytest.approx(4/20)
    assert result.metrics['oor_rate::x'] == pytest.approx(2/20)
    assert result.metrics['null_rate::count'] == pytest.approx(1/20)
    assert not result.checks['categorical_vocab::cat']
    assert not result.checks['numeric_range::x']
    assert result.n_failed == sum(not v for v in result.checks.values())
    assert any('not a privacy' in x for x in result.disclosures)


def test_duplicate_columns_refused(plan):
    frame=sample()
    frame=pd.concat([frame,frame[['x']]],axis=1)
    with pytest.raises(ValidationError,match='duplicate'):
        validate_synthetic(plan,frame,run_great_expectations=False)


def test_complex_values_fail_without_nonfinite_metrics(plan):
    frame=sample()
    frame['x']=np.full(len(frame),2+3j)
    result=validate_synthetic(plan,frame,run_great_expectations=False)
    assert not result.passed
    assert not result.checks['numeric_values::x']
    assert all(np.isfinite(v) for v in result.metrics.values())


def test_nullable_numeric_values_and_numpy_controls(plan):
    frame=sample().astype({'x':'Float64','count':'Int64'})
    frame.loc[0,['x','count']]=pd.NA
    result=validate_synthetic(plan,frame,run_great_expectations=False,
                              null_rate_tolerance=np.float64(.1),
                              numeric_range_slack=0,categorical_oov_tolerance=0)
    assert result.passed
    assert result.metrics['null_rate::x'] == pytest.approx(1/20)
