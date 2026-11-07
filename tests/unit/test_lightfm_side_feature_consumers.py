"""Real sparse side-feature matrices match independent entity-level arithmetic."""
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from buildml.core.errors import ValidationError
from buildml.recommenders.adapters.lightfm import _feature_csr


def test_entity_aggregation_order_constant_scale_and_sparse_consumer():
    frame=pd.DataFrame({'user':['b','a','b','c'],'value':[4.,2.,8.,12.],'constant':[5.]*4})
    before=frame.copy(deep=True)
    matrix=_feature_csr(frame,'user',('a','b','c'),['value','constant'])
    values=np.array([2.,6.,12.])
    mean=sum(values)/3
    scale=np.sqrt(sum((v-mean)**2 for v in values)/3)
    expected=np.column_stack([(values-mean)/scale,np.zeros(3)])
    assert isinstance(matrix,csr_matrix)
    assert matrix.dtype==np.float32
    np.testing.assert_allclose(matrix.toarray(),expected,atol=1e-6)
    np.testing.assert_allclose(matrix@np.array([2.,3.]),expected@np.array([2.,3.]),atol=1e-6)
    pd.testing.assert_frame_equal(frame,before)


@pytest.mark.parametrize('value',[np.nan,np.inf,-np.inf,1+2j])
def test_invalid_side_features_are_refused(value):
    frame=pd.DataFrame({'user':['a','b'],'value':[value,2.]})
    with pytest.raises(ValidationError,match='finite real'):
        _feature_csr(frame,'user',('a','b'),['value'])


def test_nullable_missing_side_feature_is_refused():
    frame=pd.DataFrame({'user':['a','b'],'value':pd.Series([1,pd.NA],dtype='Float64')})
    with pytest.raises(ValidationError,match='finite real'):
        _feature_csr(frame,'user',('a','b'),['value'])


def test_no_features_and_empty_catalog_shapes():
    frame=pd.DataFrame({'user':['a'],'value':[2.]})
    assert _feature_csr(frame,'user',('a',),None) is None
    matrix=_feature_csr(frame,'user',(),['value'])
    assert matrix.shape==(0,1)


@pytest.mark.parametrize('columns',[['missing'],['value']])
def test_missing_or_non_numeric_features_fail(columns):
    frame=pd.DataFrame({'user':['a'],'value':['text']})
    with pytest.raises(ValidationError):
        _feature_csr(frame,'user',('a',),columns)
