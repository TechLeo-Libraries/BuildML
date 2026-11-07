"""Frozen SSL heads preserve raw class identity and labeled-only metrics."""
import copy
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.selfsupervised.evaluate import evaluate_ssl
from buildml.selfsupervised.finetune import finetune_ssl_head
from buildml.selfsupervised.fit import fit_ssl_pretext


@pytest.fixture(scope='module')
def pretext():
    pytest.importorskip('torch')
    rng=np.random.default_rng(28)
    x=rng.normal(size=(80,2))
    frame=pd.DataFrame({'a':x[:,0],'b':x[:,1],'target':np.where(x[:,0]>0,1,2)})
    s=Session.ingest(frame).set_roles({'a':'feature','b':'feature','target':'target'}).split(test_size=.25,random_state=3)
    p,_=fit_ssl_pretext(s.dataset,s.split_plan,method='mae_tabular',epochs=2,latent_dim=3,hidden=(8,),batch_size=16)
    return s,p


@pytest.mark.parametrize('labels',[(1,2),('001','1.00')])
def test_raw_class_identity_and_independent_accuracy(pretext,labels):
    s,p=pretext
    original=s.to_pandas().copy()
    frame=original.copy()
    frame['target']=pd.Series([labels[int(v)-1] for v in original.target],dtype=object)
    data=Session.ingest(frame).set_roles({'a':'feature','b':'feature','target':'target'}).dataset
    head,fit=finetune_ssl_head(data,s.split_plan,p)
    assert set(fit.classes)==set(labels)
    json.dumps(fit.to_dict(),allow_nan=False)
    json.dumps(head.to_dict(),allow_nan=False)
    if isinstance(labels[0],int):
        # Integer/float representations refer to the same numeric classes.
        data._ensure_pandas().loc[list(s.split_plan.test_indices),'target']=frame.iloc[list(s.split_plan.test_indices)].target.astype(float).to_numpy()
    part=data._ensure_pandas().iloc[list(s.split_plan.test_indices)]
    embeddings=p.encoder_.transform(part[['a','b']].to_numpy())
    encoded=head.estimator_.predict(embeddings)
    predicted=head.estimator_._buildml_label_encoder_.inverse_transform(encoded)
    expected=float(np.mean(part.target.to_numpy()==predicted))
    result=evaluate_ssl(data,p,head,s.split_plan,partition='test')
    assert result.metrics['accuracy']==pytest.approx(expected)
    assert result.n_labeled_eval==len(part)
    assert result.n_unlabeled_eval==0


def test_mixed_text_numeric_classes_cannot_silently_collapse(pretext):
    s,p=pretext
    frame=s.to_pandas().copy()
    frame['target']=pd.Series([1,'1',2,2]*(len(frame)//4),dtype=object)
    data=Session.ingest(frame).set_roles({'a':'feature','b':'feature','target':'target'}).dataset
    with pytest.raises(ValidationError,match='class labels'):
        finetune_ssl_head(data,s.split_plan,p)


def test_saved_legacy_string_head_remains_evaluable(pretext,tmp_path):
    from sklearn.preprocessing import LabelEncoder
    s,p=pretext
    head,_=finetune_ssl_head(s.dataset,s.split_plan,p)
    legacy=copy.deepcopy(head)
    legacy.estimator_._buildml_label_encoder_=LabelEncoder().fit(['1','2'])
    del legacy.estimator_._buildml_raw_class_labels_
    legacy.classes_=('1','2')
    path=tmp_path/'legacy.joblib'
    joblib.dump(legacy,path)
    restored=joblib.load(path)
    result=evaluate_ssl(s.dataset,p,restored,s.split_plan,partition='test')
    frame=s.partition('test')
    codes=restored.estimator_.predict(p.encoder_.transform(frame[['a','b']].to_numpy()))
    labels=restored.estimator_._buildml_label_encoder_.inverse_transform(codes)
    assert result.metrics['accuracy']==pytest.approx(np.mean(frame.target.astype(str).to_numpy()==labels))
    assert any('Legacy' in w for w in result.warnings)


def test_unlabeled_only_eval_has_no_invented_metrics(pretext):
    s,p=pretext
    head,_=finetune_ssl_head(s.dataset,s.split_plan,p)
    frame=s.to_pandas().copy()
    frame['target']=frame.target.astype(float)
    frame.loc[list(s.split_plan.test_indices),'target']=np.nan
    data=Session.ingest(frame).set_roles({'a':'feature','b':'feature','target':'target'}).dataset
    result=evaluate_ssl(data,p,head,s.split_plan,partition='test')
    assert result.n_labeled_eval==0
    assert result.n_unlabeled_eval==len(s.split_plan.test_indices)
    assert result.metrics=={}


def test_unseen_raw_class_is_not_coerced_to_known_text(pretext):
    s,p=pretext
    head,_=finetune_ssl_head(s.dataset,s.split_plan,p)
    frame=s.to_pandas().copy()
    frame['target']=frame.target.astype(object)
    frame.loc[list(s.split_plan.test_indices),'target']='1'
    data=Session.ingest(frame).set_roles({'a':'feature','b':'feature','target':'target'}).dataset
    with pytest.raises(ValidationError,match='class labels'):
        evaluate_ssl(data,p,head,s.split_plan,partition='test')

