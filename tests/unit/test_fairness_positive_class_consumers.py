"""Fairness consumers must score the caller's positive class, not a column default."""
import itertools

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from buildml import Session
from buildml.core.errors import ValidationError


@pytest.fixture(scope='module', params=list(itertools.product(['strings','integers','booleans'],['logistic','pipeline','svc'])))
def fitted(request):
    labels, kind = request.param
    rng = np.random.default_rng(171)
    x = rng.normal(size=(160,2))
    positive = x[:,0]-.5*x[:,1]+rng.normal(0,.7,160)>0
    y = {'strings':np.where(positive,'warm','cold'),'integers':np.where(positive,7,2),'booleans':positive}[labels]
    frame = pd.DataFrame({'x':x[:,0],'z':x[:,1],'target':y,'group':np.where(np.arange(160)%2,'east','west')})
    s = Session.ingest(frame).set_roles({'x':'feature','z':'feature','target':'target','group':'ignore'})
    s.inject_split(train_indices=list(range(100)),validation_indices=list(range(100,130)),test_indices=list(range(130,160)))
    model = LinearSVC(C=.6) if kind=='svc' else LogisticRegression(C=.6)
    if kind=='pipeline':
        model = Pipeline([('model',model)])
    return s.fit(model,task='classification'),kind


@pytest.mark.parametrize('positive_index',[0,1])
@pytest.mark.parametrize('partition',['validation','test'])
def test_public_group_auc_and_thresholds_use_declared_class(fitted,positive_index,partition):
    s,kind = fitted
    fingerprint = joblib.hash(s.fit_result)
    before = s.to_pandas().copy(deep=True)
    estimator = s.fit_result.estimator
    model = estimator.named_steps['model'] if kind=='pipeline' else estimator
    positive = model.classes_[positive_index]
    rows = s.partition(partition)
    logits = (rows[['x','z']].to_numpy() @ model.coef_.T + model.intercept_)[:,0]
    predicted = model.classes_[(logits>0).astype(int)]
    scores = 1/(1+np.exp(-logits))
    if positive_index==0:
        scores = 1-scores
    report = s.fairness.evaluate(sensitive_column='group',partition=partition,positive_label=positive)
    truth = rows.target.to_numpy()==positive
    for group in ['east','west']:
        mask = rows.group.to_numpy()==group
        group_truth = truth[mask]
        group_pred = predicted[mask]==positive
        assert report.selection_rate_by_group[group] == pytest.approx(group_pred.mean())
        if kind=='svc':
            assert report.classical_metrics_by_group[group]['roc_auc'] is None
        else:
            pos_scores,neg_scores = scores[mask][group_truth],scores[mask][~group_truth]
            expected = np.mean([(a>b)+.5*(a==b) for a in pos_scores for b in neg_scores])
            assert report.classical_metrics_by_group[group]['roc_auc'] == pytest.approx(expected)
    if kind=='svc':
        with pytest.raises(ValidationError,match='predict_proba'):
            s.fairness.suggest_thresholds(sensitive_column='group',partition=partition,positive_label=positive)
    else:
        for target in ['demographic_parity','equal_opportunity']:
            eligible = np.ones(len(truth),dtype=bool) if target=='demographic_parity' else truth
            reference = np.mean(scores[eligible]>=.5)
            expected_thresholds = {}
            for group in ['east','west']:
                mask = (rows.group.to_numpy()==group)&eligible
                expected_thresholds[group] = min(np.linspace(0,1,17),key=lambda t:(abs(np.mean(scores[mask]>=t)-reference),t))
            suggestion = s.fairness.suggest_thresholds(sensitive_column='group',partition=partition,
                positive_label=positive,target=target,grid_size=17)
            assert suggestion.thresholds_by_group == pytest.approx(expected_thresholds)
    assert joblib.hash(s.fit_result)==fingerprint
    pd.testing.assert_frame_equal(s.to_pandas(),before)
