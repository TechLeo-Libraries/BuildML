"""Cost-matrix decisions match independent probability-weighted and realized loss."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from buildml import Session


@pytest.mark.parametrize('permutation',[(0,1,2),(2,1,0),(1,2,0)])
def test_multiclass_cost_accounting_and_axis_reordering(permutation):
    rng=np.random.default_rng(84)
    x=rng.normal(size=(180,3))
    labels=np.array(['blue','green','red'])
    target=labels[np.argmax(x+rng.normal(scale=1.2,size=x.shape),axis=1)]
    frame=pd.DataFrame(x,columns=['a','b','c'])
    frame['target']=target
    s=Session.ingest(frame).set_roles({'a':'feature','b':'feature','c':'feature','target':'target'}).split(test_size=.25,validation_size=.25,random_state=4)
    s.fit(LogisticRegression(max_iter=300),task='classification')
    costs={'blue':{'blue':0.,'green':7.,'red':2.},'green':{'blue':3.,'green':0.,'red':9.},'red':{'blue':8.,'green':1.,'red':0.}}
    classes=list(s.fit_result.estimator.classes_)
    supplied=[classes[i] for i in permutation]
    matrix=[[costs[y][a] for a in supplied] for y in supplied]
    fit=s.decision.fit(method='cost_matrix',partition='validation',cost_matrix=matrix,class_labels=supplied)
    def oracle(partition):
        rows=s.partition(partition)
        probabilities=s.fit_result.estimator.predict_proba(rows[['a','b','c']])
        losses=np.array([[sum(float(p)*costs[y][a] for y,p in zip(classes,row,strict=True)) for a in classes] for row in probabilities])
        actions=[classes[i] for i in losses.argmin(axis=1)]
        actual=sum(costs[y][a] for y,a in zip(rows.target,actions,strict=True))
        return rows,losses,actions,actual
    rows,losses,actions,realized=oracle('test')
    applied=s.decision.apply(partition='test')
    assert list(applied.decisions)==actions
    np.testing.assert_allclose(applied.scores,losses.min(axis=1),atol=1e-12)
    evaluated=s.decision.evaluate(partition='test')
    assert evaluated.realized_cost==pytest.approx(realized)
    assert evaluated.metrics['realized_cost_mean']==pytest.approx(realized/len(rows))
    assert evaluated.metrics['accuracy']==pytest.approx(np.mean(rows.target.to_numpy()==actions))
    _,validation_losses,_,validation_realized=oracle('validation')
    assert fit.metrics['realized_cost_total']==pytest.approx(validation_realized)
    assert fit.metrics['expected_cost_mean_under_proba']==pytest.approx(validation_losses.min(axis=1).mean())
    assert abs(validation_losses.min(axis=1).sum()-validation_realized)>1
    assert fit.expected_cost==pytest.approx(validation_losses.min(axis=1).sum())
