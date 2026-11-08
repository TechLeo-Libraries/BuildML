"""Real maintained SkopeRules exports preserve every learned predicate."""
import numpy as np
import pandas as pd
import pytest

from buildml.core.errors import ValidationError
from buildml.symbolic.adapters.skope_rules import _parse_skope_rule_string, induce_skope_rules
from buildml.symbolic.rules import fire_rules


@pytest.mark.parametrize('seed',[2,11])
@pytest.mark.parametrize('columns',[['annual income','rate and fee'],['c1','c0']])
def test_real_imodels_rules_match_independent_query_masks(seed,columns):
    pytest.importorskip('imodels')
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(160,2))
    frame=pd.DataFrame(x,columns=columns)
    target=((x[:,0]>0)&(x[:,1]<.5)).astype(int)
    kb,model=induce_skope_rules(frame,columns,target,task='classification',class_names=('negative','positive'),max_rules=100,n_estimators=8,precision_min=.2,recall_min=.01,random_state=seed)
    assert model.__class__.__module__.startswith('imodels')
    _,_,fired=fire_rules(frame,kb)
    aliased=frame.rename(columns=dict(zip(columns,['c0','c1'],strict=True)))
    exported=[(i,r) for i,r in enumerate(kb.rules) if r.consequent=='positive']
    assert exported
    for i,rule in exported:
        rank=int(rule.rule_id.rsplit('_',1)[1])
        original=model.rules_[rank]
        matching=aliased.query(str(original)).index
        expected=frame.index.isin(matching)
        np.testing.assert_array_equal(fired[:,i],expected)
        assert rule.confidence==pytest.approx(original.args[0])
        assert all(p.column in columns for p in rule.antecedents)
    assert any('imodels' in x for x in kb.disclosures)


@pytest.mark.parametrize('text',['c0 > 1 and unsupported()', 'c0 > 1 and alien <= 2'])
def test_parser_refuses_partial_or_unknown_predicates(text):
    with pytest.raises(ValidationError,match='rule'):
        _parse_skope_rule_string(text,['annual income'])
