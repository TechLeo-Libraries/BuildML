"""Real frozen KG ranking agrees with independent equations and strict controls."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError


@pytest.fixture(scope='module', params=[('transe', 'l1'), ('transe', 'l2'), ('distmult', 'l1')])
def fitted(request):
    method, norm = request.param
    frame = pd.DataFrame([(f'e{a}', f'r{r}', f'e{b}') for a in range(6) for b in range(6)
                          for r in range(2) if a != b], columns=['head', 'relation', 'tail'])
    session = (Session.ingest(frame).set_roles(dict.fromkeys(frame.columns, 'id'))
               .split(test_size=.2, validation_size=.2, random_state=7))
    session.kg.fit(backend='native', method=method, norm=norm, head_column='head',
                   relation_column='relation', tail_column='tail', embedding_dim=4,
                   epochs=2, random_state=19)
    return session


def query(plan, mode, oov=False):
    e0, e1, relation = plan.entity_ids[0], plan.entity_ids[1], plan.relation_ids[0]
    if mode == 'tail':
        return dict(mode=mode, heads=[e0, e0, 'outside-entity'] if oov else [e0],
                    relations=[relation, 'outside-relation', relation] if oov else [relation])
    if mode == 'head':
        return dict(mode=mode, tails=[e1, e1, 'outside-entity'] if oov else [e1],
                    relations=[relation, 'outside-relation', relation] if oov else [relation])
    return dict(mode=mode, heads=[e0, e0, 'outside-entity'] if oov else [e0],
                tails=[e1, 'outside-entity', e1] if oov else [e1])


@pytest.mark.parametrize('mode', ['tail', 'head', 'relation'])
@pytest.mark.parametrize('k', [True, np.bool_(False), 1.5, 1., '2', None, np.nan, np.inf, 0, -1])
def test_invalid_ranking_counts_cannot_replace_previous_result(fitted, mode, k):
    plan = fitted.kg_plan
    previous = fitted.kg.predict_links(**query(plan, mode), k=1)
    history = copy.deepcopy(fitted.history)
    entities, relations = plan.entity_embeddings_.copy(), plan.relation_embeddings_.copy()
    with pytest.raises(ValidationError, match='k'):
        fitted.kg.predict_links(**query(plan, mode), k=k)
    assert fitted.kg_predict_result is previous
    assert fitted.history == history
    np.testing.assert_array_equal(plan.entity_embeddings_, entities)
    np.testing.assert_array_equal(plan.relation_embeddings_, relations)


@pytest.mark.parametrize('filtered', ['false', 0, 1, None, np.nan])
def test_non_boolean_filter_cannot_silently_change_candidate_population(fitted, filtered):
    plan = fitted.kg_plan
    previous = fitted.kg.predict_links(**query(plan, 'tail'), k=1)
    history = copy.deepcopy(fitted.history)
    with pytest.raises(ValidationError, match='filtered'):
        fitted.kg.predict_links(**query(plan, 'tail'), k=1, filtered=filtered)
    assert fitted.kg_predict_result is previous
    assert fitted.history == history


@pytest.mark.parametrize('mode', ['tail', 'head', 'relation'])
@pytest.mark.parametrize('filtered', [False, True])
@pytest.mark.parametrize('k', [1, 3, 99, np.int64(2)])
def test_rankings_match_independent_equations_and_train_only_filter(fitted, mode, filtered, k):
    plan = fitted.kg_plan
    entities, relations = plan.entity_embeddings_.copy(), plan.relation_embeddings_.copy()
    known = set(map(tuple, fitted.partition('train')[['head', 'relation', 'tail']].to_numpy()))
    e0, e1, relation = plan.entity_ids[0], plan.entity_ids[1], plan.relation_ids[0]
    candidates = plan.relation_ids if mode == 'relation' else plan.entity_ids
    expected = []
    for candidate in candidates:
        triple = (candidate, relation, e1) if mode == 'head' else (
            (e0, candidate, e1) if mode == 'relation' else (e0, relation, candidate))
        if filtered and triple in known:
            continue
        h, r, t = triple
        h = entities[plan.entity_ids.index(h)]
        r = relations[plan.relation_ids.index(r)]
        t = entities[plan.entity_ids.index(t)]
        if plan.method == 'transe':
            difference = h+r-t
            value = -np.abs(difference).sum() if plan.norm == 'l1' else -np.sqrt((difference**2).sum())
        else:
            value = np.sum(h*r*t)
        expected.append((candidate, value))
    expected.sort(key=lambda pair: -pair[1])
    expected = expected[:int(k)]
    result = fitted.kg.predict_links(**query(plan, mode, oov=True), filtered=filtered, k=k)
    assert result.k == int(k) and result.n_queries == 3
    assert result.predictions[0] == tuple(pair[0] for pair in expected)
    np.testing.assert_allclose(result.scores[0], [pair[1] for pair in expected], atol=1e-12)
    assert result.predictions[1:] == ((), ()) and result.scores[1:] == ((), ())
    assert len(result.warnings) == 2
    np.testing.assert_array_equal(plan.entity_embeddings_, entities)
    np.testing.assert_array_equal(plan.relation_embeddings_, relations)


@pytest.mark.parametrize('options', [
    {'mode': 'unknown'}, {'mode': 'tail'}, {'mode': 'head'}, {'mode': 'relation'},
    {'mode': 'tail', 'heads': ['e0'], 'relations': []},
    {'mode': 'head', 'tails': ['e0'], 'relations': []},
    {'mode': 'relation', 'heads': ['e0'], 'tails': []},
])
def test_invalid_query_shape_preserves_previous_ranking(fitted, options):
    previous = fitted.kg.predict_links(**query(fitted.kg_plan, 'tail'), k=1)
    history = copy.deepcopy(fitted.history)
    with pytest.raises(ValidationError):
        fitted.kg.predict_links(**options)
    assert fitted.kg_predict_result is previous
    assert fitted.history == history


@pytest.mark.parametrize('flag', [np.bool_(False), np.bool_(True)])
def test_numpy_filter_flags_have_serializable_boolean_metadata(fitted, flag):
    result = fitted.kg.predict_links(**query(fitted.kg_plan, 'tail'), filtered=flag, k=np.int64(2))
    data = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert result.filtered is bool(flag)
    assert data['filtered'] is bool(flag)
