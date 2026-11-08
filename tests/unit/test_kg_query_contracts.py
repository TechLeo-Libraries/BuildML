"""Known graph oracles cover direction, cycles, filters and bounded paths."""
import copy

import numpy as np
import pytest

from buildml.core.errors import ValidationError
from buildml.kg.query import query_kg
from buildml.kg.results import KgPlan


@pytest.fixture
def plan():
    result = KgPlan(method='transe', head_column='h', relation_column='r', tail_column='t',
        embedding_dim=2, n_train_triples=5, n_entities=5, n_relations=2,
        entity_ids=('a','b','c','d','isolated'), relation_ids=('r','s'))
    result.entity_index_ = {name:i for i,name in enumerate(result.entity_ids)}
    result.relation_index_ = {'r':0, 's':1}
    for head, relation, tail in [(0,0,1),(0,0,1),(0,1,2),(1,0,2),(2,0,3),(3,0,0)]:
        result.out_edges_.setdefault(head, []).append((relation, tail))
        result.in_edges_.setdefault(tail, []).append((relation, head))
    return result


@pytest.mark.parametrize('direction,expected', [
    ('out', (('b','r','out'),('c','s','out'))),
    ('in', (('d','r','in'),)),
    ('both', (('b','r','out'),('c','s','out'),('d','r','in'))),
])
def test_neighbors_preserve_direction_and_deduplicate(plan, direction, expected):
    before = copy.deepcopy((plan.out_edges_, plan.in_edges_))
    result = query_kg(plan, entity='a', direction=direction)
    assert result.results == expected
    assert result.n_results == len(expected)
    assert (plan.out_edges_, plan.in_edges_) == before
    filtered = query_kg(plan, entity='a', direction=direction, relation='r')
    assert filtered.results == tuple(row for row in expected if row[1] == 'r')


@pytest.mark.parametrize('direction,expected', [('out', ('b',)), ('in', ('d',)),
                                               ('both', ('b','d'))])
def test_typed_queries_deduplicate_across_directions(plan, direction, expected):
    result = query_kg(plan, mode='typed', entity='a', relation='r', direction=direction)
    assert result.results == expected
    assert result.n_results == len(expected)


def test_shortest_directed_path_respects_hop_limit_and_cycles(plan):
    short = query_kg(plan, mode='path', source='a', target='d', max_hops=1)
    assert short.n_results == 0 and any('No path' in w for w in short.warnings)
    enough = query_kg(plan, mode='path', source='a', target='d', max_hops=np.int64(2))
    assert enough.results == (('a','s','c'), ('c','r','d'))
    assert enough.n_results == 2
    same = query_kg(plan, mode='path', source='a', target='a')
    assert not same.results and any('empty path' in d for d in same.disclosures)
    unreachable = query_kg(plan, mode='path', source='a', target='isolated', max_hops=10)
    assert not unreachable.results and unreachable.warnings


@pytest.mark.parametrize('options', [
    {'entity':'absent'}, {'entity':'a', 'relation':'absent'},
    {'mode':'typed', 'entity':'absent', 'relation':'r'},
    {'mode':'typed', 'entity':'a', 'relation':'absent'},
    {'mode':'path', 'source':'absent', 'target':'a'},
    {'mode':'path', 'source':'a', 'target':'absent'},
])
def test_unknown_vocabulary_returns_explicit_empty_result(plan, options):
    result = query_kg(plan, **options)
    assert not result.results and result.n_results == 0 and result.warnings


@pytest.mark.parametrize('options', [
    {'mode':'unknown'}, {'direction':'sideways'}, {'entity':None},
    {'mode':'typed', 'entity':'a'}, {'mode':'path', 'source':'a'},
])
def test_invalid_query_contract_is_actionable(plan, options):
    with pytest.raises(ValidationError):
        query_kg(plan, **options)


@pytest.mark.parametrize('hops', [0, -1, True, 1.5, '2', float('nan')])
def test_hop_limit_rejects_coercion_and_nonpositive_values(plan, hops):
    with pytest.raises(ValidationError, match='max_hops'):
        query_kg(plan, mode='path', source='a', target='d', max_hops=hops)
