"""Frozen graph queries agree with independent scores and train-only adjacency."""
import itertools

import numpy as np
import pandas as pd
import pytest

from buildml import Session


@pytest.fixture(params=['l1', 'l2'])
def graph_session(request):
    triples = [(h, r, t) for h, t in itertools.permutations('abcd', 2) for r in ['r', 's']]
    frame = pd.DataFrame(triples, columns=['head', 'relation', 'tail'])
    session = Session.ingest(frame).set_roles({'head': 'id', 'relation': 'id', 'tail': 'id'}).split(test_size=.25, validation_size=.25, random_state=5)
    session.kg.fit(backend='native', method='transe', head_column='head', relation_column='relation', tail_column='tail', embedding_dim=4, epochs=2, norm=request.param, random_state=6)
    return session


@pytest.mark.parametrize('mode', ['head', 'tail', 'relation'])
@pytest.mark.parametrize('filtered', [False, True])
def test_link_ranks_match_independently_scored_candidates(graph_session, mode, filtered):
    session = graph_session
    plan = session.kg_plan
    row = session.partition('train').iloc[0]
    h, r, t = row['head'], row['relation'], row['tail']
    candidates = plan.relation_ids if mode == 'relation' else plan.entity_ids
    train = set(session.partition('train').itertuples(index=False, name=None))
    expected = []
    for candidate in candidates:
        triple = (candidate, r, t) if mode == 'head' else (h, r, candidate) if mode == 'tail' else (h, candidate, t)
        if filtered and triple in train:
            continue
        hi, ri, ti = plan.entity_index_[triple[0]], plan.relation_index_[triple[1]], plan.entity_index_[triple[2]]
        difference = plan.entity_embeddings_[hi] + plan.relation_embeddings_[ri] - plan.entity_embeddings_[ti]
        score = -float(np.sum(np.abs(difference))) if plan.norm == 'l1' else -float(np.linalg.norm(difference))
        expected.append((candidate, score))
    expected.sort(key=lambda item: item[1], reverse=True)
    arguments = {'mode': mode, 'k': 10, 'filtered': filtered}
    if mode != 'head':
        arguments['heads'] = [h]
    if mode != 'tail':
        arguments['tails'] = [t]
    if mode != 'relation':
        arguments['relations'] = [r]
    before = plan.entity_embeddings_.copy()
    result = session.kg.predict_links(**arguments)
    assert result.predictions[0] == tuple(candidate for candidate, _ in expected)
    np.testing.assert_allclose(result.scores[0], [score for _, score in expected])
    np.testing.assert_array_equal(plan.entity_embeddings_, before)


@pytest.mark.parametrize('direction', ['out', 'in', 'both'])
def test_symbolic_neighbors_use_only_training_edges(graph_session, direction):
    session = graph_session
    entity = session.partition('train').iloc[0]['head']
    expected = set()
    for h, r, t in session.partition('train').itertuples(index=False, name=None):
        if direction in {'out', 'both'} and h == entity:
            expected.add((t, r, 'out'))
        if direction in {'in', 'both'} and t == entity:
            expected.add((h, r, 'in'))
    result = session.kg.query(mode='neighbors', entity=entity, direction=direction)
    assert set(result.results) == expected


def test_oov_queries_and_scores_are_disclosed_without_fabricated_candidates(graph_session):
    session = graph_session
    result = session.kg.predict_links(heads=['missing'], relations=['r'], filtered=False)
    assert result.predictions == ((),)
    assert result.warnings
    scored = session.kg.score_triples(triples=[('missing', 'r', 'a'), ('a', 'missing', 'b')])
    assert scored.unknown_entities == 1 and scored.unknown_relations == 1
    assert np.isneginf(scored.scores).all()
