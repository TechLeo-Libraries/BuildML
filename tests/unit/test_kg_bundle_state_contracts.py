"""Real KG bundles restore numerical state or refuse inconsistent fitted contracts."""

from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.kg.checkpoint import load_kg_bundle, save_kg_bundle
from buildml.kg.models import score_triples_batch


@pytest.fixture(params=[('transe', 'l1'), ('transe', 'l2'), ('distmult', 'l1')])
def fitted(request):
    method, norm = request.param
    frame = pd.DataFrame([(f'e{a}', f'r{r}', f'e{b}') for a in range(6) for b in range(6)
                          for r in range(2) if a != b], columns=['head', 'relation', 'tail'])
    s = Session.ingest(frame).set_roles(dict.fromkeys(frame.columns, 'id'))
    s.split(test_size=.2, validation_size=.2, random_state=7)
    s.kg.fit(backend='native', method=method, norm=norm, head_column='head', relation_column='relation',
             tail_column='tail', embedding_dim=4, epochs=2, random_state=19)
    return s


def _assert_restored(original, restored):
    np.testing.assert_array_equal(restored.entity_embeddings_, original.entity_embeddings_)
    np.testing.assert_array_equal(restored.relation_embeddings_, original.relation_embeddings_)
    expected_out, expected_in, triples = {}, {}, set()
    for h, r, t in zip(original.train_heads_, original.train_relations_, original.train_tails_, strict=True):
        h, r, t = int(h), int(r), int(t)
        triples.add((h, r, t))
        expected_out.setdefault(h, []).append((r, t))
        expected_in.setdefault(t, []).append((r, h))
    assert restored.true_triple_set_ == frozenset(triples)
    assert restored.out_edges_ == expected_out and restored.in_edges_ == expected_in
    assert restored.entity_index_ == {v: i for i, v in enumerate(original.entity_ids)}
    assert restored.relation_index_ == {v: i for i, v in enumerate(original.relation_ids)}


@pytest.mark.parametrize('layout', ['direct', 'redundant', 'derived', 'in_only', 'out_only'])
def test_real_kg_bundle_compatibility_layouts_reconstruct_frozen_state(fitted, layout, tmp_path):
    original = fitted.kg_plan
    root = tmp_path / 'kg'
    fitted.kg.save_bundle(root)
    payload = joblib.load(root / 'kg_plan.joblib')
    plan = payload['plan']
    if layout == 'direct':
        payload = plan
    elif layout == 'redundant':
        plan.entity_embeddings_ = plan.relation_embeddings_ = np.zeros((0, 0))
        plan.train_heads_ = plan.train_relations_ = plan.train_tails_ = np.zeros(0, dtype=np.int64)
        plan.entity_ids = plan.relation_ids = ()
        plan.entity_index_ = {}
        plan.relation_index_ = {}
        plan.true_triple_set_ = frozenset()
        plan.out_edges_ = {}
        plan.in_edges_ = {}
    elif layout == 'derived':
        plan.entity_index_ = {}
        plan.relation_index_ = {}
        plan.true_triple_set_ = frozenset()
        plan.out_edges_ = {}
        plan.in_edges_ = {}
        for key in ('true_triple_set', 'out_edges', 'in_edges'):
            payload.pop(key)
    else:
        setattr(plan, 'in_edges_' if layout == 'in_only' else 'out_edges_', {})
        payload.pop('in_edges' if layout == 'in_only' else 'out_edges')
    joblib.dump(payload, root / 'kg_plan.joblib')
    restored = load_kg_bundle(root, trusted=True)
    _assert_restored(original, restored)
    destination = Session.ingest(fitted.to_pandas()).set_roles(dict.fromkeys(fitted.to_pandas().columns, 'id'))
    destination.inject_split(train_indices=list(fitted.split_plan.train_indices),
                             validation_indices=list(fitted.split_plan.validation_indices),
                             test_indices=list(fitted.split_plan.test_indices))
    destination.kg.load_bundle(root, trusted=True)
    triples = destination.partition('test')
    expected = []
    for row in triples.itertuples(index=False):
        h = original.entity_embeddings_[original.entity_index_[row.head]]
        r = original.relation_embeddings_[original.relation_index_[row.relation]]
        t = original.entity_embeddings_[original.entity_index_[row.tail]]
        expected.append(np.sum(h*r*t) if original.method == 'distmult' else
                        -np.linalg.norm(h+r-t, ord=1 if original.norm == 'l1' else 2))
    scored = destination.kg.score_triples(partition='test')
    np.testing.assert_allclose(scored.scores, expected, atol=1e-10)


FAULTS = ['entity_rows', 'relation_width', 'nonfinite', 'duplicate_ids', 'wrong_index',
          'train_length', 'train_fractional', 'train_range', 'wrong_count', 'true_cache',
          'out_cache', 'in_cache', 'missing_embeddings', 'boolean_count',
          'index_structure', 'edge_structure', 'unhashable_ids']


@pytest.mark.parametrize('fault', FAULTS)
def test_corrupt_real_kg_contract_refuses_before_replacing_session(fitted, fault, tmp_path):
    root = tmp_path / 'kg'
    fitted.kg.save_bundle(root)
    payload = joblib.load(root / 'kg_plan.joblib')
    plan = payload['plan']
    if fault == 'entity_rows':
        plan.entity_embeddings_ = plan.entity_embeddings_[:-1]
    elif fault == 'relation_width':
        plan.relation_embeddings_ = plan.relation_embeddings_[:, :-1]
    elif fault == 'nonfinite':
        plan.entity_embeddings_[0, 0] = np.nan
    elif fault == 'duplicate_ids':
        plan.entity_ids = (*plan.entity_ids[:-1], plan.entity_ids[0])
    elif fault == 'wrong_index':
        plan.entity_index_[plan.entity_ids[0]] = 99
    elif fault == 'train_length':
        plan.train_tails_ = plan.train_tails_[:-1]
    elif fault == 'train_fractional':
        plan.train_heads_ = plan.train_heads_.astype(float)+.25
    elif fault == 'train_range':
        plan.train_relations_[0] = plan.n_relations
    elif fault == 'wrong_count':
        plan.n_train_triples += 1
    elif fault == 'true_cache':
        plan.true_triple_set_ = frozenset({(99, 99, 99)})
    elif fault in ('out_cache', 'in_cache'):
        setattr(plan, 'out_edges_' if fault == 'out_cache' else 'in_edges_', {99: [(99, 99)]})
    elif fault == 'missing_embeddings':
        plan.entity_embeddings_ = payload['entity_embeddings'] = np.zeros((0, 0))
    elif fault == 'boolean_count':
        plan.n_entities = True
    elif fault == 'index_structure':
        plan.entity_index_ = [0]
    elif fault == 'edge_structure':
        plan.out_edges_ = [0]
    else:
        plan.entity_ids = (*plan.entity_ids[:-1], ['unhashable'])
    joblib.dump(payload, root / 'kg_plan.joblib')
    before = dict(vars(fitted))
    fingerprint = joblib.hash(fitted.kg_plan)
    with pytest.raises(ValidationError):
        fitted.kg.load_bundle(root, trusted=True)
    assert vars(fitted).keys() == before.keys()
    assert all(vars(fitted)[name] is value for name, value in before.items())
    assert joblib.hash(fitted.kg_plan) == fingerprint


@pytest.mark.parametrize('method', ['complex', 'rotate'])
def test_frozen_complex_array_payloads_preserve_imaginary_state(fitted, method, tmp_path):
    # Analytic frozen representations exercise serialization/scoring only;
    # this does not invoke or simulate a PyKEEN training backend.
    base = fitted.kg_plan
    entities = base.entity_embeddings_.astype(complex) + 1j*np.roll(base.entity_embeddings_, 1, axis=1)
    relations = (base.relation_embeddings_.astype(complex)+.7j if method == 'complex'
                 else base.relation_embeddings_.copy())
    frozen = replace(base, method=method, embedding_kind=method,
                     entity_embeddings_=entities, relation_embeddings_=relations)
    root = tmp_path / 'complex'
    save_kg_bundle(root, frozen)
    payload = joblib.load(root / 'kg_plan.joblib')
    payload['plan'].entity_embeddings_ = payload['plan'].relation_embeddings_ = np.zeros((0, 0))
    joblib.dump(payload, root / 'kg_plan.joblib')
    restored = load_kg_bundle(root, trusted=True)
    np.testing.assert_array_equal(restored.entity_embeddings_, entities)
    np.testing.assert_array_equal(restored.relation_embeddings_, relations)

    expected = []
    for h, r, t in zip(base.train_heads_, base.train_relations_, base.train_tails_, strict=True):
        head, relation, tail = entities[h], relations[r], entities[t]
        expected.append(float(np.real(np.sum(head*relation*np.conj(tail)))) if method == 'complex'
                        else -float(np.sqrt(np.sum(np.abs(head*np.exp(1j*relation)-tail)**2))))
    scores = score_triples_batch(method, base.train_heads_, base.train_relations_, base.train_tails_,
                                 restored.entity_embeddings_, restored.relation_embeddings_)
    np.testing.assert_allclose(scores, expected, atol=1e-10)
